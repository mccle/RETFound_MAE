import torch
import lightning as L


from torchvision.models import resnet18
from torchmetrics.classification import AUROC, Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse
import torch
import numpy as np
# import os

from torch.utils.tensorboard import SummaryWriter
from util.datasets import build_dataset
from monai.utils import set_determinism
from monai.metrics import ConfusionMatrixMetric, ROCAUCMetric
from models_classifier import ViTClassifier
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18

# os.environ['TORCH_HOME'] = '/autofs/space/crater_001/datasets/private/mee_parkinsons/models/'

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("output_dir", metavar="output-dir", type=Path)
# parser.add_argument("--num-classes", type=int, default=1)
parser.add_argument("--hidden-features", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="GELU")
parser.add_argument("--vit-architecture", type=str, default="vit_large_patch16")
parser.add_argument("--input-size", nargs=3, type=int, default=(3, 512, 1024))
parser.add_argument("--vit-classes", type=int, default=1000)
parser.add_argument("--vit-drop", type=float, default=0.1)
parser.add_argument("--vit-non-global-pool", action="store_false")
parser.add_argument("--vit-weights", type=str, default='/autofs/space/crater_001/datasets/private/mee_parkinsons/models/RETFound_cfp_weights.pth')
parser.add_argument("--resnet", action="store_true")
parser.add_argument("--frozen", action="store_true")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--lr-step", type=int, default=15)
parser.add_argument("--lr-factor", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=4e-6)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-batches", type=int, default=200)
parser.add_argument('--num-workers', default=10, type=int)

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


class LightningResNet(L.LightningModule):
    def __init__(
            self,
            num_classes: int = 1,
            lr: float = 1e-5,
            wd: float = 4e-6,
            lr_step: int = 15,
            lr_factor: float = 0.1,
            frozen: bool = False
    ):
        super().__init__()

        weights = torch.load("/autofs/space/crater_001/datasets/private/mee_parkinsons/models/resnet18-imagenet.pth", map_location="cpu")
        model = resnet18()
        model.load_state_dict(weights)
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)

        if frozen:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True


        self.model = model
        self.loss_fn = torch.nn.BCEWithLogitsLoss() # torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.wd = wd
        self.lr_step = lr_step
        self.lr_factor = lr_factor

        if num_classes == 1:
            self.train_auroc = AUROC(task="binary")
            self.val_auroc = AUROC(task="binary")
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
        else:
            self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples, targets = batch

        logits = self.model(samples)
        # targets = targets.reshape(logits.shape)
        loss = self.loss_fn(logits, targets)

        self.train_auroc.update(logits, targets)
        self.train_acc.update(logits, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

        self.log("train/acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()


    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        logits = self.model(samples)
        # targets = targets.reshape(logits.shape)
        loss = self.loss_fn(logits, targets)

        self.val_auroc.update(logits, targets)
        self.val_acc.update(logits, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step,
            gamma=self.lr_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

def main():
    args = parser.parse_args()

    set_determinism(args.seed)

    model = LightningResNet(
        lr=args.lr,
        wd=args.wd,
        lr_step=args.lr_step,
        lr_factor=args.lr_factor,
        frozen=args.frozen
    )

    #
    # print("Loaded initial Model Weights")
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    dataset_train = build_dataset(partition='train', args=args)
    dataset_val = build_dataset(partition='val', args=args)

    train_labels = dataset_train.labels  #[x["label"] for x in train_dicts]
    train_label_type = [torch.tensor([1.]), torch.tensor([0.])] #list(set(train_labels))
    train_class_count = {str(_k): train_labels.count(_k) for _k in train_label_type}
    weights = {_k: 1.0 / train_class_count[_k] for _k in train_class_count.keys()}
    samples_weights = torch.tensor(
        [weights[str(x)] for x in tqdm(train_labels, desc="Assigning Label Weights")], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        # shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = TensorBoardLogger(save_dir=tb_dir, name="lightning_resnet")

    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir,
        monitor="val/auroc",
        mode="max",
        save_top_k=3,
        filename="{epoch:02d}-{auroc:.4f}"
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=tb_writer,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=[0],
        # limit_train_batches=250,
    )

    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)


if __name__ == "__main__":
    main()
