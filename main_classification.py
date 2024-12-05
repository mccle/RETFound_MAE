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
parser.add_argument("--input-size", type=int, default=224)
parser.add_argument("--vit-classes", type=int, default=1000)
parser.add_argument("--vit-drop", type=float, default=0.1)
parser.add_argument("--vit-non-global-pool", action="store_false")
parser.add_argument("--vit-weights", type=str, default='/autofs/space/crater_001/datasets/private/mee_parkinsons/models/RETFound_cfp_weights.pth')
parser.add_argument("--resnet", action="store_true")
parser.add_argument("--freeze-vit", action="store_true")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--lr-step", type=int, default=15)
parser.add_argument("--lr-factor", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=4e-6)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=80)
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


def main():
    args = parser.parse_args()

    set_determinism(args.seed)

    if args.resnet:
        weights = torch.load("/autofs/space/crater_001/datasets/private/mee_parkinsons/models/resnet18-imagenet.pth", map_location="cpu")
        model = resnet18() # weights=weights)
        model.load_state_dict(weights)
        model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

    else:
        model = ViTClassifier(
            num_classes=1, # args.num_classes,
            hidden_features=args.hidden_features,
            dropout=args.dropout,
            activation=args.activation,
            vit_arch=args.vit_architecture,
            vit_kwargs={
                "img_size": args.input_size,
                "num_classes": args.vit_classes,
                "drop_path_rate": args.vit_drop,
                "global_pool": args.vit_non_global_pool
            },
            vit_weights=args.vit_weights,
            freeze_feature_extraction=args.freeze_vit
        )

    print("Loaded initial Model Weights")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_factor
    )

    balanced_accuracy = ConfusionMatrixMetric("balanced accuracy")
    auroc = ROCAUCMetric()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / "tb")

    best_auroc = 0
    train_iter = 1
    val_iter = 1

    for epoch in range(1, args.epochs + 1):
        metrics = {}

        train_losses = []
        model.train()
        for samples, targets in tqdm(data_loader_train, desc=f"Training Epoch {epoch}"):
            samples, targets = samples[~torch.isnan(targets).resize(targets.size(0))].to(device), targets[~torch.isnan(targets)].to(device)

            if len(samples) == 0:
                continue

            optimizer.zero_grad()
            logits = model(samples)
            targets = targets.resize(*logits.size())
            loss = loss_fn(logits, targets)
            train_losses.append(loss.item())
            writer.add_scalar("IterLoss/train", train_losses[-1], train_iter)
            train_iter += 1

            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)

            balanced_accuracy((probs > 0.5).float(), targets)
            auroc(probs, targets)

        metrics["MeanLoss/train"] = np.mean(train_losses)

        metrics["BalancedAccuracy/train"] = balanced_accuracy.aggregate()[0].item()
        balanced_accuracy.reset()

        metrics["AUROC/train"] = auroc.aggregate()
        auroc.reset()

        val_losses = []

        with torch.no_grad():
            model.eval()
            for samples, targets in tqdm(data_loader_val, desc=f"Validating Epoch {epoch}"):
                samples, targets = samples[~torch.isnan(targets).resize(targets.size(0))].to(device), targets[~torch.isnan(targets)].to(device)

                if len(samples) == 0:
                    continue

                logits = model(samples)
                targets = targets.resize(*logits.size())
                loss = loss_fn(logits, targets)
                val_losses.append(loss.item())
                writer.add_scalar("IterLoss/val", val_losses[-1], val_iter)
                val_iter += 1

                probs = torch.sigmoid(logits)

                balanced_accuracy((probs > 0.5).float(), targets)
                auroc(probs, targets)

        metrics["MeanLoss/val"] = np.mean(val_losses)

        metrics["BalancedAccuracy/val"] = balanced_accuracy.aggregate()[0].item()
        balanced_accuracy.reset()

        metrics["AUROC/val"] = auroc.aggregate()
        auroc.reset()


        print(f"Finished epoch {epoch} with {metrics}")

        print("Saving metrics and model checkpoints")

        for metric in ["MeanLoss", "BalancedAccuracy", "AUROC"]:
            writer.add_scalars(
                metric,
                {l: metrics[f"{metric}/{l}"] for l in ["train", "val"]},
                epoch
            )

        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        lr_scheduler.step()

        torch.save(
            {
                "epoch": epoch,
                "model": model,
                "optimizer": optimizer,
                "AUROC/val": metrics["AUROC/val"],
            },
            checkpoint_dir / "last_checkpoint.pt"
        )

        if metrics["AUROC/val"] >= best_auroc:
            best_auroc = metrics["AUROC/val"]

            torch.save(
                {
                    "epoch": epoch,
                    "model": model,
                    "optimizer_state_dict": optimizer,
                    "AUROC/val": metrics["AUROC/val"],
                },
                checkpoint_dir / "best_checkpoint.pt"
            )


if __name__ == "__main__":
    main()
