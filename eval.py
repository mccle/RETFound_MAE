import argparse
import torch

from csv import DictWriter
from util.datasets import build_dataset
from monai.utils import set_determinism
from models_classifier import ViTClassifier
from models_vit import TIMMVisionTransformer
from pathlib import Path
from tqdm import tqdm


# os.environ['TORCH_HOME'] = '/autofs/space/crater_001/datasets/private/mee_parkinsons/models/'

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("checkpoint_dir", metavar="checkpoint-dir", type=Path)
parser.add_argument("output_csv", metavar="output-csv", type=Path)
# parser.add_argument("--num-classes", type=int, default=1)
parser.add_argument("--hidden-features", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="GELU")
parser.add_argument("--vit-architecture", type=str, default="vit_large_patch16")
parser.add_argument("--input-size", nargs=3, type=int, default=(3, 512, 1024))
parser.add_argument("--pretrain-size", nargs=3, type=int, default=(3, 512, 1024))
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
parser.add_argument("--batch-size", type=int, default=1)
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
parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


def main():
    args = parser.parse_args()

    checkpoints = list(args.checkpoint_dir.glob("**/*.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = [torch.load(checkpoint, map_location="cpu")["model"].to(device).eval() for checkpoint in checkpoints]

    for i, (checkpoint, model) in enumerate(zip(checkpoints, models)):
        if isinstance(model, TIMMVisionTransformer):
            model = ViTClassifier(
                num_classes=1, # args.num_classes,
                hidden_features=args.hidden_features,
                dropout=args.dropout,
                activation=args.activation,
                vit_arch=args.vit_architecture,
                vit_kwargs={
                    "img_size": args.input_size[1:],
                    "num_classes": args.vit_classes,
                    "drop_path_rate": args.vit_drop,
                    "global_pool": args.vit_non_global_pool
                },
                vit_weights=checkpoint, # args.vit_weights,
                freeze_feature_extraction=args.freeze_vit,
                train_img_size=args.pretrain_size[1:]
            )

            models[i] = model.to(device).eval()

    set_determinism(args.seed)

    dataset = build_dataset(partition='all', args=args, return_fname=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    writer = DictWriter(
        args.output_csv.open("w", newline=""),
        fieldnames=["jpgfile", "label", *[str(checkpoint) for checkpoint in checkpoints]]
    )

    writer.writeheader()

    for (ims, labels, names) in tqdm(data_loader, desc="Evaluating Models"):
        rows = [{} for _ in range(ims.size(0))]
        ims = ims.to(device)

        for (checkpoint, model) in zip(checkpoints, models):
            # model = model.to(device)
            outputs = model(ims)
            # model = model.to("cpu")

            for (row, out, label, name) in zip(rows, outputs, labels, names):
                row["jpgfile"] = name
                row["label"] = label.item()
                row[str(checkpoint)] = out.item()


        writer.writerows(rows)

    print(f"Results written to {args.output_csv}")



if __name__ == "__main__":
    main()
