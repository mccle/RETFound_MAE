import argparse
import torch
# import numpy as np
from csv import DictWriter


from torch.utils.tensorboard import SummaryWriter
from util.datasets import build_dataset
from monai.utils import set_determinism
from monai.metrics import ConfusionMatrixMetric, ROCAUCMetric
from models_classifier import ViTClassifier
from models_vit import vit_large_patch16
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("checkpoint")
parser.add_argument("output_csv", metavar="output-csv", type=Path)
parser.add_argument("--partition", choices=["val", "test", "all"], default="test")
parser.add_argument(
    "--classifier-architecture",
    type=str,
    default="ViTClassifier",
    choices=[
        "ViTClassifier",
        "RETFoundClassifier",
        "resnet18"
    ]
)
parser.add_argument("--hidden-features", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="GELU")
parser.add_argument("--vit-architecture", type=str, default="vit_large_patch16")
parser.add_argument("--input-size", type=int, default=224)
parser.add_argument("--vit-classes", type=int, default=1000)
parser.add_argument("--vit-drop", type=float, default=0.1)
parser.add_argument("--vit-non-global-pool", action="store_false")
parser.add_argument("--vit-weights", type=str, default='/autofs/space/crater_001/datasets/private/mee_parkinsons/models/RETFound_cfp_weights.pth')
#parser.add_argument("--resnet", action="store_true")
parser.add_argument("--freeze-vit", action="store_true")
parser.add_argument("--batch-size", type=int, default=80)
parser.add_argument('--num-workers', default=10, type=int)
parser.add_argument("--gpu", action="store_true")

def main():
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # print(checkpoint.keys())

    if args.classifier_architecture == "ViTClassifier": #, "resnet18"]:
        # model = ViTClassifier(
        #     num_classes=1, # args.num_classes,
        #     hidden_features=args.hidden_features,
        #     dropout=args.dropout,
        #     activation=args.activation,
        #     vit_arch=args.vit_architecture,
        #     vit_kwargs={
        #         "img_size": args.input_size,
        #         "num_classes": args.vit_classes,
        #         "drop_path_rate": args.vit_drop,
        #         "global_pool": args.vit_non_global_pool
        #     },
        #     vit_weights=args.vit_weights,
        #     freeze_feature_extraction=args.freeze_vit
        # )
        # model.load_state_dict(checkpoint["model_state_dict"])

        model = checkpoint["model"]

    elif args.classifier_architecture == "resnet18":
        model = resnet18()
        model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        model.load_state_dict(checkpoint["model_state_dict"])


    elif args.classifier_architecture == "RETFoundClassifier":
        model = vit_large_patch16(
            num_classes=2,
            drop_path_rate=0.2,
            global_pool=True,
        )

        model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    model.to(device)

    dataset_test = build_dataset(partition=args.partition, args=args)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    file_paths = deepcopy(dataset_test.file_paths)
    # rows = []

    writer = DictWriter(
        args.output_csv.open("w", newline=""),
        fieldnames=["logits", "label", "jpgfile"]
    )

    writer.writeheader()

    with torch.no_grad():
        model.eval()
        for samples, targets in tqdm(data_loader_test, desc="Running inference on test set"):
            samples, targets = samples.to(device), targets.to(device)

            logits = model(samples)

            for l, t in zip(logits, targets):
                row = {
                    "logits": l.tolist(),
                    "label": t.tolist(),
                    "jpgfile": file_paths.pop(0)
                }

                writer.writerow(row)

                #rows.append(row)

            # break

    #print(rows)


if __name__ == "__main__":
    main()


