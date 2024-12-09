import argparse
import torch

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
from csv import DictWriter
from util.datasets import build_dataset
from models_vit import vit_large_patch16
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("checkpoint")
parser.add_argument("--output-csv", metavar="output-csv", type=Path, default=None)
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

    if args.classifier_architecture in ["ViTClassifier", "resnet18"]:
        model = checkpoint["model"]

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

    output_csv = args.output_csv

    if output_csv:
        writer = DictWriter(
            args.output_csv.open("w", newline=""),
            fieldnames=["logits", "label", "jpgfile"]
        )

        writer.writeheader()

    auroc = ROCAUCMetric()
    accuracy = ConfusionMatrixMetric(metric_name="accuracy")
    balanced_accuracy = ConfusionMatrixMetric(metric_name="balanced_accuracy")


    with torch.no_grad():
        model.eval()
        for samples, labels in tqdm(data_loader_test, desc=f"Running inference on {args.partition} partition with {device}"):
            samples, labels = samples.to(device), labels.to(device)

            batch_logits = model(samples)

            for logits, label in zip(batch_logits, labels):
                jpg = file_paths.pop(0)

                # skip unreadable files
                if torch.isnan(label):
                    continue

                if output_csv:
                    row = {
                        "logits": logits.tolist(),
                        "label": label.tolist(),
                        "jpgfile": jpg
                    }

                    writer.writerow(row)

                if logits.size(0) == 1:
                    logits = logits.reshape(1, 1)
                    label = label.reshape(1, 1)

                    auroc(torch.sigmoid(logits), label)
                    accuracy(torch.round(torch.sigmoid(logits)), label)
                    balanced_accuracy(torch.round(torch.sigmoid(logits)), label)

                elif logits.size(0) == 2:
                    logits = logits.reshape(1, 2)
                    label = label.reshape(1, 1)

                    auroc(torch.softmax(logits, dim=1)[..., 0], label)
                    accuracy(torch.argmax(torch.softmax(logits, dim=1)).reshape((1, 1)), label)
                    balanced_accuracy(torch.argmax(torch.softmax(logits, dim=1)).reshape((1, 1)), label)


    print({
        "auroc": auroc.aggregate(),
        "accuracy": accuracy.aggregate()[0].item(),
        "balanced_accuracy": balanced_accuracy.aggregate()[0].item(),
    })



if __name__ == "__main__":
    main()


