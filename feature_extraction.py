import torch
import torch.nn as nn
import models_vit

from timm.models.layers import trunc_normal_
from util.pos_embed import interpolate_pos_embed
from pathlib import Path
from typing import Literal


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



num_classes = 1
hidden_features = 512
dropout = 0.2
activation = "GELU"
vit_arch = "vit_large_patch16"
vit_kwargs = {
    # "input_size": 224,
    "num_classes": 1000,
    "drop_path_rate": 0.1,
    "global_pool": True
}
vit_weights = '/autofs/space/crater_001/datasets/private/mee_parkinsons/models/mee_foundation_weights.pth'

model = getattr(models_vit, vit_arch)(**vit_kwargs)

if vit_weights:
    checkpoint = torch.load(vit_weights, map_location="cpu")

    print(f"Loading ViT weights from {vit_weights}")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)

    if vit_kwargs.get("global_pool", False):
        assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
    else:
        assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)


model.head = torch.nn.Identity()


parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("output_dir", metavar="output-dir", type=Path)
parser.add_argument("--partition", choices=["val", "test", "all"], default="all")
parser.add_argument("--batch-size", type=int, default=80)
parser.add_argument('--num-workers', default=10, type=int)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--input-size", type=int, default=224)

def main():
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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

    writer = DictWriter(
        (args.output_dir / "summary.csv").open("w", newline=""),
        fieldnames=["features", "jpgfile"]
    )

    writer.writeheader()

    with torch.no_grad():
        model.eval()
        for samples, targets in tqdm(data_loader_test, desc="Running inference on test set"):
            samples, targets = samples.to(device), targets.to(device)

            features = model(samples)

            for f, t in zip(features, targets):
                jpg = file_paths.pop(0)

                if not torch.isnan(t):

                    save_path = output_dir / Path(jpg).name.replace(".jpg", "_features.pt")

                    torch.save(f, save_path)

                    row = {
                        "features": save_path,
                        "jpgfile": jpg
                    }

                    writer.writerow(row)

                #rows.append(row)

            # break

    #print(rows)


if __name__ == "__main__":
    main()

