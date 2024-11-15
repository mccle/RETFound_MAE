# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import numpy as np
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pandas as pd
import PIL
import logging
# from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm
# from torchvision import transforms

class CSVDataset(VisionDataset):
    def __init__(self, csv, partition, transform=None, target_transform=None):
        super(CSVDataset, self).__init__(
            root=None,
            transform=transform,
            target_transform=target_transform
        )

        df = pd.read_csv(csv, dtype=str)

        if "partition" not in df.keys():
            df["partition"] = [partition] * df.shape[0]

        partition_df = df[df["partition"] == partition]

        self.file_paths = list(partition_df["jpgfile"])

        if "label" in partition_df.keys():
            self.labels = [torch.tensor([float(l)]) for l in partition_df["label"]]

        else:
            self.labels = [torch.tensor([1.0])] * len(self.file_paths)

        assert len(self.file_paths) == len(self.labels), "Mismatch between number of files and labels"

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            img = PIL.Image.open(file_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

        except Exception as error:
            logging.warning(f"{file_path} could not be opened due to {error}")

            img = PIL.Image.fromarray(np.ones((224,224))).convert("RGB")

            if self.transform:
                img = self.transform(img)

                img[:] = float("NaN")

        if self.target_transform and label is not None:
            label = self.target_transform(label)

        return img, label



def build_dataset(partition, args):

    is_train = partition == "train"

    transform = build_transform(is_train, args)
    dataset = CSVDataset(csv=args.csv, partition=partition, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
