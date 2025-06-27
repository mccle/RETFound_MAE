# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import numpy as np
import torch
import pandas as pd
import PIL
import logging

from torchvision import transforms #, datasets
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import VisionDataset
# from tqdm import tqdm
from timm.data.transforms import CenterCropOrPad

class CSVDataset(VisionDataset):
    def __init__(self, csv, partition, transform=None, target_transform=None, return_fname=False):
        super(CSVDataset, self).__init__(
            root=None,
            transform=transform,
            target_transform=target_transform
        )
        self.return_fname = return_fname

        df = pd.read_csv(csv, dtype=str)

        if "partition" not in df.keys():
            df["partition"] = [partition] * df.shape[0]

        if partition != "all":
            partition_df = df[df["partition"] == partition]

        else:
            partition_df = df

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

                # img[:] = float("NaN")

            label = torch.tensor([float("NaN")])

            return img, label

        if self.target_transform and label is not None:
            label = self.target_transform(label)

        if self.return_fname:
            return img, label, file_path

        return img, label


def build_dataset(partition, args, return_fname=False):

    is_train = partition == "train"

    transform = build_transform(is_train, args)
    dataset = CSVDataset(csv=args.csv, partition=partition, transform=transform, return_fname=return_fname)

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

    return transforms.Compose([
        # CenterCropOrPad(args.input_size[1:]),
        transforms.Resize(args.input_size[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

