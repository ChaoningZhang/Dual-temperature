# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from argparse import ArgumentParser
from pathlib import Path


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
        "custom",
    ]

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--train_dir", type=Path, default=None)
    parser.add_argument("--val_dir", type=Path, default=None)

    # dali (imagenet-100/imagenet/custom only)
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--dali_device", type=str, default="gpu")

    # custom dataset only
    parser.add_argument("--no_labels", action="store_true")


def augmentations_args(parser: ArgumentParser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--num_crops", type=int, default=2)
    parser.add_argument("--num_small_crops", type=int, default=0)

    # augmentations
    parser.add_argument("--brightness", type=float, required=True, nargs="+")
    parser.add_argument("--contrast", type=float, required=True, nargs="+")
    parser.add_argument("--saturation", type=float, required=True, nargs="+")
    parser.add_argument("--hue", type=float, required=True, nargs="+")
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")

    # for imagenet or custom dataset
    parser.add_argument("--size", type=int, default=[224], nargs="+")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")
