import os
import pathlib

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageFolder

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    """
    Returns vanilla CIFAR10/STL10 dataset.
    """
    if config.data.random_flip is False:
        tran_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            config.data.path,
            train=True,
            download=True,
            transform=tran_transform,
        )

    elif config.data.dataset == "STL10":
        # for STL10 use both train and test sets due to its small size
        train_dataset = STL10(
            config.data.path,
            split="train",
            download=True,
            transform=tran_transform,
        )
        test_dataset = STL10(
            config.data.path,
            split="test",
            download=True,
            transform=tran_transform,
        )
        dataset = ConcatDataset([train_dataset, test_dataset])

    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    return train_loader


def all_but_one_class_path_dataset(config, data_path, label_to_drop):
    """
    Returns all classes but one from a folder with labels,
    e.g.,
    ./folder
        - /0
        - /1
        - /2
        etc..
    """
    if config.data.random_flip is False:
        transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    train_dataset = ImageFolder(
        data_path,
        transform=transform,
    )

    train_idx = find_indices(train_dataset.targets, label_to_drop)
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
    )

    return train_loader


def get_forget_dataset(args, config, label_to_drop):
    """
    Returns vanilla CIFAR10/STL10 dataset.
    """
    if config.data.random_flip is False:
        tran_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            config.data.path,
            train=True,
            download=True,
            transform=tran_transform,
        )

    elif config.data.dataset == "STL10":
        # for STL10 use both train and test sets due to its small size
        train_dataset = STL10(
            config.data.path,
            split="train",
            download=True,
            transform=tran_transform,
        )
        test_dataset = STL10(
            config.data.path,
            split="test",
            download=True,
            transform=tran_transform,
        )
        dataset = ConcatDataset([train_dataset, test_dataset])

    data_remain = [data for data in dataset if data[1] != label_to_drop]
    data_forget = [data for data in dataset if data[1] == label_to_drop]
    print(len(data_remain), len(data_forget))

    remain_loader = DataLoader(
        data_remain,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    forget_loader = DataLoader(
        data_forget,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    return remain_loader, forget_loader


def all_but_one_class_dataset(config, label_to_drop):
    if config.data.random_flip is False:
        tran_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    if config.data.dataset == "CIFAR10":
        train_dataset = CIFAR10(
            config.data.path,
            train=True,
            download=True,
            transform=tran_transform,
        )

        train_idx = find_indices(train_dataset.targets, label_to_drop)
        dataset = torch.utils.data.Subset(train_dataset, train_idx)

    elif config.data.dataset == "STL10":
        # for STL10 use both train and test sets due to its small size
        train_dataset = STL10(
            config.data.path,
            split="train",
            download=True,
            transform=tran_transform,
        )
        test_dataset = STL10(
            config.data.path,
            split="test",
            download=True,
            transform=tran_transform,
        )

        train_idx = find_indices(train_dataset.labels, label_to_drop)
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        test_idx = find_indices(test_dataset.labels, label_to_drop)
        test_subset = torch.utils.data.Subset(test_dataset, test_idx)
        dataset = ConcatDataset([train_subset, test_subset])

    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    return train_loader


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, n=None):
        self.transforms = transforms

        path = pathlib.Path(img_folder)
        self.files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )

        assert n is None or n <= len(self.files)
        self.n = len(self.files) if n is None else n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if elem != condition]


def find_forget_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if elem == condition]
