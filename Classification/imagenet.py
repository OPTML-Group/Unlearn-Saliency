import os
import sys

import torch
import torchvision
from datasets.load import load_dataset
from torch.utils.data import DataLoader, Subset

# sys.path.append(".")
# from cfg import *
from tqdm import tqdm


def prepare_data(
    dataset,
    batch_size=512,
    shuffle=True,
    train_subset_indices=None,
    val_subset_indices=None,
    data_path="/localscratch/dataset",
):
    path = os.path.join(data_path, "huggingface")
    if dataset == "imagenet":
        train_set = load_dataset(
            "imagenet-1k", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "imagenet-1k", use_auth_token=True, split="validation", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomResizedCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    elif dataset == "tiny_imagenet":
        train_set = load_dataset(
            "Maysee/tiny-imagenet", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "Maysee/tiny-imagenet", use_auth_token=True, split="valid", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomCrop(64, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    elif dataset == "flowers102":
        train_set = load_dataset(
            "nelorth/oxford-flowers", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "nelorth/oxford-flowers", use_auth_token=True, split="test", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.RandomCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    else:
        raise NotImplementedError
    train_set.set_transform(transform=train_transform)
    validation_set.set_transform(transform=validation_transform)

    if train_subset_indices is not None:
        forget_indices = torch.ones_like(train_subset_indices) - train_subset_indices
        train_subset_indices = torch.nonzero(train_subset_indices)

        forget_indices = torch.nonzero(forget_indices)
        retain_set = Subset(train_set, train_subset_indices)
        forget_set = Subset(train_set, forget_indices)
    if val_subset_indices is not None:
        val_subset_indices = torch.nonzero(val_subset_indices)
        validation_set = Subset(validation_set, val_subset_indices)
    if train_subset_indices is not None:
        loaders = {
            "train": DataLoader(
                retain_set, batch_size=batch_size, num_workers=12, shuffle=shuffle
            ),
            "val": DataLoader(
                validation_set, batch_size=batch_size, num_workers=12, shuffle=shuffle
            ),
            "fog": DataLoader(
                forget_set, batch_size=batch_size, num_workers=12, shuffle=shuffle
            ),
        }
    else:
        loaders = {
            "train": DataLoader(
                train_set, batch_size=batch_size, num_workers=12, shuffle=shuffle
            ),
            "val": DataLoader(
                validation_set, batch_size=batch_size, num_workers=12, shuffle=shuffle
            ),
        }
    return loaders


def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    ys = {}
    ys["train"] = []
    ys["val"] = []
    loaders = prepare_data(
        dataset="imagenet", batch_size=1, shuffle=False, data_path="./data"
    )
    for data in tqdm(loaders["val"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["val"].append(y.item())
    for data in tqdm(loaders["train"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["train"].append(y.item())
    ys["train"] = torch.Tensor(ys["train"]).long()
    ys["val"] = torch.Tensor(ys["val"]).long()
    torch.save(ys["train"], "train_ys.pth")
    torch.save(ys["val"], "val_ys.pth")
