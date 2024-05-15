from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.dataset.features["label"].names)
        }
        self.file_to_class = {
            str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))
        }

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label


class NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


class NOT_NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/not-nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def setup_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", class_to_forget, transform)
    # train_set = Imagenette('train', transform)

    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_ga_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] == class_to_forget]

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] != class_to_forget]

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] == class_to_forget]
    train_dl = DataLoader(filtered_data, batch_size=batch_size)
    return train_dl, descriptions


def setup_forget_nsfw_data(batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFW(transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NOT_NSFW(transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl
