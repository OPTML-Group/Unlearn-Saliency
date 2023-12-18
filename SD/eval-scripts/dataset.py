import os
import numpy as np
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
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
            torch_transforms.Resize((size, size), interpolation=interpolation),
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


class Fake_Imagenette(Dataset):
    def __init__(self, data_dir, class_to_forget, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get all image files in the data folder
        self.image_files = [
            f
            for f in os.listdir(data_dir)
            if (f.endswith(".png") and not f.startswith(str(class_to_forget)))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Parse class index from the filename
        filename = self.image_files[idx]
        # print(filename)
        class_idx = int(filename.split("/")[-1].split("_")[0])

        # Load the image
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, class_idx


def setup_fid_data(class_to_forget, path, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    real_set = Fake_Imagenette(
        "imagenette_without_label_6", class_to_forget, transform=transform
    )
    real_set = [data[0] for data in real_set]

    fake_set = Fake_Imagenette(path, class_to_forget, transform=transform)
    fake_set = [data[0] for data in fake_set]

    return real_set, fake_set
