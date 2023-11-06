import argparse
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def validate(model, loader, args):
    n_samples = len(loader.dataset)
    entropy_cum_sum = 0
    forgotten_prob_cum_sum = 0
    accuracy_cum_sum = 0
    model.eval()
    for data in tqdm(iter(loader)):
        logits = model(data.to(device))

        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == args.label_of_forgotten_class).sum()
        accuracy_cum_sum += accuracy / n_samples

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy) / n_samples
        entropy_cum_sum += avg_entropy.item()
        forgotten_prob_cum_sum += (
            (probs[:, args.label_of_forgotten_class] / n_samples).sum().item()
        )

    print(f"Average entropy: {entropy_cum_sum}")
    print(f"Average prob of forgotten class: {forgotten_prob_cum_sum}")
    print(f"Average accuracy of forgotten class: {accuracy_cum_sum}")

    # Check if the CSV file exists
    csv_file_path = "results/cifar10/forget/result.csv"
    try:
        df = pd.read_csv(csv_file_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame()

    name = args.sample_path.split("/")[-4] + "/" + args.sample_path.split("/")[-3]
    result = {
        "entropy": float(entropy_cum_sum),
        "prob of forgotten class": float(forgotten_prob_cum_sum),
        "accuracy of forgotten class": float(accuracy_cum_sum.cpu()),
    }

    if name not in df.index:
        new_row = pd.DataFrame(result, index=[name])
        df = pd.concat([df, new_row])
    else:
        for metric, value in result.items():
            df.at[name, metric] = value
    print(df)
    # Save the updated DataFrame to CSV
    df.to_csv(csv_file_path)


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


def GetImageFolderLoader(path, data_type, img_size, batch_size):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    dataset = ImagePathDataset(
        path,
        transforms=transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size)

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data related settings
    parser.add_argument(
        "--sample_path", type=str, help="Path to folder containing samples"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "stl10"],
        help="name of the dataset, either cifar10 or stl10",
    )
    parser.add_argument(
        "--label_of_forgotten_class",
        type=int,
        default=0,
        help="Class label of forgotten class (for calculating average prob)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, help="test batch size for inference"
    )
    args = parser.parse_args()

    model = torchvision.models.resnet34(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(
        torch.load(f"{args.dataset}_resnet34.pth", map_location="cpu")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = GetImageFolderLoader(args.sample_path, args.dataset, 224, args.batch_size)

    validate(model, loader, args)
