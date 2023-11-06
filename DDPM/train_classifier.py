import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10, STL10


def get_dataset(args):
    """
    Returns vanilla CIFAR10/STL10 dataset (modified with train test splitting)
    """
    transform = transforms.Compose(
        [
            # transforms.Resize(32 if args.dataset == "cifar10" else 64),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if args.dataset == "cifar10":
        train_dataset = CIFAR10(
            args.data_path,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = CIFAR10(
            args.data_path,
            train=False,
            download=True,
            transform=transform,
        )

    elif args.dataset == "stl10":
        # for STL10 use both train and test sets due to its small size
        train_dataset = STL10(
            args.data_path,
            split="train",
            download=True,
            transform=transform,
        )
        test_dataset = STL10(
            args.data_path,
            split="test",
            download=True,
            transform=transform,
        )
        dataset = ConcatDataset([train_dataset, test_dataset])
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [12000, 1000]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


def get_dataset_image_folder(args):
    transform = transforms.Compose(
        [
            # transforms.Resize(32 if args.dataset == "cifar10" else 64),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(args.data_path, transform=transform)
    print(len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "stl10"], help="Dataset type")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to dataset"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--freeze_layers",
        type=bool,
        default=False,
        help="Freeze the convolution layers or not",
    )
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = torchvision.models.resnet34(pretrained=True)

    # Freeze all layers except the final layer
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    weight_decay = 5e-4

    # params_1x are the parameters of the network body, i.e., of all layers except the FC layers
    params_1x = [
        param for name, param in model.named_parameters() if "fc" not in str(name)
    ]
    optimizer = torch.optim.Adam(
        [{"params": params_1x}, {"params": model.fc.parameters(), "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=weight_decay,
    )

    train_loader, test_loader = get_dataset(args)

    # Train the model
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and labels
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print the accuracy on the test set
        print(
            "Accuracy on the test set after epoch %d: %d %%"
            % (epoch + 1, 100 * correct / total)
        )

    print("Finished fine-tuning")
    torch.save(model.state_dict(), f"{args.dataset}_resnet34.pth")
