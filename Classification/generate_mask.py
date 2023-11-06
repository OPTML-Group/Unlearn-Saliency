import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from scipy.special import erf
from trainer import validate


def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.train()

    for name, param in model.named_parameters():
        gradients[name] = 0

    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient = param.grad.data.abs()
                    gradients[name] += gradient

    else:
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # gradient = torch.norm(param.grad.data)
                    # print(param.grad.data.abs())

                    gradient = param.grad.data.abs()
                    gradients[name] += gradient

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        all_gradients = torch.cat(
            [gradient.flatten() for gradient in gradients.values()]
        )

        sigmoid_gradients = torch.abs(2 * (torch.sigmoid(all_gradients) - 0.5))
        tanh_gradients = torch.abs(torch.tanh(all_gradients))

        sigmoid_soft_dict = {}
        tanh_soft_dict = {}
        start_idx = 0
        for net_name, gradient in gradients.items():
            num_params = gradient.numel()
            end_idx = start_idx + num_params
            sigmoid_gradient = sigmoid_gradients[start_idx:end_idx]
            sigmoid_gradient = sigmoid_gradient.reshape(gradient.shape)
            sigmoid_soft_dict[net_name] = sigmoid_gradient

            tanh_gradient = tanh_gradients[start_idx:end_idx]
            tanh_gradient = tanh_gradient.reshape(gradient.shape)
            tanh_soft_dict[net_name] = tanh_gradient
            start_idx = end_idx

        torch.save(
            sigmoid_soft_dict,
            os.path.join(args.save_dir, "sigmoid_soft_{}.pt".format(i)),
        )
        torch.save(
            tanh_soft_dict, os.path.join(args.save_dir, "tanh_soft_{}.pt".format(i))
        )
        torch.save(hard_dict, os.path.join(args.save_dir, "hard_{}.pt".format(i)))


def load_pth_tar_files(folder_path):
    pth_tar_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                pth_tar_files.append(file_path)

    return pth_tar_files


def compute_gradient_ratio(mask_path):
    mask = torch.load(mask_path)
    all_elements = torch.cat([tensor.flatten() for tensor in mask.values()])
    ones_tensor = torch.ones(all_elements.shape)
    ratio = torch.sum(all_elements) / torch.sum(ones_tensor)
    name = mask_path.split("/")[-1].replace(".pt", "")
    return name, ratio


def print_gradient_ratio(mask_folder, save_path):
    ratio_dict = {}
    mask_path_list = load_pth_tar_files(mask_folder)
    for i in mask_path_list:
        name, ratio = compute_gradient_ratio(i)
        print(name, ratio)
        ratio_dict[name] = ratio.item()

    ratio_df = pd.DataFrame([ratio_dict])
    ratio_df.to_csv(save_path + "ratio_df.csv", index=False)


def main():
    args = arg_parser.parse_args()

    # print(args.choice, type(args.choice), len(args.choice))

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()
    # print(model.state_dict())

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        save_gradient_ratio(unlearn_data_loaders, model, criterion, args)


if __name__ == "__main__":
    main()
