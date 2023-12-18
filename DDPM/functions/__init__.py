import argparse
import os
from datetime import datetime

import torch.optim as optim
import yaml


def get_optimizer(config, parameters):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Optimizer {} not understood.".format(config.optim.optimizer)
        )


def get_config_and_setup_dirs(filename: str):
    with open(filename, "r") as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(
        "./results", config.data.dataset.lower(), timestamp
    )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    with open(os.path.join(config.log_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def get_mask_config_and_setup_dirs(args, filename: str):
    with open(filename, "r") as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    mask = None
    if "origin" in args.mask_path:
        mask = "origin"
    elif "inverted" in args.mask_path:
        mask = "inverted"
    elif "random" in args.mask_path:
        mask = "random"
    elif "without" in args.mask_path:
        mask = "without"
    else:
        mask = "full"

    # config.exp_root_dir = os.path.join(f"./results", config.data.dataset.lower(), f"forget/{args.alpha}_{mask}", timestamp)
    config.exp_root_dir = os.path.join(
        f"./results",
        config.data.dataset.lower(),
        "forget",
        args.method,
        f"{args.alpha}_{mask}",
        timestamp,
    )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    with open(os.path.join(config.log_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def setup_dirs(config):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(
        "./results", config.data.dataset.lower(), timestamp
    )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cycle(dl):
    while True:
        for data in dl:
            yield data


def create_class_labels(string, n_classes=10):
    if any(x.startswith("x") for x in string.split(",")):
        excluded_numbers = [int(x[1:]) for x in string.split(",") if x.startswith("x")]
        my_list = [x for x in range(n_classes) if x not in excluded_numbers]
    else:
        excluded_numbers = []
        my_list = [int(x) for x in string.split(",")]
    return my_list, excluded_numbers
