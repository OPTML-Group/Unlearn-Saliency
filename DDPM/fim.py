import argparse
import os
import sys

import numpy as np
import torch
import yaml
from functions import dict2namespace
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        required=True,
        help="Path to pretrained model for sampling",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=20,
        help="Chunking of timesteps for FIM calculation. Increase this if you are running out of GPU memory.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--fid", action="store_true")

    # parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as fp:
        config = yaml.unsafe_load(fp)
    config = dict2namespace(config)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    args, config = parse_args_and_config()
    runner = Diffusion(args, config)
    runner.save_fim()

    return 0


if __name__ == "__main__":
    sys.exit(main())

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python fim.py --config cifar10_fim.yml --ckpt_folder results/cifar10/2023_08_16_224303 --n_chunks 20
