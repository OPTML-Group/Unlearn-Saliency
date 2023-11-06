import argparse
import logging
import os
import sys
import traceback

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
        help="Path to folder with pretrained model for sampling (only necessary if sampling)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample_fid", "sample_classes", "visualization"],
        help="Sampling mode.",
    )
    parser.add_argument(
        "--n_samples_per_class",
        type=int,
        default=5000,
        help="Number of samples per class to generate.",
    )
    parser.add_argument(
        "--classes_to_generate",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Either a comma-separated string of class labels to generate e.g, '0,1,2,3', \
            otherwise prefix 'x' to drop that class , e.g., 'x0, x1' to generate all classes but 0 and 1.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
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
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=2.0,
        help="classifier-free guidance conditional strength",
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

    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_classes --n_samples_per_class 500
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python fim.py --config cifar10_fim.yml --ckpt_folder results/cifar10/2023_08_16_224303 --n_chunks 20

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_fid --n_samples_per_class 500 --classes_to_generate 'x0'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_classes --classes_to_generate "0" --n_samples_per_class 500

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python evaluator.py results/cifar10/2023_08_16_224303/fid_samples_guidance_2.0_excluded_class_0 cifar10_without_label_0

# python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode visualization --cond_scale -1
# python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_28_232149 --mode visualization --cond_scale -1
