import argparse
import os

import torch
from dataset import setup_forget_data, setup_forget_nsfw_data, setup_model
from tqdm import tqdm

def generate_mask(
    classes,
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    train_dl, descriptions = setup_forget_data(classes, batch_size, image_size)

    model.eval()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)

    gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0

    with tqdm(total=len(train_dl)) as t:
        for i, (images, labels) in enumerate(train_dl):
            optimizer.zero_grad()

            images = images.to(device)

            null_prompts = ["" for label in labels]
            prompts = [descriptions[label] for label in labels]

            forget_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}
            null_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

            forget_input, forget_emb = model.get_input(
                forget_batch, model.first_stage_key
            )
            null_input, null_emb = model.get_input(null_batch, model.first_stage_key)

            t = torch.randint(
                0, model.num_timesteps, (forget_input.shape[0],), device=device
            ).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)

            forget_out = model.apply_model(forget_noisy, t, forget_emb)
            null_out = model.apply_model(forget_noisy, t, null_emb)

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out

            loss = -criteria(noise, preds)

            loss.backward()

            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        mask_path = os.path.join("mask", str(classes))
        os.makedirs(mask_path, exist_ok=True)

        threshold_list = [0.5]
        for i in threshold_list:
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

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
            torch.save(hard_dict, os.path.join(mask_path, f"with_{str(i)}.pt"))


def generate_nsfw_mask(
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    train_dl, _ = setup_forget_nsfw_data(batch_size, image_size)

    model.eval()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)
    word_nude = "a photo of a nude person"

    gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0

    with tqdm(total=len(train_dl)) as t:
        for i, images in enumerate(train_dl):
            optimizer.zero_grad()

            images = images.to(device)
            t = torch.randint(
                0, num_timesteps, (images.shape[0],), device=device
            ).long()

            null_prompts = [""] * batch_size
            prompts = [word_nude] * batch_size

            forget_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}
            null_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

            forget_input, forget_emb = model.get_input(
                forget_batch, model.first_stage_key
            )
            null_input, null_emb = model.get_input(null_batch, model.first_stage_key)

            t = torch.randint(
                0, model.num_timesteps, (forget_input.shape[0],), device=device
            ).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)

            forget_out = model.apply_model(forget_noisy, t, forget_emb)
            null_out = model.apply_model(forget_noisy, t, null_emb)

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out

            loss = - criteria(noise, preds)
            loss.backward()

            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        threshold_list = [0.5]
        for i in threshold_list:
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

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

            torch.save(hard_dict, os.path.join("mask/nude_{}.pt".format(i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )

    parser.add_argument(
        "--classes",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="6",
    )
    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--num_timesteps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--nsfw", help="class or nsfw", type=bool, required=False, default=False
    )
    args = parser.parse_args()

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.classes)
    c_guidance = args.c_guidance
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    num_timesteps = args.num_timesteps

    if args.nsfw:
        generate_nsfw_mask(
            c_guidance,
            batch_size,
            epochs,
            lr,
            config_path,
            ckpt_path,
            diffusers_config_path,
            device,
            image_size,
            num_timesteps,
        )
    else:
        generate_mask(
            classes,
            c_guidance,
            batch_size,
            epochs,
            lr,
            config_path,
            ckpt_path,
            diffusers_config_path,
            device,
            image_size,
            num_timesteps,
        )