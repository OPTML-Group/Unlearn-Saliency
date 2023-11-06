import argparse
import glob
import os
import pdb
import random
import re
import shutil
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from convertModels import savemodelDiffusers
from dataset import setup_forget_data, setup_model, setup_remain_data
from diffusers import LMSDiscreteScheduler
from einops import rearrange, repeat
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm


def proximal_gradient(
    class_to_forget,
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_ratio,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    second_device=None,
):
    assert second_device is not None, "calc_gpu_id must be specified, e.g. 'cuda:0'"

    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)

    # set model to train
    model.train()
    losses = []

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(name)
            parameters.append(param)

    optimizer = torch.optim.Adam(parameters, lr=lr)

    name = f"compvis-pg-class_{str(class_to_forget)}-method_{train_method}-beta_{mask_ratio}-epoch_{epochs}-lr_{lr}"
    gpu1_init_params = torch.concat(
        [param.view(-1) for param in model.parameters()], dim=0
    )
    gpu2_init_params = gpu1_init_params.to(second_device)
    gpu2_params = gpu2_init_params.clone()
    n_params = gpu1_init_params.numel()
    total_steps = epochs * (len(forget_dl) + len(remain_dl))

    # TRAINING CODE
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            # with tqdm(total=10) as time:

            for i, (images, labels) in enumerate(forget_dl):
                # for i in range(1):
                optimizer.zero_grad()

                forget_images, forget_labels = next(iter(forget_dl))
                remain_images, remain_labels = next(iter(remain_dl))

                forget_prompts = [descriptions[label] for label in forget_labels]

                # player -> truck
                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]
                remain_prompts = [descriptions[label] for label in remain_labels]
                print(forget_prompts, pseudo_prompts, remain_prompts)

                """
                # debug             
                forget_images = torch.randn([batch_size, 3, 512, 512], device=device)
                remain_images = torch.randn([batch_size, 3, 512, 512], device=device)
                
                pseudo_prompts = ["haha" for i in range(batch_size)]
                forget_prompts = ["haha" for i in range(batch_size)]
                remain_prompts = ["haha" for i in range(batch_size)]
                """

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model.shared_step(remain_batch)[0]

                # forget stage
                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }

                pseudo_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": pseudo_prompts,
                }

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(
                    0,
                    model.num_timesteps,
                    (forget_input.shape[0],),
                    device=model.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model.device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                loss = forget_loss + alpha * remain_loss
                loss.backward()
                losses.append(loss.item() / batch_size)

                optimizer.step()

                with torch.no_grad():
                    ratio = int(
                        mask_ratio
                        * (
                            (
                                total_steps
                                - (epoch * (len(forget_dl) + len(remain_dl)) + i + 1)
                            )
                            / total_steps
                            * n_params
                        )
                    )

                    cnt = 0
                    for param in model.parameters():
                        gpu2_params[cnt : cnt + param.numel()] = param.view(-1).to(
                            second_device
                        )
                        cnt += param.numel()

                    gpu2_params -= gpu2_init_params
                    gpu2_params.abs_().neg_()

                    threshold = (-torch.topk(gpu2_params, ratio)[0][-1]).to(device)

                    cnt = 0
                    for param in model.parameters():
                        init_param = gpu1_init_params[cnt : cnt + param.numel()].view(
                            param.shape
                        )
                        param -= init_param

                        larger = param > threshold
                        smaller = param < -threshold
                        between = ~(larger | smaller)

                        param[larger] -= threshold
                        param[smaller] += threshold
                        param[between] = 0

                        param += init_param

                        cnt += param.numel()

                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

        prompts_path = "prompts/imagenette.csv"
        df = pd.read_csv(prompts_path)
        all_images = []

        num_inference_steps = 50
        model.eval()
        for _, row in df.iterrows():
            # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/scripts/txt2img.py#L289
            uncond_embeddings = model.get_learned_conditioning(1 * [""])
            text_embeddings = model.get_learned_conditioning(1 * [str(row.prompt)])
            print(str(row.prompt))
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            height = image_size
            width = image_size

            latents = torch.randn((1, 4, height // 8, width // 8))
            latents = latents.to(device)

            scheduler.set_timesteps(num_inference_steps)

            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(num_inference_steps)

            for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )

                # predict the noise residual
                with torch.no_grad():
                    t_unet = torch.full((2,), t, device=device)
                    noise_pred = model.apply_model(
                        latent_model_input, t_unet, text_embeddings
                    )

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            with torch.no_grad():
                image = model.decode_first_stage(latents)

            image = (image / 2 + 0.5).clamp(0, 1)
            all_images.append(image)

        grid = torch.stack(all_images, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=5)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))

        folder_path = f"models/{name}"
        os.makedirs(folder_path, exist_ok=True)
        img.save(os.path.join(f"{folder_path}", f"epoch_{epoch}.png"))

    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
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
        "--mask_ratio",
        help="mask path for stable diffusion v1-4",
        type=float,
        required=True,
        default=None,
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
        default="0",
    )
    parser.add_argument(
        "--second_device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="1",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()

    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_ratio = args.mask_ratio
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    second_device = f"cuda:{int(args.second_device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    proximal_gradient(
        classes,
        train_method,
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_ratio,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
        second_device,
    )

    # python train-scripts/certain_label.py --class_to_forget "6" --train_method full --device "7" --lr 1e-5 --mask_path "mask/truck_0.5.pt"
