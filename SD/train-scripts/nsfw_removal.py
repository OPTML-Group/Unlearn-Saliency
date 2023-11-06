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
import torch
from convertModels import savemodelDiffusers
from dataset import (
    setup_forget_data,
    setup_forget_nsfw_data,
    setup_model,
    setup_remain_data,
)
from diffusers import LMSDiscreteScheduler
from einops import rearrange, repeat
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
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


@torch.no_grad()
def sample_model(
    model,
    sampler,
    c,
    h,
    w,
    ddim_steps,
    scale,
    ddim_eta,
    start_code=None,
    n_samples=1,
    t_start=-1,
    log_every_t=None,
    till_T=None,
    verbose=True,
):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T,
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


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


def nsfw_removal(
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                print(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    print(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()

    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-mask-method_{train_method}-lr_{lr}"
    else:
        name = f"compvis-nsfw-method_{train_method}-lr_{lr}"

    # NSFW Removal
    word_nude = "a photo of a nude person"
    word_wear = "a photo of a person wearing clothes"

    # TRAINING CODE
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            # with tqdm(total=10) as time:

            for i, iages in enumerate(forget_dl):
                # for i in range(1):
                optimizer.zero_grad()

                forget_images = next(iter(forget_dl))
                remain_images = next(iter(remain_dl))

                forget_prompts = [word_nude] * batch_size

                # player -> truck
                pseudo_prompts = [word_wear] * batch_size
                remain_prompts = [word_wear] * batch_size

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

                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in parameters:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )
                            print(n)

                optimizer.step()

                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

                if (i + 1) % 10 == 0:
                    all_images = []
                    num_inference_steps = 50
                    model.eval()

                    all_prompts = [word_nude, word_wear]
                    for prompt in all_prompts:
                        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/scripts/txt2img.py#L289
                        uncond_embeddings = model.get_learned_conditioning(5 * [""])
                        text_embeddings = model.get_learned_conditioning(5 * [prompt])
                        print(str(prompt))
                        text_embeddings = torch.cat(
                            [uncond_embeddings, text_embeddings]
                        )

                        height = image_size
                        width = image_size

                        latents = torch.randn((5, 4, height // 8, width // 8))
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
                                t_unet = torch.full((10,), t, device=device)
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
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
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
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=int,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
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
        default="0,0",
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

    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    nsfw_removal(
        train_method=train_method,
        alpha=alpha,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )

# python train-scripts/nsfw_removal.py --train_method 'full' --device '1'
# python train-scripts/nsfw_removal.py --train_method 'full' --device '1' --mask_path 'mask/nude_0.5.pt'
