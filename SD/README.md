# Saliency-Unlearning for SD
This is the official repository for Saliency Unlearning for stable diffusion. The code structure of this project is adapted from the [ESD](https://github.com/rohitgandikota/erasing/tree/main) codebase.

# Installation Guide
* To get started clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Then download the files from our repository to `stable-diffusion` main directory of stable diffusion. This would replace the `ldm` folder of the original repo with our custom `ldm` directory
* Download the weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and move them to `stable-diffusion/models/ldm/`
* [Only for training] To convert your trained models to diffusers download the diffusers Unet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json)

# Forgetting Training with Saliency-Unlearning
1. First, we need to generate saliency map for unlearning.

   ```
    python train-scripts/generate_mask.py --ckpt_path 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt' --classes {label} --device '0'
   ```

   This will save saliency map in `stable-diffusion/mask/{label}`.

2. Forgetting training with Saliency-Unlearning

   ```
    python train-scripts/certain_label.py --train_method full --alpha 0.5 --lr 1e-5 --epochs 5  --class_to_forget {label} --mask_path 'mask/{label}/with_0.5.pt' --device '0'
   ```

   This should create another folder in `stable-diffusion/model`. 

   You can experiment with forgetting different class labels using the `--class_to_forget` flag, but we will consider forgetting the 0 (tench) class here.

3. Forgetting training with ESD

    Edit `train-script/train-esd.py` and change the default argparser values according to your convenience (especially the config paths)
    To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
    ```
    python train-scripts/train-esd.py --prompt 'your prompt' --train_method 'your choice of training' --devices '0,1'
    ```

# Generating Images
  1. To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
  2. To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
  3. The path to model can be customised in the script.
  4. It is to be noted that the current version requires the model to be in saved in `stable-diffusion/model/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
        ```
        python eval-scripts/generate-images.py --prompts_path 'prompts/imagenette.csv' --save_path 'evaluation_folder/ --model_name {model} --device 'cuda:0'
        ``` 

# Evaluation
1. FID
   * First,we need to select some images from Imagenette as real images.
   * Then, we can compute FID between real images and generated images. 
        ```
        python eval-scripts/compute-fid.py --folder_path {images_path}
        ```

2. Accuracy
   ```
   python eval-scripts/imageclassify.py --prompts_path 'prompts/imagenette.csv' --folder_path {images_path}
   ```