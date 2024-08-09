# Saliency-Unlearning for DDPM
This is the official repository for Saliency Unlearning for DDPM. The code structure of this project is adapted from the [DDIM](https://github.com/ermongroup/ddim) and [SA](https://github.com/clear-nus/selective-amnesia/tree/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm) codebase.

# Requirements
Install the requirements using a `conda` environment:
```
conda create --name salun-ddpm python=3.8
conda activate salun-ddpm
pip install -r requirements.txt
```

# Forgetting Training with Saliency-Unlearning

1. First train a conditional DDPM on all 10 CIFAR10/STL10 classes. 

   Specify GPUs using the `CUDA_VISIBLE_DEVICES` environment flag. 

   We demonstrate the code to run Saliency Unlearning on CIFAR10; the commands can run the STL10 experiments using the same commands but replacing config  and dataset flags accordingly.

   For instance, using two GPUs with IDs 0 and 1 on CIFAR10,

   ```
    CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_train.yml --mode train
   ```

   Similar to the VAE, a checkpoint should be saved under `results/cifar10/yyyy_mm_dd_hhmmss`. 

2. Next, we need to generate saliency map samples for unlearning.

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_saliency_unlearn.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode generate_mask
   ```

   This will save saliency map in `results/cifar10/unlearn/mask`.

3. Forgetting training with Saliency-Unlearning

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_saliency_unlearn.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode saliency_unlearn --mask_path results/cifar10/unlearn/mask/{mask_name} --alpha 1e-3 --method rl
   ```

   This should create another folder in `results/cifar10/unlearn/{method_name}`. 

   You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.

   You can experiment with forgetting different method using the `--method` flag, but we will consider forgetting with random label(rl) here.


# Evaluation
1. Image Metrics Evaluation on Classes to Remember

    First generate the sample images on the model trained in step 3.
    ```
    CUDA_VISIBLE_DEVICES="0,1" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --mode sample_fid --n_samples_per_class 5000 --classes_to_generate 'x0'
    ```
    Samples will be saved in `results/cifar10/yyyy_mm_dd_hhmmss/fid_samples_without_label_0_guidance_2.0`. We can either use `--classes_to_generate '1,2,3,4,5,6,7,8,9'` or `--classes_to_generate 'x01'` to specify that we want to generate all classes but the 0 class (as we have forgotten it).

    Next, we need samples from the reference dataset, but without the 0 class.
    ```
    python save_base_dataset.py --dataset cifar10 --label_to_forget 0
    ```
    The images should be saved in folder `./cifar10_without_label_0`.

    Now we can evaluate the image metrics
    ```
    CUDA_VISIBLE_DEVICES="0,1" python evaluator.py results/cifar10/yyyy_mm_dd_hhmmss/fid_samples_without_label_0_guidance_2.0 cifar10_without_label_0
    ```
    The metrics will be printed to the screen like such
    ```
    Inception Score: 8.198589324951172
    FID: 9.670457625511688
    sFID: 7.438950112110206
    Precision: 0.3907777777777778
    Recall: 0.7879333333333334
    ```

2. Classifier Evaluation

    First fine-tune a pretrained ResNet34 classifier for CIFAR10
    ```
    CUDA_VISIBLE_DEVICES="0" python train_classifier.py --dataset cifar10 
    ```
    The classifier checkpoint will be saved as `cifar10_resnet34.pth`.

    Generate samples of just the 0th class (500 is used for classifier evaluation in the paper)
    ```
    CUDA_VISIBLE_DEVICES="0,1" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --mode sample_classes --classes_to_generate "0" --n_samples_per_class 500
    ```
    The samples are saved in the folder `results/cifar10/yyyy_mm_dd_hhmmss/class_samples/0`.

    Finally evaluate with the trained classifier
    ```
    CUDA_VISIBLE_DEVICES="0" python classifier_evaluation.py --sample_path results/cifar10/yyyy_mm_dd_hhmmss/class_samples/0 --dataset cifar10 --label_of_forgotten_class 0
    ```
    The results will be printed to screen like such
    ```
    Classifier evaluation:
    Average entropy: 1.4654556959867477
    Average prob of forgotten class: 0.15628313273191452