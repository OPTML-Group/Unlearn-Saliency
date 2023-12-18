# Saliency-Unlearning for Classification
This is the official repository for Saliency Unlearning for Clasification. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.

# Generate Saliency Map
```
python generate_mask.py --save ${saliency_map_path} --mask ${origin_model_path} --num_indexes_to_replace 4500 --unlearn_epochs 1
```

# Unlearning
### SalUn
```
# Salun
python -u main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.1 --num_indexes_to_replace 4500 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}

# Soft-thresholding Salun
python -u main_random.py --unlearn RL_proximal --unlearn_epochs 10 --unlearn_lr 0.1 --num_indexes_to_replace 4500 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}
```


### Retrain

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1
```

### FT

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10
```

### GA

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5
```

### IU

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha}
```

### l1-sparse

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT_prune --num_indexes_to_replace 4500 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10
```