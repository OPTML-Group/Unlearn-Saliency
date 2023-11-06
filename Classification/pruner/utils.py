import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.autograd import grad
from torch.nn import Conv2d

__all__ = [
    "pruning_model",
    "pruning_model_random",
    "prune_model_custom",
    "remove_prune",
    "extract_mask",
    "reverse_mask",
    "check_sparsity",
    "check_sparsity_dict",
    "global_prune_model",
]


# Pruning operation
def pruning_model(model, px):
    print("Apply Unstructured L1 Pruning Globally (all conv layers)")
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def pruning_model_structured(model, px):
    print("Apply Unstructured L1 Pruning Globally (all conv layers)")
    # parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(
                m,
                name="weight",
                amount=px,
                dim=0,
                n=1,  # l1 loss
            )


def pruning_model_structured_channel_wise(model, px):
    print("Apply structured L1 Pruning Globally (all conv layers) channel wise")
    # parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(
                m,
                name="weight",
                amount=px,
                # Prune the second dimension, corresponding to the index of the input feature maps.
                dim=1,
                n=1,  # l1 loss
            )


def pruning_model_random(model, px):
    print("Apply Unstructured Random Pruning Globally (all conv layers)")
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def prune_model_custom(model, mask_dict):
    print("Pruning with custom mask (all conv layers)")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name + ".weight_mask"
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(
                    m, "weight", mask=mask_dict[name + ".weight_mask"]
                )
            else:
                print("Can not find [{}] in mask_dict".format(mask_name))


def remove_prune(model):
    print("Remove hooks for multiplying masks (all conv layers)")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m, "weight")


# Mask operation function
def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict


def reverse_mask(mask_dict):
    new_dict = {}
    for key in mask_dict.keys():
        new_dict[key] = 1 - mask_dict[key]

    return new_dict


# Mask statistic function


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = ", 100 * (1 - zero_sum / sum_list), "%")
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratie = None

    return remain_weight_ratie


def count_sparsity(model):
    zero_count = 0
    total_count = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            zero_count = zero_count + torch.sum(module.weight == 0)
            total_count = total_count + module.weight.nelement()

    sparsity = 100.0 * float(zero_count / total_count)

    print("Sparsity in total:")
    print(sparsity)

    return zero_count


def check_sparsity_dict(state_dict):
    sum_list = 0
    zero_sum = 0

    for key in state_dict.keys():
        if "mask" in key:
            sum_list += float(state_dict[key].nelement())
            zero_sum += float(torch.sum(state_dict[key] == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = ", 100 * (1 - zero_sum / sum_list), "%")
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratie = None

    return remain_weight_ratie


def fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx : idx + 1], targets[idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break
    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat(
        [torch.cat(_) for _ in labels]
    ).view(-1)
    return X, y


def mp_importance_score(model):
    score_dict = {}
    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, "weight")] = m.weight.data.abs()
    return score_dict


def snip_importance_score(
    model, dataloader, samples_per_class, loss_func=torch.nn.CrossEntropyLoss()
):
    score_dict = {}
    model.zero_grad()
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, model.fc.out_features, samples_per_class)
    x, y = x.to(device), y.to(device)
    loss = loss_func(model(x), y)
    loss.backward()
    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, "weight")] = m.weight.grad.data.abs()
    model.zero_grad()
    return score_dict


def grasp_importance_score(
    model, dataloader, samples_per_class, loss_func=torch.nn.CrossEntropyLoss()
):
    score_dict = {}
    model.zero_grad()
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, model.fc.out_features, samples_per_class)
    x, y = x.to(device), y.to(device)
    loss = loss_func(model(x) / 200, y)
    gs = grad(loss, model.parameters(), create_graph=True)
    model.zero_grad()
    t = sum([(g * g.data).sum() for g in gs])
    t.backward()

    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            score_dict[(m, "weight")] = -m.weight.data * m.weight.grad.data
    model.zero_grad()
    return score_dict


def synflow_importance_score(
    model,
    dataloader,
):
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    model.eval()  # Crucial! BatchNorm will break the conservation laws for synaptic saliency
    model.zero_grad()
    score_dict = {}
    signs = linearize(model)

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(next(model.parameters()).device)
    output = model(input)
    torch.sum(output).backward()

    for m in model.modules():
        if isinstance(m, (Conv2d,)):
            if hasattr(m, "weight_orig"):
                score_dict[(m, "weight")] = (
                    m.weight_orig.grad.data * m.weight.data
                ).abs()
            else:
                score_dict[(m, "weight")] = (m.weight.grad.data * m.weight.data).abs()
    model.zero_grad()
    nonlinearize(model, signs)
    return score_dict


def global_prune_model(
    model, ratio, method, dataloader=None, structured=False, sample_per_classes=25
):
    if method == "mp":
        score_dict = mp_importance_score(model)
    elif method == "snip":
        score_dict = snip_importance_score(model, dataloader, sample_per_classes)
    elif method == "grasp":
        score_dict = grasp_importance_score(model, dataloader, sample_per_classes)
    elif method == "synflow":
        pass
    else:
        raise NotImplementedError(f"Pruning Method {method} not Implemented")

    if method == "synflow":
        iteration_number = (
            100  # In SynFlow Paper, an iteration number of 100 performs well
        )
        each_ratio = 1 - (1 - ratio) ** (1 / iteration_number)
        for _ in range(iteration_number):
            score_dict = synflow_importance_score(model, dataloader)
            if structured:
                pass
            else:
                prune.global_unstructured(
                    parameters=score_dict.keys(),
                    pruning_method=prune.L1Unstructured,
                    amount=each_ratio,
                    importance_scores=score_dict,
                )
    else:
        prune.global_unstructured(
            parameters=score_dict.keys(),
            pruning_method=prune.L1Unstructured,
            amount=ratio,
            importance_scores=score_dict,
        )
