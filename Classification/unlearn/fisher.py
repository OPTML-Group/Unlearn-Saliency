import copy

import torch
from torch.autograd import grad
from tqdm import tqdm


def fisher_information_martix(model, train_dl, device):
    model.eval()
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    total = 0
    for i, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]

        epsilon = 1e-8
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(
                prediction, model.parameters(), retain_graph=True, create_graph=False
            )
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total

    return fisher_approximation


def fisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]

    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    fisher_approximation = fisher_information_martix(model, retain_loader, device)
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(args.alpha / fisher_approximation[i]).clamp(
            max=1e-3
        ) * torch.empty_like(parameter).normal_(0, 1)
        noise = noise * 10 if parameter.shape[-1] == 10 else noise
        print(torch.max(noise))
        parameter.data = parameter.data + noise
    return model


def hessian(dataset, model, loss_fn, args):
    model.eval()
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0

    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = torch.nn.functional.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        p.grad2_acc /= len(train_loader)


def get_mean_var(p, args, is_base_dist=False):
    var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.shape[0] == args.num_classes:
        var = var.clamp(max=1e2)
    var = args.alpha * var
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())

    if p.shape[0] == args.num_classes and (
        (args.num_indexes_to_replace == 4500 and args.dataset == "cifar10")
        or (args.num_indexes_to_replace == 450 and args.dataset == "cifar100")
    ):
        mu[args.class_to_replace] = 0
        var[args.class_to_replace] = 0.0001
    if p.shape[0] == args.num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
    return mu, var


def fisher_new(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]
    dataset = retain_loader.dataset
    for p in model.parameters():
        p.data0 = copy.deepcopy(p.data.clone())
    hessian(dataset, model, criterion, args)
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, args, False)
        p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
    return model
