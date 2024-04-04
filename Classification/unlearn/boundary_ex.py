import time

import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn


def expand_model(model):
    last_fc_name = None
    last_fc_layer = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_fc_name = name
            last_fc_layer = module

    if last_fc_name is None:
        raise ValueError("No Linear layer found in the model.")

    num_classes = last_fc_layer.out_features

    bias = last_fc_layer.bias is not None

    new_last_fc_layer = nn.Linear(
        in_features=last_fc_layer.in_features,
        out_features=num_classes + 1,
        bias=bias,
        device=last_fc_layer.weight.device,
        dtype=last_fc_layer.weight.dtype,
    )

    with torch.no_grad():
        new_last_fc_layer.weight[:-1] = last_fc_layer.weight
        if bias:
            new_last_fc_layer.bias[:-1] = last_fc_layer.bias

    parts = last_fc_name.split(".")
    current_module = model
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    setattr(current_module, parts[-1], new_last_fc_layer)


@iterative_unlearn
def boundary_expanding_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None
):
    train_loader = data_loaders["forget"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        image = image.cuda()
        target = target.cuda()

        target_label = torch.ones_like(target)
        target_label *= args.num_classes
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target_label)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def boundary_expanding(data_loaders, model, criterion, args, mask=None):
    expand_model(model)
    return boundary_expanding_iter(data_loaders, model, criterion, args, mask)
