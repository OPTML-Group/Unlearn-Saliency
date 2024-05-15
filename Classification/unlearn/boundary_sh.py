import copy
import time

import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn


def discretize(x):
    return torch.round(x * 255) / 255


def FGSM_perturb(x, y, model=None, bound=None, criterion=None):
    device = model.parameters().__next__().device
    model.zero_grad()
    x_adv = x.detach().clone().requires_grad_(True).to(device)

    pred = model(x_adv)
    loss = criterion(pred, y)
    loss.backward()

    grad_sign = x_adv.grad.data.detach().sign()
    x_adv = x_adv + grad_sign * bound
    x_adv = discretize(torch.clamp(x_adv, 0.0, 1.0))

    return x_adv.detach()


@iterative_unlearn
def boundary_shrink_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, test_model=None
):
    assert test_model is not None

    bound = 0.1  # hard coding in the paper

    train_loader = data_loaders["forget"]
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        test_model.eval()
        image_adv = FGSM_perturb(
            image, target, model=test_model, bound=bound, criterion=criterion
        )

        adv_outputs = test_model(image_adv)
        adv_label = torch.argmax(adv_outputs, dim=1)

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, adv_label)

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


def boundary_shrink(data_loaders, model: nn.Module, criterion, args, mask=None):
    device = model.parameters().__next__().device
    test_model = copy.deepcopy(model).to(device)
    return boundary_shrink_iter(
        data_loaders, model, criterion, args, mask, test_model=test_model
    )
