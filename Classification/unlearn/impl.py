import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
            )
            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
