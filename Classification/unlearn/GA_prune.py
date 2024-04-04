import sys
import time

import torch
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pruner import *
from trainer import train, validate
from utils import *

from .impl import iterative_unlearn


def GA(train_loader, model, criterion, optimizer, epoch, args):
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

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
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


def GA_prune(data_loaders, model, criterion, args):
    best_sa = 0
    train_loader = data_loaders["forget"]
    test_loader = data_loaders["test"]
    val_loader = data_loaders["val"]
    all_result = {}
    all_result["train_ta"] = []
    all_result["test_ta"] = []
    all_result["val_ta"] = []
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    start_epoch = 0
    start_state = 0
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    print(
        "######################################## Start Standard Training Iterative Pruning ########################################"
    )

    for state in range(start_state, args.pruning_times):
        print("******************************************")
        print("pruning state", state)
        print("******************************************")

        check_sparsity(model)
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            print(optimizer.state_dict()["param_groups"][0]["lr"])
            if state == 0:
                if (epoch) == args.rewind_epoch:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.save_dir, "epoch_{}_rewind_weight.pt".format(epoch + 1)
                        ),
                    )
                    if args.prune_type == "rewind_lt":
                        initalization = deepcopy(model.state_dict())
            acc = GA(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            tacc = validate(val_loader, model, criterion, args)
            # evaluate on test set
            test_tacc = validate(test_loader, model, criterion, args)

            scheduler.step()

            all_result["train_ta"].append(acc)
            all_result["val_ta"].append(tacc)
            all_result["test_ta"].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint(
                {
                    "state": state,
                    "result": all_result,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_sa": best_sa,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "init_weight": initalization,
                },
                is_SA_best=is_best_sa,
                pruning=state,
                save_path=args.save_dir,
            )

            # plot training curve
            plt.plot(all_result["train_ta"], label="train_acc")
            plt.plot(all_result["val_ta"], label="val_acc")
            plt.plot(all_result["test_ta"], label="test_acc")
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
            plt.close()
            print("one epoch duration:{}".format(time.time() - start_time))

        # report result
        check_sparsity(model)
        print("Performance on the test data set")
        test_tacc = validate(test_loader, model, criterion, args)
        if len(all_result["val_ta"]) != 0:
            val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
            print(
                "* best SA = {}, Epoch = {}".format(
                    all_result["test_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
                )
            )

        all_result = {}
        all_result["train_ta"] = []
        all_result["test_ta"] = []
        all_result["val_ta"] = []
        best_sa = 0
        start_epoch = 0

        if args.prune_type == "pt":
            print("* loading pretrained weight")
            initalization = torch.load(
                os.path.join(args.save_dir, "0model_SA_best.pth.tar"),
                map_location=torch.device("cuda:" + str(args.gpu)),
            )["state_dict"]

        # pruning and rewind
        if args.random_prune:
            print("random pruning")
            pruning_model_random(model, args.rate)
        else:
            print("L1 pruning")
            pruning_model(model, args.rate)

        remain_weight = check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        # weight rewinding
        # rewind, initialization is a full model architecture without masks
        if state < args.pruning_times - 1:
            model.load_state_dict(initalization, strict=False)
            prune_model_custom(model, current_mask)
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )
            if args.rewind_epoch:
                # learning rate rewinding
                for _ in range(args.rewind_epoch):
                    scheduler.step()
    return model
