import os
from collections import OrderedDict

import arg_parser
import pruner
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import trainer
import unlearn
import utils


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    utils.setup_seed(args.seed)
    # prepare dataset
    poison_label = args.class_to_replace
    args.class_to_replace = -1

    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    forget_loader, retain_loader = utils.get_unlearn_loader(marked_loader, args)

    def poison_func(data, target):
        import numpy as np

        poisoned_data = np.copy(data)
        poisoned_target = np.zeros_like(target) + poison_label
        poisoned_data[
            :, -2 - args.trigger_size : -2, -2 - args.trigger_size : -2, :
        ] *= 0
        # batch_data[:, -2, -2, :] *= 0
        # batch_data[:, -2, -2, :] *= 0
        return poisoned_data, poisoned_target

    (
        poisoned_loader,
        unpoisoned_loader,
        poisoned_train_loader,
        poisoned_test_loader,
    ) = utils.get_poisoned_loader(
        forget_loader, retain_loader, test_loader, poison_func, args
    )
    unlearn_data_loaders = OrderedDict(
        retain=unpoisoned_loader,
        forget=poisoned_loader,
        val=val_loader,
        test=test_loader,
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        # ================================training================================

        utils.setup_seed(args.train_seed)

        if args.mask and os.path.exists(args.mask):
            checkpoint = torch.load(args.mask, map_location=device)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)

            """
            current_mask = pruner.extract_mask(checkpoint)
            pruner.prune_model_custom(model, current_mask)
            pruner.check_sparsity(model)
            """
        else:
            optimizer, scheduler = trainer.get_optimizer_and_scheduler(model, args)

            trainer.train_with_rewind(
                model, optimizer, scheduler, poisoned_train_loader, criterion, args
            )
            os.makedirs(os.path.dirname(args.mask), exist_ok=True)
            torch.save(model.state_dict(), args.mask)

        # ================================validate before================================

        evaluation_result = {}
        evaluation_result["test_acc"] = trainer.validate(
            test_loader, model, criterion, args
        )
        evaluation_result["attack_acc"] = trainer.validate(
            poisoned_test_loader, model, criterion, args
        )

        # ================================unlearn================================

        utils.setup_seed(args.train_seed)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)

    if evaluation_result is None:
        evaluation_result = {}

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # ================================validate after================================

    if "test_acc_unlearn" not in evaluation_result:
        evaluation_result["test_acc_unlearn"] = trainer.validate(
            test_loader, model, criterion, args
        )
    if "attack_acc_unlearn" not in evaluation_result:
        evaluation_result["attack_acc_unlearn"] = trainer.validate(
            poisoned_test_loader, model, criterion, args
        )

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()

"""
import os
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict

import utils
import unlearn
from trainer import validate
# import pruner

import arg_parser


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    utils.setup_seed(args.seed)
    # prepare dataset
    poison_label = args.class_to_replace
    args.class_to_replace = -1

    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(
        args)
    model.cuda()

    forget_loader, retain_loader = utils.get_unlearn_loader(
        marked_loader, args)

    def poison_func(data, target):
        import numpy as np
        poisoned_data = np.copy(data)
        poisoned_target = np.zeros_like(target) + poison_label
        poisoned_data[:, -2 - args.trigger_size:-
                      2, -2 - args.trigger_size:-2, :] *= 0
        return poisoned_data, poisoned_target

    poisoned_loader, unpoisoned_loader, poisoned_train_loader, poisoned_test_loader = utils.get_poisoned_loader(
        forget_loader, retain_loader, test_loader, poison_func, args)
    unlearn_data_loaders = OrderedDict(
        retain=unpoisoned_loader,
        forget=poisoned_loader,
        val=val_loader,
        test=test_loader)

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
    
        if args.mask and os.path.exists(args.mask):
            checkpoint = torch.load(args.mask, map_location=device)
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            

        # ================================pruning================================

        if args.mask and os.path.exists(args.mask):
            checkpoint = torch.load(args.mask, map_location=device)
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            current_mask = pruner.extract_mask(checkpoint)
            pruner.prune_model_custom(model, current_mask)
            pruner.check_sparsity(model)
        else:
            prune_method = pruner.get_prune_method(args.prune)

            prune_method(model, poisoned_train_loader,
                        test_loader, criterion, args)
            os.makedirs(os.path.dirname(args.mask), exist_ok=True)
            torch.save(model.state_dict(), args.mask)

        
        # ================================validate before================================

        evaluation_result = {}
        evaluation_result['test_acc'] = validate(
            test_loader, model, criterion, args)
        evaluation_result['attack_acc'] = validate(
            poisoned_test_loader, model, criterion, args)

        # ================================unlearn================================

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)

    if evaluation_result is None:
        evaluation_result = {}

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # ================================validate after================================

    if 'test_acc_unlearn' not in evaluation_result:
        evaluation_result['test_acc_unlearn'] = validate(
            test_loader, model, criterion, args)
    if 'attack_acc_unlearn' not in evaluation_result:
        evaluation_result['attack_acc_unlearn'] = validate(
            poisoned_test_loader, model, criterion, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == '__main__':
    main()
"""
