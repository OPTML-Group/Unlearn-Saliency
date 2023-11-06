import pruner

from .FT import FT_iter
from .impl import iterative_unlearn

prune_step = 2


@iterative_unlearn
def FT_prune_bi(data_loaders, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()

    # prune
    prune_rate = 1 - (1 - args.rate) ** (
        1 / ((args.unlearn_epochs - 1) // prune_step + 1)
    )

    if (args.unlearn_epochs - epoch) % prune_step == 0:
        if args.random_prune:
            print("random pruning")
            pruner.pruning_model_random(model, prune_rate)
        else:
            print("L1 pruning")
            pruner.pruning_model(model, prune_rate)

    pruner.check_sparsity(model)

    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args)
