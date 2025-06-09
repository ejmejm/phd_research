"""
This script is used to run experiments for better understanding how long it takes for new
features to mature and converge.
More specifically, the goal of this script is to produce metrics that explain how long it
takes for a feature to become "safe", and how long it takes for the associated weight to converge
as a function of the optimal utility of the feature and the number of other existing features.
"""


from collections import OrderedDict
import logging
from typing import  Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.experiment_helpers import *
from phd.feature_search.scripts.old.feature_maturity_experiment import *


CONVERGENCE_N_SAMPLES = 1_000_000


logger = logging.getLogger(__name__)


def compute_optimal_stats(
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the optimal weights and utilities given the task and model structure."""

    device = next(model.parameters()).device

    inputs, targets = next(task.get_iterator(batch_size=CONVERGENCE_N_SAMPLES))
    inputs = inputs.to(device)
    targets = targets.to(device)

    loss = np.inf

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_features = model.layers[-1].in_features
    init_lr = 1.0 / np.sqrt(n_features)
    optimizer = torch.optim.SGD(trainable_params, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.5,
        patience = 10,
        verbose = False,
    )
    step = 0
    min_lr = init_lr * 0.001
    
    while True:
        # Compute optimal weights
        outputs, param_inputs = model(inputs)
        loss = criterion(outputs, targets)

        scheduler.step(loss.item())
        
        # Stop if learning rate gets too small
        if optimizer.param_groups[0]['lr'] <= min_lr:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    logger.info(f'Optimal model converged in {step} steps with a loss of {loss.item():.5f}')

    with torch.no_grad():
        weight_layer = model.layers[-1].weight
        optimal_weights = weight_layer.data
        feature_weight_sums = weight_layer.abs().sum(dim=0)

        layer_inputs = param_inputs[weight_layer]
        input_magnitudes = layer_inputs.abs().mean(dim=0)

        optimal_utilities = input_magnitudes * feature_weight_sums

    return optimal_weights, optimal_utilities, loss.item()


def reset_model(model: MLP):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data.zero_()


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: nn.Module,
        cbp_tracker: CBPTracker,
    ):
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    feature_layer = model.layers[-2]
    feature_layer._forward_hooks = OrderedDict()
    
    # Initialize accumulators
    total_pruned = 0

    while step < cfg.train.total_steps:
        
        # Compute utilities with optimal weights
        reset_model(model) # Set all trainable params to zero
        optimal_weights, optimal_utilities, loss = compute_optimal_stats(task, model, criterion)
        optimal_utilities = optimal_utilities.cpu().numpy()

        # Prune the lowest utility feature
        prune_feature_idx = np.argmin(optimal_utilities)
        cbp_tracker._prune_layer(feature_layer, [prune_feature_idx])
        total_pruned += 1
    
        # Log the results
        wandb.log({
            'pruned_utility': optimal_utilities[prune_feature_idx],
            'total_pruned': total_pruned,
            'loss': loss,
            'min_utility': optimal_utilities.min(),
            'max_utility': optimal_utilities.max(),
            'mean_utility': optimal_utilities.mean(),
        })
    
    pbar.close()
    wandb.finish()


@hydra.main(config_path='../../conf', config_name='feature_maturity_defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)
    cbp_tracker._utility_reset_mode = 'zero'
    model.to(cfg.get('optimal_weight_device', cfg.device))

    run_experiment(cfg, task, model, criterion, cbp_tracker)


if __name__ == '__main__':
    main()
