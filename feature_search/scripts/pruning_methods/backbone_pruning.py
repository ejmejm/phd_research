import logging
import os
import sys
from typing import Iterator, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from idbd import IDBD
from tasks import NonlinearGEOFFTask
from run_experiment import *
from scripts.feature_maturity_experiment import *


logger = logging.getLogger(__name__)


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        cbp_tracker: CBPTracker,
    ):

    # Training loop
    step = 0
    prev_pruned_idx = np.nan
    prune_layer = model.layers[-2]
    pruned_newest_feature = None
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    n_steps_since_log = 0
    total_pruned = 0
    target_buffer = []

    while step < cfg.train.total_steps:

        # Generate batch of data
        inputs, targets = next(task_iterator)
        target_buffer.extend(targets.view(-1).tolist())
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            total_pruned += sum([len(idxs) for idxs in pruned_idxs.values()])
            if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
                new_pruned_idx = pruned_idxs[prune_layer][0]
                pruned_newest_feature = new_pruned_idx == prev_pruned_idx
        
        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            optimizer.step(loss, outputs, param_inputs)
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate metrics
        loss_acc += loss.item()
        cumulative_loss += loss.item()
        n_steps_since_log += 1
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_acc / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
            }
            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'], accuracy=metrics['accuracy'])
            
            # Reset accumulators
            loss_acc = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1
        pbar.update(1)

    pbar.close()
    wandb.finish()


@hydra.main(config_path='../../conf', config_name='feature_maturity_defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)
    cbp_tracker._utility_reset_mode = 'zero' # TODO: Decide if I want to use zero reset mode or not

    run_experiment(cfg, task, task_iterator, model, criterion, optimizer, cbp_tracker)


if __name__ == '__main__':
    main()
