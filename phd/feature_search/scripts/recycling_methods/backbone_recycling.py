import logging
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.experiment_helpers import *
from phd.feature_search.scripts.feature_maturity_experiment import *


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
    prev_pruned_idxs = set()
    prune_layer = model.layers[-2]
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_accum = 0.0
    pruned_accum = 0
    pruned_newest_feature_accum = 0
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
            n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
            total_pruned += n_pruned

            if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
                pruned_accum += len(pruned_idxs[prune_layer])
                for new_pruned_idx in pruned_idxs[prune_layer]:
                    pruned_newest_feature_accum += int(new_pruned_idx in prev_pruned_idxs)
                prev_pruned_idxs = set(pruned_idxs[prune_layer])
        
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
        loss_accum += loss.item()
        cumulative_loss += loss.item()
        n_steps_since_log += 1
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_accum / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
            }

            if pruned_accum > 0:
                metrics['fraction_pruned_were_new'] = pruned_newest_feature_accum / pruned_accum
                pruned_newest_feature_accum = 0
                pruned_accum = 0

            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'])
            pbar.update(cfg.train.log_freq)
            
            # Reset accumulators
            loss_accum = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1

    pbar.close()
    wandb.finish()


@hydra.main(config_path='../../conf', config_name='rupam_task')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)

    run_experiment(cfg, task, task_iterator, model, criterion, optimizer, cbp_tracker)


if __name__ == '__main__':
    main()
