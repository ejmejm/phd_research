"""
This script nearly replicates the task used in Rupam and Rich's 2013 paper,
"Representation Search through Generate and Test": http://incompleteideas.net/papers/MS-AAAIws-2013.pdf.

Note that there are some minor differences, notably, all of the LTU units in the task share the
same threshold of 0, and there is no target noise. Support for gadient descent in the first layer of
the prediction network while using autostep in the second layer is also not supported.
"""

from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.models import LTU
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.experiment_helpers import *
from phd.research_utils.logging import *


def prepare_ltu_geoff_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_components(cfg)

    assert isinstance(task, NonlinearGEOFFTask)
    
    assert cfg.model.weight_init_method == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.weight_init == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.model.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"

    if cbp_tracker is not None:
        cbp_tracker.incoming_weight_init = 'binary'
    
    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed_from_string(base_seed, 'task_init_generator'))
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.0 # 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    torch.manual_seed(seed_from_string(base_seed, 'experiment_setup'))

    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker


def run_experiment(
        cfg: DictConfig,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        cbp_tracker: CBPTracker,
    ):
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    cumulant_mean = 0.0
    cumulant_square_mean = 0.0
    total_pruned = 0
    cumulant_gamma = 0.999
    target_buffer = []
    
    while step < cfg.train.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        target_buffer.extend(targets.view(-1).tolist())
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        if cfg.train.standardize_cumulants:
            with torch.no_grad():
                targets, cumulant_mean, cumulant_square_mean = standardize_targets(
                    targets, cumulant_mean, cumulant_square_mean, cumulant_gamma, step)

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            total_pruned += sum([len(idxs) for idxs in pruned_idxs.values()])

        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)
        
        # loss += torch.randn_like(loss)

        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            retain_graph = optimizer.version == 'squared_grads'
            loss.backward(retain_graph=retain_graph)
            optimizer.step(outputs, param_inputs)
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate metrics
        loss_acc += loss.item()
        cumulative_loss += loss.item()
        n_steps_since_log += 1
        
        # Calculate and accumulate accuracy for classification
        if isinstance(criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            accuracy_acc += accuracy
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_acc / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'accuracy': accuracy_acc / n_steps_since_log if isinstance(criterion, nn.CrossEntropyLoss) else None,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
            }
            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            log_metrics(metrics, cfg, step=step)
            
            pbar.set_postfix(loss=metrics['loss'], accuracy=metrics['accuracy'])
            
            # Reset accumulators
            loss_acc = 0.0
            accuracy_acc = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1
        pbar.update(1)
    
    pbar.close()


@hydra.main(config_path='../conf', config_name='rupam_task')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    init_experiment(cfg.project, cfg)
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_ltu_geoff_experiment(cfg)
    run_experiment(cfg, task_iterator, model, criterion, optimizer, cbp_tracker)
    finish_experiment()


if __name__ == '__main__':
    main()
