"""
This script is used to run experiments for better understanding how long it takes for new
features to mature and converge.
More specifically, the goal of this script is to produce metrics that explain how long it
takes for a feature to become "safe", and how long it takes for the associated weight to converge
as a function of the optimal utility of the feature and the number of other existing features.
"""


from collections import OrderedDict
import copy
import logging
import os
import sys
from typing import Iterator, List, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from feature_recycling import reset_input_weights
from idbd import IDBD, RMSPropIDBD
from models import LTU
from tasks import NonlinearGEOFFTask
from run_experiment import *


CONVERGENCE_N_SAMPLES = 1_000_000
OPTIMAL_WEIGHT_LOSS_THRESHOLD = 0.0001
OPTIMAL_WEIGHT_SIMILARITY_THRESHOLD = 0.02
CONVERGENCE_STEPS = 5000


logger = logging.getLogger(__name__)


def prepare_experiment(cfg: DictConfig):
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
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
    )
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker


def check_sequence_convergence(sequence: List[float], threshold: float = 0.01) -> bool:
    """
    Check if the sequence has converged, by checking if the standard deviation is less than a threshold.
    """
    return np.std(sequence) < threshold


def compute_optimal_weights(
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: nn.Module,
        device: str,
    ) -> torch.Tensor:
    original_device = next(model.parameters()).device
    model = copy.deepcopy(model)
    model.layers[1]._forward_hooks = OrderedDict() # Remove CBP hook
    model.to(device)

    inputs, targets = next(task.get_iterator(batch_size=CONVERGENCE_N_SAMPLES))
    inputs = inputs.to(device)
    targets = targets.to(device)

    avg_squared_target = targets.square().mean().item()
    loss = np.inf

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_features = model.layers[-1].in_features
    init_lr = 1.0 / np.sqrt(n_features)
    optimizer = torch.optim.Adam(trainable_params, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1.0,
        end_factor = 0.01,
        total_iters = 1000,
    )
    
    loss_history = []
    step = 0
    convergence_threshold = OPTIMAL_WEIGHT_LOSS_THRESHOLD * avg_squared_target
    while len(loss_history) < 10 or not check_sequence_convergence(loss_history, convergence_threshold):
        # Compute optimal weights
        outputs, param_inputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_history.append(loss.item())
        loss_history = loss_history[-10:]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step < 1000:
            scheduler.step()
            step += 1

    logger.info(f'Model converged in {step} steps with a loss of {np.mean(loss_history):.5f}')

    return model.layers[-1].weight.data.to(original_device)


def compute_weight_convergence_steps(
        model: MLP,
        optimal_weights: torch.Tensor,
        convergence_steps: torch.Tensor,
        step: int,
    ) -> torch.Tensor:

    convergence_mask = torch.isclose(
        model.layers[-1].weight,
        optimal_weights,
        rtol = OPTIMAL_WEIGHT_SIMILARITY_THRESHOLD,
    )

    convergence_steps = convergence_steps.clone()
    convergence_steps[~convergence_mask] = torch.inf
    convergence_steps[convergence_mask] = torch.minimum(
        convergence_steps[convergence_mask], torch.tensor(step))

    return convergence_steps


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
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    cumulant_mean = 0.0
    cumulant_square_mean = 0.0
    total_pruned = 0
    step_since_feature_pruned = 0
    cumulant_gamma = 0.999
    target_buffer = []

    weight_convergence_steps = torch.full_like(model.layers[-1].weight, torch.inf)
    optimal_weights = compute_optimal_weights(task, model, criterion, cfg.get('optimal_weight_device', cfg.device))
    
    # Buffers used to check for convergence
    recent_targets = []
    recent_losses = []

    while step < cfg.train.total_steps:

        # Generate batch of data
        inputs, targets = next(task_iterator)

        flat_targets = targets.view(-1).tolist()
        target_buffer.extend(flat_targets)
        recent_targets.extend(flat_targets)
        recent_targets = recent_targets[-CONVERGENCE_STEPS * len(flat_targets):]
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        if cfg.train.standardize_cumulants:
            raise NotImplementedError("Need to implement for computing optimal weights first")
            with torch.no_grad():
                targets, cumulant_mean, cumulant_square_mean = standardize_targets(
                    targets, cumulant_mean, cumulant_square_mean, cumulant_gamma, step)

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            total_pruned += sum([len(idxs) for idxs in pruned_idxs.values()])

        feature_pruned = cbp_tracker and pruned_idxs.get(model.layers[-1], 0) > 0
        if feature_pruned:
            print('FEATURE PRUNED')
            step_since_feature_pruned = 0
            # When a feature is pruned, we need to calculate the new optimal weights
            optimal_weights = compute_optimal_weights(task, model, criterion, cfg.device)
        else:
            step_since_feature_pruned += 1
        
        # Update the convergence tracker for each feature
        weight_convergence_steps = compute_weight_convergence_steps(
            model, optimal_weights, weight_convergence_steps, step) 

        # convergence_threshold = OPTIMAL_WEIGHT_LOSS_THRESHOLD * np.mean(np.array(target_buffer) ** 2)
        # prior_loss_avg = np.mean(recent_losses[:CONVERGENCE_STEPS // 2])
        # recent_loss_avg = np.mean(recent_losses[CONVERGENCE_STEPS // 2:])
        # print(f'Recent loss: {np.mean(recent_losses):.5f}, Std: {np.std(recent_losses):.5f}, Treshold: {convergence_threshold:.5f}, Prior loss: {prior_loss_avg:.5f}, Recent loss: {recent_loss_avg:.5f}')
        # if len(recent_losses) >= CONVERGENCE_STEPS and prior_loss_avg - recent_loss_avg < convergence_threshold:
        #     logger.info(f'Model converged in {step} steps with a loss of {np.mean(recent_losses):.5f}')
        #     break

        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)
        recent_losses.append(loss.item())
        recent_losses = recent_losses[-CONVERGENCE_STEPS:]

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
                'accuracy': accuracy_acc / n_steps_since_log if isinstance(criterion, nn.CrossEntropyLoss) else None,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
            }
            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'], accuracy=metrics['accuracy'])
            
            # Reset accumulators
            loss_acc = 0.0
            accuracy_acc = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1
        pbar.update(1)
    
    pbar.close()
    wandb.finish()


@hydra.main(config_path='../conf', config_name='feature_maturity_defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)

    run_experiment(cfg, task, task_iterator, model, criterion, optimizer, cbp_tracker)


if __name__ == '__main__':
    main()
