import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from phd.feature_search.core.feature_recycling import reset_input_weights
from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.experiment_helpers import *
from phd.research_utils.logging import *


@hydra.main(config_path='../conf', config_name='defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    init_experiment(cfg.project, cfg)
    
    task, task_iterator, model, criterion, optimizer, input_recycler, cbp_tracker = \
        prepare_components(cfg)

    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulant_stats = StandardizationStats(gamma=0.99)
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    total_pruned = 0
    target_buffer = []
    
    while step < cfg.train.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        
        # Get recycled features
        features, recycled_features = input_recycler.step(
            batch_size = inputs.size(0),
            real_features = inputs,
            first_layer_weights = model.get_first_layer_weights(),
            step_num = step
        )
        
        if cfg.train.standardize_cumulants:
            with torch.no_grad():
                targets, cumulant_stats = standardize_targets(targets, cumulant_stats)
        
        target_buffer.extend(targets.view(-1).tolist())
        features, targets = features.to(cfg.device), targets.to(cfg.device)

        # Reset weights and optimizer states for recycled features
        reset_input_weights(recycled_features, model, optimizer, cfg)
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            total_pruned += sum([len(idxs) for idxs in pruned_idxs.values()])

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
            # Add recycler statistics
            metrics.update(input_recycler.get_statistics(step, model, optimizer))
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
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
