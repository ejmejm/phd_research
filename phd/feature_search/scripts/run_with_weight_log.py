import os
import sys
from typing import Dict
import os.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from phd.feature_search.core.idbd import IDBD, RMSPropIDBD
from phd.feature_search.core.models import MLP
from phd.feature_search.core.experiment_helpers import *


def get_model_statistics(model: MLP, param_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics about individual weights, biases, inputs and total influence paths for each layer.
    
    Args:
        model: The MLP model to analyze
        param_inputs: Dictionary mapping weight parameters to their inputs
        
    Returns:
        Dictionary containing individual weight, bias, input values and total influence paths for each layer
    """
    stats = {}
    
    # Compute statistics for each layer
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        # Individual weights
        for in_idx in range(layer.weight.shape[1]):
            for out_idx in range(layer.weight.shape[0]):
                weight_val = layer.weight[out_idx, in_idx].item()
                stats[f'layer_{i}/w_{in_idx}_{out_idx}'] = weight_val
        
        # Individual biases (if exists)
        if layer.bias is not None:
            for idx in range(layer.bias.shape[0]):
                bias_val = layer.bias[idx].item()
                stats[f'layer_{i}/b_{idx}'] = bias_val
        
        layer_inputs = param_inputs[layer.weight]
        assert len(layer_inputs.shape) == 1
        for idx in range(layer_inputs.shape[0]):
            stats[f'layer_{i}/input_{idx}'] = layer_inputs[idx].item()
    
    # Compute total influence paths for first layer inputs only if 2-layer network with 1 output
    if len(linear_layers) == 2 and linear_layers.weight.shape[0] == 1:
        n_inputs = linear_layers[0].weight.shape[1]
        for input_idx in range(n_inputs):
            total_influence = 0.0
            # For each hidden unit in first layer
            for hidden_idx in range(linear_layers[0].weight.shape[0]):
                path_product = linear_layers[0].weight[hidden_idx, input_idx].item()
                # Multiply by weight to output (assuming 1 output)
                path_product *= linear_layers[1].weight[0, hidden_idx].item()
                total_influence += path_product
            stats[f'input_{input_idx}_influence'] = total_influence
    
    return stats


@hydra.main(config_path='../conf', config_name='defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    set_seed(cfg.seed)
    
    if not cfg.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Initialize wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.project, config=wandb_config, allow_val_change=True)
    
    task = prepare_task(cfg)
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim=cfg.task.n_features,
        output_dim=cfg.model.output_dim,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        weight_init_method=cfg.model.weight_init_method,
        activation=cfg.model.activation,
        device=cfg.device
    ).to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    optimizer = prepare_optimizer(model, cfg)
    
    # Initialize feature recycler
    recycler = InputRecycler(
        n_features=cfg.task.n_features,
        n_real_features=cfg.task.n_real_features,
        distractor_chance=cfg.input_recycling.distractor_chance,
        recycle_rate=cfg.input_recycling.recycle_rate,
        utility_decay=cfg.input_recycling.utility_decay,
        use_cbp_utility=cfg.input_recycling.use_cbp_utility,
        feature_protection_steps=cfg.input_recycling.feature_protection_steps,
        device=cfg.device,
    )
    
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    target_buffer = []
    
    while step < cfg.train.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        target_buffer.extend(targets.view(-1).tolist())
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        # Get recycled features
        features, recycled_features = recycler.step(
            batch_size=inputs.size(0),
            real_features=inputs,
            first_layer_weights=model.get_first_layer_weights(),
            step_num=step
        )

        # Reset weights and optimizer states for recycled features
        reset_feature_weights(recycled_features, model, optimizer, cfg)
        
        # Forward pass
        outputs, param_inputs = model(features)
        param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, RMSPropIDBD):
            loss.backward(create_graph=True)
            optimizer.step(param_inputs)
        elif isinstance(optimizer, IDBD):
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
            }
            # Add recycler statistics
            metrics.update(recycler.get_statistics(step, model, optimizer))
            # Add model statistics
            metrics.update(get_model_statistics(model, param_inputs))
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

if __name__ == '__main__':
    main()
