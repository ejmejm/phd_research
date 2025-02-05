import math
import os
import random
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from adam import Adam
from feature_recycling import InputRecycler, CBPTracker
from idbd import IDBD, RMSPropIDBD
from models import MLP
from tasks import DummyTask, GEOFFTask, NonlinearGEOFFTask


omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))


def reset_feature_weights(idxs: Union[int, Sequence[int]], model: MLP, optimizer: optim.Optimizer, cfg: DictConfig):
    """Reset the weights and associated optimizer state for a feature."""
    if isinstance(idxs, Sequence) and len(idxs) == 0:
        return
    
    first_layer = model.layers[0]
    
    # Reset weights
    if cfg.model.weight_init_method == 'zeros':
        with torch.no_grad():
            first_layer.weight[:, idxs] = 0
    elif cfg.model.weight_init_method == 'kaiming_uniform':
        fan = first_layer.weight.shape[1] # fan_in
        gain = 1
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            first_layer.weight[:, idxs] = first_layer.weight[:, idxs].uniform_(-bound, bound)
    else:
        raise ValueError(f'Invalid weight initialization method: {cfg.model.weight_init_method}')

    # Reset optimizer states
    if isinstance(optimizer, Adam):
        # Reset Adam state for the specific feature
        state = optimizer.state[first_layer.weight]
        if len(state) > 0: # State is only populated after the first call to step
            state['step'][:, idxs] = 0
            state['exp_avg'][:, idxs] = 0
            state['exp_avg_sq'][:, idxs] = 0
            if 'max_exp_avg_sq' in state:  # For AMSGrad
                state['max_exp_avg_sq'][:, idxs] = 0
    elif isinstance(optimizer, IDBD):
        state = optimizer.state[first_layer.weight]
        state['beta'][:, idxs] = math.log(cfg.train.learning_rate)
        state['h'][:, idxs] = 0
    else:
        raise ValueError(f'Invalid optimizer type: {type(optimizer)}')


def prepare_task(cfg: DictConfig):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'dummy':
        return DummyTask(cfg.task.n_features, cfg.model.output_dim, cfg.task.type)
    elif cfg.task.name.lower() == 'static_linear_geoff':
        # Non-stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, -1, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'linear_geoff':
        # Stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, 20, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features=cfg.task.n_real_features,
            flip_rate=cfg.task.flip_rate,
            n_layers=cfg.task.n_layers,
            hidden_dim=cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale=cfg.task.weight_scale,
            activation=cfg.task.activation,
            sparsity=cfg.task.sparsity,
            weight_init=cfg.task.weight_init,
        )


def prepare_optimizer(model: nn.Module, cfg: DictConfig):
    """Prepare the optimizer based on configuration."""
    if cfg.train.optimizer == 'adam':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'rmsprop':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            betas=(0, 0.999),
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'sgd_momentum':
        return optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
        )
    elif cfg.train.optimizer == 'idbd':
        return IDBD(
            model.parameters(),
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            version=cfg.idbd.version,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'rmsprop_idbd':
        return RMSPropIDBD(
            model.parameters(),
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            trace_diagonal_approx=cfg.idbd.diagonal_approx,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f'Invalid optimizer type: {cfg.train.optimizer}')


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_statistics(model: MLP, features: torch.Tensor, param_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics about the model's weights, biases, and layer inputs.
    
    Args:
        model: The MLP model to analyze
        features: Input features to compute layer activations
        param_inputs: Dictionary mapping weight parameters to their inputs
        
    Returns:
        Dictionary containing various model statistics
    """
    stats = {}
    
    # Compute statistics for each layer
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        # Weight norms
        weight_l1 = torch.norm(layer.weight, p=1).item() / layer.weight.numel()

        stats[f'layer_{i}/weight_l1'] = weight_l1
        
        # Bias norms (if exists)
        if layer.bias is not None:
            bias_l1 = torch.norm(layer.bias, p=1).item() / layer.bias.numel()
            stats[f'layer_{i}/bias_l1'] = bias_l1
        
        # Input norms
        if i == 0:
            input_l1 = torch.norm(features, p=1, dim=1).mean().item() / features.shape[1]
        else:
            layer_inputs = param_inputs[layer.weight]
            input_l1 = torch.norm(layer_inputs, p=1, dim=-1).mean().item() / layer_inputs.shape[-1]
        stats[f'layer_{i}/input_l1'] = input_l1
    
    return stats


def standardize_targets(
    targets: torch.Tensor,
    cumulant_mean: float,
    cumulant_square_mean: float,
    cumulant_gamma: float,
    step: int,
) -> torch.Tensor:
    """Standardize targets using a running mean and variance."""
    cumulant_mean = cumulant_gamma * cumulant_mean + (1 - cumulant_gamma) * targets.mean()
    cumulant_square_mean = cumulant_gamma * cumulant_square_mean + (1 - cumulant_gamma) * targets.square().mean()
    bias_correction = 1 / (1 - cumulant_gamma ** (step + 1))
    curr_mean = cumulant_mean * bias_correction
    curr_square_mean = cumulant_square_mean * bias_correction
    std_dev = (curr_square_mean - curr_mean.square()).sqrt()
    std_dev = 1 if std_dev == 0 else std_dev
    targets = (targets - curr_mean) / std_dev
    return targets, cumulant_mean, cumulant_square_mean



@hydra.main(config_path='conf', config_name='defaults')
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
        n_start_real_features=cfg.input_recycling.get('n_start_real_features', -1),
        device=cfg.device,
    )
    
    # Initialize CBP tracker
    if cfg.feature_recycling.use_cbp_utility:
        cbp_tracker = CBPTracker(
            optimizer = optimizer,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate=cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.input_recycling.feature_protection_steps,
        )
        cbp_tracker.track_sequential(model.layers)
    else:
        cbp_tracker = None
    

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
    cumulant_gamma = 0.999
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
        
        if cfg.train.standardize_cumulants:
            with torch.no_grad():
                targets, cumulant_mean, cumulant_square_mean = standardize_targets(
                    targets, cumulant_mean, cumulant_square_mean, cumulant_gamma, step)

        # Reset weights and optimizer states for recycled features
        reset_feature_weights(recycled_features, model, optimizer, cfg)
        if cbp_tracker is not None:
            cbp_tracker.prune_features()

        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, RMSPropIDBD):
            loss.backward(create_graph=True)
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            optimizer.step(param_inputs)
        elif isinstance(optimizer, IDBD):
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
            }
            # Add recycler statistics
            metrics.update(recycler.get_statistics(step, model, optimizer))
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

if __name__ == '__main__':
    main()
