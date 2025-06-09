import hashlib
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import omegaconf
from omegaconf import DictConfig

from .adam import Adam
from .feature_recycling import InputRecycler, CBPTracker
from .idbd import IDBD, RMSPropIDBD
from .models import MLP
from .tasks import DummyTask, GEOFFTask, NonlinearGEOFFTask


# Only register resolver if it hasn't been registered yet
if not omegaconf.OmegaConf.has_resolver('eval'):
    omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))


def prepare_task(cfg: DictConfig, seed: Optional[int] = None):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'dummy':
        return DummyTask(cfg.task.n_features, cfg.model.output_dim, cfg.task.type)
    elif cfg.task.name.lower() == 'static_linear_geoff':
        # Non-stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, -1, cfg.task.n_real_features, seed=seed)
    elif cfg.task.name.lower() == 'linear_geoff':
        # Stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, 20, cfg.task.n_real_features, seed=seed)
    elif cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features=cfg.task.n_real_features,
            flip_rate=cfg.task.flip_rate,
            n_layers=cfg.task.n_layers,
            n_stationary_layers=cfg.task.n_stationary_layers,
            hidden_dim=cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale=cfg.task.weight_scale,
            activation=cfg.task.activation,
            sparsity=cfg.task.sparsity,
            weight_init=cfg.task.weight_init,
            seed=seed,
        )


def prepare_optimizer(
        model: Union[nn.Module, List[nn.Module]], 
        optimizer_name: str,
        optimizer_kwargs: DictConfig,
    ):
    """Prepare the optimizer based on configuration.
    
    Uses default values for parameters not specified in config, while allowing
    irrelevant parameters to be specified without causing errors.
    """
    if isinstance(model, nn.Module):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = []
        for m in model:
            trainable_params.extend([p for p in m.parameters() if p.requires_grad])

    def _extract_kwargs(param_names, defaults = None):
        """Extract specified parameters from config, using defaults when not provided."""
        if defaults is None:
            defaults = {}
        
        kwargs = {}
        for param_name in param_names:
            value = optimizer_kwargs.get(param_name)
            if value is not None:
                kwargs[param_name] = value
            elif param_name in defaults:
                kwargs[param_name] = defaults[param_name]
        return kwargs

    if optimizer_name == 'adam':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        # Map learning_rate to lr for consistency with PyTorch API
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        return Adam(trainable_params, **kwargs)
        
    elif optimizer_name == 'rmsprop':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        return Adam(
            trainable_params,
            betas=(0, 0.999),
            **kwargs
        )
        
    elif optimizer_name == 'sgd':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        return optim.SGD(trainable_params, **kwargs)
        
    elif optimizer_name == 'sgd_momentum':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0, 'momentum': 0.9})
        if 'learning_rate' in kwargs:
            kwargs['lr'] = kwargs.pop('learning_rate')
        return optim.SGD(trainable_params, **kwargs)
        
    elif optimizer_name == 'idbd':
        kwargs = _extract_kwargs(
            ['learning_rate', 'meta_learning_rate', 'version', 'weight_decay', 'autostep'], 
            {'version': 'squared_grads', 'weight_decay': 0, 'autostep': True}
        )
        # Map learning_rate to init_lr for IDBD API
        if 'learning_rate' in kwargs:
            kwargs['init_lr'] = kwargs.pop('learning_rate')
        if 'meta_learning_rate' in kwargs:
            kwargs['meta_lr'] = kwargs.pop('meta_learning_rate')
        return IDBD(trainable_params, **kwargs)
        
    else:
        raise ValueError(f'Invalid optimizer type: {optimizer_name}')


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_from_string(seed: Optional[int], string: str) -> Optional[int]:
    """Deterministic hash of a string."""
    if seed is None:
        return random.randint(0, 2**32)
    return seed + int(hashlib.md5(string.encode()).hexdigest(), 16) % (2**32)


def get_model_statistics(
    model: MLP,
    features: torch.Tensor,
    param_inputs: Dict[str, torch.Tensor],
    masks: Optional[List[torch.Tensor]] = None,
    metric_prefix: str = '',
) -> Dict[str, float]:
    """
    Compute statistics about the model's weights, biases, and layer inputs.
    
    Args:
        model: The MLP model to analyze
        features: Input features to compute layer activations
        param_inputs: Dictionary mapping weight parameters to their inputs
        masks: Optional list of boolean masks for computing statistics only on certain
            units in each layer. If provided, must contain one mask per layer.
        
    Returns:
        Dictionary containing various model statistics
    """
    stats = {}
    
    # Compute statistics for each layer
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    
    if masks is not None:
        # Verify we have one mask per layer
        assert len(masks) == len(linear_layers), \
            f"Expected {len(linear_layers)} masks but got {len(masks)}"
    
    for i, layer in enumerate(linear_layers):
        mask = masks[i] if masks is not None else \
            torch.ones(layer.weight.shape[1], dtype=torch.bool, device=layer.weight.device)
            # torch.ones_like(layer.weight[:,0], dtype=torch.bool)
        
        # Weight norms (masked)
        weight_masked = layer.weight[:, mask]
        if weight_masked.numel() > 0:
            weight_l1 = torch.norm(weight_masked, p=1).item() / weight_masked.numel()
            stats[f'layer_{i}/{metric_prefix}weight_l1'] = weight_l1
        else:
            stats[f'layer_{i}/{metric_prefix}weight_l1'] = 0.0
        
        # # Bias norms (if exists, masked)
        # if layer.bias is not None:
        #     bias_masked = layer.bias[mask]
        #     bias_l1 = torch.norm(bias_masked, p=1).item() / bias_masked.numel()
        #     stats[f'layer_{i}/bias_l1'] = bias_l1
        
        # Input norms (masked)
        if mask.sum() > 0:
            if i == 0:
                input_l1 = torch.norm(features[:, mask], p=1, dim=1) / mask.sum()
                input_l1 = input_l1.mean().item()
            else:
                layer_inputs = param_inputs[layer.weight]
                if layer_inputs.ndim == 1:
                    layer_inputs = layer_inputs.unsqueeze(0)
                input_l1 = torch.norm(layer_inputs[:, mask], p=1, dim=-1) / mask.sum()
                input_l1 = input_l1.mean().item()
        else:
            input_l1 = 0.0
        stats[f'layer_{i}/{metric_prefix}input_l1'] = input_l1
    
    return stats


class StandardizationStats(nn.Module):
    """Holds running statistics for standardization."""
    def __init__(
        self,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.register_buffer('running_mean', torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer('step', torch.tensor(0, device=device))
        self.gamma = gamma


@torch.no_grad()
def standardize_targets(
    targets: torch.Tensor,
    stats: StandardizationStats,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, StandardizationStats]:
    """Exponentially-weighted Welford normalisation (EW-Welford).

    Normalises a 2-D tensor of shape ``(batch, 1)`` so that its running mean
    approaches zero and its running standard deviation approaches one, while
    keeping **O(1)** state and compute per call.  Statistics adapt to concept
    drift via the forgetting factor ``gamma``.

    Args:
        targets: Input tensor of shape ``(batch, 1)`` on any device / dtype.
        stats: StandardizationStats object containing running statistics.
        eps: Small constant added for numerical stability; safeguards against
            division by zero and negative variance caused by round-off.

    Returns:
        Tuple containing:
            - **z** (*torch.Tensor*): Normalised tensor with the same shape as
              ``targets``.
            - **new_stats** (*StandardizationStats*): Updated running statistics.

    Example:
        ```python
        stats = StandardizationStats(gamma=0.99, device="cuda")

        for batch in data_stream:               # batch shape: (B, 1)
            batch = batch.cuda()
            z, stats = standardize_targets(batch, stats)
            # ... use z for loss / back-prop ...
        ```
    """
    # --------------------------------------------------------------------- #
    # 1. Normalize the current batch using statistics **from the prev step**.
    # --------------------------------------------------------------------- #
    var_safe = stats.running_var.clamp_min(eps)  # ensure σ² ≥ eps
    std = torch.sqrt(var_safe)
    z = (targets - stats.running_mean) / std

    # --------------------------------------------------------------------- #
    # 2. Update running statistics with the batch mean (EW-Welford update).
    # --------------------------------------------------------------------- #
    alpha = 1.0 - stats.gamma                    # EW learning rate
    batch_mean = targets.mean()                  # scalar (dim == 1)
    delta = batch_mean - stats.running_mean
    stats.running_mean.add_(alpha * delta)       # μ_t

    delta2 = batch_mean - stats.running_mean
    stats.running_var.mul_(stats.gamma).add_(alpha * delta * delta2)

    # Numerical hygiene: clamp and squash accidental NaNs.
    stats.running_var.clamp_min_(eps)
    if torch.isnan(stats.running_var):
        stats.running_var.fill_(eps)

    stats.step.add_(1)
    return z, stats


def prepare_components(cfg: DictConfig, model: Optional[nn.Module] = None):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    if model is None:
        model = MLP(
            input_dim = cfg.task.n_features,
            output_dim = cfg.model.output_dim,
            n_layers = cfg.model.n_layers,
            hidden_dim = cfg.model.hidden_dim,
            weight_init_method = cfg.model.weight_init_method,
            activation = cfg.model.activation,
            n_frozen_layers = cfg.model.n_frozen_layers,
            seed = seed_from_string(base_seed, 'model'),
        )
    model.to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
    
    # Initialize feature recycler
    recycler = InputRecycler(
        n_features = cfg.task.n_features,
        n_real_features = cfg.task.n_real_features,
        distractor_chance = cfg.input_recycling.distractor_chance,
        recycle_rate = cfg.input_recycling.recycle_rate,
        utility_decay = cfg.input_recycling.utility_decay,
        use_cbp_utility = cfg.input_recycling.use_cbp_utility,
        feature_protection_steps = cfg.input_recycling.feature_protection_steps,
        n_start_real_features = cfg.input_recycling.get('n_start_real_features', -1),
        device = 'cpu',
        seed = seed_from_string(base_seed, 'recycler'),
    )
    
    # Initialize CBP tracker
    if cfg.feature_recycling.use_cbp_utility:
        cbp_tracker = CBPTracker(
            optimizer = optimizer,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate = cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.feature_recycling.feature_protection_steps,
            seed = seed_from_string(base_seed, 'cbp_tracker'),
        )
        cbp_tracker.track_sequential(model.layers)
    else:
        cbp_tracker = None
        
    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker