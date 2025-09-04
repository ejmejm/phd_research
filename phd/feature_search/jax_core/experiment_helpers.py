from ctypes import c_int32
import hashlib
import random
from typing import Optional, Tuple

import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, PRNGKeyArray
import numpy as np
import omegaconf
from omegaconf import DictConfig
import optax

from .models import MLP
from .optimizer import EqxOptimizer
from .tasks.geoff import NonlinearGEOFFTask
from .utils import tree_replace


# Only register resolver if it hasn't been registered yet
if not omegaconf.OmegaConf.has_resolver('eval'):
    omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))
    
if not omegaconf.OmegaConf.has_resolver('as_tuple'):
    omegaconf.OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))


def prepare_task(cfg: DictConfig, seed: Optional[int] = None):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features = cfg.task.n_real_features,
            flip_rate = cfg.task.flip_rate,
            n_layers = cfg.task.n_layers,
            n_stationary_layers = cfg.task.n_stationary_layers,
            hidden_dim = cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale = cfg.task.weight_scale,
            activation = cfg.task.activation,
            sparsity = cfg.task.sparsity,
            weight_init = cfg.task.weight_init,
            seed = seed,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task.name}")


def prepare_optimizer(
        model: eqx.Module, 
        optimizer_name: str,
        optimizer_kwargs: DictConfig,
        filter_spec: Optional[PyTree] = None,
    )-> EqxOptimizer:
    """Prepare the optimizer based on configuration.
    
    Uses default values for parameters not specified in config, while allowing
    irrelevant parameters to be specified without causing errors.
    """
    if filter_spec is None:
        filter_spec = jax.tree.map(lambda _: False, model)
        filter_spec = eqx.tree_at(
            lambda x: x.layers[model.n_frozen_layers:],
            filter_spec,
            jax.tree.map(lambda _: True, model.layers[model.n_frozen_layers:]),
        )

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
        optimizer = optax.adam(learning_rate=kwargs['learning_rate'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec)
        
    elif optimizer_name == 'rmsprop':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        optimizer = optax.rmsprop(learning_rate=kwargs['learning_rate'], decay=0.999)
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec)
        
    elif optimizer_name == 'sgd':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        optimizer = optax.sgd(learning_rate=kwargs['learning_rate'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec)
    
    elif optimizer_name == 'sgd_momentum':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0, 'momentum': 0.9})
        optimizer = optax.sgd(learning_rate=kwargs['learning_rate'], momentum=kwargs['momentum'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec)
        
    elif optimizer_name == 'idbd':
        raise NotImplementedError('IDBD optimizer is not implemented for JAX yet.')
        # kwargs = _extract_kwargs(
        #     ['learning_rate', 'meta_learning_rate', 'version', 'weight_decay', 'autostep', 'step_size_decay'], 
        #     {'version': 'squared_grads', 'weight_decay': 0, 'autostep': True}
        # )
        # # Map learning_rate to init_lr for IDBD API
        # if 'learning_rate' in kwargs:
        #     kwargs['init_lr'] = kwargs.pop('learning_rate')
        # if 'meta_learning_rate' in kwargs:
        #     kwargs['meta_lr'] = kwargs.pop('meta_learning_rate')
        # return IDBD(trainable_params, **kwargs)
        
    else:
        raise ValueError(f'Invalid optimizer type: {optimizer_name}')


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        jax.random.seed(seed)
        np.random.seed(seed)


def seed_from_string(seed: Optional[int], string: str) -> Optional[int]:
    """Deterministic hash of a string."""
    if seed is None:
        return random.randint(0, 2**32)
    return seed + int(hashlib.md5(string.encode()).hexdigest(), 16) % (2**32)


def rng_from_string(rng: Optional[PRNGKeyArray], string: str) -> PRNGKeyArray:
    """Rng key based on prior key + deterministic hash of a string."""
    if rng is None:
        return jax.random.key(random.randint(0, 2**31))
    string_int = c_int32(int(hashlib.md5(string.encode()).hexdigest(), 16))
    return jax.random.fold_in(rng, string_int)


# def get_model_statistics(
#     model: MLP,
#     features: Float[Array, 'batch features'],
#     param_inputs: List[Float[Array, 'batch features']],
#     masks: Optional[List[Bool[Array, 'features']]] = None,
#     metric_prefix: str = '',
# ) -> Dict[str, float]:
#     """
#     Compute statistics about the model's weights, biases, and layer inputs.
    
#     Args:
#         model: The MLP model to analyze
#         features: Input features to compute layer activations
#         param_inputs: List of input arrays to each layer (from model forward pass)
#         masks: Optional list of boolean masks for computing statistics only on certain
#             units in each layer. If provided, must contain one mask per layer.
#         metric_prefix: Prefix to add to metric names
        
#     Returns:
#         Dictionary containing various model statistics
#     """
    
#     def compute_layer_stats(i: int, layer: Any, layer_input: Array, mask: Optional[Array]) -> Tuple[float, float]:
#         """Compute statistics for a single layer."""
#         # Create default mask if none provided
#         if mask is None:
#             mask = jnp.ones(layer.weight.shape[1], dtype=bool)
        
#         # Weight norms (masked)
#         weight_masked = layer.weight[:, mask]
#         weight_l1 = jnp.where(
#             weight_masked.size > 0,
#             jnp.linalg.norm(weight_masked, ord=1) / weight_masked.size,
#             0.0
#         )
        
#         # Input norms (masked)
#         mask_sum = jnp.sum(mask)
#         input_l1 = jnp.where(
#             mask_sum > 0,
#             jnp.mean(jnp.linalg.norm(layer_input[:, mask], ord=1, axis=-1) / mask_sum),
#             0.0
#         )
        
#         return weight_l1, input_l1
    
#     # Get all linear layers (in JAX MLP, all layers are linear)
#     n_layers = len(model.layers)
    
#     # Verify param_inputs length matches number of layers
#     assert len(param_inputs) == n_layers, \
#         f"Expected {n_layers} param_inputs but got {len(param_inputs)}"
    
#     if masks is not None:
#         assert len(masks) == n_layers, \
#             f"Expected {n_layers} masks but got {len(masks)}"
    
#     # Compute statistics for each layer
#     stats = {}
#     for i, layer in enumerate(model.layers):
#         mask = masks[i] if masks is not None else None
#         layer_input = param_inputs[i]
        
#         # Ensure layer_input is 2D (add batch dimension if needed)
#         if layer_input.ndim == 1:
#             layer_input = jnp.expand_dims(layer_input, axis=0)
        
#         weight_l1, input_l1 = compute_layer_stats(i, layer, layer_input, mask)
        
#         stats[f'layer_{i}/{metric_prefix}weight_l1'] = float(weight_l1)
#         stats[f'layer_{i}/{metric_prefix}input_l1'] = float(input_l1)
    
#     return stats



class StandardizationStats(eqx.Module):
    """Holds running statistics for standardization."""
    def __init__(self, gamma: float = 0.99):
        self.running_mean = jnp.zeros(1)
        self.running_var = jnp.ones(1)
        self.step = jnp.zeros(1)
        self.gamma = gamma


def standardize_targets(
    targets: Float[Array, 'batch 1'],
    stats: StandardizationStats,
    eps: float = 1e-8,
) -> Tuple[Float[Array, 'batch 1'], StandardizationStats]:
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
    var_safe = jnp.clip(stats.running_var, min=eps) # ensure σ² ≥ eps
    std = jnp.sqrt(var_safe)
    z = (targets - stats.running_mean) / std

    # --------------------------------------------------------------------- #
    # 2. Update running statistics with the batch mean (EW-Welford update).
    # --------------------------------------------------------------------- #
    alpha = 1.0 - stats.gamma                    # EW learning rate
    batch_mean = targets.mean()                  # scalar (dim == 1)
    delta = batch_mean - stats.running_mean
    running_mean = stats.running_mean + alpha * delta

    delta2 = batch_mean - running_mean
    running_var = stats.running_var * stats.gamma + alpha * delta * delta2
    running_var = jnp.clip(running_var, min=eps)

    # Numerical hygiene: clamp and squash accidental NaNs.
    if jnp.isnan(running_var):
        running_var[:] = eps

    return z, tree_replace(
        stats,
        running_mean = running_mean,
        running_var = running_var,
        step = stats.step + 1,
    )


def prepare_components(cfg: DictConfig, model: Optional[eqx.Module] = None):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    
    task = prepare_task(cfg, seed=seed_from_string(rng, 'task'))
    
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
            key = rng_from_string(rng, 'model'),
        )
    model.to(cfg.device)
    
    criterion = (optax.softmax_cross_entropy if cfg.task.type == 'classification'
                else optax.l2_loss)
    optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
    
    # # TODO: Implement CBP tracker in JAX
    # # Initialize CBP tracker
    # if cfg.feature_recycling.use_cbp_utility:
    #     cbp_tracker = CBPTracker(
    #         optimizer = optimizer,
    #         replace_rate = cfg.feature_recycling.recycle_rate,
    #         decay_rate = cfg.feature_recycling.utility_decay,
    #         maturity_threshold = cfg.feature_recycling.feature_protection_steps,
    #         initial_step_size_method = cfg.feature_recycling.initial_step_size_method,
    #         seed = seed_from_string(base_seed, 'cbp_tracker'),
    #     )
    #     cbp_tracker.track_sequential(model.layers)
    # else:
    #     cbp_tracker = None
    
    cbp_tracker = None
    
    return task, model, criterion, optimizer, cbp_tracker