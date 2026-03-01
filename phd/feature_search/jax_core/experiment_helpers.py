from ctypes import c_int32
import hashlib
import logging
import random
from typing import Optional, Tuple

import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray
import numpy as np
import omegaconf
from omegaconf import DictConfig
import optax

from .feature_recycling import CBPTracker
from phd.jax_core.models import MLP
from phd.jax_core.optimizers import EqxOptimizer, optax_idbd, optax_upgd, custom_optax_adam
from phd.jax_core.tasks.geoff import BinaryRegressionTask, CoreTransientBinaryTask, InputChangingGEOFFTask, NonlinearGEOFFTask
from phd.jax_core.tasks.summation import SummationTask
from phd.jax_core.utils import tree_replace


logger = logging.getLogger(__name__)


# Only register resolver if it hasn't been registered yet
if not omegaconf.OmegaConf.has_resolver('eval'):
    omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))
    
if not omegaconf.OmegaConf.has_resolver('as_tuple'):
    omegaconf.OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))


def prepare_task(cfg: DictConfig, seed: Optional[int] = None):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features = cfg.task.n_features,
            n_outputs = cfg.task.n_outputs,
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
    elif cfg.task.name.lower() == 'input_changing_geoff':
        cfg.task.type = 'regression'
        return InputChangingGEOFFTask(
            n_features = cfg.task.n_features,
            n_outputs = cfg.task.n_outputs,
            flip_rate = cfg.task.flip_rate,
            n_layers = cfg.task.n_layers,
            n_stationary_layers = cfg.task.n_stationary_layers,
            hidden_dim = cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale = cfg.task.weight_scale,
            activation = cfg.task.activation,
            sparsity = cfg.task.sparsity,
            weight_init = cfg.task.weight_init,
            input_bounds = cfg.task.input_bounds,
            input_subspace_range = cfg.task.input_subspace_range,
            input_change_freq = cfg.task.input_change_freq,
            max_input_center_change = cfg.task.max_input_center_change,
            seed = seed,
        )
    elif cfg.task.name.lower() == 'binary_regression':
        cfg.task.type = 'regression'
        return BinaryRegressionTask(
            n_features = cfg.task.n_features,
            n_outputs = cfg.task.n_outputs,
            flip_rate = cfg.task.flip_rate,
            n_stationary_layers = cfg.task.n_stationary_layers,
            hidden_dim = cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale = cfg.task.weight_scale,
            sparsity = cfg.task.sparsity,
            input_bounds = cfg.task.input_bounds,
            input_subspace_range = cfg.task.input_subspace_range,
            input_change_freq = cfg.task.input_change_freq,
            max_input_center_change = cfg.task.max_input_center_change,
            seed = seed,
        )
    elif cfg.task.name.lower() == 'core_transient_binary':
        cfg.task.type = 'regression'
        return CoreTransientBinaryTask(
            n_features = cfg.task.n_features,
            n_core_layers = cfg.task.n_core_layers,
            core_hidden_dim = cfg.task.core_hidden_dim,
            core_sparsity = cfg.task.core_sparsity,
            n_transient_layers = cfg.task.n_transient_layers,
            transient_hidden_dim = cfg.task.transient_hidden_dim,
            transient_sparsity = cfg.task.transient_sparsity,
            transient_activation_rate = cfg.task.transient_activation_rate,
            n_outputs = cfg.task.n_outputs,
            weight_scale = cfg.task.weight_scale,
            input_bounds = cfg.task.input_bounds,
            input_subspace_range = cfg.task.input_subspace_range,
            input_change_freq = cfg.task.input_change_freq,
            max_input_center_change = cfg.task.max_input_center_change,
            n_calibration_samples = cfg.task.get('n_calibration_samples', 10000),
            seed = seed,
        )
    elif cfg.task.name.lower() == 'summation':
        cfg.task.type = 'regression'
        return SummationTask(
            n_features = cfg.task.n_features,
            subset_size = cfg.task.subset_size,
            change_subset_freq = cfg.task.change_subset_freq,
            flip_multiplier_freq = cfg.task.flip_multiplier_freq,
            input_min = cfg.task.input_min,
            input_max = cfg.task.input_max,
            initial_multiplier = cfg.task.initial_multiplier,
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
        weight_decay = kwargs.pop('weight_decay')
        kwargs['lr'] = kwargs.pop('learning_rate')
        optimizer = custom_optax_adam(**kwargs)
        if weight_decay != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(weight_decay))
        return EqxOptimizer(optimizer, model, filter_spec, name='adam')
        
    elif optimizer_name == 'rmsprop':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        weight_decay = kwargs.pop('weight_decay')
        kwargs['lr'] = kwargs.pop('learning_rate')
        kwargs['betas'] = (0, 0.999)
        optimizer = custom_optax_adam(**kwargs)
        if weight_decay != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(weight_decay))
        return EqxOptimizer(optimizer, model, filter_spec, name='rmsprop')
        
    elif optimizer_name == 'sgd':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        optimizer = optax.sgd(learning_rate=kwargs['learning_rate'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec, name='sgd')
    
    elif optimizer_name == 'sgd_momentum':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0, 'momentum': 0.9})
        optimizer = optax.sgd(learning_rate=kwargs['learning_rate'], momentum=kwargs['momentum'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec, name='sgd')
        
    elif optimizer_name == 'idbd':
        kwargs = _extract_kwargs(
            [
                'learning_rate', 'meta_learning_rate', 'weight_decay', 'autostep',
                'step_size_decay', 'version', 'shadow_weight_threshold_factor',
            ],
            {
                'version': 'prediction_grads', 'weight_decay': 0, 'autostep': True,
                'step_size_decay': 0.0, 'shadow_weight_threshold_factor': 0.0,
            },
        )
        kwargs['init_lr'] = kwargs.pop('learning_rate')
        kwargs['meta_lr'] = kwargs.pop('meta_learning_rate')
        optimizer = optax_idbd(**kwargs)
        return EqxOptimizer(optimizer, model, filter_spec, name='idbd')
    
    elif optimizer_name == 'upgd':
        kwargs = _extract_kwargs(
            ['learning_rate', 'weight_decay', 'beta_utility', 'sigma'],
            {'learning_rate': 1e-5, 'weight_decay': 0.001, 'beta_utility': 0.999, 'sigma': 0.001},
        )
        kwargs['lr'] = kwargs.pop('learning_rate')
        optimizer = optax_upgd(**kwargs)
        return EqxOptimizer(optimizer, model, filter_spec, name='upgd')
        
    else:
        raise ValueError(f'Invalid optimizer type: {optimizer_name}')


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
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


class StandardizationStats(eqx.Module):
    """Holds running statistics for standardization."""
    running_mean: Float[Array, ''] = eqx.field(default=1, converter=jnp.asarray)
    running_var: Float[Array, ''] = eqx.field(default=1, converter=jnp.asarray)
    step: Int[Array, ''] = eqx.field(default=0, converter=jnp.asarray)
    gamma: float = eqx.field(default=0.99)


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
    running_var = jnp.where(jnp.isnan(running_var), eps, running_var)

    return z, tree_replace(
        stats,
        running_mean = running_mean,
        running_var = running_var,
        step = stats.step + 1,
    )


def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    use_bias = cfg.model.get('use_bias', True)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim = cfg.task.n_features,
        output_dim = cfg.task.get('n_outputs', 1),
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim + int(use_bias),
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        use_normalize_and_project = cfg.model.get('use_normalize_and_project', False),
        key = rng_from_string(rng, 'model'),
    )
    
    criterion = (optax.softmax_cross_entropy if cfg.task.type == 'classification'
                else lambda x, y: jnp.square(y - x).mean())
    optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
    
    # Determine if we need separate optimizers for the intermediate and output layers
    repr_optimizer_name = cfg.get('representation_optimizer', {}).get('name')
    assert repr_optimizer_name != 'idbd', "IDBD is not supported for the representation optimizer!"
    n_repr_trainable_layers = max(0, len(model.layers) - 1 - model.n_frozen_layers)
    
    if repr_optimizer_name is not None and n_repr_trainable_layers > 0:
        base_filter_spec = jax.tree.map(lambda _: False, model)
        repr_filter_spec = eqx.tree_at(
            lambda x: x.layers[model.n_frozen_layers:len(model.layers) - 1],
            base_filter_spec,
            jax.tree.map(lambda _: True, model.layers[model.n_frozen_layers:len(model.layers) - 1]),
        )
        output_filter_spec = eqx.tree_at(
            lambda x: x.layers[-1],
            base_filter_spec,
            jax.tree.map(lambda _: True, model.layers[-1]),
        )
        
        # Use separate optimizers for the intermediate and output layers
        repr_optimizer = prepare_optimizer(model, repr_optimizer_name, cfg.representation_optimizer, filter_spec=repr_filter_spec)
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer, filter_spec=output_filter_spec)
        logger.info(f"Using separate optimizers for the intermediate and output layers: {repr_optimizer_name} and {cfg.optimizer.name}")
    else:
        # Only use one optimizer
        repr_optimizer = None
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
        logger.info(f"Using single optimizer: {cfg.optimizer.name}")
    
    # Initialize CBP tracker
    if cfg.feature_recycling.use_cbp_utility:
        cbp_tracker = CBPTracker(
            model = model,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate = cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.feature_recycling.feature_protection_steps,
            initial_step_size_method = cfg.feature_recycling.initial_step_size_method,
            incoming_weight_init = cfg.feature_recycling.incoming_weight_init,
            filter_spec = None, # Don't forget to add if doing more than 2 layers
            rng = rng_from_string(rng, 'cbp_tracker'),
        )
    else:
        cbp_tracker = None
        
    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker