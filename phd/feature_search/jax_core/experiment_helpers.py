from typing import Optional

import equinox as eqx
from equinox import nn
import jax
import omegaconf
from omegaconf import DictConfig
import optax

from .tasks.geoff import NonlinearGEOFFTask
from .optimizer import EqxOptimizer


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
    )-> EqxOptimizer:
    """Prepare the optimizer based on configuration.
    
    Uses default values for parameters not specified in config, while allowing
    irrelevant parameters to be specified without causing errors.
    """
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
        trainable_params = eqx.filter(model, filter_spec)
        opt_state = optimizer.init(trainable_params)
        return EqxOptimizer(optimizer, opt_state)
        
    elif optimizer_name == 'rmsprop':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        optimizer = optax.rmsprop(learning_rate=kwargs['learning_rate'], decay=0.999)
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        trainable_params = eqx.filter(model, filter_spec)
        opt_state = optimizer.init(trainable_params)
        return EqxOptimizer(optimizer, opt_state)
        
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
        trainable_params = eqx.filter(model, filter_spec)
        opt_state = optimizer.init(trainable_params)
        return EqxOptimizer(optimizer, opt_state)
        
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
