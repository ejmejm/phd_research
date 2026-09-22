"""Optimizers.

The paper uses plain SGD throughout. Adam is kept because the connectivity
algorithms (SET, DEEP-R) reset per-weight optimizer state when a connection is
pruned or regrown, which needs the moment estimates to be addressable per
weight -- ``AdamState`` exposes them as pytrees shaped like the parameters.
"""

import logging
from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
from optax._src import base
from omegaconf import DictConfig

from .utils import tree_unzip


logger = logging.getLogger(__name__)


class EqxOptimizer(eqx.Module):
    """An optax transformation bound to an equinox model and a filter spec."""

    name: str = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    filter_spec: Optional[PyTree] = eqx.field(default=None, static=True)
    state: PyTree

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        model: eqx.Module,
        filter_spec: Optional[PyTree] = None,
        name: Optional[str] = None,
    ):
        self.optimizer = optimizer
        self.filter_spec = filter_spec

        if filter_spec is not None:
            trainable_params = eqx.filter(model, filter_spec)
        else:
            trainable_params = model

        self.state = self.optimizer.init(trainable_params)
        self.name = name

    def with_update(self, grads, model) -> Tuple[PyTree, 'EqxOptimizer']:
        """Return parameter updates and an optimizer carrying the new state."""
        if self.filter_spec is not None:
            if isinstance(grads, tuple):
                grads = tuple(eqx.filter(g, self.filter_spec) for g in grads)
            else:
                grads = eqx.filter(grads, self.filter_spec)
            model = eqx.filter(model, self.filter_spec)

        updates, new_state = self.optimizer.update(grads, self.state, model)
        return updates, eqx.tree_at(lambda x: x.state, self, new_state)


class AdamState(NamedTuple):
    """Adam state with per-parameter step counts and moments."""
    lr: base.Updates
    step: base.Updates
    exp_avg: base.Updates
    exp_avg_sq: Optional[base.Updates] = None


def custom_optax_adam(
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> base.GradientTransformation:
    """Adam with per-parameter step counts.

    optax's built-in Adam keeps one scalar step count for the whole tree. Here
    the count is per parameter, so a connection that is pruned and later
    regrown can have its bias correction restarted in isolation.
    """
    def init_fn(params):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')

        learning_rate = jnp.array(lr, dtype=jnp.float32)
        step = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.int32), params)
        exp_avg = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        exp_avg_sq = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)

        return AdamState(lr=learning_rate, step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

    def update_fn(updates, state, params):
        loss_grads = updates
        lr_, step, exp_avg, exp_avg_sq = state

        def _adam_update(step, exp_avg, exp_avg_sq, grad):
            step += 1

            exp_avg = exp_avg * betas[0] + grad * (1 - betas[0])
            exp_avg_sq = exp_avg_sq * betas[1] + grad**2 * (1 - betas[1])

            bias_correction1 = 1 - betas[0]**step
            bias_correction2 = 1 - betas[1]**step

            step_size = lr_ / bias_correction1
            denom = jnp.sqrt(exp_avg_sq / bias_correction2) + eps
            param_update = exp_avg / denom * -step_size

            return param_update, step, exp_avg, exp_avg_sq

        results = jax.tree.map(_adam_update, step, exp_avg, exp_avg_sq, loss_grads)
        param_updates, step, exp_avg, exp_avg_sq = tree_unzip(results, 4)

        state = AdamState(lr=lr_, step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        return param_updates, state

    return base.GradientTransformation(init_fn, update_fn)


def prepare_optimizer(
    model: eqx.Module,
    optimizer_name: str,
    optimizer_kwargs: DictConfig,
    filter_spec: Optional[PyTree] = None,
) -> EqxOptimizer:
    """Build an optimizer from config, ignoring keys it does not use.

    Configs are shared across methods, so an unused key (say ``momentum`` on
    an SGD run) must not be an error.
    """
    def _extract_kwargs(param_names, defaults=None):
        defaults = defaults or {}
        kwargs = {}
        for param_name in param_names:
            value = optimizer_kwargs.get(param_name)
            if value is not None:
                kwargs[param_name] = value
            elif param_name in defaults:
                kwargs[param_name] = defaults[param_name]
        return kwargs

    if optimizer_name == 'sgd':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        optimizer = optax.sgd(learning_rate=kwargs['learning_rate'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec, name='sgd')

    elif optimizer_name == 'sgd_momentum':
        kwargs = _extract_kwargs(
            ['learning_rate', 'weight_decay', 'momentum'],
            {'weight_decay': 0, 'momentum': 0.9},
        )
        optimizer = optax.sgd(
            learning_rate=kwargs['learning_rate'], momentum=kwargs['momentum'])
        if kwargs['weight_decay'] != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(kwargs['weight_decay']))
        return EqxOptimizer(optimizer, model, filter_spec, name='sgd_momentum')

    elif optimizer_name == 'adam':
        kwargs = _extract_kwargs(['learning_rate', 'weight_decay'], {'weight_decay': 0})
        weight_decay = kwargs.pop('weight_decay')
        kwargs['lr'] = kwargs.pop('learning_rate')
        optimizer = custom_optax_adam(**kwargs)
        if weight_decay != 0:
            optimizer = optax.chain(optimizer, optax.add_decayed_weights(weight_decay))
        return EqxOptimizer(optimizer, model, filter_spec, name='adam')

    raise ValueError(f'Invalid optimizer type: {optimizer_name}')
