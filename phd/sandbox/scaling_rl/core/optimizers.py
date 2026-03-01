"""ObGD optimizer for StreamAC.

Reference (PyTorch): phd/streaming_rl/core/obgd.py
"""
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from phd.jax_core.utils import tree_replace


class ObGDOptimizer(eqx.Module):
    """Observe-Batch Gradient Descent (ObGD) optimizer with eligibility traces.

    Implements the per-step update rule from StreamAC:
        e_t   = γλ * e_{t-1} + ∇L
        α     = lr / max(1, max(|δ|, 1) * Σ|e_t| * lr * κ)
        θ    -= α * δ * e_t
        e_t   = 0  (on episode end)

    Follows the same spirit as EqxOptimizer but with an extended
    with_update(grads, delta, reset) signature required by StreamAC.
    """
    lr: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    lamda: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    traces: Any  # Same pytree structure as model, zero-initialised arrays

    def __init__(self, model: eqx.Module, lr: float, gamma: float, lamda: float, kappa: float):
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.traces = jax.tree.map(jnp.zeros_like, model)

    def with_update(
        self,
        grads: Any,
        delta: Float[Array, ''],
        reset: Array,
    ) -> Tuple[Any, 'ObGDOptimizer']:
        """Compute parameter updates and return updated optimizer.

        Args:
            grads:  Gradient pytree with same structure as the model.
            delta:  Scalar TD error used to scale the update.
            reset:  Boolean scalar; True when the episode just ended.

        Returns:
            (updates, new_optimizer) where updates is a pytree compatible
            with eqx.apply_updates and new_optimizer has updated traces.
        """
        # Update eligibility traces: e = γλ·e + ∇L
        new_traces = jax.tree.map(
            lambda e, g: self.gamma * self.lamda * e + g,
            self.traces, grads,
        )

        # Adaptive step-size: prevent exploding updates
        z_sum = jnp.array(
            [jnp.sum(jnp.abs(e)) for e in jax.tree.leaves(new_traces)]
        ).sum()
        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        dot = delta_bar * z_sum * self.lr * self.kappa
        step_size = jnp.where(dot > 1.0, self.lr / dot, self.lr)

        # Parameter updates: θ -= step_size * δ * e
        updates = jax.tree.map(lambda e: -step_size * delta * e, new_traces)

        # Reset traces at episode boundaries
        final_traces = jax.tree.map(
            lambda e: jnp.where(reset, jnp.zeros_like(e), e),
            new_traces,
        )
        return updates, tree_replace(self, traces=final_traces)
