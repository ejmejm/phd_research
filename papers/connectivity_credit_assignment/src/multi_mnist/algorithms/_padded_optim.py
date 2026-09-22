"""Resetting per-parameter optimizer state when connections are rewired.

A regrown connection is a new parameter that happens to reuse a slot. Carrying
the previous occupant's Adam moments into it would apply a stale step to a
weight that has nothing to do with them, so they are zeroed. Plain SGD has no
per-parameter state and is unaffected.
"""

import equinox as eqx
import jax.numpy as jnp

from ..optim import AdamState, EqxOptimizer
from ..utils import tree_replace


def reset_optimizer_at(optimizer: EqxOptimizer, w1_reset, w2_reset) -> EqxOptimizer:
    """Zero Adam state for a ``PaddedMLP`` wherever the reset masks are True."""
    state = optimizer.state
    if not isinstance(state, AdamState):
        return optimizer

    def _reset_field(field, dtype):
        return eqx.tree_at(
            lambda f: (f.W1, f.W2), field,
            (jnp.where(w1_reset, jnp.asarray(0, dtype=dtype), field.W1),
             jnp.where(w2_reset, jnp.asarray(0, dtype=dtype), field.W2)),
        )

    new_state = AdamState(
        lr=state.lr,
        step=_reset_field(state.step, jnp.int32),
        exp_avg=_reset_field(state.exp_avg, jnp.float32),
        exp_avg_sq=_reset_field(state.exp_avg_sq, jnp.float32),
    )
    return tree_replace(optimizer, state=new_state)
