"""Weight initializers shared by the models.

``PaddedMLP`` has its own variant that takes an explicit fan-in, because a
block-sparse unit's fan-in is not the width of its weight row.
"""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray


@partial(jax.jit, static_argnums=(1, 2))
def lecun_uniform(
    key: PRNGKeyArray, shape: Tuple[int, ...], in_dim: Optional[int] = None,
) -> Array:
    """LeCun uniform initialization; fan-in defaults to the last axis."""
    in_dim = shape[-1] if in_dim is None else in_dim
    bound = jnp.sqrt(3.0) / jnp.sqrt(in_dim)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


@partial(jax.jit, static_argnames=('shape', 'in_dim'))
def kaiming_uniform(
    key: PRNGKeyArray, shape: Tuple[int, ...], gain: float = 1.0,
    in_dim: Optional[int] = None,
) -> Array:
    """Kaiming uniform initialization; fan-in defaults to the last axis."""
    in_dim = shape[-1] if in_dim is None else in_dim
    bound = gain * jnp.sqrt(3.0) / jnp.sqrt(in_dim)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)
