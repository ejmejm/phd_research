"""Activation functions available to every model."""

import jax
import jax.numpy as jnp
from jax import Array


def ltu(x: Array, threshold: float = 0.0) -> Array:
    """Linear threshold unit with a sigmoid surrogate gradient.

    The forward pass is a step function; the backward pass uses the derivative
    of a sigmoid so the unit is still trainable by backpropagation.
    """
    @jax.custom_vjp
    def ltu_fn(x):
        return (x > threshold).astype(jnp.float32)

    def ltu_fwd(x):
        return ltu_fn(x), x

    def ltu_bwd(res, g):
        x = res
        sigmoid_val = jax.nn.sigmoid(x - threshold)
        return (g * sigmoid_val * (1 - sigmoid_val),)

    ltu_fn.defvjp(ltu_fwd, ltu_bwd)
    return ltu_fn(x)


ACTIVATION_MAP = {
    'relu': jax.nn.relu,
    'leaky_relu': jax.nn.leaky_relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'ltu': ltu,
    'swish': jax.nn.swish,
    'linear': jax.nn.identity,
}
