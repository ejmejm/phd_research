"""The layers with their own backward passes, for algorithms.py.

    ltu          a linear threshold unit. The backward pass uses the slope of the sigmoid.
    linear       weights @ inputs. The backward pass to the inputs uses shadow weights in place of the real weights.
    Standardize  standardizes the LTUs' activations, as batch norm with running averages over time, as in
                 sharables/network_autostep/network_autostep.py.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@jax.custom_vjp
def ltu(pre_activations: Array) -> Array:
    """1 where the pre-activation is above 0, else 0. The backward pass uses the slope of the sigmoid."""
    return (pre_activations > 0).astype(jnp.float32)


def ltu_forward(pre_activations):
    return ltu(pre_activations), pre_activations


def ltu_backward(pre_activations, output_gradient):
    sigmoid = jax.nn.sigmoid(pre_activations)
    return (output_gradient * sigmoid * (1 - sigmoid),)


ltu.defvjp(ltu_forward, ltu_backward)


@jax.custom_vjp
def linear(weights: Array, backward_weights: Array, inputs: Array) -> Array:
    """weights @ inputs. The backward pass to the inputs uses backward_weights in place of the weights; this is how
    shadow weights enter the backward pass."""
    return weights @ inputs


def linear_forward(weights, backward_weights, inputs):
    return weights @ inputs, (backward_weights, inputs)


def linear_backward(saved, output_gradient):
    backward_weights, inputs = saved
    return jnp.outer(output_gradient, inputs), jnp.zeros_like(backward_weights), backward_weights.T @ output_gradient


linear.defvjp(linear_forward, linear_backward)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def standardize(activations, firing_rate, error_mean, error_activation_mean, units, variance_floor):
    """(activations - firing_rate) / std for the `units`, and 0 for the others, where std = sqrt(firing_rate (1 -
    firing_rate)) is the std of an LTU's activation."""
    return standardize_forward(activations, firing_rate, error_mean, error_activation_mean, units, variance_floor)[0]


def standardize_forward(activations, firing_rate, error_mean, error_activation_mean, units, variance_floor):
    variance = firing_rate * (1 - firing_rate)
    std = jnp.sqrt(jnp.maximum(variance, variance_floor))
    standardized = jnp.where(units, (activations - firing_rate) / std, 0.0)
    return standardized, (standardized, std, variance > variance_floor, error_mean, error_activation_mean, units)


def standardize_backward(variance_floor, saved, output_gradient):
    # Batch norm's backward pass: dL/dactivations = (dL/dout - mean(dL/dshift) - out * mean(dL/dscale)) / std. While
    # the variance is at its floor, std does not depend on the activations, so the scale term drops out. The running
    # averages get no gradient. This pass is affine, not linear, in dL/dout: it is not 0 where dL/dout is 0. So it is 0
    # outside the `units`, and each unit must be standardized in exactly one call.
    standardized, std, above_floor, error_mean, error_activation_mean, units = saved
    centered = output_gradient - error_mean - jnp.where(above_floor, standardized * error_activation_mean, 0.0)
    return jnp.where(units, centered / std, 0.0), None, None, None, None


standardize.defvjp(standardize_forward, standardize_backward)


class Standardize(eqx.Module):
    """Standardizes the LTUs' activations with running averages. The output is scale * standardize(activations) +
    shift, where shift and scale stay at 0 and 1. We keep them for their gradients: dL/dshift is the error arriving at
    each LTU, and dL/dscale is that error times the LTU's standardized activation."""
    firing_rate: Float[Array, 'ltus']             # running average of each LTU's activation
    error_mean: Float[Array, 'ltus']              # running average of dL/dshift
    error_activation_mean: Float[Array, 'ltus']   # running average of dL/dscale
    shift: Float[Array, 'ltus']                   # always 0
    scale: Float[Array, 'ltus']                   # always 1
    rate: float = eqx.field(static=True)          # of the running averages
    variance_floor: float = eqx.field(static=True)

    def __init__(self, n_ltus: int, rate: float, variance_floor: float):
        self.firing_rate = jnp.full(n_ltus, 0.5)
        self.error_mean = jnp.zeros(n_ltus)
        self.error_activation_mean = jnp.zeros(n_ltus)
        self.shift = jnp.zeros(n_ltus)
        self.scale = jnp.ones(n_ltus)
        self.rate, self.variance_floor = rate, variance_floor

    def running_average(self, average, value):
        return average + self.rate * (value - average)

    def __call__(self, activations: Float[Array, 'ltus'], units: Bool[Array, 'ltus']) -> Float[Array, 'ltus']:
        """The standardized activations of the `units`, and 0 for the others."""
        firing_rate = self.running_average(self.firing_rate, activations)   # includes this sample
        standardized = standardize(activations, firing_rate, self.error_mean, self.error_activation_mean, units,
                                   self.variance_floor)
        return jnp.where(units, self.scale * standardized + self.shift, 0.0)

    def update_statistics(self, activations: Float[Array, 'ltus'], gradients: 'Standardize') -> 'Standardize':
        """The running averages, updated from this sample's activations and the gradients of shift and scale."""
        return eqx.tree_at(lambda norm: (norm.firing_rate, norm.error_mean, norm.error_activation_mean), self,
                           (self.running_average(self.firing_rate, activations),
                            self.running_average(self.error_mean, gradients.shift),
                            self.running_average(self.error_activation_mean, gradients.scale)))

    def restart(self, ltus: Bool[Array, 'ltus']) -> 'Standardize':
        """Fresh running averages for these LTUs."""
        return eqx.tree_at(lambda norm: (norm.firing_rate, norm.error_mean, norm.error_activation_mean), self,
                           (jnp.where(ltus, 0.5, self.firing_rate), jnp.where(ltus, 0.0, self.error_mean),
                            jnp.where(ltus, 0.0, self.error_activation_mean)))
