"""Predictive Coding network with closed-form iPC update rules.

Layer convention (generative direction, matching Salvatori et al. 2024):
  Layer L = input (clamped to observation)
  Layer 0 = output (clamped to target in supervised setting)
  weights[l] has shape (dim[l], dim[l+1]) and predicts x[l] from f(x[l+1])
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray

from phd.jax_core.models import ACTIVATION_MAP


# Activation derivatives (needed for value node updates)
ACTIVATION_DERIV_MAP = {
    'relu': lambda x: (x > 0).astype(jnp.float32),
    'tanh': lambda x: 1.0 - jnp.tanh(x) ** 2,
    'sigmoid': lambda x: jax.nn.sigmoid(x) * (1.0 - jax.nn.sigmoid(x)),
    'linear': lambda x: jnp.ones_like(x),
}


class PCNetwork(eqx.Module):
    """Predictive Coding network with explicit weight matrices.

    The network stores L weight matrices for L layers. The generative model
    predicts x[l] from f(x[l+1]) via weights[l].

    Value nodes are NOT stored here — they live in TrainState and persist
    across observations in the streaming setting.
    """
    layer_dims: tuple = eqx.field(static=True)      # (dim_0, dim_1, ..., dim_L)
    num_layers: int = eqx.field(static=True)         # L
    activation_name: str = eqx.field(static=True)

    weights: list  # weights[l] shape: (dim[l], dim[l+1])


def init_pc_network(
    layer_dims: Tuple[int, ...],
    activation: str,
    key: PRNGKeyArray,
) -> PCNetwork:
    """Initialize a PCNetwork with LeCun uniform weights.

    Args:
        layer_dims: (dim_0, dim_1, ..., dim_L) where dim_0 = output, dim_L = input
        activation: activation function name
        key: PRNG key
    """
    L = len(layer_dims) - 1
    keys = jax.random.split(key, L)

    weights = []
    for l in range(L):
        fan_in = layer_dims[l + 1]  # input dimension for this weight matrix
        bound = jnp.sqrt(3.0 / fan_in)
        w = jax.random.uniform(
            keys[l], (layer_dims[l], layer_dims[l + 1]),
            minval=-bound, maxval=bound,
        )
        weights.append(w)

    return PCNetwork(
        layer_dims=tuple(layer_dims),
        num_layers=L,
        activation_name=activation,
        weights=weights,
    )


def pc_forward_pass(
    network: PCNetwork,
    x_input: jnp.ndarray,
) -> list:
    """Run the generative model top-down to initialize value nodes.

    Starting from x[L] = x_input, compute:
      x[l] = f(weights[l] @ f(x[l+1]))  for l = L-1, ..., 1

    Returns value_nodes: list of L-1 arrays [x[1], x[2], ..., x[L-1]].
    """
    f = ACTIVATION_MAP[network.activation_name]
    L = network.num_layers

    value_nodes = [None] * (L - 1)  # indices 0..L-2 → layers 1..L-1
    x_above = x_input  # start at layer L

    for l in range(L - 1, 0, -1):
        # Prediction of layer l from layer l+1
        mu = network.weights[l] @ f(x_above)
        # Apply activation to get value node (except we store pre-activation for
        # consistency — actually, the standard iPC formulation has x[l] as the
        # node value and applies f() when computing predictions from it.
        # So x[l] = mu[l] is the forward-init value (the prediction itself).
        x_above = mu
        value_nodes[l - 1] = mu  # store at index l-1 for layer l

    return value_nodes


def ipc_step(
    network: PCNetwork,
    value_nodes: list,
    x_input: jnp.ndarray,
    y_target: jnp.ndarray,
    gamma: float,
    alpha: float,
) -> Tuple[PCNetwork, list, dict]:
    """One step of incremental Predictive Coding (simultaneous value + weight update).

    Args:
        network: PCNetwork with current weights
        value_nodes: list of L-1 arrays [x[1], ..., x[L-1]] (hidden layer values)
        x_input: input observation (clamped at layer L)
        y_target: target (clamped at layer 0), e.g. one-hot label
        gamma: inference learning rate (value node update)
        alpha: weight learning rate

    Returns:
        (updated_network, updated_value_nodes, info_dict)
    """
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    # Build full node list: all_x[0] = y_target, all_x[1..L-1] = value_nodes, all_x[L] = x_input
    all_x = [y_target] + list(value_nodes) + [x_input]

    # Compute predictions and errors for all layers
    predictions = []
    errors = []
    for l in range(L):
        mu_l = network.weights[l] @ f(all_x[l + 1])
        predictions.append(mu_l)
        errors.append(all_x[l] - mu_l)

    # Update hidden value nodes (layers 1 to L-1)
    new_value_nodes = []
    for l in range(1, L):
        # Top-down prediction error at this layer
        top_down = -errors[l]
        # Bottom-up error signal from layer below
        bottom_up = f_prime(all_x[l]) * (network.weights[l - 1].T @ errors[l - 1])
        x_new = all_x[l] + gamma * (top_down + bottom_up)
        new_value_nodes.append(x_new)

    # Update weights (all layers)
    new_weights = []
    weight_update_norms = []
    for l in range(L):
        delta_w = alpha * jnp.outer(errors[l], f(all_x[l + 1]))
        new_weights.append(network.weights[l] + delta_w)
        weight_update_norms.append(jnp.sqrt(jnp.sum(delta_w ** 2)))

    new_network = PCNetwork(
        layer_dims=network.layer_dims,
        num_layers=network.num_layers,
        activation_name=network.activation_name,
        weights=new_weights,
    )

    # Metrics
    layer_errors = jnp.array([jnp.sum(e ** 2) for e in errors])
    total_energy = jnp.sum(layer_errors)

    info = {
        'total_energy': total_energy,
        'layer_errors': layer_errors,
        'weight_update_norms': jnp.array(weight_update_norms),
    }

    return new_network, new_value_nodes, info
