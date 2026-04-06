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
    output_node: jnp.ndarray,
    x_input: jnp.ndarray,
    gamma: float,
    alpha: float,
    has_target: bool = True,
    y_target: jnp.ndarray = None,
) -> Tuple[PCNetwork, list, jnp.ndarray, dict]:
    """One step of incremental Predictive Coding (simultaneous value + weight update).

    When has_target=True, layer 0 is clamped to y_target (supervised).
    When has_target=False, layer 0 is free and updated via prediction error.
    Value node and weight updates happen in both cases.

    Args:
        network: PCNetwork with current weights
        value_nodes: list of L-1 arrays [x[1], ..., x[L-1]] (hidden layer values)
        output_node: current output node value (layer 0); used when has_target=False,
            updated to the network's prediction of layer 0 when has_target=True
        x_input: input observation (clamped at layer L)
        gamma: inference learning rate (value node update)
        alpha: weight learning rate
        has_target: if True, clamp layer 0 to y_target; if False, layer 0 is free
        y_target: target (clamped at layer 0 when has_target=True)

    Returns:
        (updated_network, updated_value_nodes, updated_output_node, info_dict)
    """
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    # Layer 0: clamped to target or free output node
    if has_target:
        all_x = [y_target] + list(value_nodes) + [x_input]
    else:
        all_x = [output_node] + list(value_nodes) + [x_input]

    # Compute predictions and errors for all layers
    errors = []
    for l in range(L):
        mu_l = network.weights[l] @ f(all_x[l + 1])
        errors.append(all_x[l] - mu_l)

    # Update hidden value nodes (layers 1 to L-1)
    new_value_nodes = []
    for l in range(1, L):
        top_down = -errors[l]
        bottom_up = f_prime(all_x[l]) * (network.weights[l - 1].T @ errors[l - 1])
        x_new = all_x[l] + gamma * (top_down + bottom_up)
        new_value_nodes.append(x_new)

    # Update output node (layer 0)
    if has_target:
        # After learning, set output_node to the network's prediction for layer 0
        new_output_node = network.weights[0] @ f(new_value_nodes[0])
    else:
        # Free node: move toward prediction (only top-down signal, no layer below)
        new_output_node = all_x[0] - gamma * errors[0]

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

    return new_network, new_value_nodes, new_output_node, info


def ipc_step_grads(
    network: PCNetwork,
    value_nodes: list,
    x_input: jnp.ndarray,
    y_target: jnp.ndarray,
    gamma: float,
) -> Tuple[list, list, dict]:
    """Compute iPC value node updates and weight gradients without applying them.

    Same as ipc_step but returns the weight gradients (local Hebbian outer
    products) instead of applying them directly. This allows the caller to
    pass gradients through an optimizer (e.g. Adam) before applying.

    Returns:
        (updated_value_nodes, weight_grads, info_dict)
        where weight_grads[l] has shape (dim[l], dim[l+1]).
    """
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    all_x = [y_target] + list(value_nodes) + [x_input]

    predictions = []
    errors = []
    for l in range(L):
        mu_l = network.weights[l] @ f(all_x[l + 1])
        predictions.append(mu_l)
        errors.append(all_x[l] - mu_l)

    # Update hidden value nodes (layers 1 to L-1)
    new_value_nodes = []
    for l in range(1, L):
        top_down = -errors[l]
        bottom_up = f_prime(all_x[l]) * (network.weights[l - 1].T @ errors[l - 1])
        x_new = all_x[l] + gamma * (top_down + bottom_up)
        new_value_nodes.append(x_new)

    # Compute weight gradients (negative because optax minimizes)
    weight_grads = []
    for l in range(L):
        grad_l = -jnp.outer(errors[l], f(all_x[l + 1]))
        weight_grads.append(grad_l)

    layer_errors = jnp.array([jnp.sum(e ** 2) for e in errors])
    total_energy = jnp.sum(layer_errors)

    info = {
        'total_energy': total_energy,
        'layer_errors': layer_errors,
    }

    return new_value_nodes, weight_grads, info
