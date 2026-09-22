"""Building a sparse ``DynamicNetwork`` and sizing it to a connection budget.

Both SET and DEEP-R start from an Erdos-Renyi random topology at a target
connection count, so that construction lives here rather than in either
algorithm.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from ..utils import tree_replace
from .dynamic_network import DynamicNetwork, build_outgoing_indices, sync_outgoing_weights


# One hidden layer, so every layer index into DynamicNetwork is 0.
HIDDEN_LAYER = 0


class StructureModel(eqx.Module):
    """Thin wrapper giving DynamicNetwork the model interface the trainer expects."""
    network: DynamicNetwork

    def __call__(self, x):
        return self.network(x)

    def post_update(self):
        """Mirror incoming weights into the outgoing-indexed copy.

        DynamicNetwork's custom VJP reads outgoing weights on the backward
        pass, so they must track the incoming weights after every update.
        """
        return tree_replace(self, network=sync_outgoing_weights(self.network))


def model_filter_spec(model: StructureModel):
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: (m.network.weights, m.network.output_weights),
        spec, (True, True),
    )


def _init_erdos_renyi_structure(
    network, key, hidden, input_dim, output_dim, max_conns, p_w1, p_w2,
):
    """Build the initial sparse topology for both bipartite layers.

    `p_w1` and `p_w2` are the (pre-clipped) per-edge inclusion probabilities
    for each bipartite layer. Clipped to [0, 1] before sampling.
    """
    k_w1_mask, k_w1_w, k_w2_mask, k_w2_w = jax.random.split(key, 4)

    # --- W1 (input -> hidden) ---
    p_w1 = jnp.minimum(jnp.asarray(p_w1, dtype=jnp.float32), 1.0)
    mask_w1 = jax.random.bernoulli(k_w1_mask, p_w1, (hidden, input_dim))

    # Per row, pack active column indices into the leading slots; pad with -1.
    cols = jnp.broadcast_to(
        jnp.arange(input_dim, dtype=jnp.int32), mask_w1.shape,
    )
    score = jnp.where(mask_w1, cols, input_dim + cols)  # actives sort first
    sort_order = jnp.argsort(score, axis=-1)
    sorted_cols = jnp.take_along_axis(cols, sort_order, axis=-1)
    sorted_active = jnp.take_along_axis(mask_w1, sort_order, axis=-1)
    take_cols = sorted_cols[:, :max_conns]
    take_active = sorted_active[:, :max_conns]
    new_idx_row = jnp.where(take_active, take_cols, jnp.int32(-1))

    fan_in_per_row = take_active.sum(axis=-1).astype(jnp.float32)
    raw_w = jax.random.uniform(k_w1_w, (hidden, max_conns), minval=-1.0, maxval=1.0)
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_per_row, 1.0))
    w1 = raw_w * bound[:, None] * take_active.astype(jnp.float32)

    new_input_indices = network.input_indices.at[HIDDEN_LAYER, :hidden].set(new_idx_row)
    new_weights = network.weights.at[HIDDEN_LAYER, :hidden].set(w1)
    new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, :hidden].set(1)

    # --- W2 (hidden -> output) ---
    p_w2 = jnp.minimum(jnp.asarray(p_w2, dtype=jnp.float32), 1.0)
    mask_w2 = jax.random.bernoulli(k_w2_mask, p_w2, (output_dim, hidden))
    fan_in_w2 = mask_w2.sum(axis=-1).astype(jnp.float32)
    raw_w2 = jax.random.uniform(k_w2_w, (output_dim, hidden), minval=-1.0, maxval=1.0)
    bound_w2 = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_w2, 1.0))
    w2 = raw_w2 * bound_w2[:, None] * mask_w2.astype(jnp.float32)

    new_output_mask = network.output_mask.at[:, input_dim:input_dim + hidden].set(
        mask_w2.astype(network.output_mask.dtype),
    )
    new_output_weights = network.output_weights.at[:, input_dim:input_dim + hidden].set(w2)

    return tree_replace(
        network,
        input_indices=new_input_indices,
        weights=new_weights,
        unit_mask=new_unit_mask,
        output_mask=new_output_mask,
        output_weights=new_output_weights,
    )


def init_sparse_model(
    cfg: DictConfig, input_dim: int, output_dim: int,
    hidden_units: int, max_conns: int, max_fan_out: int,
    p_w1: float, p_w2: float,
    *, key: PRNGKeyArray,
) -> StructureModel:
    activation = cfg.model.activation

    network = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=1,
        max_units_per_layer=hidden_units,
        max_connections_per_unit=max_conns,
        activations=(activation,),
        max_fan_out=max_fan_out,
        init_strategy='empty',
        key=key,
    )

    network = _init_erdos_renyi_structure(
        network, key, hidden_units, input_dim, output_dim, max_conns, p_w1, p_w2,
    )
    network = build_outgoing_indices(network)
    return StructureModel(network=network)


def derive_sizes(cfg: DictConfig, input_dim: int, output_dim: int):
    """Derive ``hidden_units``, ``max_connections_per_unit``, ``max_fan_out``,
    and the per-layer edge probabilities ``(p_w1, p_w2)`` from the SET budget
    configuration.

    Two init modes are supported (selected by ``cfg.model.sparse.init_mode``):

    - ``epsilon`` (default, original SET): user provides ``model.sparse.epsilon`` and
      ``model.sparse.connection_budget``; hidden width is derived as
      ``floor((budget/ε - input_dim - output_dim) / 2)``. Per-layer edge
      probabilities follow the Erdős-Rényi formula
      ``p_layer = ε(n+m)/(n·m)``, which gives *different* densities for W1
      and W2 when (n, m) are asymmetric (the typical case here).

    - ``uniform_p``: both bipartite layers share a single edge probability
      ``p = budget / ((input_dim + output_dim) · hidden_units)``. Hidden width
      is taken directly from ``model.initial_hidden_units``. ``ε`` is not consulted.
      This keeps W1 and W2 at the *same* density at every scale, and yields
      constant per-hidden-unit W1 fan-in (``p·input_dim``) when budget and H
      are scaled proportionally.
    """
    init_mode = str(cfg.model.sparse.get('init_mode', 'epsilon'))
    budget = float(cfg.model.sparse.connection_budget)

    if init_mode == 'uniform_p':
        hidden = int(cfg.model.initial_hidden_units)
        if hidden < 1:
            raise ValueError(
                f'target_hidden_units must be >= 1, got {hidden}'
            )
        p = budget / ((input_dim + output_dim) * hidden)
        if p > 1.0:
            raise ValueError(
                f'uniform_p init: budget={budget} requires p={p:.4f} > 1 at '
                f'hidden={hidden}, input={input_dim}, output={output_dim}. '
                f'Reduce budget or increase hidden_units.'
            )
        p_w1 = p
        p_w2 = p
        expected_w1_fan_in = p * input_dim
        expected_w1_col_fan = p * hidden
    elif init_mode == 'epsilon':
        epsilon = float(cfg.model.sparse.epsilon)
        hidden = int(np.floor(
            (budget / epsilon - input_dim - output_dim) / 2.0,
        ))
        if hidden < 1:
            raise ValueError(
                f'connection_budget={budget} with epsilon={epsilon}, '
                f'input_dim={input_dim}, output_dim={output_dim} yields '
                f'hidden_units={hidden}; budget too small.',
            )
        p_w1 = min(1.0, epsilon * (input_dim + hidden) / (input_dim * hidden))
        p_w2 = min(1.0, epsilon * (output_dim + hidden) / (output_dim * hidden))
        expected_w1_fan_in = epsilon * (input_dim + hidden) / hidden
        expected_w1_col_fan = epsilon * (input_dim + hidden) / input_dim
    else:
        raise ValueError(
            f"model.sparse.init_mode must be 'epsilon' or 'uniform_p', got {init_mode!r}"
        )

    safety = float(cfg.model.sparse.get('fan_in_safety_factor', 2.0))
    max_conns = int(min(input_dim, max(1, np.ceil(safety * expected_w1_fan_in))))
    # max_fan_out caps the per-buffer-position outgoing array used by the
    # backward pass through the hidden layer. With one hidden layer it's
    # bounded by W1's expected column fan-in. We require at least 2x headroom
    # over the expected mean so SET has room for the column degree distribution
    # to grow (binomial → power-law).
    fan_out_factor = max(2.0, safety)
    max_fan_out = int(min(
        hidden,
        max(8, np.ceil(fan_out_factor * expected_w1_col_fan) + 4),
    ))
    return hidden, max_conns, max_fan_out, p_w1, p_w2
