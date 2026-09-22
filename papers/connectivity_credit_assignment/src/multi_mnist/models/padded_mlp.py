"""The two-layer MLP used for every figure in the paper.

Connectivity is expressed as a mask over a dense weight matrix rather than as
a sparse data structure. That is deliberate: the paper compares dense and
block-sparse networks at matched parameter counts, and a masked dense matmul
makes the two numerically identical apart from which weights are allowed to
be non-zero. It is *not* an efficient sparse implementation -- a block-sparse
network here costs exactly what a dense one costs.

SET and DEEP-R run here too, on an ``init_strategy: sparse`` model. That
costs the dense matmul whatever the density, which is affordable at the sizes
in this repository and buys two things the sparse representation cannot: a
connection budget that is preserved per *layer* rather than per unit, as both
papers specify, and ``structure_diagnostics`` including ``path_purity``. Use
``DynamicNetwork`` when the network is large enough that paying the dense cost
stops being acceptable; use ``BlockSparseMLP`` for fast block-sparse runs.
"""

import warnings
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from ..activations import ACTIVATION_MAP
from ..utils import tree_replace


class PaddedMLP(eqx.Module):
    """Two-layer MLP padded to ``max_hidden`` units.

    Inactive units and connections stay in the arrays but are zeroed by masks
    in the forward pass, so the shapes are constant and JAX never recompiles
    when connectivity changes.
    """

    W1: jax.Array            # (max_hidden, input_dim)
    W2: jax.Array            # (output_dim, max_hidden)
    unit_mask: jax.Array     # (max_hidden,) 1 where the unit slot is in use
    w1_mask: jax.Array       # (max_hidden, input_dim) connection mask
    w2_mask: jax.Array       # (output_dim, max_hidden)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    max_hidden: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __call__(self, x):
        h = ACTIVATION_MAP[self.activation]((self.W1 * self.w1_mask) @ x)
        h = h * self.unit_mask
        out = (self.W2 * self.w2_mask) @ h
        return out, h


def lecun_uniform(key, shape, fan_in):
    """LeCun uniform with an explicitly supplied fan-in.

    Block-sparse units see only their own task's inputs, so the fan-in that
    sets the scale is not the width of the weight row.
    """
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in, 1.0).astype(jnp.float32))
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * bound


def init_model(
    cfg: DictConfig, input_dim: int, output_dim: int, n_tasks: int,
    *, key: PRNGKeyArray,
) -> PaddedMLP:
    """Build a ``PaddedMLP`` under the configured connectivity pattern.

    ``dense`` connects every input to every hidden unit and every hidden unit
    to every output. ``block_sparse`` partitions the hidden units evenly over
    tasks and connects each unit only to the inputs and outputs of its task,
    yielding ``n_tasks`` independent sub-networks. ``sparse`` draws an
    Erdos-Renyi random topology at the density set by ``model.sparse.epsilon``,
    which is the starting point SET and DEEP-R evolve from.
    """
    max_hidden = int(cfg.model.max_hidden_units)
    initial = int(cfg.model.initial_hidden_units)
    strategy = cfg.model.init_strategy
    activation = cfg.model.activation

    W1 = jnp.zeros((max_hidden, input_dim), dtype=jnp.float32)
    W2 = jnp.zeros((output_dim, max_hidden), dtype=jnp.float32)
    w1_mask = jnp.zeros_like(W1)
    w2_mask = jnp.zeros_like(W2)
    unit_mask = jnp.concatenate([
        jnp.ones((initial,), dtype=jnp.float32),
        jnp.zeros((max_hidden - initial,), dtype=jnp.float32),
    ]) if initial > 0 else jnp.zeros((max_hidden,), dtype=jnp.float32)

    if initial > 0:
        # Split into 6 even though only the first two are used. The published
        # runs split into 6 (the unused four seeded connectivity patterns that
        # are not part of the paper), and `split(key, 2)[0] != split(key, 6)[0]`,
        # so narrowing this would change every model initialization.
        kw1, kw2 = jax.random.split(key, 6)[:2]

        if strategy == 'dense':
            W1 = W1.at[:initial].set(lecun_uniform(kw1, (initial, input_dim), input_dim))
            W2 = W2.at[:, :initial].set(lecun_uniform(kw2, (output_dim, initial), initial))
            w1_mask = w1_mask.at[:initial].set(1.0)
            w2_mask = w2_mask.at[:, :initial].set(1.0)

        elif strategy == 'sparse':
            # Erdos-Renyi, as in the SET paper: each bipartite layer gets edge
            # probability p = epsilon * (n_in + n_out) / (n_in * n_out). This
            # is the same construction as models/sparse_init.py, materialized
            # as a mask over a dense matrix instead of per-unit slot arrays --
            # so SET and DEEP-R can run here, with path_purity available and
            # no per-unit cap on how far a degree distribution can drift.
            epsilon = float(cfg.model.sparse.epsilon)
            p_w1 = min(1.0, epsilon * (input_dim + initial) / (input_dim * initial))
            p_w2 = min(1.0, epsilon * (output_dim + initial) / (output_dim * initial))

            km1, kv1 = jax.random.split(kw1)
            km2, kv2 = jax.random.split(kw2)
            m1 = jax.random.bernoulli(km1, p_w1, (initial, input_dim)).astype(jnp.float32)
            m2 = jax.random.bernoulli(km2, p_w2, (output_dim, initial)).astype(jnp.float32)

            # Scale by the realized per-row fan-in, matching sparse_init.py.
            v1 = jax.random.uniform(kv1, (initial, input_dim), minval=-1.0, maxval=1.0)
            v2 = jax.random.uniform(kv2, (output_dim, initial), minval=-1.0, maxval=1.0)
            b1 = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(m1.sum(axis=-1), 1.0))
            b2 = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(m2.sum(axis=-1), 1.0))

            W1 = W1.at[:initial].set(v1 * b1[:, None] * m1)
            W2 = W2.at[:, :initial].set(v2 * b2[:, None] * m2)
            w1_mask = w1_mask.at[:initial].set(m1)
            w2_mask = w2_mask.at[:, :initial].set(m2)

        elif strategy == 'block_sparse':
            assert n_tasks > 1, 'block_sparse init requires n_tasks > 1'
            assert initial >= n_tasks, (
                f'block_sparse init requires initial_hidden_units ({initial}) >= '
                f'n_tasks ({n_tasks}) so each task gets at least one unit'
            )
            input_dim_per_task = input_dim // n_tasks
            output_dim_per_task = output_dim // n_tasks
            base_units = initial // n_tasks
            extra_units = initial % n_tasks

            block_w1_mask = jnp.zeros((initial, input_dim), dtype=jnp.float32)
            block_w2_mask = jnp.zeros((output_dim, initial), dtype=jnp.float32)

            if extra_units == 0:
                W1_init = lecun_uniform(kw1, (initial, input_dim), input_dim_per_task)
                W2_init = lecun_uniform(kw2, (output_dim, initial), base_units)
                for t in range(n_tasks):
                    u0, u1 = t * base_units, (t + 1) * base_units
                    i0, i1 = t * input_dim_per_task, (t + 1) * input_dim_per_task
                    o0, o1 = t * output_dim_per_task, (t + 1) * output_dim_per_task
                    block_w1_mask = block_w1_mask.at[u0:u1, i0:i1].set(1.0)
                    block_w2_mask = block_w2_mask.at[o0:o1, u0:u1].set(1.0)
                W1 = W1.at[:initial].set(W1_init * block_w1_mask)
                W2 = W2.at[:, :initial].set(W2_init * block_w2_mask)
            else:
                warnings.warn(
                    f'block_sparse: initial_hidden_units ({initial}) not divisible by '
                    f'n_tasks ({n_tasks}); first {extra_units} task(s) receive '
                    f'{base_units + 1} units, remaining receive {base_units}',
                    stacklevel=2,
                )
                W1_full = jnp.zeros((initial, input_dim), dtype=jnp.float32)
                W2_full = jnp.zeros((output_dim, initial), dtype=jnp.float32)
                kw1_tasks = jax.random.split(kw1, n_tasks)
                kw2_tasks = jax.random.split(kw2, n_tasks)
                u_off = 0
                for t in range(n_tasks):
                    upt = base_units + (1 if t < extra_units else 0)
                    u0, u1 = u_off, u_off + upt
                    u_off = u1
                    i0, i1 = t * input_dim_per_task, (t + 1) * input_dim_per_task
                    o0, o1 = t * output_dim_per_task, (t + 1) * output_dim_per_task
                    W1_block = lecun_uniform(
                        kw1_tasks[t], (upt, input_dim_per_task), input_dim_per_task)
                    W2_block = lecun_uniform(
                        kw2_tasks[t], (output_dim_per_task, upt), upt)
                    W1_full = W1_full.at[u0:u1, i0:i1].set(W1_block)
                    W2_full = W2_full.at[o0:o1, u0:u1].set(W2_block)
                    block_w1_mask = block_w1_mask.at[u0:u1, i0:i1].set(1.0)
                    block_w2_mask = block_w2_mask.at[o0:o1, u0:u1].set(1.0)
                W1 = W1.at[:initial].set(W1_full)
                W2 = W2.at[:, :initial].set(W2_full)

            w1_mask = w1_mask.at[:initial].set(block_w1_mask)
            w2_mask = w2_mask.at[:, :initial].set(block_w2_mask)

        else:
            raise ValueError(f'Unknown init_strategy: {strategy}')

    return PaddedMLP(
        W1=W1, W2=W2, unit_mask=unit_mask,
        w1_mask=w1_mask, w2_mask=w2_mask,
        input_dim=input_dim, output_dim=output_dim,
        max_hidden=max_hidden, activation=activation,
    )


def model_filter_spec(model: PaddedMLP):
    """Only W1 and W2 are trainable; masks are structure, not parameters."""
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(lambda m: (m.W1, m.W2), spec, (True, True))


def fill_masks_to_dense(model: PaddedMLP, initial_hidden_units: int) -> PaddedMLP:
    """Flip a block-sparse mask to all-ones over the active region.

    This is the dense-transition intervention of Figure 3. W1 and W2 already
    hold exactly 0 at the cross-task positions, so the flip is
    value-preserving: the forward pass is unchanged at the moment it happens,
    and the network only diverges once gradients start flowing into the
    newly-active weights on the following step.
    """
    h = initial_hidden_units
    return tree_replace(
        model,
        w1_mask=model.w1_mask.at[:h, :].set(1.0),
        w2_mask=model.w2_mask.at[:, :h].set(1.0),
    )


def structure_diagnostics(model: PaddedMLP, n_tasks: int) -> dict:
    """Per-seed structure statistics computed from a vmapped model.

    Returns a dict whose values are (n_seeds,) jax arrays.

    ``path_purity``: per-output mean of (#same-task paths) / (#total paths).
    A path is an (output, hidden, input) triple with active w2 and w1
    connections; multiple hidden units between the same (output, input)
    pair count as multiple paths. Outputs with no paths are excluded from
    the mean. 1.0 = every output's paths land in its own task,
    1/n_tasks = uniform mixing. Trivially 1.0 when n_tasks == 1.
    """
    unit_active = model.unit_mask                      # (S, max_hidden)
    fan_in_per_unit = model.w1_mask.sum(axis=-1)        # (S, max_hidden)
    fan_out_per_unit = model.w2_mask.sum(axis=-2)       # (S, max_hidden)
    n_active = jnp.maximum(unit_active.sum(axis=-1), 1) # avoid /0

    stats = {
        'active_units': unit_active.sum(axis=-1),
        'active_connections': fan_in_per_unit.sum(axis=-1) + fan_out_per_unit.sum(axis=-1),
        'mean_fan_in': (fan_in_per_unit * unit_active).sum(axis=-1) / n_active,
        'mean_fan_out': (fan_out_per_unit * unit_active).sum(axis=-1) / n_active,
    }

    if n_tasks > 1:
        # paths[s, o, i] = #hidden units routing input i → output o
        # (multiplicity via different hidden units is preserved).
        paths = model.w2_mask @ model.w1_mask                       # (S, output_dim, input_dim)
        out_per_task = paths.shape[-2] // n_tasks
        in_per_task = paths.shape[-1] // n_tasks
        grouped = paths.reshape(
            *paths.shape[:-2], n_tasks, out_per_task, n_tasks, in_per_task,
        )
        # paths_by_task[s, t_out, o_within, t_in] = #paths from output
        # (t_out, o_within) terminating at any input in task t_in.
        paths_by_task = grouped.sum(axis=-1)                        # (S, T, out_per_task, T)
        total_per_output = paths_by_task.sum(axis=-1)               # (S, T, out_per_task)
        # diagonal over (t_out, t_in): output's own-task path count.
        # jnp.diagonal appends the diagonal axis at the end.
        same_per_output = jnp.diagonal(paths_by_task, axis1=-3, axis2=-1)  # (S, out_per_task, T)
        same_per_output = jnp.swapaxes(same_per_output, -1, -2)            # (S, T, out_per_task)

        output_active = total_per_output > 0
        purity_per_output = jnp.where(
            output_active,
            same_per_output / jnp.maximum(total_per_output, 1),
            0.0,
        )
        sum_purity = purity_per_output.sum(axis=(-1, -2))                   # (S,)
        n_active_outputs = output_active.sum(axis=(-1, -2)).astype(jnp.float32)
        stats['path_purity'] = sum_purity / jnp.maximum(n_active_outputs, 1.0)
    else:
        stats['path_purity'] = jnp.ones_like(stats['active_units'], dtype=jnp.float32)

    return stats
