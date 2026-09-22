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
    yielding ``n_tasks`` independent sub-networks. ``sparse`` draws a random
    topology for SET and DEEP-R to evolve from, at a density set by
    ``model.sparse.init_mode``: ``epsilon`` is SET's shape-dependent
    Erdos-Renyi formula, ``uniform_p`` is DEEP-R's flat per-layer density.
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
            # --- density, per layer ---
            mode = str(cfg.model.sparse.get('init_mode', 'epsilon'))
            if mode == 'epsilon':
                # SET, Eq. 1: p = eps*(n_in + n_out)/(n_in * n_out) per bipartite
                # layer. Density depends on layer shape, so W1 and W2 end up at
                # different densities whenever (n_in, n_out) are asymmetric.
                epsilon = float(cfg.model.sparse.epsilon)
                p_w1 = epsilon * (input_dim + initial) / (input_dim * initial)
                p_w2 = epsilon * (output_dim + initial) / (output_dim * initial)
                # Clipping a saturated probability to 1 would be silently
                # destructive: the layer comes out fully dense, so an evolution
                # step cannot change its connectivity, but pruning still zeroes
                # zeta of its weights at every event and "regrows" them at 0 --
                # a periodic partial reset of the layer masquerading as
                # structure learning. Refuse instead. The paper's own default
                # epsilon=20 saturates on any layer with few enough outputs.
                for name, p, n_out in (('W1', p_w1, initial), ('W2', p_w2, output_dim)):
                    if p >= 1.0:
                        raise ValueError(
                            f'model.sparse.epsilon={epsilon} saturates {name}: '
                            f'p={p:.3f} >= 1 at hidden={initial}, n_out={n_out}, '
                            f'so the layer would be fully dense and could not '
                            f'evolve. Lower epsilon, or widen the layer.')
            elif mode == 'uniform_p':
                # DEEP-R, Appendix A: a per-layer density, with the output layer
                # deliberately much denser -- the paper reports 0.75/2.3/22.8 x a
                # global p0 across its three layers, and warns that "the
                # performance dropped drastically if the output layer was
                # initialized to be very sparse". `w2_density_ratio` is that
                # skew; the two densities are then solved against the shared
                # connection budget so the arms stay budget-matched:
                #     p1 * I * H + p2 * O * H = budget,  p2 = ratio * p1
                ratio = float(cfg.model.sparse.get('w2_density_ratio', 1.0))
                budget = float(cfg.model.sparse.connection_budget)
                p_w1 = budget / (initial * (input_dim + ratio * output_dim))
                p_w2 = ratio * p_w1
                if p_w2 > 1.0:
                    raise ValueError(
                        f'w2_density_ratio={ratio} needs p_w2={p_w2:.3f} <= 1 at '
                        f'hidden={initial}. The output layer saturates: lower the '
                        f'ratio, or widen the network.')
            else:
                raise ValueError(
                    f"model.sparse.init_mode must be 'epsilon' or 'uniform_p', "
                    f'got {mode!r}')

            km1, kv1 = jax.random.split(kw1)
            km2, kv2 = jax.random.split(kw2)
            m1 = jax.random.bernoulli(km1, p_w1, (initial, input_dim)).astype(jnp.float32)
            m2 = jax.random.bernoulli(km2, p_w2, (output_dim, initial)).astype(jnp.float32)

            # --- weight values ---
            # Both schemes draw at the DENSE fan-in and then mask, which is what
            # both papers do -- neither rescales to the realized sparse fan-in.
            weight_init = str(cfg.model.sparse.get('weight_init', 'glorot'))
            if weight_init == 'glorot':
                # SET's runs were Keras with default layer initializers, i.e.
                # glorot_uniform over the dense shape. The paper itself only
                # says "initialize ANN model".
                b1 = jnp.sqrt(6.0 / (input_dim + initial))
                b2 = jnp.sqrt(6.0 / (initial + output_dim))
                v1 = jax.random.uniform(kv1, (initial, input_dim), minval=-b1, maxval=b1)
                v2 = jax.random.uniform(kv2, (output_dim, initial), minval=-b2, maxval=b2)
            elif weight_init == 'normal':
                # DEEP-R, Appendix A: "theta = (1/sqrt(n_in)) N(0,1) c where n_in
                # is the number of afferent neurons" -- the dense layer width.
                # DEEP-R stores theta = |w_0| with an independent random sign;
                # drawing w_0 directly is the same distribution, because w_0 is
                # symmetric about zero. See algorithms/deep_r.py:_init_signs.
                v1 = jax.random.normal(kv1, (initial, input_dim)) / jnp.sqrt(float(input_dim))
                v2 = jax.random.normal(kv2, (output_dim, initial)) / jnp.sqrt(float(initial))
            else:
                raise ValueError(
                    f"model.sparse.weight_init must be 'glorot' or 'normal', "
                    f'got {weight_init!r}')

            W1 = W1.at[:initial].set(v1 * m1)
            W2 = W2.at[:, :initial].set(v2 * m2)
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
