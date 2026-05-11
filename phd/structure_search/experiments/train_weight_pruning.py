"""Simple connection-pruning / unit-generation training for a 2-layer MLP.

Padded-dense representation: W1 of shape (max_hidden, input_dim) and
W2 of shape (output_dim, max_hidden), with masks marking active units
and connections. Pruning is connection-level (smallest contribution
utility); generation creates whole new hidden units, sampling each new
unit's incoming/outgoing fan-in from a configurable range.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import Optional, Tuple

# Resolve relative MLflow tracking URI before Hydra changes CWD
_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
if _uri.startswith('sqlite:///') and not os.path.isabs(_uri[len('sqlite:///'):]):
    os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{os.path.abspath(_uri[len("sqlite:///"):])}'

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer, set_seed, rng_from_string,
)
from phd.jax_core.models import ACTIVATION_MAP
from phd.jax_core.optimizers import EqxOptimizer
from phd.jax_core.optimizers.adam import AdamState
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees, tree_replace
from phd.research_utils.logging import (
    init_experiment, init_child_runs, import_logger, bind_to_active_run,
    log_metrics, log_child_metrics, finish_child_runs, finish_experiment,
)
from phd.structure_search.data import load_dataset, ParallelMNISTStream


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PaddedMLP(eqx.Module):
    """Dense 2-layer MLP with padding to support up to ``max_hidden`` units.

    Inactive units / connections are zeroed via masks; their weights stay in
    the array but contribute nothing to the forward pass.
    """
    W1: jax.Array            # (max_hidden, input_dim)
    W2: jax.Array            # (output_dim, max_hidden)
    unit_mask: jax.Array     # (max_hidden,) — 1 if unit slot is in use
    w1_mask: jax.Array       # (max_hidden, input_dim) — connection mask
    w2_mask: jax.Array       # (output_dim, max_hidden)
    w1_utility: jax.Array    # (max_hidden, input_dim) — EMA of contribution
    w2_utility: jax.Array    # (output_dim, max_hidden)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    max_hidden: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __call__(self, x):
        h = ACTIVATION_MAP[self.activation]((self.W1 * self.w1_mask) @ x)
        h = h * self.unit_mask
        out = (self.W2 * self.w2_mask) @ h
        return out, h


def _lecun_uniform(key, shape, fan_in):
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in, 1.0).astype(jnp.float32))
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * bound


def init_model(
    cfg: DictConfig, input_dim: int, output_dim: int, n_tasks: int,
    *, key: PRNGKeyArray,
) -> PaddedMLP:
    """Build a PaddedMLP per the configured init strategy."""
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
        kw1, kw2, kperm_in, kperm_out, kk_in, kk_out = jax.random.split(key, 6)
        if strategy == 'dense':
            W1 = W1.at[:initial].set(_lecun_uniform(kw1, (initial, input_dim), input_dim))
            W2 = W2.at[:, :initial].set(_lecun_uniform(kw2, (output_dim, initial), initial))
            w1_mask = w1_mask.at[:initial].set(1.0)
            w2_mask = w2_mask.at[:, :initial].set(1.0)
        elif strategy == 'block_sparse':
            assert n_tasks > 1, 'block_sparse init requires n_tasks > 1 (parallel_mnist)'
            assert initial % n_tasks == 0, 'initial_hidden_units must be divisible by n_tasks'
            input_dim_per_task = input_dim // n_tasks
            output_dim_per_task = output_dim // n_tasks
            units_per_task = initial // n_tasks
            W1_init = _lecun_uniform(kw1, (initial, input_dim), input_dim_per_task)
            W2_init = _lecun_uniform(kw2, (output_dim, initial), units_per_task)
            block_w1_mask = jnp.zeros((initial, input_dim), dtype=jnp.float32)
            block_w2_mask = jnp.zeros((output_dim, initial), dtype=jnp.float32)
            for t in range(n_tasks):
                u0, u1 = t * units_per_task, (t + 1) * units_per_task
                i0, i1 = t * input_dim_per_task, (t + 1) * input_dim_per_task
                o0, o1 = t * output_dim_per_task, (t + 1) * output_dim_per_task
                block_w1_mask = block_w1_mask.at[u0:u1, i0:i1].set(1.0)
                block_w2_mask = block_w2_mask.at[o0:o1, u0:u1].set(1.0)
            W1 = W1.at[:initial].set(W1_init * block_w1_mask)
            W2 = W2.at[:, :initial].set(W2_init * block_w2_mask)
            w1_mask = w1_mask.at[:initial].set(block_w1_mask)
            w2_mask = w2_mask.at[:, :initial].set(block_w2_mask)
        elif strategy == 'single_output_block_sparse':
            # Like block_sparse on the input side (each unit sees all
            # input_dim_per_task inputs of its task), but each unit fans out
            # to exactly one output, with units split evenly across the
            # output_dim_per_task outputs within each task.
            assert n_tasks > 1, 'single_output_block_sparse init requires n_tasks > 1 (parallel_mnist)'
            assert initial % n_tasks == 0, 'initial_hidden_units must be divisible by n_tasks'
            input_dim_per_task = input_dim // n_tasks
            output_dim_per_task = output_dim // n_tasks
            units_per_task = initial // n_tasks
            assert units_per_task % output_dim_per_task == 0, (
                'initial_hidden_units / n_tasks must be divisible by output_dim_per_task '
                'so units distribute evenly across outputs'
            )
            units_per_output = units_per_task // output_dim_per_task
            W1_init = _lecun_uniform(kw1, (initial, input_dim), input_dim_per_task)
            W2_init = _lecun_uniform(kw2, (output_dim, initial), units_per_output)
            block_w1_mask = jnp.zeros((initial, input_dim), dtype=jnp.float32)
            block_w2_mask = jnp.zeros((output_dim, initial), dtype=jnp.float32)
            for t in range(n_tasks):
                ut0 = t * units_per_task
                i0, i1 = t * input_dim_per_task, (t + 1) * input_dim_per_task
                o0 = t * output_dim_per_task
                block_w1_mask = block_w1_mask.at[ut0:ut0 + units_per_task, i0:i1].set(1.0)
                for o in range(output_dim_per_task):
                    u0 = ut0 + o * units_per_output
                    u1 = u0 + units_per_output
                    block_w2_mask = block_w2_mask.at[o0 + o, u0:u1].set(1.0)
            W1 = W1.at[:initial].set(W1_init * block_w1_mask)
            W2 = W2.at[:, :initial].set(W2_init * block_w2_mask)
            w1_mask = w1_mask.at[:initial].set(block_w1_mask)
            w2_mask = w2_mask.at[:, :initial].set(block_w2_mask)
        elif strategy == 'random_sparse':
            range_in = tuple(cfg.model.init_random_range_in)
            range_out = tuple(cfg.model.init_random_range_out)
            k_ins = jax.random.randint(kk_in, (initial,), range_in[0], range_in[1] + 1)
            k_outs = jax.random.randint(kk_out, (initial,), range_out[0], range_out[1] + 1)
            in_perm_keys = jax.random.split(kperm_in, initial)
            out_perm_keys = jax.random.split(kperm_out, initial)
            in_weight_keys = jax.random.split(kw1, initial)
            out_weight_keys = jax.random.split(kw2, initial)

            def per_unit(in_perm_key, out_perm_key, in_weight_key, out_weight_key, k_in, k_out):
                in_perm = jax.random.permutation(in_perm_key, input_dim)
                out_perm = jax.random.permutation(out_perm_key, output_dim)
                in_keep = (jnp.arange(input_dim) < k_in).astype(jnp.float32)
                out_keep = (jnp.arange(output_dim) < k_out).astype(jnp.float32)
                w1_mask_row = jnp.zeros(input_dim).at[in_perm].set(in_keep)
                w2_mask_col = jnp.zeros(output_dim).at[out_perm].set(out_keep)
                W1_row = _lecun_uniform(in_weight_key, (input_dim,), k_in.astype(jnp.float32)) * w1_mask_row
                W2_col = _lecun_uniform(out_weight_key, (output_dim,), 1.0) * w2_mask_col
                return w1_mask_row, w2_mask_col, W1_row, W2_col

            w1_mask_rows, w2_mask_cols, W1_rows, W2_cols = jax.vmap(per_unit)(
                in_perm_keys, out_perm_keys, in_weight_keys, out_weight_keys, k_ins, k_outs,
            )
            W1 = W1.at[:initial].set(W1_rows)
            W2 = W2.at[:, :initial].set(W2_cols.T)
            w1_mask = w1_mask.at[:initial].set(w1_mask_rows)
            w2_mask = w2_mask.at[:, :initial].set(w2_mask_cols.T)
        else:
            raise ValueError(f'Unknown init_strategy: {strategy}')

    return PaddedMLP(
        W1=W1, W2=W2, unit_mask=unit_mask,
        w1_mask=w1_mask, w2_mask=w2_mask,
        w1_utility=jnp.zeros_like(W1), w2_utility=jnp.zeros_like(W2),
        input_dim=input_dim, output_dim=output_dim,
        max_hidden=max_hidden, activation=activation,
    )


def _model_filter_spec(model: PaddedMLP):
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(lambda m: (m.W1, m.W2), spec, (True, True))


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: PaddedMLP
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray


# ---------------------------------------------------------------------------
# Optimizer state reset (for Adam, where each weight has its own age)
# ---------------------------------------------------------------------------

def _reset_optimizer_at(
    optimizer: EqxOptimizer, w1_reset: jax.Array, w2_reset: jax.Array,
) -> EqxOptimizer:
    """Zero the per-parameter Adam state at positions where reset masks are True.

    SGD has no per-param state, so this is a no-op for it.
    """
    state = optimizer.state
    if not isinstance(state, AdamState):
        return optimizer

    def _reset_field(field, fill_dtype):
        new_W1 = jnp.where(w1_reset, jnp.asarray(0, dtype=fill_dtype), field.W1)
        new_W2 = jnp.where(w2_reset, jnp.asarray(0, dtype=fill_dtype), field.W2)
        return eqx.tree_at(lambda f: (f.W1, f.W2), field, (new_W1, new_W2))

    new_state = AdamState(
        lr=state.lr,
        step=_reset_field(state.step, jnp.int32),
        exp_avg=_reset_field(state.exp_avg, jnp.float32),
        exp_avg_sq=_reset_field(state.exp_avg_sq, jnp.float32),
    )
    return tree_replace(optimizer, state=new_state)


# ---------------------------------------------------------------------------
# Restructure: prune connections, then generate units
# ---------------------------------------------------------------------------

def _prune_connections(model: PaddedMLP, prune_count: int):
    """Prune the ``prune_count`` smallest-utility active connections.

    Returns updated (W1, W2, w1_mask, w2_mask, w1_utility, w2_utility, unit_mask,
    pruned_w1, pruned_w2, n_pruned). The ``pruned_*`` boolean arrays mark the
    positions that were just zeroed (used to reset optimizer state).
    """
    n_w1_entries = model.w1_mask.size
    util_flat = jnp.concatenate([
        model.w1_utility.reshape(-1), model.w2_utility.reshape(-1),
    ])
    mask_flat = jnp.concatenate([
        model.w1_mask.reshape(-1), model.w2_mask.reshape(-1),
    ])
    score = jnp.where(mask_flat > 0, util_flat, jnp.inf)
    # smallest-K via top_k on negated score
    _, smallest_idx = jax.lax.top_k(-score, prune_count)
    is_active_pick = jnp.isfinite(score[smallest_idx])
    n_pruned = is_active_pick.sum().astype(jnp.int32)

    prune_flat = jnp.zeros_like(mask_flat).at[smallest_idx].set(is_active_pick.astype(jnp.float32))
    pruned_w1 = prune_flat[:n_w1_entries].reshape(model.w1_mask.shape)
    pruned_w2 = prune_flat[n_w1_entries:].reshape(model.w2_mask.shape)

    keep_w1 = 1.0 - pruned_w1
    keep_w2 = 1.0 - pruned_w2
    new_W1 = model.W1 * keep_w1
    new_W2 = model.W2 * keep_w2
    new_w1_mask = model.w1_mask * keep_w1
    new_w2_mask = model.w2_mask * keep_w2
    new_w1_utility = model.w1_utility * keep_w1
    new_w2_utility = model.w2_utility * keep_w2

    # Deactivate units that have lost all in- OR out-connections
    has_in = (new_w1_mask.sum(axis=1) > 0).astype(jnp.float32)
    has_out = (new_w2_mask.sum(axis=0) > 0).astype(jnp.float32)
    new_unit_mask = model.unit_mask * has_in * has_out

    return (new_W1, new_W2, new_w1_mask, new_w2_mask,
            new_w1_utility, new_w2_utility, new_unit_mask,
            pruned_w1.astype(bool), pruned_w2.astype(bool), n_pruned)


def _generate_units(
    model: PaddedMLP, optimizer: EqxOptimizer, rng: PRNGKeyArray,
    *, max_units: int, range_in: Tuple[int, int],
    range_out: Tuple[int, int], connection_budget: float,
) -> Tuple[PaddedMLP, EqxOptimizer, jax.Array]:
    """Generate up to ``max_units`` new hidden units, capped by connection_budget.

    A new unit is generated only if the network currently has room for one
    with the *worst-case* fan-in/fan-out (``range_in[1] + range_out[1]``)
    without exceeding ``connection_budget`` total active connections. The
    actual cost is ``k_in + k_out`` (sampled uniformly from the ranges).
    """
    max_cost = float(range_in[1] + range_out[1])
    median_util = _median_active_util(model)
    cur_conns_init = (model.w1_mask.sum() + model.w2_mask.sum()).astype(jnp.float32)

    def step(carry, key):
        model, optimizer, cur_conns, n_gen = carry
        k_in_key, k_out_key, in_key, out_key, w_in_key, w_out_key = jax.random.split(key, 6)

        free_slot = jnp.argmax(model.unit_mask == 0)
        has_slot = jnp.any(model.unit_mask == 0)
        can_afford = (cur_conns + max_cost) <= connection_budget
        do_gen = has_slot & can_afford

        k_in = jax.random.randint(k_in_key, (), range_in[0], range_in[1] + 1)
        k_out = jax.random.randint(k_out_key, (), range_out[0], range_out[1] + 1)

        in_perm = jax.random.permutation(in_key, model.input_dim)
        out_perm = jax.random.permutation(out_key, model.output_dim)
        in_keep = (jnp.arange(model.input_dim) < k_in).astype(jnp.float32)
        out_keep = (jnp.arange(model.output_dim) < k_out).astype(jnp.float32)
        new_w1_row = jnp.zeros(model.input_dim).at[in_perm].set(in_keep)
        new_w2_col = jnp.zeros(model.output_dim).at[out_perm].set(out_keep)

        new_W1_row = _lecun_uniform(w_in_key, (model.input_dim,), k_in.astype(jnp.float32)) * new_w1_row
        new_W2_col = jnp.zeros(model.output_dim)  # output weights init to zero so no shock
        del w_out_key  # kept in split for API symmetry

        # Apply changes only when do_gen; otherwise leave the slot as-is.
        def _set_row(arr, idx, new_row):
            return arr.at[idx].set(jnp.where(do_gen, new_row, arr[idx]))

        def _set_col(arr, idx, new_col):
            return arr.at[:, idx].set(jnp.where(do_gen, new_col, arr[:, idx]))

        new_model = tree_replace(
            model,
            W1=_set_row(model.W1, free_slot, new_W1_row),
            W2=_set_col(model.W2, free_slot, new_W2_col),
            w1_mask=_set_row(model.w1_mask, free_slot, new_w1_row),
            w2_mask=_set_col(model.w2_mask, free_slot, new_w2_col),
            w1_utility=_set_row(model.w1_utility, free_slot, new_w1_row * median_util),
            w2_utility=_set_col(model.w2_utility, free_slot, new_w2_col * median_util),
            unit_mask=model.unit_mask.at[free_slot].set(
                jnp.where(do_gen, 1.0, model.unit_mask[free_slot]),
            ),
        )

        # Reset optimizer state at newly active positions
        w1_reset = jnp.zeros_like(model.W1, dtype=bool)
        w2_reset = jnp.zeros_like(model.W2, dtype=bool)
        w1_reset = w1_reset.at[free_slot].set((new_w1_row > 0) & do_gen)
        w2_reset = w2_reset.at[:, free_slot].set((new_w2_col > 0) & do_gen)
        new_optimizer = _reset_optimizer_at(optimizer, w1_reset, w2_reset)

        cost = (k_in + k_out).astype(jnp.float32)
        new_cur_conns = jnp.where(do_gen, cur_conns + cost, cur_conns)
        new_n_gen = n_gen + do_gen.astype(jnp.int32)

        return (new_model, new_optimizer, new_cur_conns, new_n_gen), None

    keys = jax.random.split(rng, max_units)
    (model, optimizer, _, n_generated), _ = jax.lax.scan(
        step, (model, optimizer, cur_conns_init, jnp.int32(0)), keys,
    )
    return model, optimizer, n_generated


def _median_active_util(model: PaddedMLP) -> jax.Array:
    """Median of utilities over currently-active connections (NaN-safe)."""
    util = jnp.concatenate([model.w1_utility.reshape(-1), model.w2_utility.reshape(-1)])
    mask = jnp.concatenate([model.w1_mask.reshape(-1), model.w2_mask.reshape(-1)])
    masked = jnp.where(mask > 0, util, jnp.nan)
    med = jnp.nanmedian(masked)
    return jnp.where(jnp.isnan(med), 0.0, med)


def restructure(
    state: TrainState, prune_count: int, max_units_per_event: int,
    range_in: Tuple[int, int], range_out: Tuple[int, int],
    connection_budget: float,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    (new_W1, new_W2, new_w1_mask, new_w2_mask, new_w1_utility, new_w2_utility,
     new_unit_mask, pruned_w1, pruned_w2, n_pruned) = _prune_connections(
        state.model, prune_count,
    )
    model = tree_replace(
        state.model, W1=new_W1, W2=new_W2,
        w1_mask=new_w1_mask, w2_mask=new_w2_mask,
        w1_utility=new_w1_utility, w2_utility=new_w2_utility,
        unit_mask=new_unit_mask,
    )
    optimizer = _reset_optimizer_at(state.optimizer, pruned_w1, pruned_w2)

    next_rng, gen_rng = jax.random.split(state.rng)
    model, optimizer, n_generated = _generate_units(
        model, optimizer, gen_rng,
        max_units=max_units_per_event, range_in=range_in, range_out=range_out,
        connection_budget=connection_budget,
    )
    return tree_replace(
        state, model=model, optimizer=optimizer, rng=next_rng,
    ), n_pruned, n_generated


# ---------------------------------------------------------------------------
# Step + scan
# ---------------------------------------------------------------------------

def train_step(
    state: TrainState, data, *, num_classes: int, n_tasks: int, decay: float,
):
    images, labels = data                          # labels: (B, K), images: (B, K*C_in)
    one_hot = jax.nn.one_hot(labels, num_classes)   # (B, K, C)

    def loss_fn(model):
        outputs, h = jax.vmap(model)(images)
        outputs_r = outputs.reshape(-1, n_tasks, num_classes)
        log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
        loss_per_batch = -jnp.sum(one_hot * log_probs, axis=-1)  # shape (B, K)
        loss = jnp.mean(jnp.sum(loss_per_batch, axis=1), axis=0)
        return loss, (outputs_r, h)

    (loss, (outputs_r, h)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(state.model)
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()

    updates, new_optimizer = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    # EMA of contribution utility = |input_activation| * |weight|, gated by mask.
    x_abs = jnp.abs(images).mean(axis=0)
    h_abs = jnp.abs(h).mean(axis=0)
    w1u = (decay * new_model.w1_utility
           + (1 - decay) * jnp.abs(new_model.W1) * x_abs) * new_model.w1_mask
    w2u = (decay * new_model.w2_utility
           + (1 - decay) * jnp.abs(new_model.W2) * h_abs) * new_model.w2_mask

    new_state = tree_replace(
        state,
        model=tree_replace(new_model, w1_utility=w1u, w2_utility=w2u),
        optimizer=new_optimizer,
        step=state.step + 1,
    )
    return new_state, jnp.stack([loss, correct])




# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _standardize(images: np.ndarray, ref: Optional[np.ndarray] = None):
    """Per-pixel mean-0 var-1 standardization. ref provides stats if given."""
    src = ref if ref is not None else images
    mean = src.mean(axis=0, keepdims=True)
    std = src.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (images - mean) / std


def prepare_experiment(cfg: DictConfig):
    seeds = cfg.seed
    n_tasks = int(cfg.task.n_tasks)
    permute_period = int(cfg.task.permute_period)
    standardize = bool(cfg.task.get('standardize', False))
    eval_freq = int(cfg.train.get('eval_freq', 0))

    raw_train_images, labels, num_classes, input_dim_per_task = load_dataset('mnist', split='train')
    images = _standardize(raw_train_images) if standardize else raw_train_images

    test_images = test_labels = None
    if eval_freq > 0:
        test_images, test_labels, _, _ = load_dataset('mnist', split='test')
        if standardize:
            test_images = _standardize(test_images, ref=raw_train_images)

    input_dim = n_tasks * input_dim_per_task
    output_dim = n_tasks * num_classes

    streams, train_states = [], []
    for seed in seeds:
        rng = jax.random.key(seed)
        streams.append(ParallelMNISTStream(
            images=images, labels=labels, n_tasks=n_tasks,
            batch_size=cfg.train.batch_size, seed=seed,
            permute_period=permute_period,
            test_images=test_images, test_labels=test_labels,
        ))

        model = init_model(
            cfg, input_dim, output_dim, n_tasks,
            key=rng_from_string(rng, 'model'),
        )
        optimizer = prepare_optimizer(
            model, cfg.optimizer.name, cfg.optimizer,
            filter_spec=_model_filter_spec(model),
        )
        train_states.append(TrainState(
            model=model, optimizer=optimizer,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    n_active_conns = int(
        train_states[0].model.w1_mask.sum() + train_states[0].model.w2_mask.sum()
    )
    print(f'PaddedMLP: padded_params={n_params}, active_connections={n_active_conns}, '
          f'initial_units={int(train_states[0].model.unit_mask.sum())}, seeds={seeds}')

    return stack_pytrees(train_states), streams, num_classes, n_tasks


# ---------------------------------------------------------------------------
# Structure diagnostics (computed outside JIT, on the vmapped model)
# ---------------------------------------------------------------------------

def _structure_diagnostics(model: PaddedMLP, n_tasks: int) -> dict:
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


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _eval_forward(model, images, labels, num_classes, n_tasks):
    outputs, _ = jax.vmap(model)(images)
    one_hot = jax.nn.one_hot(labels, num_classes)
    outputs_r = outputs.reshape(-1, n_tasks, num_classes)
    log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
    loss = jnp.mean(jnp.sum(-jnp.sum(one_hot * log_probs, axis=-1), axis=1))
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()
    return loss, correct


def evaluate_test(batched_model, test_images, test_labels,
                  num_classes: int, n_tasks: int, batch_size: int = 512):
    """Evaluate batched (vmapped-over-seeds) model on a test set, chunked."""
    @jax.jit
    def _eval_chunk(model, imgs, lbls):
        return jax.vmap(
            lambda m: _eval_forward(m, imgs, lbls, num_classes, n_tasks),
        )(model)

    n_test = test_images.shape[0]
    total_loss = total_acc = None
    n_chunks = 0
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        chunk_imgs = jnp.array(test_images[start:end])
        chunk_lbls = jnp.array(test_labels[start:end])
        cl, ca = _eval_chunk(batched_model, chunk_imgs, chunk_lbls)
        total_loss = cl if total_loss is None else total_loss + cl
        total_acc = ca if total_acc is None else total_acc + ca
        n_chunks += 1
    return total_loss / n_chunks, total_acc / n_chunks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig, train_state: TrainState, streams,
                   num_classes: int, n_tasks: int):
    log_freq = int(cfg.train.log_freq)
    num_log_periods = int(cfg.train.total_steps) // log_freq
    structure_search_enabled = bool(cfg.structure_search.enabled)
    prune_freq = int(cfg.structure_search.prune_frequency) if structure_search_enabled else log_freq
    eval_freq = int(cfg.train.get('eval_freq', 0))
    n_test_samples_cfg = cfg.train.get('n_test_samples', None)
    n_test_samples = int(n_test_samples_cfg) if n_test_samples_cfg is not None else None

    train_step_fn = partial(
        train_step, num_classes=num_classes, n_tasks=n_tasks,
        decay=float(cfg.structure_search.decay_rate),
    )

    if structure_search_enabled:
        assert log_freq % prune_freq == 0, \
            f'log_freq={log_freq} must be divisible by prune_frequency={prune_freq}'
        prune_cycles_per_log = log_freq // prune_freq
        restructure_fn = partial(
            restructure,
            prune_count=int(cfg.structure_search.prune_count),
            max_units_per_event=int(cfg.structure_search.max_units_per_event),
            range_in=tuple(cfg.structure_search.gen_range_in),
            range_out=tuple(cfg.structure_search.gen_range_out),
            connection_budget=float(cfg.structure_search.connection_budget),
        )

        def prune_cycle(state, cycle_data):
            state, metrics = jax.lax.scan(train_step_fn, state, cycle_data, unroll=SCAN_UNROLL)
            state, n_pruned, n_generated = restructure_fn(state)
            return state, (metrics, n_pruned, n_generated)

        def scan_log_period(state, data):
            data = jax.tree.map(
                lambda x: x.reshape(prune_cycles_per_log, prune_freq, *x.shape[1:]), data,
            )
            state, (metrics, pruned, generated) = jax.lax.scan(prune_cycle, state, data)
            return state, metrics.reshape(-1, *metrics.shape[2:]), pruned, generated
    else:
        def scan_log_period(state, data):
            state, metrics = jax.lax.scan(train_step_fn, state, data, unroll=SCAN_UNROLL)
            return state, metrics, None, None

    vmapped_scan = jax.jit(jax.vmap(scan_log_period))

    all_losses, all_accs, all_per_seed_losses, all_per_seed_accs = [], [], [], []
    all_test_losses, all_test_accs = [], []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))
    cumulative_pruned = 0.0
    cumulative_generated = 0.0

    for _ in range(num_log_periods):
        batch = [s.sample_batch(log_freq) for s in streams]
        imgs = jnp.array(np.stack([b[0] for b in batch]))
        lbls = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics, pruned, generated = vmapped_scan(train_state, (imgs, lbls))

        # metrics: (n_seeds, log_freq, 2)
        per_seed_loss = metrics[..., 0].mean(axis=1)
        per_seed_acc = metrics[..., 1].mean(axis=1)
        mean_loss, mean_acc = float(per_seed_loss.mean()), float(per_seed_acc.mean())
        std_loss, std_acc = float(per_seed_loss.std()), float(per_seed_acc.std())
        step = int(train_state.step[0].item())

        structure_metrics = {
            k: float(v.mean()) for k, v in
            _structure_diagnostics(train_state.model, n_tasks).items()
        }
        if pruned is not None:
            cumulative_pruned += float(pruned.sum(axis=1).mean())
            cumulative_generated += float(generated.sum(axis=1).mean())
            structure_metrics['cumulative_pruned'] = cumulative_pruned
            structure_metrics['cumulative_generated'] = cumulative_generated

        test_metrics_dict = {}
        if eval_freq > 0 and step % eval_freq == 0:
            t_imgs, t_lbls = streams[0].get_test_batch()
            if n_test_samples is not None and n_test_samples < t_imgs.shape[0]:
                t_imgs = t_imgs[:n_test_samples]
                t_lbls = t_lbls[:n_test_samples]
            test_loss, test_acc = evaluate_test(
                train_state.model, t_imgs, t_lbls, num_classes, n_tasks,
            )
            mean_test_loss = float(test_loss.mean())
            mean_test_acc = float(test_acc.mean())
            all_test_losses.append(mean_test_loss)
            all_test_accs.append(mean_test_acc)
            test_metrics_dict = {
                'test_loss': mean_test_loss,
                'test_accuracy': mean_test_acc,
            }

        if logging_active:
            log_futures.append(log_executor.submit(
                _bg_log, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(),
                structure_metrics, test_metrics_dict, cfg, step,
            ))

        all_losses.append(mean_loss)
        all_accs.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accs.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        postfix = {
            'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}',
            'units': f'{structure_metrics["active_units"]:.0f}',
            'conn': f'{structure_metrics["active_connections"]:.0f}',
        }
        if test_metrics_dict:
            postfix['t_acc'] = f'{test_metrics_dict["test_accuracy"]:.4f}'
        pbar.set_postfix(postfix)

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()
    return (train_state, all_losses, all_accs,
            all_per_seed_losses, all_per_seed_accs,
            all_test_losses, all_test_accs)


def _bg_log(mean_loss, std_loss, mean_acc, std_acc,
            per_seed_loss, per_seed_acc, structure_metrics,
            test_metrics_dict, cfg, step):
    metrics = {
        'loss': mean_loss, 'loss_std': std_loss,
        'accuracy': mean_acc, 'accuracy_std': std_acc,
    }
    metrics.update(structure_metrics)
    metrics.update(test_metrics_dict)
    log_metrics(metrics, cfg, step=step)
    log_child_metrics(
        {'loss': per_seed_loss, 'accuracy': per_seed_acc}, cfg, step=step,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_config(cfg: DictConfig) -> dict:
    configure_jax(cfg)
    import_logger(cfg)
    bind_to_active_run(cfg)

    if cfg.seed is None:
        cfg.seed = [int(np.random.randint(0, 1_000_000_000))]
    elif isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    if cfg.get('log_individual_seeds', False) and not cfg.get('mlflow', False):
        raise ValueError('log_individual_seeds requires mlflow logging.')

    set_seed(cfg.seed[0])
    init_child_runs(cfg.seed, cfg)

    train_state, streams, num_classes, n_tasks = prepare_experiment(cfg)
    (train_state, all_losses, all_accs, all_per_seed_losses, all_per_seed_accs,
     all_test_losses, all_test_accs) = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks,
    )

    n_tail = max(1, len(all_losses) // 10)
    summary = {
        'average_loss': float(np.mean(all_losses)),
        'asymptotic_loss': float(np.mean(all_losses[-n_tail:])),
        'asymptotic_accuracy': float(np.mean(all_accs[-n_tail:])),
        'final_active_units': float(train_state.model.unit_mask.sum(axis=-1).mean()),
        'final_active_connections': float(
            (train_state.model.w1_mask.sum(axis=(-1, -2))
             + train_state.model.w2_mask.sum(axis=(-1, -2))).mean()
        ),
    }
    if all_test_losses:
        n_test_tail = max(1, len(all_test_losses) // 10)
        summary['asymptotic_test_loss'] = float(np.mean(all_test_losses[-n_test_tail:]))
        summary['asymptotic_test_accuracy'] = float(np.mean(all_test_accs[-n_test_tail:]))
    print(f'Average loss: {summary["average_loss"]:.4f} | '
          f'Asymptotic loss: {summary["asymptotic_loss"]:.4f} | '
          f'Asymptotic acc: {summary["asymptotic_accuracy"]:.4f}')
    if all_test_losses:
        print(f'Asymptotic test loss: {summary["asymptotic_test_loss"]:.4f} | '
              f'Asymptotic test acc: {summary["asymptotic_test_accuracy"]:.4f}')
    log_metrics(summary, cfg)

    if all_per_seed_losses:
        per_seed_losses = np.stack(all_per_seed_losses)
        per_seed_accs = np.stack(all_per_seed_accs)
        log_child_metrics({
            'average_loss': per_seed_losses.mean(axis=0).tolist(),
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
        }, cfg)

    finish_child_runs(cfg)
    return summary


@hydra.main(config_path='../conf', config_name='train_weight_pruning', version_base='1.1')
def main(cfg: DictConfig) -> None:
    cfg = init_experiment(cfg.project, cfg)
    try:
        run_config(cfg)
    finally:
        finish_experiment(cfg)


if __name__ == '__main__':
    main()
