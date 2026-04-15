"""Shared utilities and JAX experiment core for local_pruning_progression.

Contains all the JAX training code (forward, loss, utilities,
build_run_fn, run_variant, aggregate_results) and shared helpers
(mask samplers, metrics, MLflow, data loading). Step scripts import
from here and only define their sweep config + objective function.
"""

import os
import sys
from typing import Any, Dict

# Make phd/structure_search/ and the repo root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from phd.jax_core.utils import configure_jax

# Configure JAX persistent XLA compile cache + device before any jit.
configure_jax(OmegaConf.create(dict(
    jax_jit_cache_dir='/tmp/jax_cache',
    device='gpu',
)))


# ═════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════

MLFLOW_PROJECT = 'local_pruning_progression'

N_TASKS = 2
NUM_CLASSES = 10
INPUT_PER_TASK = 784
INPUT_DIM = INPUT_PER_TASK * N_TASKS       # 1568
OUTPUT_DIM = NUM_CLASSES * N_TASKS         # 20
BUDGET = 1500
EMA_DECAY = 0.998
PERMUTE_PERIOD = 4000
SPP = 50                                    # default steps per prune event
PPE = 1                                     # connections pruned per event
N_CYCLES = 4500                             # default 4500 * 50 = 225k steps
TOTAL_STEPS = N_CYCLES * SPP                # 225000 — target for step-based sweeps
EVAL_WINDOW_CYCLES = 800                    # last 40k steps (at default SPP)
EVAL_WINDOW_STEPS = EVAL_WINDOW_CYCLES * SPP
WINDOW_LOG_CYCLES = 100                     # 5k step granularity (at default SPP)
WINDOW_LOG_STEPS = WINDOW_LOG_CYCLES * SPP
N_SEEDS = 20
BASE_SEED = 42


# ═════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════

_MNIST_CACHE = {}

def load_mnist():
    if 'data' not in _MNIST_CACHE:
        from data import load_dataset
        images, labels, _, _ = load_dataset('mnist', split='train')
        _MNIST_CACHE['data'] = (
            jnp.array(images, dtype=jnp.float32),
            jnp.array(labels, dtype=jnp.int32),
        )
    return _MNIST_CACHE['data']


# ═════════════════════════════════════════════════════════════════════
# Stats
# ═════════════════════════════════════════════════════════════════════

def ci95(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    return float(1.96 * arr.std(ddof=0) / np.sqrt(len(arr)))


# ═════════════════════════════════════════════════════════════════════
# MLflow helpers
# ═════════════════════════════════════════════════════════════════════

_DEFAULT_MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'


def resolve_mlflow_tracking_uri() -> str:
    uri = os.environ.get('MLFLOW_TRACKING_URI', _DEFAULT_MLFLOW_TRACKING_URI)
    prefix = 'sqlite:///'
    if uri.startswith(prefix) and not os.path.isabs(uri[len(prefix):]):
        uri = f'sqlite:///{os.path.abspath(uri[len(prefix):])}'
    return uri


def resolve_optuna_tracking_uri() -> str:
    default = 'sqlite:///optuna.db'
    uri = os.environ.get('OPTUNA_TRACKING_URI', default)
    prefix = 'sqlite:///'
    if uri.startswith(prefix) and not os.path.isabs(uri[len(prefix):]):
        uri = f'sqlite:///{os.path.abspath(uri[len(prefix):])}'
    return uri


def log_result_metrics(results: dict):
    """Log aggregated metrics to the currently-active MLflow run.

    Logs final scalars (mean + CI95) and a per-5k-step loss trajectory
    (mean across seeds) so convergence is visible in MLflow.
    """
    import mlflow
    for name, arr in [('final_loss', results['final_losses']),
                      ('alignment', results['alignments']),
                      ('separation_f1', results['separation_f1s']),
                      ('purity', results['purities']),
                      ('entropy', results['entropies'])]:
        mlflow.log_metric(name, float(arr.mean()))
        mlflow.log_metric(f'{name}_ci95', ci95(arr))

    # Loss trajectory (mean across seeds, one point per 5k-step window).
    windowed = np.asarray(results['windowed_loss'])   # (S, T)
    steps = np.asarray(results['window_steps'])       # (T,)
    mean_traj = windowed.mean(axis=0)
    for t, s in enumerate(steps):
        mlflow.log_metric('loss_window_5k', float(mean_traj[t]), step=int(s))


# ═════════════════════════════════════════════════════════════════════
# Fixed-mask samplers
# ═════════════════════════════════════════════════════════════════════

def sample_fixed_mask_random(key, output_dim: int, input_dim: int,
                             budget: int) -> jnp.ndarray:
    total = output_dim * input_dim
    noise = jax.random.uniform(key, (total,))
    picks = jnp.argsort(noise)[:budget]
    flat = jnp.zeros(total, dtype=jnp.int32).at[picks].set(1)
    return flat.reshape(output_dim, input_dim)


def sample_fixed_mask_intask(key, output_dim: int, input_dim: int,
                             budget: int, n_tasks: int = 2) -> jnp.ndarray:
    assert budget % n_tasks == 0
    out_per = output_dim // n_tasks
    in_per = input_dim // n_tasks
    budget_per = budget // n_tasks
    block = out_per * in_per
    M = jnp.zeros((output_dim, input_dim), dtype=jnp.int32)
    keys = jax.random.split(key, n_tasks)
    for t in range(n_tasks):
        noise = jax.random.uniform(keys[t], (block,))
        picks = jnp.argsort(noise)[:budget_per]
        flat = jnp.zeros(block, dtype=jnp.int32).at[picks].set(1)
        sub = flat.reshape(out_per, in_per)
        M = M.at[t * out_per:(t + 1) * out_per,
                 t * in_per:(t + 1) * in_per].set(sub)
    return M


def sample_init_mask_dynamic(key, output_dim, input_dim, budget):
    return sample_fixed_mask_random(key, output_dim, input_dim, budget)


# ═════════════════════════════════════════════════════════════════════
# Purity / entropy / alignment
# ═════════════════════════════════════════════════════════════════════

def purity_entropy_linear(M, input_per_task=784, n_tasks=2):
    M = np.asarray(M)
    purs, ents = [], []
    for u in range(M.shape[0]):
        counts = np.array([
            int(M[u, t * input_per_task:(t + 1) * input_per_task].sum())
            for t in range(n_tasks)])
        total = counts.sum()
        if total == 0:
            continue
        purs.append(counts.max() / total)
        ps = counts / total
        e = sum(-p * np.log2(p) for p in ps if p > 0)
        ents.append(e / np.log2(n_tasks))
    if not purs:
        return 0.0, 1.0
    return float(np.mean(purs)), float(np.mean(ents))


def batch_purity_entropy_linear(all_M, input_per_task=784, n_tasks=2):
    all_M = np.asarray(all_M)
    S = all_M.shape[0]
    purs, ents = np.zeros(S), np.zeros(S)
    for s in range(S):
        purs[s], ents[s] = purity_entropy_linear(
            all_M[s], input_per_task, n_tasks)
    return purs, ents


def task_alignment_linear(M, input_per_task=784, n_tasks=2, num_classes=10):
    M = np.asarray(M)
    total = int(M.sum())
    if total == 0:
        return 0.0
    aligned = 0
    for t in range(n_tasks):
        out_lo, out_hi = t * num_classes, (t + 1) * num_classes
        in_lo, in_hi = t * input_per_task, (t + 1) * input_per_task
        aligned += int(M[out_lo:out_hi, in_lo:in_hi].sum())
    return aligned / total


def batch_task_alignment_linear(all_M, input_per_task=784, n_tasks=2,
                                num_classes=10):
    all_M = np.asarray(all_M)
    S = all_M.shape[0]
    out = np.zeros(S)
    for s in range(S):
        out[s] = task_alignment_linear(
            all_M[s], input_per_task, n_tasks, num_classes)
    return out


def task_separation_f1_linear(M, input_per_task=784, n_tasks=2,
                              num_classes=10):
    """F1 of connectivity as a binary classifier of 'input-task == output-task'.

    TP: active connection & input in same task as output
    FP: active connection & input in different task
    FN: inactive connection & input in same task
    (TN: inactive connection & input in different task — unused)

    Precision == task_alignment. Recall = TP / (TP + FN) = fraction of
    same-task inputs that are connected. F1 penalizes both cross-task
    connections (low precision) and missed same-task inputs (low recall),
    so it can't be gamed by keeping only a handful of aligned connections.
    """
    M = np.asarray(M)
    tp = 0
    for t in range(n_tasks):
        out_lo, out_hi = t * num_classes, (t + 1) * num_classes
        in_lo, in_hi = t * input_per_task, (t + 1) * input_per_task
        tp += int(M[out_lo:out_hi, in_lo:in_hi].sum())
    total_active = int(M.sum())
    fp = total_active - tp
    possible_aligned = n_tasks * num_classes * input_per_task
    fn = possible_aligned - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def batch_task_separation_f1_linear(all_M, input_per_task=784, n_tasks=2,
                                    num_classes=10):
    all_M = np.asarray(all_M)
    S = all_M.shape[0]
    out = np.zeros(S)
    for s in range(S):
        out[s] = task_separation_f1_linear(
            all_M[s], input_per_task, n_tasks, num_classes)
    return out


# ═════════════════════════════════════════════════════════════════════
# Model + utility functions
# ═════════════════════════════════════════════════════════════════════

def forward(W, M, x):
    return (W * M) @ x


def loss_fn(W, M, x, y):
    logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))


def contribution_utility(W, M, x, y):
    return jnp.abs(x)[None, :] * jnp.abs(W) * M


def signed_utility(W, M, x, y):
    logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
    softmax = jax.nn.softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(y, NUM_CLASSES)
    e = (onehot - softmax).reshape(-1)
    c = W * x[None, :]
    u = jnp.abs(e[:, None] + c) - jnp.abs(e[:, None])
    return u * M


UTILITY_FNS = {
    'contribution': contribution_utility,
    'signed': signed_utility,
}


# ═════════════════════════════════════════════════════════════════════
# Prune + generate
# ═════════════════════════════════════════════════════════════════════

def prune_and_generate_one(W, M, U, rng):
    prune_key, gen_key, noise_key = jax.random.split(rng, 3)

    noise = jax.random.uniform(noise_key, M.shape,
                               minval=-1e-10, maxval=1e-10)
    scores = jnp.where(M == 1, U + noise, jnp.inf)
    prune_idx = jnp.argmin(scores.reshape(-1))

    flat_M = M.reshape(-1).at[prune_idx].set(0)
    flat_W = W.reshape(-1).at[prune_idx].set(0.0)
    flat_U = U.reshape(-1).at[prune_idx].set(0.0)
    M, W, U = flat_M.reshape(M.shape), flat_W.reshape(W.shape), flat_U.reshape(U.shape)

    gen_noise = jax.random.uniform(gen_key, M.shape)
    gen_scores = jnp.where(M == 0, gen_noise, 2.0)
    gen_idx = jnp.argmin(gen_scores.reshape(-1))

    n_active = jnp.sum(M).astype(jnp.float32)
    mean_u = jnp.where(n_active > 0, jnp.sum(U) / n_active, 0.0)

    flat_M = M.reshape(-1).at[gen_idx].set(1)
    flat_W = W.reshape(-1).at[gen_idx].set(0.0)
    flat_U = U.reshape(-1).at[gen_idx].set(mean_u)
    return flat_W.reshape(W.shape), flat_M.reshape(M.shape), flat_U.reshape(U.shape)


# ═════════════════════════════════════════════════════════════════════
# Build JIT+vmap run function
# ═════════════════════════════════════════════════════════════════════

_RUN_FN_CACHE = {}


def build_run_fn(mnist_images, mnist_labels, variant: str,
                 n_cycles: int = N_CYCLES, budget: int = BUDGET,
                 utility_fn: str = 'contribution'):
    assert variant in ('dynamic', 'fixed_random', 'fixed_intask')
    assert utility_fn in UTILITY_FNS
    is_dynamic = (variant == 'dynamic')
    utility_impl = UTILITY_FNS[utility_fn]

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr):
        rng, mkey = jax.random.split(rng)
        if variant == 'dynamic':
            M = sample_init_mask_dynamic(mkey, OUTPUT_DIM, INPUT_DIM, budget)
        elif variant == 'fixed_random':
            M = sample_fixed_mask_random(mkey, OUTPUT_DIM, INPUT_DIM, budget)
        else:
            M = sample_fixed_mask_intask(mkey, OUTPUT_DIM, INPUT_DIM,
                                         budget, N_TASKS)

        W = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        U = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)

        def train_step(carry, inputs):
            W, M, U, step, perm0, perm1 = carry
            data_key, perm_key = inputs
            x, y_raw = make_sample(data_key)
            y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])
            loss_val, g = jax.value_and_grad(loss_fn)(W, M, x, y)
            W = W - lr * g * M
            u = utility_impl(W, M, x, y)
            U = EMA_DECAY * U + (1.0 - EMA_DECAY) * u
            step = step + 1
            should_perm = (step >= PERMUTE_PERIOD) & (step % PERMUTE_PERIOD == 0)
            pk1, pk2 = jax.random.split(perm_key)
            which = jax.random.randint(pk1, (), 0, N_TASKS)
            new_perm = jax.random.permutation(pk2, NUM_CLASSES).astype(jnp.int32)
            perm0 = jnp.where(should_perm & (which == 0), new_perm, perm0)
            perm1 = jnp.where(should_perm & (which == 1), new_perm, perm1)
            return (W, M, U, step, perm0, perm1), loss_val

        def prune_cycle(carry, _):
            W, M, U, step, perm0, perm1, rng = carry
            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, SPP)
            perm_keys = jax.random.split(pk, SPP)
            (W, M, U, step, perm0, perm1), losses = jax.lax.scan(
                train_step, (W, M, U, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()
            if is_dynamic:
                rng, prune_key = jax.random.split(rng)
                W, M, U = prune_and_generate_one(W, M, U, prune_key)
            return (W, M, U, step, perm0, perm1, rng), cycle_loss

        (W, M, U, step, perm0, perm1, rng), per_cycle_loss = jax.lax.scan(
            prune_cycle, (W, M, U, step, perm0, perm1, rng),
            None, length=n_cycles)
        return M, per_cycle_loss

    @jax.jit
    def run_all(rngs, lr):
        return jax.vmap(lambda r: run_one(r, lr))(rngs)

    return run_all


def get_run_fn(variant, n_seeds, n_cycles=N_CYCLES, budget=BUDGET,
               utility_fn='contribution'):
    key = (variant, n_seeds, n_cycles, budget, utility_fn)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist()
        _RUN_FN_CACHE[key] = build_run_fn(
            mnist_images, mnist_labels, variant, n_cycles,
            budget=budget, utility_fn=utility_fn)
    return _RUN_FN_CACHE[key]


# ═════════════════════════════════════════════════════════════════════
# Aggregate + run
# ═════════════════════════════════════════════════════════════════════

def aggregate_results(all_M, all_cycle_loss, n_cycles=N_CYCLES):
    all_M = np.asarray(all_M)
    all_cycle_loss = np.asarray(all_cycle_loss)
    S = all_cycle_loss.shape[0]
    final_losses = all_cycle_loss[:, -EVAL_WINDOW_CYCLES:].mean(axis=1)
    n_windows = n_cycles // WINDOW_LOG_CYCLES
    trimmed = all_cycle_loss[:, :n_windows * WINDOW_LOG_CYCLES]
    windowed = trimmed.reshape(S, n_windows, WINDOW_LOG_CYCLES).mean(axis=2)
    window_steps = np.arange(1, n_windows + 1) * WINDOW_LOG_CYCLES * SPP
    purs, ents = batch_purity_entropy_linear(all_M, INPUT_PER_TASK, N_TASKS)
    aligns = batch_task_alignment_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)
    f1s = batch_task_separation_f1_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)
    return dict(
        final_losses=final_losses, purities=purs, entropies=ents,
        alignments=aligns, separation_f1s=f1s,
        windowed_loss=windowed, window_steps=window_steps,
    )


def run_variant(variant, lr, n_seeds=N_SEEDS, n_cycles=N_CYCLES,
                budget=BUDGET, utility_fn='contribution'):
    """Run one configuration across seeds, return aggregated results dict."""
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn(variant, n_seeds, n_cycles, budget, utility_fn)
    all_M, all_cycle_loss = run_fn(rngs, jnp.float32(lr))
    jax.block_until_ready((all_M, all_cycle_loss))
    return aggregate_results(all_M, all_cycle_loss, n_cycles)


# ═════════════════════════════════════════════════════════════════════
# Step 4 — Threshold pruning from fully connected
# ═════════════════════════════════════════════════════════════════════

def prune_nonpositive(W, M, U):
    """Prune all active connections with EMA utility <= 0. No generation."""
    should_prune = (M == 1) & (U <= 0.0)
    n_pruned = jnp.sum(should_prune)
    keep = 1 - should_prune.astype(jnp.int32)
    return W * keep, M * keep, U * keep, n_pruned


def build_run_fn_step4(mnist_images, mnist_labels, max_cycles=N_CYCLES,
                       spp=SPP):
    """Build JIT+vmap run function for step 4 threshold pruning.

    Starts fully connected, prunes all connections with signed utility <= 0
    every `spp` steps. Stops pruning after 3 consecutive zero-prune events.
    Training continues post-convergence for the eval window.

    Note: each cycle runs `spp` training steps THEN prunes. So the first
    prune event is at step `spp`, not at step 0 — the utility EMA always
    has at least `spp` steps of warmup before any pruning decision.
    """
    utility_impl = signed_utility

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr):
        rng, _ = jax.random.split(rng)
        M = jnp.ones((OUTPUT_DIM, INPUT_DIM), dtype=jnp.int32)
        W = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        U = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        consecutive_zeros = jnp.array(0, dtype=jnp.int32)
        converged = jnp.array(False)
        converge_cycle = jnp.array(max_cycles, dtype=jnp.int32)

        def train_step(carry, inputs):
            W, M, U, step, perm0, perm1 = carry
            data_key, perm_key = inputs
            x, y_raw = make_sample(data_key)
            y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])
            loss_val, g = jax.value_and_grad(loss_fn)(W, M, x, y)
            W = W - lr * g * M
            u = utility_impl(W, M, x, y)
            U = EMA_DECAY * U + (1.0 - EMA_DECAY) * u
            step = step + 1
            should_perm = (step >= PERMUTE_PERIOD) & (
                step % PERMUTE_PERIOD == 0)
            pk1, pk2 = jax.random.split(perm_key)
            which = jax.random.randint(pk1, (), 0, N_TASKS)
            new_perm = jax.random.permutation(
                pk2, NUM_CLASSES).astype(jnp.int32)
            perm0 = jnp.where(
                should_perm & (which == 0), new_perm, perm0)
            perm1 = jnp.where(
                should_perm & (which == 1), new_perm, perm1)
            return (W, M, U, step, perm0, perm1), loss_val

        def prune_cycle(carry, cycle_idx):
            (W, M, U, step, perm0, perm1, rng,
             consecutive_zeros, converged, converge_cycle) = carry
            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, spp)
            perm_keys = jax.random.split(pk, spp)
            (W, M, U, step, perm0, perm1), losses = jax.lax.scan(
                train_step, (W, M, U, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()

            # Prune (skip if already converged)
            W_p, M_p, U_p, n_pruned = prune_nonpositive(W, M, U)
            W = jnp.where(converged, W, W_p)
            M = jnp.where(converged, M, M_p)
            U = jnp.where(converged, U, U_p)
            n_pruned = jnp.where(converged, 0, n_pruned)

            # Convergence tracking
            new_consec = jnp.where(
                n_pruned == 0,
                consecutive_zeros + 1,
                jnp.array(0, dtype=jnp.int32))
            new_consec = jnp.where(converged, consecutive_zeros, new_consec)
            newly_converged = (~converged) & (new_consec >= 3)
            converged = converged | newly_converged
            converge_cycle = jnp.where(
                newly_converged, cycle_idx, converge_cycle)

            n_active = jnp.sum(M)
            return (W, M, U, step, perm0, perm1, rng,
                    new_consec, converged, converge_cycle), \
                   (cycle_loss,
                    n_pruned.astype(jnp.int32),
                    n_active.astype(jnp.int32))

        init_carry = (W, M, U, step, perm0, perm1, rng,
                      consecutive_zeros, converged, converge_cycle)
        final_carry, (per_cycle_loss, per_cycle_pruned,
                      per_cycle_active) = jax.lax.scan(
            prune_cycle, init_carry,
            jnp.arange(max_cycles, dtype=jnp.int32))

        final_M = final_carry[1]
        final_converge_cycle = final_carry[9]
        return (final_M, per_cycle_loss, per_cycle_pruned,
                per_cycle_active, final_converge_cycle)

    @jax.jit
    def run_all(rngs, lr):
        return jax.vmap(lambda r: run_one(r, lr))(rngs)

    return run_all


def get_run_fn_step4(n_seeds, max_cycles=N_CYCLES, spp=SPP):
    key = ('step4', n_seeds, max_cycles, spp)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist()
        _RUN_FN_CACHE[key] = build_run_fn_step4(
            mnist_images, mnist_labels, max_cycles, spp)
    return _RUN_FN_CACHE[key]


def aggregate_results_step4(all_M, all_cycle_loss, all_pruned, all_active,
                            converge_cycles, max_cycles=N_CYCLES, spp=SPP):
    all_M = np.asarray(all_M)
    all_cycle_loss = np.asarray(all_cycle_loss)
    all_pruned = np.asarray(all_pruned)
    all_active = np.asarray(all_active)
    converge_cycles = np.asarray(converge_cycles)
    S = all_cycle_loss.shape[0]

    # Eval window: last EVAL_WINDOW_STEPS of training
    eval_window_cycles = min(EVAL_WINDOW_STEPS // spp, max_cycles)
    if eval_window_cycles < 1:
        eval_window_cycles = 1
    final_losses = all_cycle_loss[:, -eval_window_cycles:].mean(axis=1)

    # Trajectory windows: every WINDOW_LOG_STEPS of training
    window_log_cycles = max(1, WINDOW_LOG_STEPS // spp)
    n_windows = max_cycles // window_log_cycles
    n_trim = n_windows * window_log_cycles
    trimmed = all_cycle_loss[:, :n_trim]
    windowed = trimmed.reshape(S, n_windows, window_log_cycles).mean(axis=2)
    window_steps = np.arange(1, n_windows + 1) * window_log_cycles * spp

    trimmed_pruned = all_pruned[:, :n_trim]
    pruned_windowed = trimmed_pruned.reshape(
        S, n_windows, window_log_cycles).sum(axis=2).mean(axis=0)
    trimmed_active = all_active[:, :n_trim]
    active_windowed = trimmed_active.reshape(
        S, n_windows, window_log_cycles)[:, :, -1].mean(axis=0)

    purs, ents = batch_purity_entropy_linear(all_M, INPUT_PER_TASK, N_TASKS)
    aligns = batch_task_alignment_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)
    f1s = batch_task_separation_f1_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)

    # Per-seed final budget (n_active at convergence). For seeds that
    # never converged, converge_cycles == max_cycles, so clip.
    cc_clipped = np.minimum(converge_cycles, max_cycles - 1)
    final_budgets = all_active[np.arange(S), cc_clipped].astype(np.float64)

    # Convergence steps (= converge_cycles * spp) for easier reporting
    converge_steps = converge_cycles.astype(np.float64) * spp

    return dict(
        final_losses=final_losses, purities=purs, entropies=ents,
        alignments=aligns, separation_f1s=f1s,
        windowed_loss=windowed, window_steps=window_steps,
        converge_cycles=converge_cycles.astype(np.float64),
        converge_steps=converge_steps,
        final_budgets=final_budgets,
        pruned_trajectory=all_pruned, active_trajectory=all_active,
        pruned_windowed=pruned_windowed, active_windowed=active_windowed,
        spp=spp,
    )


def run_threshold_variant(lr, n_seeds=N_SEEDS, spp=SPP,
                          total_steps=TOTAL_STEPS):
    """Run step 4 threshold pruning across seeds, return aggregated results.

    Holds total training steps constant across spp values — n_cycles is
    derived as total_steps // spp so each run sees the same amount of
    training.
    """
    max_cycles = total_steps // spp
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn_step4(n_seeds, max_cycles, spp)
    out = run_fn(rngs, jnp.float32(lr))
    all_M, all_cycle_loss, all_pruned, all_active, converge_cycles = out
    jax.block_until_ready(out)
    return aggregate_results_step4(
        all_M, all_cycle_loss, all_pruned, all_active,
        converge_cycles, max_cycles, spp)


def log_result_metrics_step4(results):
    """Log step 4 metrics: base metrics + convergence + trajectories."""
    import mlflow
    log_result_metrics(results)

    for name, arr in [('converge_cycle', results['converge_cycles']),
                      ('converge_step', results['converge_steps']),
                      ('final_budget', results['final_budgets'])]:
        mlflow.log_metric(name, float(np.mean(arr)))
        mlflow.log_metric(f'{name}_ci95', ci95(arr))

    window_steps = results['window_steps']
    pruned_w = results['pruned_windowed']
    active_w = results['active_windowed']
    for t, s in enumerate(window_steps):
        mlflow.log_metric('n_pruned_window', float(pruned_w[t]),
                          step=int(s))
        mlflow.log_metric('n_active', float(active_w[t]),
                          step=int(s))
