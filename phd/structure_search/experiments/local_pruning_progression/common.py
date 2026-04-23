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


def load_mnist_normalized():
    """MNIST with per-pixel standardization so input std ≈ 1.

    Subtracts per-pixel mean and divides by per-pixel std (floored at 1e-3
    so near-constant "dead" pixels don't amplify noise). Used by step 6 so
    the statistical-threshold formula can assume σ_x = 1 without having to
    track a per-weight noise scale at runtime.
    """
    if 'data_normalized' not in _MNIST_CACHE:
        images_jnp, labels = load_mnist()
        images = np.asarray(images_jnp)
        mean = images.mean(axis=0, keepdims=True)
        std = images.std(axis=0, keepdims=True)
        normalized = (images - mean) / np.maximum(std, 1e-3)
        _MNIST_CACHE['data_normalized'] = (
            jnp.array(normalized, dtype=jnp.float32), labels,
        )
    return _MNIST_CACHE['data_normalized']


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
    pairs = [('final_loss', results['final_losses']),
             ('alignment', results['alignments']),
             ('separation_f1', results['separation_f1s']),
             ('purity', results['purities']),
             ('entropy', results['entropies'])]
    if 'final_accuracies' in results:
        pairs.insert(1, ('final_accuracy', results['final_accuracies']))
    for name, arr in pairs:
        mlflow.log_metric(name, float(arr.mean()))
        mlflow.log_metric(f'{name}_ci95', ci95(arr))

    # Loss trajectory (mean across seeds, one point per 5k-step window).
    windowed = np.asarray(results['windowed_loss'])   # (S, T)
    steps = np.asarray(results['window_steps'])       # (T,)
    mean_traj = windowed.mean(axis=0)
    for t, s in enumerate(steps):
        mlflow.log_metric('loss_window_5k', float(mean_traj[t]), step=int(s))
    if 'windowed_acc' in results:
        acc_traj = np.asarray(results['windowed_acc']).mean(axis=0)
        for t, s in enumerate(steps):
            mlflow.log_metric('acc_window_5k', float(acc_traj[t]),
                              step=int(s))


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


def bce_loss_fn(W, M, x, y):
    """Per-output BCE with a 20-dim binary target (two 1's, one per task).

    Scaled by /N_TASKS to keep the per-task loss magnitude roughly
    comparable to softmax CE — makes LR comparisons across loss variants
    less apples-vs-oranges.
    """
    logits = forward(W, M, x)
    target = jax.nn.one_hot(y, NUM_CLASSES).reshape(-1)
    lp = (target * jax.nn.log_sigmoid(logits)
          + (1.0 - target) * jax.nn.log_sigmoid(-logits))
    return -jnp.sum(lp) / float(N_TASKS)


LOSS_FNS = {
    'softmax_ce': loss_fn,
    'bce': bce_loss_fn,
}


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


def bce_utility(W, M, x, y):
    pre_act = forward(W, M, x)
    target = jax.nn.one_hot(y, NUM_CLASSES).reshape(-1)
    pre_act_removed = pre_act[:, None] - x[None, :] * W
    lp = (target * jax.nn.log_sigmoid(pre_act)
          + (1.0 - target) * jax.nn.log_sigmoid(-pre_act))
    lp_removed = (target[:, None] * jax.nn.log_sigmoid(pre_act_removed)
                  + (1.0 - target)[:, None]
                  * jax.nn.log_sigmoid(-pre_act_removed))
    u = (-lp_removed) - (-lp[:, None])
    return u * M


def softmax_ce_utility(W, M, x, y):
    """Per-weight LOO utility under per-task softmax CE (the training loss).

    Removing weight W[k, j] changes only logit[k] by d = −x[j]·W[k, j];
    all other logits are untouched. Exploiting that, the LOO NLL delta
    collapses to

        U[k, j] = −d · 𝟙[k%NUM_CLASSES == y[k // NUM_CLASSES]]
                  + log(1 + p_k · (exp(d) − 1))

    where p_k = softmax(logits_task_t)[k % NUM_CLASSES] is the per-task
    softmax probability at output k. Positive U ⇒ removing hurts NLL ⇒
    keep.
    """
    logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
    p = jax.nn.softmax(logits, axis=-1).reshape(-1)
    k_idx = jnp.arange(OUTPUT_DIM)
    y_match = (k_idx % NUM_CLASSES
               == y[k_idx // NUM_CLASSES]).astype(jnp.float32)
    d = -x[None, :] * W
    u = (-d * y_match[:, None]
         + jnp.log1p(p[:, None] * jnp.expm1(d)))
    return u * M


UTILITY_FNS = {
    'contribution': contribution_utility,
    'signed': signed_utility,
    'bce': bce_utility,
    'softmax_ce': softmax_ce_utility,
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
                 utility_fn: str = 'contribution',
                 loss_fn_name: str = 'softmax_ce'):
    assert variant in ('dynamic', 'fixed_random', 'fixed_intask')
    assert utility_fn in UTILITY_FNS
    assert loss_fn_name in LOSS_FNS
    is_dynamic = (variant == 'dynamic')
    utility_impl = UTILITY_FNS[utility_fn]
    loss_impl = LOSS_FNS[loss_fn_name]

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
            loss_val, g = jax.value_and_grad(loss_impl)(W, M, x, y)
            # Per-task accuracy from the same pre-update forward.
            logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
            acc_val = jnp.mean((jnp.argmax(logits, axis=-1) == y)
                               .astype(jnp.float32))
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
            return (W, M, U, step, perm0, perm1), (loss_val, acc_val)

        def prune_cycle(carry, _):
            W, M, U, step, perm0, perm1, rng = carry
            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, SPP)
            perm_keys = jax.random.split(pk, SPP)
            (W, M, U, step, perm0, perm1), (losses, accs) = jax.lax.scan(
                train_step, (W, M, U, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()
            cycle_acc = accs.mean()
            if is_dynamic:
                rng, prune_key = jax.random.split(rng)
                W, M, U = prune_and_generate_one(W, M, U, prune_key)
            return (W, M, U, step, perm0, perm1, rng), (cycle_loss, cycle_acc)

        (W, M, U, step, perm0, perm1, rng), (per_cycle_loss,
                                              per_cycle_acc) = jax.lax.scan(
            prune_cycle, (W, M, U, step, perm0, perm1, rng),
            None, length=n_cycles)
        return M, per_cycle_loss, per_cycle_acc

    @jax.jit
    def run_all(rngs, lr):
        return jax.vmap(lambda r: run_one(r, lr))(rngs)

    return run_all


def get_run_fn(variant, n_seeds, n_cycles=N_CYCLES, budget=BUDGET,
               utility_fn='contribution', loss_fn_name='softmax_ce'):
    key = (variant, n_seeds, n_cycles, budget, utility_fn, loss_fn_name)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist()
        _RUN_FN_CACHE[key] = build_run_fn(
            mnist_images, mnist_labels, variant, n_cycles,
            budget=budget, utility_fn=utility_fn,
            loss_fn_name=loss_fn_name)
    return _RUN_FN_CACHE[key]


# ═════════════════════════════════════════════════════════════════════
# Aggregate + run
# ═════════════════════════════════════════════════════════════════════

def aggregate_results(all_M, all_cycle_loss, all_cycle_acc=None,
                      n_cycles=N_CYCLES):
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
    out = dict(
        final_losses=final_losses, purities=purs, entropies=ents,
        alignments=aligns, separation_f1s=f1s,
        windowed_loss=windowed, window_steps=window_steps,
    )
    if all_cycle_acc is not None:
        all_cycle_acc = np.asarray(all_cycle_acc)
        out['final_accuracies'] = all_cycle_acc[:, -EVAL_WINDOW_CYCLES:].mean(axis=1)
        acc_trim = all_cycle_acc[:, :n_windows * WINDOW_LOG_CYCLES]
        out['windowed_acc'] = acc_trim.reshape(
            S, n_windows, WINDOW_LOG_CYCLES).mean(axis=2)
    return out


def run_variant(variant, lr, n_seeds=N_SEEDS, n_cycles=N_CYCLES,
                budget=BUDGET, utility_fn='contribution',
                loss_fn_name='softmax_ce'):
    """Run one configuration across seeds, return aggregated results dict."""
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn(variant, n_seeds, n_cycles, budget, utility_fn,
                        loss_fn_name)
    all_M, all_cycle_loss, all_cycle_acc = run_fn(rngs, jnp.float32(lr))
    jax.block_until_ready((all_M, all_cycle_loss, all_cycle_acc))
    return aggregate_results(all_M, all_cycle_loss, all_cycle_acc, n_cycles)


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
                            converge_cycles, max_cycles=N_CYCLES, spp=SPP,
                            all_cycle_acc=None):
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

    out = dict(
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
    if all_cycle_acc is not None:
        all_cycle_acc = np.asarray(all_cycle_acc)
        out['final_accuracies'] = all_cycle_acc[:, -eval_window_cycles:].mean(axis=1)
        trimmed_acc = all_cycle_acc[:, :n_trim]
        out['windowed_acc'] = trimmed_acc.reshape(
            S, n_windows, window_log_cycles).mean(axis=2)
    return out


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


# ═════════════════════════════════════════════════════════════════════
# Step 6 — Statistical-confidence threshold pruning
# ═════════════════════════════════════════════════════════════════════

# Precomputed constants in the τ formula:
#   K = C · (1 − β) / (1 + β)
# The leading C depends on the utility. For signed utility the per-sample
# noise has the form |N(0,σ²)| − const, so Var ∝ (1 − 2/π)·σ² and
# C = 1 − 2/π. For BCE, the per-sample term is (σ(pre_act) − target)·w·x
# (no absolute value), so its per-sample variance is bounded by |w·x|·σ_x
# directly and C = 1.
STAT_PRUNE_K_SIGNED = (1.0 - 2.0 / np.pi) * (1.0 - EMA_DECAY) / (1.0 + EMA_DECAY)
STAT_PRUNE_K_BCE = (1.0 - EMA_DECAY) / (1.0 + EMA_DECAY)
STAT_PRUNE_K = STAT_PRUNE_K_SIGNED  # default / back-compat

UTILITY_K = {
    'signed': STAT_PRUNE_K_SIGNED,
    'bce': STAT_PRUNE_K_BCE,
}


def prune_statistical(W, M, U, step, z_alpha, K=STAT_PRUNE_K):
    """Prune active connections where the bias-corrected EMA utility is
    below the per-weight statistical confidence threshold τ_w.

    With σ_x = 1 (inputs normalized upfront), the threshold is
        τ_w = −z_α · |w| · sqrt(K · (1 + β^t) / (1 − β^t))
    and the EMA is bias-corrected as
        U_corr = U / (1 − β^t).
    A weight is pruned iff U_corr < τ_w, i.e. the utility is bad enough
    that we are 1 − α confident it isn't just noise.

    `K` selects the utility-specific leading constant (STAT_PRUNE_K_SIGNED
    for step 6 signed utility, STAT_PRUNE_K_BCE for step 8 BCE utility).

    t = step count (= age of the weight, since weights are never
    regenerated in this code path).
    """
    t = step.astype(jnp.float32)
    beta = jnp.float32(EMA_DECAY)
    beta_t = jnp.power(beta, t)
    # Floor to avoid divide-by-zero at t=0. In practice the first prune
    # is at step = spp > 0, so 1 − β^t > 0 already — belt and braces.
    one_minus = jnp.maximum(1.0 - beta_t, 1e-12)
    U_corr = U / one_minus
    ratio = (1.0 + beta_t) / one_minus
    tau = -z_alpha * jnp.abs(W) * jnp.sqrt(K * ratio)
    should_prune = (M == 1) & (U_corr < tau)
    n_pruned = jnp.sum(should_prune)
    keep = 1 - should_prune.astype(jnp.int32)
    return W * keep, M * keep, U * keep, n_pruned


def build_run_fn_step6(mnist_images, mnist_labels, max_cycles=N_CYCLES,
                       spp=SPP, utility_fn='signed',
                       loss_fn_name='softmax_ce'):
    """Build JIT+vmap run function for step 6 statistical-threshold pruning.

    Like step 4 but swaps the U ≤ 0 rule for a per-weight statistical
    confidence threshold. z_alpha is passed as a runtime argument so a
    single JIT compile covers every CI value in the sweep.

    Expects `mnist_images` to be pre-normalized (per-pixel std ≈ 1) so
    the formula can assume σ_x = 1 without runtime bookkeeping.

    `utility_fn` selects the per-connection utility; the threshold
    constant K is picked accordingly. `loss_fn_name` selects the training
    loss ('softmax_ce' for step 6, 'bce' for step 8's BCE-matched variant).
    Per-step accuracy is tracked alongside loss for cross-variant
    comparison.
    """
    assert utility_fn in UTILITY_FNS
    assert utility_fn in UTILITY_K, (
        f"utility_fn {utility_fn!r} has no K defined for the threshold")
    assert loss_fn_name in LOSS_FNS
    utility_impl = UTILITY_FNS[utility_fn]
    loss_impl = LOSS_FNS[loss_fn_name]
    K = UTILITY_K[utility_fn]

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr, z_alpha):
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
            loss_val, g = jax.value_and_grad(loss_impl)(W, M, x, y)
            logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
            acc_val = jnp.mean((jnp.argmax(logits, axis=-1) == y)
                               .astype(jnp.float32))
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
            return (W, M, U, step, perm0, perm1), (loss_val, acc_val)

        def prune_cycle(carry, cycle_idx):
            (W, M, U, step, perm0, perm1, rng,
             consecutive_zeros, converged, converge_cycle) = carry
            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, spp)
            perm_keys = jax.random.split(pk, spp)
            (W, M, U, step, perm0, perm1), (losses, accs) = jax.lax.scan(
                train_step, (W, M, U, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()
            cycle_acc = accs.mean()

            # Prune (skip if already converged)
            W_p, M_p, U_p, n_pruned = prune_statistical(
                W, M, U, step, z_alpha, K)
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
                   (cycle_loss, cycle_acc,
                    n_pruned.astype(jnp.int32),
                    n_active.astype(jnp.int32))

        init_carry = (W, M, U, step, perm0, perm1, rng,
                      consecutive_zeros, converged, converge_cycle)
        final_carry, (per_cycle_loss, per_cycle_acc, per_cycle_pruned,
                      per_cycle_active) = jax.lax.scan(
            prune_cycle, init_carry,
            jnp.arange(max_cycles, dtype=jnp.int32))

        final_M = final_carry[1]
        final_converge_cycle = final_carry[9]
        return (final_M, per_cycle_loss, per_cycle_acc, per_cycle_pruned,
                per_cycle_active, final_converge_cycle)

    @jax.jit
    def run_all(rngs, lr, z_alpha):
        return jax.vmap(lambda r: run_one(r, lr, z_alpha))(rngs)

    return run_all


def get_run_fn_step6(n_seeds, max_cycles=N_CYCLES, spp=SPP,
                     utility_fn='signed', loss_fn_name='softmax_ce'):
    key = ('step6', n_seeds, max_cycles, spp, utility_fn, loss_fn_name)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist_normalized()
        _RUN_FN_CACHE[key] = build_run_fn_step6(
            mnist_images, mnist_labels, max_cycles, spp, utility_fn,
            loss_fn_name)
    return _RUN_FN_CACHE[key]


def run_statistical_variant(lr, z_alpha, n_seeds=N_SEEDS, spp=SPP,
                            total_steps=TOTAL_STEPS, utility_fn='signed',
                            loss_fn_name='softmax_ce'):
    """Run step 6 statistical-threshold pruning across seeds.

    Holds total training steps constant across SPP values: n_cycles is
    derived as total_steps // spp so each run sees the same amount of
    training.

    `utility_fn` selects the per-connection utility ('signed' for step 6,
    'bce' for step 8). `loss_fn_name` selects the training loss
    ('softmax_ce' default; 'bce' for step 8's BCE-matched variant).
    """
    max_cycles = total_steps // spp
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn_step6(n_seeds, max_cycles, spp, utility_fn,
                              loss_fn_name)
    out = run_fn(rngs, jnp.float32(lr), jnp.float32(z_alpha))
    (all_M, all_cycle_loss, all_cycle_acc, all_pruned, all_active,
     converge_cycles) = out
    jax.block_until_ready(out)
    return aggregate_results_step4(
        all_M, all_cycle_loss, all_pruned, all_active,
        converge_cycles, max_cycles, spp, all_cycle_acc=all_cycle_acc)


# ═════════════════════════════════════════════════════════════════════
# Step 7 — Demand-driven connection generation
# ═════════════════════════════════════════════════════════════════════

# Demand EMA decay (~100-update memory; faster than utility EMA β=0.998
# because demand updates are sparse — at most a few per cycle).
BETA_D = 0.99
# Connection auto-resolves into demand EMA at this age (= 1/(1−β),
# the effective sample size of the utility EMA — by then the EMA has
# fully matured even without a threshold crossing).
DEMAND_N_EFF = 500
# Softmax allocation temperature. Raw signed-utility magnitudes are
# typically O(0.01-0.1), so T=0.01 gives meaningful concentration.
SOFTMAX_T = 0.01

# Allocation method codes (so build_run_fn_step7 can branch in pure Python
# at compile time without dragging string-handling into JAX).
_ALLOC_CLIPPED_LINEAR = 'clipped_linear'
_ALLOC_SOFTMAX = 'softmax'


def _bias_correction_factors(A):
    """Return (one_minus, ratio) used by both U_corr and τ_w.

    Uses per-connection age A (int32, shape M) instead of the global step
    counter — so newly-generated connections get the wide CI / generous
    threshold appropriate for an immature EMA.
    """
    t = A.astype(jnp.float32)
    beta = jnp.float32(EMA_DECAY)
    beta_t = jnp.power(beta, t)
    one_minus = jnp.maximum(1.0 - beta_t, 1e-12)
    ratio = (1.0 + beta_t) / one_minus
    return one_minus, ratio


def _resolve_and_update_demand(W, M, U, A, R, D, z_alpha):
    """Per-cycle resolution + demand EMA update.

    Returns (D_new, R_new, U_corr, tau_w). U_corr and tau_w are returned
    so the prune step doesn't have to recompute them.
    """
    one_minus, ratio = _bias_correction_factors(A)
    U_corr = U / one_minus
    tau_w = -z_alpha * jnp.abs(W) * jnp.sqrt(STAT_PRUNE_K * ratio)

    # Symmetric two-sided + age fallback. Only resolve once per
    # incarnation (R==0) and only for active connections (M==1).
    crossed_neg = U_corr < tau_w
    crossed_pos = U_corr > -tau_w
    aged_out = A >= DEMAND_N_EFF
    newly_resolved = (M == 1) & (R == 0) & (
        crossed_neg | crossed_pos | aged_out)
    nr_f = newly_resolved.astype(jnp.float32)

    # Per-output-neuron count and mean of the bias-corrected utility of
    # newly-resolved connections.
    m_per_neuron = jnp.sum(nr_f, axis=1)                  # (OUTPUT_DIM,)
    sum_per_neuron = jnp.sum(nr_f * U_corr, axis=1)       # (OUTPUT_DIM,)
    safe_m = jnp.maximum(m_per_neuron, 1.0)
    u_mean_per_neuron = sum_per_neuron / safe_m

    # Closed-form: applying d ← β_d·d + (1−β_d)·u_mean repeatedly m_i
    # times collapses to d ← β_d^m_i·d + (1−β_d^m_i)·u_mean. For m_i=0,
    # β_d^0 = 1, so D[i] is unchanged.
    decay_pow = jnp.power(jnp.float32(BETA_D), m_per_neuron)
    D_new = decay_pow * D + (1.0 - decay_pow) * u_mean_per_neuron

    R_new = R | newly_resolved.astype(jnp.int32)
    return D_new, R_new, U_corr, tau_w


def _prune_step7(W, M, U, A, R, U_corr, tau_w):
    """Prune connections with U_corr < τ_w. Zero out all per-connection
    state (W, U, A, R) for pruned slots so a future generation event into
    the same slot starts from a clean state.
    """
    should_prune = (M == 1) & (U_corr < tau_w)
    n_pruned = jnp.sum(should_prune)
    keep = 1 - should_prune.astype(jnp.int32)
    keep_f = keep.astype(jnp.float32)
    return (W * keep_f, M * keep, U * keep_f, A * keep, R * keep,
            n_pruned)


def _allocate(D, method):
    """Compute per-output-neuron allocation probabilities from demand D."""
    if method == _ALLOC_CLIPPED_LINEAR:
        pos = jnp.maximum(D, 0.0)
        s = jnp.sum(pos)
        n = D.shape[0]
        uniform = jnp.full((n,), 1.0 / n, dtype=jnp.float32)
        return jnp.where(s > 0, pos / jnp.maximum(s, 1e-12), uniform)
    elif method == _ALLOC_SOFTMAX:
        return jax.nn.softmax(D / jnp.float32(SOFTMAX_T))
    else:
        raise ValueError(f'Unknown allocation method: {method}')


def _generate(W, M, U, A, R, n_pruned, p, rng):
    """Sample n_pruned new connections via Gumbel-max top-k.

    Only empty slots (M==0) are eligible. Each empty slot's score is
    log(p[output]) + Gumbel(0,1) + tiny uniform tie-breaker. We pick
    the n_pruned highest-scoring empty slots and mark them active. New
    slots inherit W=U=A=R=0 (which they already are, since pruning
    zeroed them and untouched-empty slots were never written).
    """
    g_key, n_key = jax.random.split(rng)
    log_p = jnp.log(p + 1e-12)[:, None]                   # (OUTPUT_DIM, 1)
    u = jax.random.uniform(g_key, M.shape,
                           minval=1e-12, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    tiny = 1e-12 * jax.random.uniform(n_key, M.shape)
    raw_scores = log_p + gumbel + tiny
    neg_inf = jnp.full(M.shape, -jnp.inf, dtype=jnp.float32)
    scores = jnp.where(M == 0, raw_scores, neg_inf)

    flat = scores.reshape(-1)
    sorted_desc = -jnp.sort(-flat)                        # descending
    idx = jnp.maximum(n_pruned.astype(jnp.int32) - 1, 0)
    threshold = jnp.where(
        n_pruned > 0,
        sorted_desc[idx],
        jnp.float32(jnp.inf))
    to_generate = (M == 0) & (scores >= threshold)
    n_generated = jnp.sum(to_generate)
    add = to_generate.astype(jnp.int32)
    return W, M + add, U, A, R, n_generated


def build_run_fn_step7(mnist_images, mnist_labels, n_cycles=N_CYCLES,
                       spp=SPP, allocation_method=_ALLOC_CLIPPED_LINEAR):
    """Build JIT+vmap run function for step 7 demand-driven generation.

    Starts from a random sparse mask at BUDGET=1500, then every `spp`
    train steps: resolve newly-mature connections into per-output-neuron
    demand D, statistically prune (CI-based), then generate exactly
    n_pruned_this_cycle new connections via Gumbel-max top-k weighted by
    `_allocate(D, allocation_method)`. Active count stays at BUDGET.

    Expects pre-normalized MNIST (load_mnist_normalized) so τ_w can use
    σ_x = 1.
    """
    utility_impl = signed_utility

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr, z_alpha):
        rng, mkey = jax.random.split(rng)
        M = sample_init_mask_dynamic(
            mkey, OUTPUT_DIM, INPUT_DIM, BUDGET)
        W = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        U = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        A = jnp.zeros((OUTPUT_DIM, INPUT_DIM), dtype=jnp.int32)
        R = jnp.zeros((OUTPUT_DIM, INPUT_DIM), dtype=jnp.int32)
        D = jnp.zeros((OUTPUT_DIM,), dtype=jnp.float32)
        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)

        def train_step(carry, inputs):
            W, M, U, A, step, perm0, perm1 = carry
            data_key, perm_key = inputs
            x, y_raw = make_sample(data_key)
            y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])
            loss_val, g = jax.value_and_grad(loss_fn)(W, M, x, y)
            W = W - lr * g * M
            u = utility_impl(W, M, x, y)
            U = EMA_DECAY * U + (1.0 - EMA_DECAY) * u
            A = A + M
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
            return (W, M, U, A, step, perm0, perm1), loss_val

        def cycle(carry, _):
            (W, M, U, A, R, D, step, perm0, perm1, rng) = carry
            rng, tk, pk, gk = jax.random.split(rng, 4)
            data_keys = jax.random.split(tk, spp)
            perm_keys = jax.random.split(pk, spp)
            (W, M, U, A, step, perm0, perm1), losses = jax.lax.scan(
                train_step, (W, M, U, A, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()

            # Resolve + demand update
            D, R, U_corr, tau_w = _resolve_and_update_demand(
                W, M, U, A, R, D, z_alpha)

            # Prune
            W, M, U, A, R, n_pruned = _prune_step7(
                W, M, U, A, R, U_corr, tau_w)

            # Generate
            p = _allocate(D, allocation_method)
            W, M, U, A, R, n_generated = _generate(
                W, M, U, A, R, n_pruned, p, gk)

            n_active = jnp.sum(M)
            return (W, M, U, A, R, D, step, perm0, perm1, rng), \
                   (cycle_loss,
                    n_pruned.astype(jnp.int32),
                    n_generated.astype(jnp.int32),
                    n_active.astype(jnp.int32),
                    D)

        init_carry = (W, M, U, A, R, D, step, perm0, perm1, rng)
        final_carry, (per_cycle_loss, per_cycle_pruned,
                      per_cycle_generated, per_cycle_active,
                      per_cycle_D) = jax.lax.scan(
            cycle, init_carry, None, length=n_cycles)

        final_M = final_carry[1]
        return (final_M, per_cycle_loss, per_cycle_pruned,
                per_cycle_generated, per_cycle_active, per_cycle_D)

    @jax.jit
    def run_all(rngs, lr, z_alpha):
        return jax.vmap(lambda r: run_one(r, lr, z_alpha))(rngs)

    return run_all


def get_run_fn_step7(n_seeds, n_cycles=N_CYCLES, spp=SPP,
                     allocation_method=_ALLOC_CLIPPED_LINEAR):
    key = ('step7', n_seeds, n_cycles, spp, allocation_method)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist_normalized()
        _RUN_FN_CACHE[key] = build_run_fn_step7(
            mnist_images, mnist_labels, n_cycles, spp, allocation_method)
    return _RUN_FN_CACHE[key]


def aggregate_results_step7(all_M, all_cycle_loss, all_pruned,
                            all_generated, all_active, all_D,
                            n_cycles=N_CYCLES, spp=SPP):
    """Step-7 aggregator. Like step-4's, but with no convergence/budget
    fields, plus per-cycle generation count and per-output-neuron demand
    trajectory.
    """
    all_M = np.asarray(all_M)
    all_cycle_loss = np.asarray(all_cycle_loss)
    all_pruned = np.asarray(all_pruned)
    all_generated = np.asarray(all_generated)
    all_active = np.asarray(all_active)
    all_D = np.asarray(all_D)                  # (S, n_cycles, OUTPUT_DIM)
    S = all_cycle_loss.shape[0]

    eval_window_cycles = min(EVAL_WINDOW_STEPS // spp, n_cycles)
    if eval_window_cycles < 1:
        eval_window_cycles = 1
    final_losses = all_cycle_loss[:, -eval_window_cycles:].mean(axis=1)

    window_log_cycles = max(1, WINDOW_LOG_STEPS // spp)
    n_windows = n_cycles // window_log_cycles
    n_trim = n_windows * window_log_cycles
    trimmed = all_cycle_loss[:, :n_trim]
    windowed = trimmed.reshape(S, n_windows, window_log_cycles).mean(axis=2)
    window_steps = np.arange(1, n_windows + 1) * window_log_cycles * spp

    pruned_windowed = all_pruned[:, :n_trim].reshape(
        S, n_windows, window_log_cycles).sum(axis=2).mean(axis=0)
    generated_windowed = all_generated[:, :n_trim].reshape(
        S, n_windows, window_log_cycles).sum(axis=2).mean(axis=0)
    active_windowed = all_active[:, :n_trim].reshape(
        S, n_windows, window_log_cycles)[:, :, -1].mean(axis=0)

    # Demand trajectory: take the last D snapshot in each window per
    # neuron, then mean across seeds. Shape (n_windows, OUTPUT_DIM).
    D_trim = all_D[:, :n_trim, :]
    demand_windowed = D_trim.reshape(
        S, n_windows, window_log_cycles, OUTPUT_DIM)[:, :, -1, :].mean(axis=0)

    purs, ents = batch_purity_entropy_linear(all_M, INPUT_PER_TASK, N_TASKS)
    aligns = batch_task_alignment_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)
    f1s = batch_task_separation_f1_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)

    return dict(
        final_losses=final_losses, purities=purs, entropies=ents,
        alignments=aligns, separation_f1s=f1s,
        windowed_loss=windowed, window_steps=window_steps,
        pruned_windowed=pruned_windowed,
        generated_windowed=generated_windowed,
        active_windowed=active_windowed,
        demand_windowed=demand_windowed,
        spp=spp,
    )


def run_generation_variant(lr, allocation_method, ci=0.9, n_seeds=N_SEEDS,
                           spp=SPP, n_cycles=N_CYCLES):
    """Run step 7 demand-driven generation across seeds."""
    import scipy.stats
    z_alpha = float(scipy.stats.norm.ppf(ci))
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn_step7(n_seeds, n_cycles, spp, allocation_method)
    out = run_fn(rngs, jnp.float32(lr), jnp.float32(z_alpha))
    all_M, all_cycle_loss, all_pruned, all_generated, all_active, all_D = out
    jax.block_until_ready(out)
    return aggregate_results_step7(
        all_M, all_cycle_loss, all_pruned, all_generated, all_active,
        all_D, n_cycles, spp)


def log_result_metrics_step7(results):
    """Log step 7 metrics: base metrics + generation count + per-output
    demand trajectory."""
    import mlflow
    log_result_metrics(results)

    window_steps = results['window_steps']
    pruned_w = results['pruned_windowed']
    generated_w = results['generated_windowed']
    active_w = results['active_windowed']
    demand_w = results['demand_windowed']                  # (n_windows, 20)
    for t, s in enumerate(window_steps):
        s_int = int(s)
        mlflow.log_metric('n_pruned_window', float(pruned_w[t]), step=s_int)
        mlflow.log_metric('n_generated_window',
                          float(generated_w[t]), step=s_int)
        mlflow.log_metric('n_active', float(active_w[t]), step=s_int)
        for i in range(OUTPUT_DIM):
            mlflow.log_metric(f'demand_unit_{i:02d}',
                              float(demand_w[t, i]), step=s_int)


# ═════════════════════════════════════════════════════════════════════
# Step 7 diagnostics — per-task stat tracking
# ═════════════════════════════════════════════════════════════════════

DIAG_WINDOW_STEPS = 500   # log every 500 steps for diagnostic resolution


def build_run_fn_step7_diag(mnist_images, mnist_labels, n_cycles=N_CYCLES,
                            spp=SPP,
                            allocation_method=_ALLOC_CLIPPED_LINEAR,
                            permute_task=None, permute_period=PERMUTE_PERIOD,
                            task_scales=None, random_task=None):
    """Diagnostic variant of step 7 with per-task stat tracking.

    permute_task: None = no permutation, 0 or 1 = permute only that task.
    permute_period: steps between permutations.
    task_scales: list/tuple of length N_TASKS, or None. Multiplies logits
                 per task before softmax/loss. [2.0, 1.0] doubles task 0.
    random_task: None = normal labels, 0 or 1 = replace that task's label
                 with a uniform-random class each step.
    """
    scales = jnp.ones((N_TASKS,), dtype=jnp.float32)
    if task_scales is not None:
        scales = jnp.array(task_scales, dtype=jnp.float32)
    nc = NUM_CLASSES

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def loss_fn_diag(W, M, x, y):
        logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
        logits = logits * scales[:, None]
        lp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(
            jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))

    def utility_fn_diag(W, M, x, y):
        logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
        logits = logits * scales[:, None]
        sm = jax.nn.softmax(logits, axis=-1)
        onehot = jax.nn.one_hot(y, NUM_CLASSES)
        e = (onehot - sm).reshape(-1)
        c = W * x[None, :]
        u = jnp.abs(e[:, None] + c) - jnp.abs(e[:, None])
        return u * M

    def run_one(rng, lr, z_alpha):
        rng, mkey = jax.random.split(rng)
        M = sample_init_mask_dynamic(mkey, OUTPUT_DIM, INPUT_DIM, BUDGET)
        W = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        U = jnp.zeros((OUTPUT_DIM, INPUT_DIM))
        A = jnp.zeros((OUTPUT_DIM, INPUT_DIM), dtype=jnp.int32)
        R = jnp.zeros((OUTPUT_DIM, INPUT_DIM), dtype=jnp.int32)
        D = jnp.zeros((OUTPUT_DIM,), dtype=jnp.float32)
        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)

        def train_step(carry, inputs):
            W, M, U, A, step, perm0, perm1 = carry
            data_key, perm_key, rand_key = inputs
            x, y_raw = make_sample(data_key)
            y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])

            if random_task is not None:
                y = y.at[random_task].set(
                    jax.random.randint(rand_key, (), 0, NUM_CLASSES))

            loss_val, g = jax.value_and_grad(loss_fn_diag)(W, M, x, y)
            W = W - lr * g * M
            u = utility_fn_diag(W, M, x, y)
            U = EMA_DECAY * U + (1.0 - EMA_DECAY) * u
            A = A + M
            step = step + 1

            if permute_task is not None:
                should_perm = (step >= permute_period) & (
                    step % permute_period == 0)
                new_perm = jax.random.permutation(
                    perm_key, NUM_CLASSES).astype(jnp.int32)
                if permute_task == 0:
                    perm0 = jnp.where(should_perm, new_perm, perm0)
                else:
                    perm1 = jnp.where(should_perm, new_perm, perm1)

            return (W, M, U, A, step, perm0, perm1), loss_val

        def cycle(carry, _):
            (W, M, U, A, R, D, step, perm0, perm1, rng) = carry
            rng, tk, pk, gk, rk = jax.random.split(rng, 5)
            data_keys = jax.random.split(tk, spp)
            perm_keys = jax.random.split(pk, spp)
            rand_keys = jax.random.split(rk, spp)
            (W, M, U, A, step, perm0, perm1), losses = jax.lax.scan(
                train_step, (W, M, U, A, step, perm0, perm1),
                (data_keys, perm_keys, rand_keys))
            cycle_loss = losses.mean()

            D, R, U_corr, tau_w = _resolve_and_update_demand(
                W, M, U, A, R, D, z_alpha)

            M_pre = M
            W, M, U, A, R, n_pruned = _prune_step7(
                W, M, U, A, R, U_corr, tau_w)
            M_post_prune = M

            p = _allocate(D, allocation_method)
            W, M, U, A, R, n_generated = _generate(
                W, M, U, A, R, n_pruned, p, gk)

            # Per-task stats from M snapshots
            pruned_mask = (M_pre == 1) & (M_post_prune == 0)
            gen_mask = (M_post_prune == 0) & (M == 1)
            n_pruned_t0 = jnp.sum(pruned_mask[:nc, :])
            n_pruned_t1 = jnp.sum(pruned_mask[nc:, :])
            n_gen_t0 = jnp.sum(gen_mask[:nc, :])
            n_gen_t1 = jnp.sum(gen_mask[nc:, :])
            n_active_t0 = jnp.sum(M[:nc, :])
            n_active_t1 = jnp.sum(M[nc:, :])

            return (W, M, U, A, R, D, step, perm0, perm1, rng), \
                   (cycle_loss,
                    n_pruned.astype(jnp.int32),
                    n_generated.astype(jnp.int32),
                    jnp.sum(M).astype(jnp.int32),
                    D,
                    n_pruned_t0.astype(jnp.int32),
                    n_pruned_t1.astype(jnp.int32),
                    n_gen_t0.astype(jnp.int32),
                    n_gen_t1.astype(jnp.int32),
                    n_active_t0.astype(jnp.int32),
                    n_active_t1.astype(jnp.int32))

        init_carry = (W, M, U, A, R, D, step, perm0, perm1, rng)
        final_carry, outputs = jax.lax.scan(
            cycle, init_carry, None, length=n_cycles)

        final_M = final_carry[1]
        return (final_M,) + outputs

    @jax.jit
    def run_all(rngs, lr, z_alpha):
        return jax.vmap(lambda r: run_one(r, lr, z_alpha))(rngs)

    return run_all


def get_run_fn_step7_diag(n_seeds, n_cycles=N_CYCLES, spp=SPP,
                          allocation_method=_ALLOC_CLIPPED_LINEAR,
                          permute_task=None, permute_period=PERMUTE_PERIOD,
                          task_scales=None, random_task=None):
    key = ('step7_diag', n_seeds, n_cycles, spp, allocation_method,
           permute_task, permute_period,
           tuple(task_scales) if task_scales else None,
           random_task)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist_normalized()
        _RUN_FN_CACHE[key] = build_run_fn_step7_diag(
            mnist_images, mnist_labels, n_cycles, spp,
            allocation_method, permute_task, permute_period,
            task_scales, random_task)
    return _RUN_FN_CACHE[key]


def aggregate_results_step7_diag(all_M, all_cycle_loss, all_pruned,
                                 all_gen, all_active, all_D,
                                 all_pruned_t0, all_pruned_t1,
                                 all_gen_t0, all_gen_t1,
                                 all_active_t0, all_active_t1,
                                 n_cycles=N_CYCLES, spp=SPP,
                                 window_log_steps=DIAG_WINDOW_STEPS):
    all_M = np.asarray(all_M)
    all_cycle_loss = np.asarray(all_cycle_loss)
    all_D = np.asarray(all_D)
    S = all_cycle_loss.shape[0]

    eval_window_cycles = min(EVAL_WINDOW_STEPS // spp, n_cycles)
    if eval_window_cycles < 1:
        eval_window_cycles = 1
    final_losses = all_cycle_loss[:, -eval_window_cycles:].mean(axis=1)

    wlc = max(1, window_log_steps // spp)
    n_win = n_cycles // wlc
    n_trim = n_win * wlc
    window_steps = np.arange(1, n_win + 1) * wlc * spp

    def _window_sum_mean(arr):
        a = np.asarray(arr)[:, :n_trim]
        return a.reshape(S, n_win, wlc).sum(axis=2).mean(axis=0)

    def _window_last_mean(arr):
        a = np.asarray(arr)[:, :n_trim]
        return a.reshape(S, n_win, wlc)[:, :, -1].mean(axis=0)

    def _window_loss(arr):
        a = np.asarray(arr)[:, :n_trim]
        return a.reshape(S, n_win, wlc).mean(axis=2)

    windowed_loss = _window_loss(all_cycle_loss)

    # Demand per task: mean of D[:nc] and D[nc:] per window
    nc = NUM_CLASSES
    D_trim = all_D[:, :n_trim, :]
    D_win = D_trim.reshape(S, n_win, wlc, OUTPUT_DIM)[:, :, -1, :]
    demand_t0_w = D_win[:, :, :nc].mean(axis=2).mean(axis=0)   # (n_win,)
    demand_t1_w = D_win[:, :, nc:].mean(axis=2).mean(axis=0)
    demand_full_w = D_win.mean(axis=0)                          # (n_win, 20)

    purs, ents = batch_purity_entropy_linear(all_M, INPUT_PER_TASK, N_TASKS)
    aligns = batch_task_alignment_linear(
        all_M, INPUT_PER_TASK, N_TASKS, NUM_CLASSES)

    return dict(
        final_losses=final_losses, purities=purs, entropies=ents,
        alignments=aligns,
        windowed_loss=windowed_loss, window_steps=window_steps,
        pruned_w=_window_sum_mean(all_pruned),
        gen_w=_window_sum_mean(all_gen),
        active_w=_window_last_mean(all_active),
        demand_t0_w=demand_t0_w,
        demand_t1_w=demand_t1_w,
        demand_full_w=demand_full_w,
        pruned_t0_w=_window_sum_mean(all_pruned_t0),
        pruned_t1_w=_window_sum_mean(all_pruned_t1),
        gen_t0_w=_window_sum_mean(all_gen_t0),
        gen_t1_w=_window_sum_mean(all_gen_t1),
        active_t0_w=_window_last_mean(all_active_t0),
        active_t1_w=_window_last_mean(all_active_t1),
        spp=spp,
    )


def run_generation_diag(lr, ci=0.9, n_seeds=N_SEEDS, spp=SPP,
                        n_cycles=N_CYCLES,
                        allocation_method=_ALLOC_CLIPPED_LINEAR,
                        permute_task=None, permute_period=PERMUTE_PERIOD,
                        task_scales=None, random_task=None,
                        window_log_steps=DIAG_WINDOW_STEPS):
    import scipy.stats
    z_alpha = float(scipy.stats.norm.ppf(ci))
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn_step7_diag(
        n_seeds, n_cycles, spp, allocation_method,
        permute_task, permute_period, task_scales, random_task)
    out = run_fn(rngs, jnp.float32(lr), jnp.float32(z_alpha))
    jax.block_until_ready(out)
    (all_M, all_cycle_loss, all_pruned, all_gen, all_active, all_D,
     all_pruned_t0, all_pruned_t1, all_gen_t0, all_gen_t1,
     all_active_t0, all_active_t1) = out
    return aggregate_results_step7_diag(
        all_M, all_cycle_loss, all_pruned, all_gen, all_active, all_D,
        all_pruned_t0, all_pruned_t1, all_gen_t0, all_gen_t1,
        all_active_t0, all_active_t1,
        n_cycles, spp, window_log_steps)


def log_result_metrics_step7_diag(results, t0_label='task_0',
                                  t1_label='task_1'):
    """Log diagnostic step 7 metrics with per-task labels."""
    import mlflow
    for name, arr in [('final_loss', results['final_losses']),
                      ('alignment', results['alignments']),
                      ('purity', results['purities']),
                      ('entropy', results['entropies'])]:
        mlflow.log_metric(name, float(arr.mean()))
        mlflow.log_metric(f'{name}_ci95', ci95(arr))

    ws = results['window_steps']
    for t, s in enumerate(ws):
        si = int(s)
        mlflow.log_metric('loss_window',
                          float(results['windowed_loss'].mean(axis=0)[t]),
                          step=si)
        mlflow.log_metric('n_pruned', float(results['pruned_w'][t]),
                          step=si)
        mlflow.log_metric('n_generated', float(results['gen_w'][t]),
                          step=si)
        mlflow.log_metric('n_active', float(results['active_w'][t]),
                          step=si)
        mlflow.log_metric(f'demand_{t0_label}',
                          float(results['demand_t0_w'][t]), step=si)
        mlflow.log_metric(f'demand_{t1_label}',
                          float(results['demand_t1_w'][t]), step=si)
        mlflow.log_metric(f'pruned_{t0_label}',
                          float(results['pruned_t0_w'][t]), step=si)
        mlflow.log_metric(f'pruned_{t1_label}',
                          float(results['pruned_t1_w'][t]), step=si)
        mlflow.log_metric(f'generated_{t0_label}',
                          float(results['gen_t0_w'][t]), step=si)
        mlflow.log_metric(f'generated_{t1_label}',
                          float(results['gen_t1_w'][t]), step=si)
        mlflow.log_metric(f'active_{t0_label}',
                          float(results['active_t0_w'][t]), step=si)
        mlflow.log_metric(f'active_{t1_label}',
                          float(results['active_t1_w'][t]), step=si)


# ═════════════════════════════════════════════════════════════════════
# Step 10 — Statistical-threshold pruning of input weights in a 2-layer
# LTU network. Only the input → hidden weights are pruned; the hidden →
# output weights are trained but never pruned. Each hidden unit is
# wired to exactly one output (fixed 1-to-1 routing, 5 units per output)
# so every unit inherits a task identity from its output, letting us
# score pruning quality with a per-unit F1 against the unit's initial
# 256-candidate fanin.
# ═════════════════════════════════════════════════════════════════════

from phd.jax_core.models import ltu as _ltu  # noqa: E402

N_HIDDEN = 100
HIDDEN_PER_OUTPUT = N_HIDDEN // OUTPUT_DIM   # 5
FANIN_BUDGET = 256

# Per-utility K constants: BCE-based utilities use K = (1−β)/(1+β);
# signed-based utilities (both finite and ±∞ target variants) use
# K = (1 − 2/π) · (1 − β) / (1 + β).
STAT_PRUNE_K_SIGNED_BINARY = STAT_PRUNE_K_SIGNED
STAT_PRUNE_K_BCE_LTU = STAT_PRUNE_K_BCE
STAT_PRUNE_K_SIGNED_INF = STAT_PRUNE_K_SIGNED


def sample_fanin_mask_random(key, n_hidden: int = N_HIDDEN,
                             input_dim: int = INPUT_DIM,
                             fanin: int = FANIN_BUDGET) -> jnp.ndarray:
    """Per-hidden-unit random fanin mask: each row has exactly `fanin`
    ones sampled uniformly from `input_dim` inputs (no task bias)."""
    keys = jax.random.split(key, n_hidden)

    def one_row(k):
        noise = jax.random.uniform(k, (input_dim,))
        picks = jnp.argsort(noise)[:fanin]
        row = jnp.zeros(input_dim, dtype=jnp.int32).at[picks].set(1)
        return row

    return jax.vmap(one_row)(keys)


def hidden_to_output_map(n_hidden: int = N_HIDDEN,
                         hidden_per_output: int = HIDDEN_PER_OUTPUT
                         ) -> jnp.ndarray:
    """Fixed sequential routing: unit i → output i // hidden_per_output."""
    return jnp.arange(n_hidden) // hidden_per_output


def hidden_unit_task_ids(n_hidden: int = N_HIDDEN,
                         hidden_per_output: int = HIDDEN_PER_OUTPUT,
                         num_classes: int = NUM_CLASSES) -> jnp.ndarray:
    """Task identity of each hidden unit (derived from its output)."""
    return (jnp.arange(n_hidden) // hidden_per_output) // num_classes


def forward_2layer(W_in, M_in, W_out, x):
    """2-layer forward pass.

    Returns (logits, h, z1) where z1 is the hidden pre-activation (used
    by the per-weight utilities) and h is the LTU output (binary 0/1).

    W_in: (N_HIDDEN, INPUT_DIM), M_in: (N_HIDDEN, INPUT_DIM),
    W_out: (N_HIDDEN, OUTPUT_DIM).
    """
    z1 = (W_in * M_in) @ x                             # (N_HIDDEN,)
    h = _ltu(z1)                                       # (N_HIDDEN,) in {0, 1}
    logits = h @ W_out                                 # (OUTPUT_DIM,)
    return logits, h, z1


def loss_fn_2layer(W_in, M_in, W_out, x, y):
    """Softmax CE on the 2-layer forward."""
    logits, _, _ = forward_2layer(W_in, M_in, W_out, x)
    logits_task = logits.reshape(N_TASKS, NUM_CLASSES)
    lp = jax.nn.log_softmax(logits_task, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))


# ─────────────────────────────────────────────────────────────────────
# Hidden-unit flip utility and target derivation
# ─────────────────────────────────────────────────────────────────────

def flip_utility_hidden(h, W_out, logits, y):
    """Per-hidden-unit flip utility under per-task softmax CE.

    For each unit i, compute the softmax-CE NLL delta if we were to flip
    h[i] from its current binary value to 1 - h[i]. Summed across tasks.

    Returns U_flip: (N_HIDDEN,) — positive means flipping hurts (keep),
    negative means flipping helps (this unit's current value is wrong).
    """
    logits_task = logits.reshape(N_TASKS, NUM_CLASSES)                   # (T, C)
    # W_out is (N_HIDDEN, OUTPUT_DIM); per-task view (N_HIDDEN, T, C)
    W_out_task = W_out.reshape(N_HIDDEN, N_TASKS, NUM_CLASSES)
    flip = 1.0 - 2.0 * h                                                 # (N_HIDDEN,)
    # New task-logits per hidden unit: (N_HIDDEN, T, C)
    logits_flip = logits_task[None, :, :] + flip[:, None, None] * W_out_task
    lp_task = jax.nn.log_softmax(logits_task, axis=-1)                   # (T, C)
    lp_flip = jax.nn.log_softmax(logits_flip, axis=-1)                   # (H, T, C)
    task_idx = jnp.arange(N_TASKS)
    nll = -lp_task[task_idx, y]                                          # (T,)
    nll_flip = -lp_flip[:, task_idx, y]                                  # (H, T)
    return (nll_flip - nll[None, :]).sum(axis=-1)                        # (H,)


def compute_ltu_targets(h, U_flip):
    """Derive per-unit binary LTU targets and the `informative` mask.

    target_h = better of {current, flipped} (by loss).
    informative_h = the unit is either currently active or should be active.
    """
    should_flip = U_flip < 0.0
    target_h = jnp.where(should_flip, 1.0 - h, h)
    informative_h = (h > 0.0) | (target_h > 0.0)
    return target_h, informative_h


# ─────────────────────────────────────────────────────────────────────
# Per-weight input-layer utility functions (4 variants)
# ─────────────────────────────────────────────────────────────────────
# All take: W_in (H, D), M_in (H, D), x (D,), target_h (H,), h (H,),
# informative_h (H,). Sign convention: U > 0 ⇒ removing the weight
# hurts the per-unit loss ⇒ keep this weight.

def bce_ltu_utility(W_in, M_in, x, target_h, h, informative_h):
    """Per-weight BCE remove utility on each hidden unit independently.

    `pre_act` is the hidden pre-activation z1. No informative gating.
    """
    z1 = (W_in * M_in) @ x                                         # (H,)
    pre_act_removed = z1[:, None] - x[None, :] * W_in              # (H, D)
    lp = (target_h * jax.nn.log_sigmoid(z1)
          + (1.0 - target_h) * jax.nn.log_sigmoid(-z1))            # (H,)
    lp_removed = (target_h[:, None] * jax.nn.log_sigmoid(pre_act_removed)
                  + (1.0 - target_h)[:, None]
                  * jax.nn.log_sigmoid(-pre_act_removed))          # (H, D)
    u = (-lp_removed) - (-lp[:, None])                             # (H, D)
    return u * M_in


def bce_ltu_utility_informative(W_in, M_in, x, target_h, h, informative_h):
    """Same as bce_ltu_utility but zeroed on uninformative hidden units."""
    u = bce_ltu_utility(W_in, M_in, x, target_h, h, informative_h)
    return u * informative_h[:, None].astype(u.dtype)


def signed_ltu_utility(W_in, M_in, x, target_h, h, informative_h):
    """Signed utility with binary target and continuous (sigmoid) prediction.

    Closest analog to step 3's 1-layer `signed_utility`, replacing the
    softmax-CE error with (binary LTU target − sigmoid(z1)).
    """
    z1 = (W_in * M_in) @ x                                         # (H,)
    e = target_h - jax.nn.sigmoid(z1)                              # (H,)
    c = W_in * x[None, :]                                          # (H, D)
    u = jnp.abs(e[:, None] + c) - jnp.abs(e[:, None])
    return u * M_in


def signed_ltu_utility_inf(W_in, M_in, x, target_h, h, informative_h):
    """Signed utility in the ±∞-target limit: u = (2·target − 1) · w · x.

    target=1 → contribution c is aligned (+c is "going the right way");
    target=0 → +|c| contribution hurts ⇒ utility = −c.
    """
    c = W_in * x[None, :]                                          # (H, D)
    u = (2.0 * target_h - 1.0)[:, None] * c
    return u * M_in


LTU_UTILITY_FNS = {
    'bce_ltu': bce_ltu_utility,
    'bce_ltu_informative': bce_ltu_utility_informative,
    'signed_ltu': signed_ltu_utility,
    'signed_ltu_inf': signed_ltu_utility_inf,
}


LTU_UTILITY_K = {
    'bce_ltu':             STAT_PRUNE_K_BCE_LTU,
    'bce_ltu_informative': STAT_PRUNE_K_BCE_LTU,
    'signed_ltu':          STAT_PRUNE_K_SIGNED_BINARY,
    'signed_ltu_inf':      STAT_PRUNE_K_SIGNED_INF,
    'no_prune':            STAT_PRUNE_K_BCE_LTU,  # unused; any value fine
}


# ─────────────────────────────────────────────────────────────────────
# Per-hidden-unit F1 against the initial fanin candidate pool
# ─────────────────────────────────────────────────────────────────────

def per_unit_fanin_f1(M_initial: np.ndarray, M_final: np.ndarray,
                      input_per_task: int = INPUT_PER_TASK,
                      hidden_per_output: int = HIDDEN_PER_OUTPUT,
                      num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Per-hidden-unit F1 on the initial fanin candidate pool.

    For each hidden unit, treat pruning as a binary classifier over its
    initial-fanin candidates: "keep" = predicted positive, "same-task" =
    ground-truth positive (task of input matches task of the unit's
    output). Returns an (N_HIDDEN,)-shape array; uses NaN for units with
    no same-task candidates (undefined recall).
    """
    M_initial = np.asarray(M_initial)
    M_final = np.asarray(M_final)
    H, D = M_initial.shape
    unit_task = (np.arange(H) // hidden_per_output) // num_classes       # (H,)
    input_task = np.arange(D) // input_per_task                          # (D,)
    same_task = unit_task[:, None] == input_task[None, :]                # (H, D)
    init = M_initial > 0
    final = M_final > 0
    tp = (final & init & same_task).sum(axis=1)
    fp = (final & init & ~same_task).sum(axis=1)
    fn = (~final & init & same_task).sum(axis=1)
    f1 = np.full(H, np.nan, dtype=np.float64)
    denom = 2 * tp + fp + fn
    nz = denom > 0
    f1[nz] = 2.0 * tp[nz] / denom[nz]
    return f1


def batch_per_unit_fanin_f1(all_M_initial, all_M_final,
                            input_per_task=INPUT_PER_TASK,
                            hidden_per_output=HIDDEN_PER_OUTPUT,
                            num_classes=NUM_CLASSES) -> np.ndarray:
    """Mean (across hidden units) per-unit fanin F1, per seed. Returns
    (S,)-shape."""
    all_M_initial = np.asarray(all_M_initial)
    all_M_final = np.asarray(all_M_final)
    S = all_M_initial.shape[0]
    out = np.zeros(S, dtype=np.float64)
    for s in range(S):
        f1 = per_unit_fanin_f1(all_M_initial[s], all_M_final[s],
                               input_per_task, hidden_per_output, num_classes)
        out[s] = np.nanmean(f1)
    return out


def batch_per_unit_task_counts(all_M_final,
                               input_per_task=INPUT_PER_TASK,
                               hidden_per_output=HIDDEN_PER_OUTPUT,
                               num_classes=NUM_CLASSES):
    """Per-seed average (across hidden units) count of kept same-task and
    cross-task input connections. Returns (avg_same, avg_cross), each
    (S,)-shape."""
    all_M_final = np.asarray(all_M_final)
    S, H, D = all_M_final.shape
    unit_task = (np.arange(H) // hidden_per_output) // num_classes       # (H,)
    input_task = np.arange(D) // input_per_task                          # (D,)
    same_task = unit_task[:, None] == input_task[None, :]                # (H, D)
    kept = all_M_final > 0                                               # (S, H, D)
    same = (kept & same_task[None, :, :]).sum(axis=2)                    # (S, H)
    cross = (kept & ~same_task[None, :, :]).sum(axis=2)                  # (S, H)
    return same.mean(axis=1).astype(np.float64), cross.mean(axis=1).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────
# Step 10 build + run
# ─────────────────────────────────────────────────────────────────────

def build_run_fn_step10(mnist_images, mnist_labels, max_cycles=N_CYCLES,
                        spp=SPP, utility_fn='signed_ltu_inf',
                        prune_enabled=True):
    """Build JIT+vmap run function for step 10 2-layer pruning.

    Architecture: input (INPUT_DIM=1568) → N_HIDDEN=100 LTU → softmax CE.
    Each hidden unit starts with FANIN_BUDGET=256 random input candidates.
    Statistical-threshold pruning operates on W_in only; W_out trains
    freely and is never pruned.

    `utility_fn` selects the per-weight utility driving the threshold.
    If `prune_enabled=False`, the whole pruning machinery is skipped
    (no_prune baseline).

    Expects `mnist_images` to be pre-normalized (σ_x ≈ 1) so the
    threshold formula can assume σ_x = 1.
    """
    assert utility_fn in LTU_UTILITY_FNS
    utility_impl = LTU_UTILITY_FNS[utility_fn]
    K = LTU_UTILITY_K[utility_fn]

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr, z_alpha):
        rng, mkey, wkey_in, wkey_out = jax.random.split(rng, 4)
        M_in_init = sample_fanin_mask_random(mkey)                     # (H, D)
        M_in = M_in_init
        # Kaiming-like init for W_in over the 256-fanin pool.
        w_in_bound = jnp.sqrt(3.0 / float(FANIN_BUDGET))
        W_in = jax.random.uniform(wkey_in, (N_HIDDEN, INPUT_DIM),
                                  minval=-w_in_bound, maxval=w_in_bound)
        W_in = W_in * M_in                                             # zero outside fanin
        # W_out: scalar per hidden unit only into its one target output;
        # stored as (N_HIDDEN, OUTPUT_DIM) with a fixed routing mask so
        # gradients only flow to the diagonal-like entries.
        h2o = hidden_to_output_map()
        W_out_mask = jax.nn.one_hot(h2o, OUTPUT_DIM)                   # (H, O)
        w_out_bound = jnp.sqrt(3.0 / float(HIDDEN_PER_OUTPUT))
        W_out = jax.random.uniform(wkey_out, (N_HIDDEN, OUTPUT_DIM),
                                   minval=-w_out_bound,
                                   maxval=w_out_bound) * W_out_mask

        U = jnp.zeros((N_HIDDEN, INPUT_DIM))
        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        consecutive_zeros = jnp.array(0, dtype=jnp.int32)
        converged = jnp.array(not prune_enabled)
        converge_cycle = jnp.array(0 if not prune_enabled else max_cycles,
                                   dtype=jnp.int32)

        def train_step(carry, inputs):
            W_in, M_in, W_out, U, step, perm0, perm1 = carry
            data_key, perm_key = inputs
            x, y_raw = make_sample(data_key)
            y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])

            def loss_both(W_in, W_out):
                return loss_fn_2layer(W_in, M_in, W_out, x, y)

            loss_val, (gW_in, gW_out) = jax.value_and_grad(
                loss_both, argnums=(0, 1))(W_in, W_out)
            # Accuracy from the same forward.
            logits, h, _ = forward_2layer(W_in, M_in, W_out, x)
            acc_val = jnp.mean(
                (jnp.argmax(logits.reshape(N_TASKS, NUM_CLASSES), axis=-1) == y)
                .astype(jnp.float32))

            W_in = W_in - lr * gW_in * M_in
            # Only update W_out entries along the fixed 1-to-1 routing.
            W_out = W_out - lr * gW_out * W_out_mask

            # Per-weight utility. Need h and target_h derived from flip
            # utility on the output layer.
            U_flip = flip_utility_hidden(h, W_out, logits, y)
            target_h, informative_h = compute_ltu_targets(h, U_flip)
            u = utility_impl(W_in, M_in, x, target_h, h,
                             informative_h.astype(jnp.float32))
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
            return (W_in, M_in, W_out, U, step, perm0, perm1), (loss_val,
                                                                 acc_val)

        def prune_cycle(carry, cycle_idx):
            (W_in, M_in, W_out, U, step, perm0, perm1, rng,
             consecutive_zeros, converged, converge_cycle) = carry
            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, spp)
            perm_keys = jax.random.split(pk, spp)
            (W_in, M_in, W_out, U, step, perm0,
             perm1), (losses, accs) = jax.lax.scan(
                train_step,
                (W_in, M_in, W_out, U, step, perm0, perm1),
                (data_keys, perm_keys))
            cycle_loss = losses.mean()
            cycle_acc = accs.mean()

            # Prune (only when enabled and not already converged).
            if prune_enabled:
                W_in_p, M_in_p, U_p, n_pruned = prune_statistical(
                    W_in, M_in, U, step, z_alpha, K)
                W_in = jnp.where(converged, W_in, W_in_p)
                M_in = jnp.where(converged, M_in, M_in_p)
                U = jnp.where(converged, U, U_p)
                n_pruned = jnp.where(converged, 0, n_pruned)
            else:
                n_pruned = jnp.int32(0)

            new_consec = jnp.where(
                n_pruned == 0,
                consecutive_zeros + 1,
                jnp.array(0, dtype=jnp.int32))
            new_consec = jnp.where(converged, consecutive_zeros, new_consec)
            newly_converged = (~converged) & (new_consec >= 3)
            converged = converged | newly_converged
            converge_cycle = jnp.where(
                newly_converged, cycle_idx, converge_cycle)

            n_active = jnp.sum(M_in)
            return (W_in, M_in, W_out, U, step, perm0, perm1, rng,
                    new_consec, converged, converge_cycle), \
                   (cycle_loss, cycle_acc,
                    n_pruned.astype(jnp.int32),
                    n_active.astype(jnp.int32))

        init_carry = (W_in, M_in, W_out, U, step, perm0, perm1, rng,
                      consecutive_zeros, converged, converge_cycle)
        final_carry, (per_cycle_loss, per_cycle_acc, per_cycle_pruned,
                      per_cycle_active) = jax.lax.scan(
            prune_cycle, init_carry,
            jnp.arange(max_cycles, dtype=jnp.int32))

        final_M_in = final_carry[1]
        final_converge_cycle = final_carry[10]
        return (M_in_init, final_M_in, per_cycle_loss, per_cycle_acc,
                per_cycle_pruned, per_cycle_active, final_converge_cycle)

    @jax.jit
    def run_all(rngs, lr, z_alpha):
        return jax.vmap(lambda r: run_one(r, lr, z_alpha))(rngs)

    return run_all


def get_run_fn_step10(n_seeds, max_cycles=N_CYCLES, spp=SPP,
                      utility_fn='signed_ltu_inf', prune_enabled=True):
    key = ('step10', n_seeds, max_cycles, spp, utility_fn, prune_enabled)
    if key not in _RUN_FN_CACHE:
        mnist_images, mnist_labels = load_mnist_normalized()
        _RUN_FN_CACHE[key] = build_run_fn_step10(
            mnist_images, mnist_labels, max_cycles, spp, utility_fn,
            prune_enabled)
    return _RUN_FN_CACHE[key]


def aggregate_results_step10(M_in_init, M_in_final, all_cycle_loss,
                             all_cycle_acc, all_pruned, all_active,
                             converge_cycles, max_cycles=N_CYCLES, spp=SPP):
    """Wraps aggregate_results_step4's budget/convergence/loss/acc aggregation
    and adds a per-hidden-unit fanin-F1 metric unique to step 10."""
    all_M_final = np.asarray(M_in_final)
    # `aggregate_results_step4` expects an `all_M` for its existing linear
    # purity/entropy/alignment metrics. Those don't make sense in the
    # 2-layer fanin setup, but passing the final mask keeps the function
    # signature happy; we simply ignore those fields downstream.
    out = aggregate_results_step4(
        all_M_final, all_cycle_loss, all_pruned, all_active,
        converge_cycles, max_cycles, spp, all_cycle_acc=all_cycle_acc)
    out['fanin_f1s'] = batch_per_unit_fanin_f1(M_in_init, M_in_final)
    same, cross = batch_per_unit_task_counts(M_in_final)
    out['kept_same_task'] = same
    out['kept_cross_task'] = cross
    return out


def run_2layer_variant(lr, z_alpha, utility_fn='signed_ltu_inf',
                       prune_enabled=True, n_seeds=N_SEEDS, spp=SPP,
                       total_steps=TOTAL_STEPS):
    """Run step 10 2-layer pruning across seeds.

    If `utility_fn == 'no_prune'`, prune_enabled is forced to False and
    a dummy utility is used (ignored at runtime).
    """
    if utility_fn == 'no_prune':
        prune_enabled = False
        utility_fn_for_build = 'signed_ltu_inf'  # unused
    else:
        utility_fn_for_build = utility_fn
    max_cycles = total_steps // spp
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn_step10(n_seeds, max_cycles, spp,
                               utility_fn_for_build, prune_enabled)
    out = run_fn(rngs, jnp.float32(lr), jnp.float32(z_alpha))
    (all_M_init, all_M_final, all_cycle_loss, all_cycle_acc,
     all_pruned, all_active, converge_cycles) = out
    jax.block_until_ready(out)
    return aggregate_results_step10(
        all_M_init, all_M_final, all_cycle_loss, all_cycle_acc,
        all_pruned, all_active, converge_cycles, max_cycles, spp)


def log_result_metrics_step10(results):
    """Like log_result_metrics_step4 plus the fanin-F1 metric and
    per-hidden-unit average kept-connection counts split by task."""
    import mlflow
    log_result_metrics_step4(results)
    for name, arr in [('fanin_f1', results['fanin_f1s']),
                      ('kept_same_task', results['kept_same_task']),
                      ('kept_cross_task', results['kept_cross_task'])]:
        mlflow.log_metric(name, float(np.mean(arr)))
        mlflow.log_metric(f'{name}_ci95', ci95(arr))
