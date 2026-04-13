"""Step 1 — linear model, global pruning on non-stationary multi-MNIST.

Runs three configurations (dynamic prune+generate, random-fixed baseline,
within-task-fixed baseline), each with its own LR sweep. Logs aggregated
metrics to MLflow under project ``local_pruning_progression``.

Usage:
    python 01_linear_global.py                 # full LR sweep + finals
    python 01_linear_global.py --dry-run       # small config, sanity check
    python 01_linear_global.py --variant dynamic --lr 1e-3 --final-only
"""

import argparse
import os
import sys
import time

# Make sibling modules and phd/structure_search/data.py importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', '..'))  # repo root

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from phd.jax_core.utils import configure_jax

# Configure JAX's persistent XLA compile cache before any jit happens.
configure_jax(OmegaConf.create(dict(
    jax_jit_cache_dir='/tmp/jax_cache',
    device='gpu',
)))

from common import (
    batch_purity_entropy_linear,
    ci95,
    mlflow_log_aggregate,
    mlflow_log_trajectory,
    mlflow_start,
    sample_fixed_mask_intask,
    sample_fixed_mask_random,
    sample_init_mask_dynamic,
)

# ─── Configuration ────────────────────────────────────────────────────

N_TASKS = 2
NUM_CLASSES = 10
INPUT_PER_TASK = 784
INPUT_DIM = INPUT_PER_TASK * N_TASKS       # 1568
OUTPUT_DIM = NUM_CLASSES * N_TASKS         # 20
BUDGET = 1500
EMA_DECAY = 0.998
PERMUTE_PERIOD = 4000
SPP = 50                                    # steps per prune event
PPE = 1                                     # connections pruned per event
N_CYCLES_DEFAULT = 4500                     # 4500 * 50 = 225k steps (3x turnover)
EVAL_WINDOW_CYCLES = 800                    # last 40k steps
WINDOW_LOG_CYCLES = 100                     # 5k step granularity
N_SEEDS_SWEEP = 5
N_SEEDS_FINAL = 20
BASE_SEED = 42

DEFAULT_LR_GRID = [1.5625e-4, 6.25e-4, 2.5e-3, 1e-2, 4e-2]


# ─── Data ─────────────────────────────────────────────────────────────

def load_mnist():
    from data import load_dataset
    images, labels, _, _ = load_dataset('mnist', split='train')
    return jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)


# ─── Model ────────────────────────────────────────────────────────────

def forward(W, M, x):
    return (W * M) @ x


def loss_fn(W, M, x, y):
    logits = forward(W, M, x).reshape(N_TASKS, NUM_CLASSES)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))


# ─── Prune + generate ─────────────────────────────────────────────────

def prune_and_generate_one(W, M, U, rng):
    """Prune the lowest-utility active connection (with tie-break noise),
    then add one new connection in a uniformly-random empty slot. New
    connection's utility = mean of currently-active utilities; weight = 0."""
    prune_key, gen_key, noise_key = jax.random.split(rng, 3)

    # --- Prune the min-utility active slot ---
    noise = jax.random.uniform(noise_key, M.shape,
                               minval=-1e-10, maxval=1e-10)
    scores = jnp.where(M == 1, U + noise, jnp.inf)
    prune_idx = jnp.argmin(scores.reshape(-1))

    flat_M = M.reshape(-1).at[prune_idx].set(0)
    flat_W = W.reshape(-1).at[prune_idx].set(0.0)
    flat_U = U.reshape(-1).at[prune_idx].set(0.0)
    M = flat_M.reshape(M.shape)
    W = flat_W.reshape(W.shape)
    U = flat_U.reshape(U.shape)

    # --- Generate: pick random empty slot, initialize utility to mean ---
    gen_noise = jax.random.uniform(gen_key, M.shape)
    gen_scores = jnp.where(M == 0, gen_noise, 2.0)
    gen_idx = jnp.argmin(gen_scores.reshape(-1))

    n_active = jnp.sum(M).astype(jnp.float32)
    mean_u = jnp.where(n_active > 0, jnp.sum(U) / n_active, 0.0)

    flat_M = M.reshape(-1).at[gen_idx].set(1)
    flat_W = W.reshape(-1).at[gen_idx].set(0.0)
    flat_U = U.reshape(-1).at[gen_idx].set(mean_u)
    return (flat_W.reshape(W.shape), flat_M.reshape(M.shape),
            flat_U.reshape(U.shape))


# ─── Run one seed ─────────────────────────────────────────────────────

def build_run_fn(mnist_images, mnist_labels, variant: str, n_cycles: int):
    """Return a JIT+vmap-ed run function:
        run_all(rngs, lr) -> (final_M, per_cycle_loss)
    `variant` ∈ {'dynamic', 'fixed_random', 'fixed_intask'}.
    """
    assert variant in ('dynamic', 'fixed_random', 'fixed_intask')
    is_dynamic = (variant == 'dynamic')

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng, lr):
        # --- Init mask ---
        rng, mkey = jax.random.split(rng)
        if variant == 'dynamic':
            M = sample_init_mask_dynamic(mkey, OUTPUT_DIM, INPUT_DIM, BUDGET)
        elif variant == 'fixed_random':
            M = sample_fixed_mask_random(mkey, OUTPUT_DIM, INPUT_DIM, BUDGET)
        else:  # fixed_intask
            M = sample_fixed_mask_intask(mkey, OUTPUT_DIM, INPUT_DIM,
                                         BUDGET, N_TASKS)

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

            loss, g = jax.value_and_grad(loss_fn)(W, M, x, y)
            W = W - lr * g * M

            u = jnp.abs(x)[None, :] * jnp.abs(W) * M
            U = EMA_DECAY * U + (1.0 - EMA_DECAY) * u

            step = step + 1

            # Permutation event at step % PERMUTE_PERIOD == 0 (step >= PERMUTE_PERIOD)
            should_perm = (step >= PERMUTE_PERIOD) & (step % PERMUTE_PERIOD == 0)
            pk1, pk2 = jax.random.split(perm_key)
            which = jax.random.randint(pk1, (), 0, N_TASKS)
            new_perm = jax.random.permutation(pk2, NUM_CLASSES).astype(jnp.int32)
            perm0 = jnp.where(should_perm & (which == 0), new_perm, perm0)
            perm1 = jnp.where(should_perm & (which == 1), new_perm, perm1)

            return (W, M, U, step, perm0, perm1), loss

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


# ─── Aggregation ──────────────────────────────────────────────────────

def aggregate_results(all_M, all_cycle_loss, n_cycles: int):
    """Compute final_loss (per seed), window trajectory (per seed), purity,
    entropy. Returns a dict of numpy arrays."""
    all_M = np.asarray(all_M)
    all_cycle_loss = np.asarray(all_cycle_loss)  # (S, n_cycles)
    S = all_cycle_loss.shape[0]

    # Final-40k loss: mean of last EVAL_WINDOW_CYCLES per seed.
    final_losses = all_cycle_loss[:, -EVAL_WINDOW_CYCLES:].mean(axis=1)

    # Window trajectory: mean over WINDOW_LOG_CYCLES blocks.
    n_windows = n_cycles // WINDOW_LOG_CYCLES
    trimmed = all_cycle_loss[:, :n_windows * WINDOW_LOG_CYCLES]
    windowed = trimmed.reshape(S, n_windows, WINDOW_LOG_CYCLES).mean(axis=2)
    window_steps = np.arange(1, n_windows + 1) * WINDOW_LOG_CYCLES * SPP

    purs, ents = batch_purity_entropy_linear(all_M, INPUT_PER_TASK, N_TASKS)

    return dict(
        final_losses=final_losses,
        purities=purs,
        entropies=ents,
        windowed_loss=windowed,
        window_steps=window_steps,
    )


# ─── Runner ──────────────────────────────────────────────────────────

# Cache compiled run fns across LR sweeps. Key = (variant, n_seeds, n_cycles).
# The persistent XLA cache (configure_jax) handles cross-process caching; this
# dict avoids rebuilding the traced+vmapped Python wrapper each call.
_RUN_FN_CACHE = {}


def get_run_fn(variant: str, n_seeds: int, n_cycles: int,
               mnist_images, mnist_labels):
    key = (variant, n_seeds, n_cycles)
    if key not in _RUN_FN_CACHE:
        _RUN_FN_CACHE[key] = build_run_fn(
            mnist_images, mnist_labels, variant, n_cycles)
    return _RUN_FN_CACHE[key]


def run_variant(variant: str, lr: float, n_seeds: int, n_cycles: int,
                mnist_images, mnist_labels):
    """Run one (variant, lr) configuration across n_seeds and return
    aggregated results."""
    rngs = jax.random.split(jax.random.key(BASE_SEED), n_seeds)
    run_fn = get_run_fn(variant, n_seeds, n_cycles, mnist_images, mnist_labels)

    t0 = time.time()
    all_M, all_cycle_loss = run_fn(rngs, jnp.float32(lr))
    jax.block_until_ready((all_M, all_cycle_loss))
    elapsed = time.time() - t0

    results = aggregate_results(all_M, all_cycle_loss, n_cycles)
    results['elapsed'] = elapsed
    return results


def summarize(results, label: str):
    fl = results['final_losses']
    pu = results['purities']
    en = results['entropies']
    print(f'  {label:<50} loss={fl.mean():.4f}+/-{ci95(fl):.4f}  '
          f'pur={pu.mean():.3f}+/-{ci95(pu):.3f}  '
          f'ent={en.mean():.3f}+/-{ci95(en):.3f}  '
          f'({results["elapsed"]:.0f}s)')


def log_run(run_name: str, variant: str, lr: float, n_seeds: int,
            n_cycles: int, results, log_trajectory: bool = True):
    import mlflow
    mlflow_module = mlflow_start(run_name, dict(
        step=1,
        variant=variant,
        budget=BUDGET,
        ema_decay=EMA_DECAY,
        spp=SPP,
        ppe=PPE,
        n_cycles=n_cycles,
        total_steps=n_cycles * SPP,
        permute_period=PERMUTE_PERIOD,
        n_seeds=n_seeds,
        lr=lr,
        eval_window_steps=EVAL_WINDOW_CYCLES * SPP,
    ))
    try:
        mlflow_log_aggregate(mlflow_module, 'final_loss',
                             results['final_losses'])
        mlflow_log_aggregate(mlflow_module, 'purity', results['purities'])
        mlflow_log_aggregate(mlflow_module, 'entropy', results['entropies'])
        if log_trajectory:
            mlflow_log_trajectory(mlflow_module, 'loss_window_5k',
                                  results['windowed_loss'],
                                  results['window_steps'])
    finally:
        mlflow_module.end_run()


def lr_sweep(variant: str, lr_grid, n_cycles, mnist_images, mnist_labels,
             log_sweep_runs: bool = True):
    """Sweep LRs at N_SEEDS_SWEEP seeds. Returns (best_lr, sweep_dict).
    Does NOT handle edge-of-grid extension — caller should widen the grid
    if the minimum sits at an endpoint."""
    print(f'\n=== LR sweep: variant={variant} (n_seeds={N_SEEDS_SWEEP}) ===')
    results_per_lr = {}
    for lr in lr_grid:
        r = run_variant(variant, lr, N_SEEDS_SWEEP, n_cycles,
                        mnist_images, mnist_labels)
        results_per_lr[lr] = r
        summarize(r, f'lr={lr:.3e}')
        if log_sweep_runs:
            log_run(f'step1_{variant}_lr={lr:.3e}', variant, lr,
                    N_SEEDS_SWEEP, n_cycles, r, log_trajectory=False)

    mean_losses = {lr: r['final_losses'].mean() for lr, r in results_per_lr.items()}
    best_lr = min(mean_losses, key=mean_losses.get)
    sorted_lrs = sorted(mean_losses.keys())
    idx = sorted_lrs.index(best_lr)
    at_edge = (idx == 0 or idx == len(sorted_lrs) - 1)
    print(f'  Best LR: {best_lr:.3e} (loss={mean_losses[best_lr]:.4f})')
    if at_edge:
        print(f'  !! Best LR at edge of grid — extend grid and rerun !!')
    return best_lr, results_per_lr, at_edge


# ─── Main ────────────────────────────────────────────────────────────

def run_full(n_cycles: int, lr_grid, mnist_images, mnist_labels):
    """Full protocol for step 1: LR sweep + 20-seed final for each of
    the three variants."""
    print(f'\n{"#" * 66}\n# STEP 1 — linear, global pruning\n{"#" * 66}')
    print(f'  n_cycles={n_cycles}  total_steps={n_cycles * SPP}  '
          f'spp={SPP}  budget={BUDGET}')
    print(f'  lr_grid={lr_grid}')

    variants = ['dynamic', 'fixed_random', 'fixed_intask']
    best_lrs = {}
    for v in variants:
        best_lr, _, at_edge = lr_sweep(v, lr_grid, n_cycles,
                                       mnist_images, mnist_labels)
        best_lrs[v] = best_lr
        if at_edge:
            print(f'  WARNING: {v} best LR at edge — widen grid and rerun.')

    print(f'\n=== Final 20-seed runs at best LR ===')
    for v in variants:
        lr = best_lrs[v]
        r = run_variant(v, lr, N_SEEDS_FINAL, n_cycles,
                        mnist_images, mnist_labels)
        summarize(r, f'step1_{v} (lr={lr:.3e})')
        log_run(f'step1_{v}', v, lr, N_SEEDS_FINAL, n_cycles, r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Tiny config to verify plumbing')
    p.add_argument('--variant', default=None,
                   choices=['dynamic', 'fixed_random', 'fixed_intask'])
    p.add_argument('--lr', type=float, default=None,
                   help='Single LR (skip sweep)')
    p.add_argument('--final-only', action='store_true',
                   help='Run only the 20-seed final, not the sweep '
                        '(requires --variant and --lr)')
    p.add_argument('--lr-grid', type=str, default=None,
                   help='Comma-separated LR values (overrides default)')
    p.add_argument('--n-cycles', type=int, default=None)
    args = p.parse_args()

    print('Loading MNIST...')
    mnist_images, mnist_labels = load_mnist()
    print(f'  {mnist_images.shape[0]} images')

    if args.dry_run:
        n_cycles = 100               # 5k steps
        lr_grid = [1e-3, 4e-3]
        n_seeds = 2
        print('\n=== DRY RUN ===')
        for v in ['dynamic', 'fixed_random', 'fixed_intask']:
            for lr in lr_grid:
                r = run_variant(v, lr, n_seeds, n_cycles,
                                mnist_images, mnist_labels)
                summarize(r, f'{v} lr={lr:.3e}')
        print('dry run ok')
        return

    n_cycles = args.n_cycles if args.n_cycles else N_CYCLES_DEFAULT
    if args.lr_grid:
        lr_grid = [float(x) for x in args.lr_grid.split(',')]
    else:
        lr_grid = DEFAULT_LR_GRID

    if args.final_only:
        assert args.variant and args.lr, '--final-only requires --variant and --lr'
        r = run_variant(args.variant, args.lr, N_SEEDS_FINAL, n_cycles,
                        mnist_images, mnist_labels)
        summarize(r, f'step1_{args.variant} (lr={args.lr:.3e})')
        log_run(f'step1_{args.variant}', args.variant, args.lr,
                N_SEEDS_FINAL, n_cycles, r)
        return

    if args.variant and args.lr:
        r = run_variant(args.variant, args.lr, N_SEEDS_SWEEP, n_cycles,
                        mnist_images, mnist_labels)
        summarize(r, f'{args.variant} lr={args.lr:.3e}')
        return

    run_full(n_cycles, lr_grid, mnist_images, mnist_labels)


if __name__ == '__main__':
    main()
