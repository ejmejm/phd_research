"""Batch size ablation: fixed turnover, varying batch size.

All 4 experiments prune the same total connections (~10,000 = 1.8x turnover)
over 1M steps, but with different batch sizes. Uses EMA utility with
median-reset after each prune event.

  1/event  x 10000 events x 100 steps/prune  = 10000 pruned, 1M steps
  10/event x 1000 events  x 1000 steps/prune = 10000 pruned, 1M steps
  50/event x 200 events   x 5000 steps/prune = 10000 pruned, 1M steps
  200/event x 50 events   x 20000 steps/prune = 10000 pruned, 1M steps

20 seeds, vmapped. 100 inputs/task, 10 classes/task, budget=5500.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

N_TASKS = 2
N_SEEDS = 20
BASE_SEED = 42

IN_PER_TASK = 100
OUT_PER_TASK = 10
INPUT_DIM = IN_PER_TASK * N_TASKS
OUTPUT_DIM = OUT_PER_TASK * N_TASKS
N_HIDDEN_MAX = 30
BUDGET = 5500
MAX_GEN_PER_CYCLE = 5
LR = 1e-3
EMA_DECAY = 0.999

# Fixed target
_tgt_key = jax.random.key(999)
HYPER = [jax.random.normal(jax.random.split(_tgt_key)[i], (OUT_PER_TASK, IN_PER_TASK)) * 0.1
         for i in range(N_TASKS)]

# ══════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════

def forward(Wi, Wo, Mi, Mo, x):
    h = jax.nn.leaky_relu((Wi * Mi) @ x)
    return (Wo * Mo) @ h, h

def contribution_utility(Wi, Wo, Mi, Mo, x):
    _, h = forward(Wi, Wo, Mi, Mo, x)
    return (jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi,
            jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo)

def loss_ce(Wi, Wo, Mi, Mo, x, y):
    raw, _ = forward(Wi, Wo, Mi, Mo, x)
    logits = raw.reshape(N_TASKS, OUT_PER_TASK)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, OUT_PER_TASK) * lp, axis=-1))

def sample_fn(key):
    k1, k2 = jax.random.split(key)
    xs = [jax.random.normal(k1, (IN_PER_TASK,)),
          jax.random.normal(k2, (IN_PER_TASK,))]
    labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(HYPER, xs)])
    return jnp.concatenate(xs), labels

# ══════════════════════════════════════════════════════════════════════
# PRUNING & GENERATION
# ══════════════════════════════════════════════════════════════════════

def prune_n(Ui, Uo, Mi, Mo, n, rng):
    ui = jnp.where(Mi.reshape(-1) == 1, Ui.reshape(-1), jnp.inf)
    uo = jnp.where(Mo.reshape(-1) == 1, Uo.reshape(-1), jnp.inf)
    all_u = jnp.concatenate([ui, uo])
    noise = jax.random.uniform(rng, all_u.shape, minval=-1e-10, maxval=1e-10)
    perturbed = jnp.where(jnp.isinf(all_u), all_u, all_u + noise)
    thresh = jnp.sort(perturbed)[n]
    thresh = jnp.minimum(thresh, 1e30)
    active = jnp.concatenate([(Mi.reshape(-1) == 1), (Mo.reshape(-1) == 1)])
    prune = (perturbed < thresh) & active
    ni = Mi.size
    return (Mi * (1 - prune[:ni].reshape(Mi.shape).astype(jnp.int32)),
            Mo * (1 - prune[ni:].reshape(Mo.shape).astype(jnp.int32)))

def cleanup_dead(Mi, Mo):
    return Mi * (Mo.sum(axis=0) > 0)[:, None]

def gen_full(Wi, Wo, Mi, Mo, key):
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has = jnp.any(~active)
    bound = jnp.sqrt(3.0 / INPUT_DIM)
    w = jax.random.uniform(key, (INPUT_DIM,), minval=-bound, maxval=bound)
    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(1), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(1), Mo)
    return Wi, Wo, Mi, Mo

# ══════════════════════════════════════════════════════════════════════
# EMA UTILITY WITH MEDIAN RESET
# ══════════════════════════════════════════════════════════════════════

def median_of_active(U, M):
    """Median utility among active connections."""
    flat_u = jnp.where(M.reshape(-1) == 1, U.reshape(-1), jnp.nan)
    # Sort; nans go to end
    sorted_u = jnp.sort(jnp.where(jnp.isnan(flat_u), jnp.inf, flat_u))
    n_active = (M.reshape(-1) == 1).sum()
    mid = n_active // 2
    return sorted_u[mid]

def reset_utility_to_median(Ui, Uo, Mi, Mo):
    """Reset all active utilities to the current median."""
    # Compute median across ALL active connections (input + output)
    all_u = jnp.concatenate([
        jnp.where(Mi.reshape(-1) == 1, Ui.reshape(-1), jnp.inf),
        jnp.where(Mo.reshape(-1) == 1, Uo.reshape(-1), jnp.inf),
    ])
    sorted_u = jnp.sort(all_u)
    n_active = (Mi == 1).sum() + (Mo == 1).sum()
    med = sorted_u[n_active // 2]
    Ui = jnp.where(Mi == 1, med, 0.0)
    Uo = jnp.where(Mo == 1, med, 0.0)
    return Ui, Uo

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT BUILDER
# ══════════════════════════════════════════════════════════════════════

def make_run_fn(spp, nprunes, ppe):
    def run_one(rng):
        Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
        Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
        Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)
        Ui = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Uo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))

        # Init 1 unit
        rng, gk = jax.random.split(rng)
        Wi, Wo, Mi, Mo = gen_full(Wi, Wo, Mi, Mo, gk)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, Ui, Uo, rng = carry

            # ── Train with EMA utility ──
            rng, tk = jax.random.split(rng)
            tkeys = jax.random.split(tk, spp)

            def train_body(carry, key):
                Wi, Wo, Ui, Uo = carry
                x, y = sample_fn(key)
                gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
                Wi = Wi - LR * gi * Mi
                Wo = Wo - LR * go * Mo
                ui, uo = contribution_utility(Wi, Wo, Mi, Mo, x)
                Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
                Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo
                return (Wi, Wo, Ui, Uo), None

            (Wi, Wo, Ui, Uo), _ = jax.lax.scan(
                train_body, (Wi, Wo, Ui, Uo), tkeys)

            # ── Prune ──
            rng, pk = jax.random.split(rng)
            Mi, Mo = prune_n(Ui, Uo, Mi, Mo, ppe, pk)
            Mi = cleanup_dead(Mi, Mo)

            # ── Reset utility to median ──
            Ui, Uo = reset_utility_to_median(Ui, Uo, Mi, Mo)

            # ── Generate to fill budget ──
            def gen_one(carry, _):
                Wi, Wo, Mi, Mo, rng = carry
                nc = Mi.sum() + Mo.sum()
                need = nc < BUDGET
                rng, gk = jax.random.split(rng)
                Wg, Wog, Mg, Mog = gen_full(Wi, Wo, Mi, Mo, gk)
                Wi = jnp.where(need, Wg, Wi)
                Wo = jnp.where(need, Wog, Wo)
                Mi = jnp.where(need, Mg, Mi)
                Mo = jnp.where(need, Mog, Mo)
                return (Wi, Wo, Mi, Mo, rng), None

            (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
                gen_one, (Wi, Wo, Mi, Mo, rng), None, length=MAX_GEN_PER_CYCLE)

            return (Wi, Wo, Mi, Mo, Ui, Uo, rng), None

        (Wi, Wo, Mi, Mo, Ui, Uo, _), _ = jax.lax.scan(
            prune_cycle, (Wi, Wo, Mi, Mo, Ui, Uo, rng), None, length=nprunes)
        return Mi, Mo

    @jax.jit
    def run_all(rngs):
        return jax.vmap(run_one)(rngs)
    return run_all

# ══════════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════════

def compute_purities(all_Mi, all_Mo):
    all_Mi, all_Mo = np.array(all_Mi), np.array(all_Mo)
    ns = all_Mi.shape[0]
    purs, ents = np.zeros(ns), np.zeros(ns)
    for s in range(ns):
        Mi, Mo = all_Mi[s], all_Mo[s]
        ups, ues = [], []
        for k in range(N_HIDDEN_MAX):
            if not (Mi[k].any() or Mo[:, k].any()):
                continue
            ic = [int(Mi[k, t * IN_PER_TASK:(t + 1) * IN_PER_TASK].sum())
                  for t in range(N_TASKS)]
            ti = sum(ic)
            if ti == 0:
                continue
            ups.append(max(ic) / ti)
            ues.append(sum(-c / ti * np.log2(c / ti + 1e-15) for c in ic if c > 0))
        purs[s] = np.mean(ups) if ups else 0.0
        ents[s] = np.mean(ues) if ues else 1.0
    return purs, ents

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENTS: fixed turnover ~1.8x, varying batch size
# ══════════════════════════════════════════════════════════════════════

# Total pruned = 10000 for all (10000/5500 = 1.8x turnover)
# Total steps = 1M for all
EXPERIMENTS = [
    ('1/event x 10000 events (100 spp)',
     dict(spp=100,   nprunes=10000, ppe=1)),
    ('10/event x 1000 events (1000 spp)',
     dict(spp=1000,  nprunes=1000,  ppe=10)),
    ('50/event x 200 events (5000 spp)',
     dict(spp=5000,  nprunes=200,   ppe=50)),
    ('200/event x 50 events (20000 spp)',
     dict(spp=20000, nprunes=50,    ppe=200)),
]

def main():
    results = {}
    for name, kw in EXPERIMENTS:
        spp, nprunes, ppe = kw['spp'], kw['nprunes'], kw['ppe']
        total_pruned = nprunes * ppe
        turnover = total_pruned / BUDGET

        print(f'\n{"=" * 65}')
        print(f'{name}')
        print(f'  spp={spp} events={nprunes} ppe={ppe} '
              f'total_pruned={total_pruned} turnover={turnover:.1f}x')

        run_fn = make_run_fn(spp, nprunes, ppe)
        rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

        t0 = time.time()
        print(f'  Compiling + running {N_SEEDS} seeds ...', end=' ', flush=True)
        all_Mi, all_Mo = run_fn(rngs)
        jax.block_until_ready((all_Mi, all_Mo))
        print(f'{time.time() - t0:.1f}s')

        purs, ents = compute_purities(all_Mi, all_Mo)
        mp, sp = purs.mean(), 1.96 * purs.std() / np.sqrt(N_SEEDS)
        me, se = ents.mean(), 1.96 * ents.std() / np.sqrt(N_SEEDS)
        print(f'  Per-seed purity: {np.array2string(purs, precision=2)}')
        print(f'  >>> Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}')
        results[name] = (mp, sp, me, se)

    print(f'\n{"=" * 70}')
    print(f'SUMMARY: Fixed 1.8x turnover, 1M steps, EMA + median reset')
    print(f'{"=" * 70}')
    print(f'{"Config":<45} {"Purity":>14} {"Entropy":>14}')
    print('-' * 70)
    for name, (mp, sp, me, se) in results.items():
        print(f'{name:<45} {mp:.3f}+/-{sp:.3f}   {me:.3f}+/-{se:.3f}')

if __name__ == '__main__':
    main()
