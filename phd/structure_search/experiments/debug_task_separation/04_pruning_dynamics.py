"""Debug task separation: pruning dynamics at the 100-input breakpoint.

Uses the 100-input/task synthetic problem (step 5 from 03_scaling_progression)
where separation first breaks down, and tests whether adjusting the pruning
dynamics can recover it.

Three axes:
  A. More training time between prunes (clearer utility signal?)
  B. More total prune events (more sculpting time?)
  C. Larger prune step size / batch pruning (faster sculpting per event?)

Plus:
  D. Matched 10x turnover via different paths
  E. Smaller connection budget (easier to sculpt?)

All experiments: 20 seeds, vmapped, contribution utility, full generation,
SGD, stationary, fully connected (100in/task, 10class/task).
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

N_TASKS = 2
N_SEEDS = 20
BASE_SEED = 42

# Fixed problem: 100 inputs/task, 10 classes/task
IN_PER_TASK = 100
OUT_PER_TASK = 10
INPUT_DIM = IN_PER_TASK * N_TASKS   # 200
OUTPUT_DIM = OUT_PER_TASK * N_TASKS  # 20
N_HIDDEN_MAX = 30
CONNS_PER_UNIT = INPUT_DIM + OUTPUT_DIM  # 220 for full gen
MAX_GEN_PER_CYCLE = 5
LR = 1e-3

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

# ══════════════════════════════════════════════════════════════════════
# PRUNING (supports batch: prune N lowest at once)
# ══════════════════════════════════════════════════════════════════════

def prune_n(Ui, Uo, Mi, Mo, n, rng):
    """Prune the n lowest-utility active connections."""
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

# ══════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════

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
# DATA
# ══════════════════════════════════════════════════════════════════════

# Fixed target: 10 random hyperplanes per task (shared across all experiments)
_tgt_key = jax.random.key(999)
HYPER = [jax.random.normal(jax.random.split(_tgt_key)[i], (OUT_PER_TASK, IN_PER_TASK)) * 0.1
         for i in range(N_TASKS)]

def sample_fn(key):
    k1, k2 = jax.random.split(key)
    xs = [jax.random.normal(k1, (IN_PER_TASK,)),
          jax.random.normal(k2, (IN_PER_TASK,))]
    labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(HYPER, xs)])
    return jnp.concatenate(xs), labels

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT BUILDER
# ══════════════════════════════════════════════════════════════════════

def make_run_fn(spp, nprunes, ppe, budget):
    """Build vmapped experiment: spp steps/prune, nprunes cycles, ppe pruned/event."""

    def run_one(rng):
        Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
        Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
        Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)

        # Init 1 unit
        rng, gk = jax.random.split(rng)
        Wi, Wo, Mi, Mo = gen_full(Wi, Wo, Mi, Mo, gk)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, rng = carry

            # ── Train ──
            rng, tk = jax.random.split(rng)
            tkeys = jax.random.split(tk, spp)

            def train_body(carry, key):
                Wi, Wo, Ui, Uo = carry
                x, y = sample_fn(key)
                gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
                Wi = Wi - LR * gi * Mi
                Wo = Wo - LR * go * Mo
                ui, uo = contribution_utility(Wi, Wo, Mi, Mo, x)
                return (Wi, Wo, Ui + ui, Uo + uo), None

            zi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
            zo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
            (Wi, Wo, Ui, Uo), _ = jax.lax.scan(
                train_body, (Wi, Wo, zi, zo), tkeys)
            Ui, Uo = Ui / spp, Uo / spp

            # ── Prune ──
            rng, pk = jax.random.split(rng)
            Mi, Mo = prune_n(Ui, Uo, Mi, Mo, ppe, pk)
            Mi = cleanup_dead(Mi, Mo)

            # ── Generate to fill budget ──
            def gen_one(carry, _):
                Wi, Wo, Mi, Mo, rng = carry
                nc = Mi.sum() + Mo.sum()
                need = nc < budget
                rng, gk = jax.random.split(rng)
                Wg, Wog, Mg, Mog = gen_full(Wi, Wo, Mi, Mo, gk)
                Wi = jnp.where(need, Wg, Wi)
                Wo = jnp.where(need, Wog, Wo)
                Mi = jnp.where(need, Mg, Mi)
                Mo = jnp.where(need, Mog, Mo)
                return (Wi, Wo, Mi, Mo, rng), None

            (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
                gen_one, (Wi, Wo, Mi, Mo, rng), None, length=MAX_GEN_PER_CYCLE)

            return (Wi, Wo, Mi, Mo, rng), None

        (Wi, Wo, Mi, Mo, _), _ = jax.lax.scan(
            prune_cycle, (Wi, Wo, Mi, Mo, rng), None, length=nprunes)
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
# RUNNER
# ══════════════════════════════════════════════════════════════════════

DEFAULT_BUDGET = 5500

def run_experiment(name, spp, nprunes, ppe, budget=DEFAULT_BUDGET):
    total_pruned = nprunes * ppe
    turnover = total_pruned / budget
    total_steps = nprunes * spp

    print(f'\n{"=" * 70}')
    print(f'{name}')
    print(f'  steps/prune={spp}  prune_events={nprunes}  pruned/event={ppe}')
    print(f'  total_pruned={total_pruned}  turnover={turnover:.1f}x  '
          f'total_steps={total_steps/1e6:.1f}M  budget={budget}')

    run_fn = make_run_fn(spp, nprunes, ppe, budget)
    rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

    t0 = time.time()
    print(f'  Compiling + running {N_SEEDS} seeds ...', end=' ', flush=True)
    all_Mi, all_Mo = run_fn(rngs)
    jax.block_until_ready((all_Mi, all_Mo))
    elapsed = time.time() - t0
    print(f'{elapsed:.1f}s')

    purs, ents = compute_purities(all_Mi, all_Mo)
    mp, sp = purs.mean(), 1.96 * purs.std() / np.sqrt(N_SEEDS)
    me, se = ents.mean(), 1.96 * ents.std() / np.sqrt(N_SEEDS)
    print(f'  >>> Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}')
    return mp, sp, me, se, turnover, total_steps

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # === Baseline (from 03_scaling step 5) ===
    ('Baseline',
     dict(spp=5000, nprunes=200, ppe=1)),

    # === A: More training time between prunes ===
    ('A1: 5x train/prune',
     dict(spp=25000, nprunes=200, ppe=1)),
    ('A2: 25x train/prune',
     dict(spp=125000, nprunes=200, ppe=1)),

    # === B: More total prune events (longer runs) ===
    ('B1: 10x events (10M steps)',
     dict(spp=5000, nprunes=2000, ppe=1)),
    ('B2: 10x events, fast (1M)',
     dict(spp=500, nprunes=2000, ppe=1)),

    # === C: Batch pruning (prune N connections per event) ===
    ('C1: 10/event',
     dict(spp=5000, nprunes=200, ppe=10)),
    ('C2: 50/event',
     dict(spp=5000, nprunes=200, ppe=50)),
    ('C3: 275/event (10x turn)',
     dict(spp=5000, nprunes=200, ppe=275)),

    # === D: 10x turnover via different paths ===
    ('D1: 55/ev x 1000ev (5M)',
     dict(spp=5000, nprunes=1000, ppe=55)),
    ('D2: 10/ev x 5500ev fast',
     dict(spp=1000, nprunes=5500, ppe=10)),

    # === E: Smaller budget (tighter network) ===
    ('E1: budget=1000',
     dict(spp=5000, nprunes=200, ppe=1, budget=1000)),
    ('E2: budget=1000, 10/ev',
     dict(spp=5000, nprunes=200, ppe=10, budget=1000)),
    ('E3: budget=1000, 10x turn',
     dict(spp=5000, nprunes=200, ppe=50, budget=1000)),
]

def main():
    results = {}
    for name, kw in EXPERIMENTS:
        mp, sp, me, se, turn, tsteps = run_experiment(name, **kw)
        results[name] = (mp, sp, me, se, turn, tsteps)

    print(f'\n{"=" * 85}')
    print(f'SUMMARY ({N_SEEDS} seeds, 100 inputs/task, 10 classes/task)')
    print(f'{"=" * 85}')
    print(f'{"Experiment":<30} {"Turn":>5} {"Steps":>6} {"Purity":>14} {"Entropy":>14}')
    print('-' * 85)
    for name, (mp, sp, me, se, turn, tsteps) in results.items():
        flag = ' ***' if mp > 0.8 else (' <--' if mp < 0.6 else '')
        print(f'{name:<30} {turn:>4.1f}x {tsteps/1e6:>5.1f}M '
              f'{mp:.3f}+/-{sp:.3f}   {me:.3f}+/-{se:.3f}{flag}')

if __name__ == '__main__':
    main()
