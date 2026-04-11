"""Check whether Adam's separation is real (task-pure full units) or
just sparsity (units collapsed to a few connections that happen to be
from one task).

Runs Adam and SGD baseline on MNIST, reports per-unit connection counts
alongside purity. 20 seeds.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
import time

N_TASKS = 2
N_SEEDS = 20
BASE_SEED = 42
EMA_DECAY = 0.999
MAX_GEN_PER_CYCLE = 5
NUM_CLASSES = 10
INPUT_PER_TASK = 784
INPUT_DIM = INPUT_PER_TASK * N_TASKS
OUTPUT_DIM = NUM_CLASSES * N_TASKS
N_HIDDEN_MAX = 80
MAX_CONNS = 64
BUDGET = 5000
SPP = 8000
LR = 1e-3
N_EVAL = 200

def load_mnist():
    from data import load_dataset
    images, labels, _, _ = load_dataset('mnist', split='train')
    return jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)

def forward(Wi, Wo, Mi, Mo, x):
    h = jax.nn.leaky_relu((Wi * Mi) @ x)
    return (Wo * Mo) @ h, h

def loss_ce(Wi, Wo, Mi, Mo, x, y):
    raw, _ = forward(Wi, Wo, Mi, Mo, x)
    logits = raw.reshape(N_TASKS, NUM_CLASSES)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))

def compute_utility(Wi, Wo, Mi, Mo, x):
    _, h = forward(Wi, Wo, Mi, Mo, x)
    return (jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi,
            jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo)

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

def reset_utility_to_median(Ui, Uo, Mi, Mo):
    all_u = jnp.concatenate([
        jnp.where(Mi.reshape(-1) == 1, Ui.reshape(-1), jnp.inf),
        jnp.where(Mo.reshape(-1) == 1, Uo.reshape(-1), jnp.inf),
    ])
    n_active = (Mi == 1).sum() + (Mo == 1).sum()
    med = jnp.sort(all_u)[n_active // 2]
    return jnp.where(Mi == 1, med, 0.0), jnp.where(Mo == 1, med, 0.0)

def gen_unit(Wi, Wo, Mi, Mo, key):
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has = jnp.any(~active)
    k1, k2 = jax.random.split(key)
    nc = min(MAX_CONNS, INPUT_DIM)
    noise = jax.random.uniform(k1, (INPUT_DIM,))
    perm = jnp.argsort(noise)
    im = jnp.zeros(INPUT_DIM, dtype=jnp.int32).at[perm[:nc]].set(1)
    bound = jnp.sqrt(3.0 / nc)
    w = jax.random.uniform(k2, (INPUT_DIM,), minval=-bound, maxval=bound) * im
    om = jnp.ones(OUTPUT_DIM, dtype=jnp.int32)
    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(im), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(om), Mo)
    return Wi, Wo, Mi, Mo

# ══════════════════════════════════════════════════════════════════════

def make_run_fn(mnist_images, mnist_labels, use_adam=False):
    ppe = max(1, round(0.005 * BUDGET))
    n_prunes = max(1, round(3.0 * BUDGET) // ppe)

    if use_adam:
        opt = optax.adam(LR)

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    def run_one(rng):
        Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
        Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
        Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)
        Ui = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Uo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))

        rng, gk = jax.random.split(rng)
        Wi, Wo, Mi, Mo = gen_unit(Wi, Wo, Mi, Mo, gk)

        if use_adam:
            opt_state = opt.init((Wi, Wo))
        else:
            opt_state = jnp.array(0.0)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng = carry
            rng, tk = jax.random.split(rng)
            tkeys = jax.random.split(tk, SPP)

            def train_body(carry, key):
                Wi, Wo, Ui, Uo, opt_state = carry
                x, y = make_sample(key)
                gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
                if use_adam:
                    updates, opt_state = opt.update((gi, go), opt_state, (Wi, Wo))
                    Wi = Wi + updates[0] * Mi
                    Wo = Wo + updates[1] * Mo
                else:
                    Wi = Wi - LR * gi * Mi
                    Wo = Wo - LR * go * Mo
                ui, uo = compute_utility(Wi, Wo, Mi, Mo, x)
                Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
                Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo
                return (Wi, Wo, Ui, Uo, opt_state), None

            (Wi, Wo, Ui, Uo, opt_state), _ = jax.lax.scan(
                train_body, (Wi, Wo, Ui, Uo, opt_state), tkeys)

            rng, pk = jax.random.split(rng)
            Mi, Mo = prune_n(Ui, Uo, Mi, Mo, ppe, pk)
            Mi = cleanup_dead(Mi, Mo)
            Ui, Uo = reset_utility_to_median(Ui, Uo, Mi, Mo)

            def gen_one(carry, _):
                Wi, Wo, Mi, Mo, rng = carry
                nc = Mi.sum() + Mo.sum()
                need = nc < BUDGET
                rng, gk = jax.random.split(rng)
                Wg, Wog, Mg, Mog = gen_unit(Wi, Wo, Mi, Mo, gk)
                Wi = jnp.where(need, Wg, Wi)
                Wo = jnp.where(need, Wog, Wo)
                Mi = jnp.where(need, Mg, Mi)
                Mo = jnp.where(need, Mog, Mo)
                return (Wi, Wo, Mi, Mo, rng), None
            (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
                gen_one, (Wi, Wo, Mi, Mo, rng), None, length=MAX_GEN_PER_CYCLE)

            return (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng), None

        (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng), _ = jax.lax.scan(
            prune_cycle, (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng),
            None, length=n_prunes)

        return Mi, Mo

    @jax.jit
    def run_all(rngs):
        return jax.vmap(run_one)(rngs)
    return run_all

# ══════════════════════════════════════════════════════════════════════
# DETAILED ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze(all_Mi, all_Mo, label):
    all_Mi, all_Mo = np.array(all_Mi), np.array(all_Mo)
    ns = all_Mi.shape[0]

    all_in_counts = []     # total input conns per unit
    all_out_counts = []    # total output conns per unit
    all_task0_in = []      # task 0 input conns per unit
    all_task1_in = []      # task 1 input conns per unit
    all_purities = []
    seed_purities = []
    seed_mean_in = []

    for s in range(ns):
        Mi, Mo = all_Mi[s], all_Mo[s]
        s_purs = []
        s_ins = []
        for k in range(N_HIDDEN_MAX):
            if not (Mi[k].any() or Mo[:, k].any()):
                continue
            t0 = int(Mi[k, :INPUT_PER_TASK].sum())
            t1 = int(Mi[k, INPUT_PER_TASK:].sum())
            ti = t0 + t1
            to = int(Mo[:, k].sum())
            if ti == 0:
                continue
            pur = max(t0, t1) / ti
            all_in_counts.append(ti)
            all_out_counts.append(to)
            all_task0_in.append(t0)
            all_task1_in.append(t1)
            all_purities.append(pur)
            s_purs.append(pur)
            s_ins.append(ti)
        seed_purities.append(np.mean(s_purs) if s_purs else 0.0)
        seed_mean_in.append(np.mean(s_ins) if s_ins else 0.0)

    all_in = np.array(all_in_counts)
    all_out = np.array(all_out_counts)
    all_pur = np.array(all_purities)
    sp = np.array(seed_purities)
    smi = np.array(seed_mean_in)

    print(f'\n{"=" * 65}')
    print(f'{label}')
    print(f'  Purity:      {sp.mean():.3f} +/- {1.96*sp.std()/np.sqrt(ns):.3f}')
    print(f'  Mean in/unit: {smi.mean():.1f} +/- {1.96*smi.std()/np.sqrt(ns):.1f}')
    print(f'  Active units: {len(all_in) / ns:.0f} per seed')
    print(f'  Total conns:  {(all_in.sum() + all_out.sum()) / ns:.0f} per seed')

    print(f'\n  Input connections per unit:')
    print(f'    mean={all_in.mean():.1f}  median={np.median(all_in):.0f}'
          f'  min={all_in.min()}  max={all_in.max()}  std={all_in.std():.1f}')

    print(f'  Output connections per unit:')
    print(f'    mean={all_out.mean():.1f}  median={np.median(all_out):.0f}'
          f'  min={all_out.min()}  max={all_out.max()}  std={all_out.std():.1f}')

    # Purity vs connection count: bin by input count
    print(f'\n  Purity by input connection count:')
    bins = [(1, 5), (6, 15), (16, 30), (31, 50), (51, 64)]
    for lo, hi in bins:
        mask = (all_in >= lo) & (all_in <= hi)
        if mask.sum() == 0:
            continue
        bp = all_pur[mask]
        print(f'    {lo:>2}-{hi:<2} conns: n={mask.sum():>5}  '
              f'purity={bp.mean():.3f} +/- {bp.std():.3f}')

    # Distribution of input counts
    print(f'\n  Input count distribution:')
    for threshold in [10, 20, 30, 40, 50, 60, 64]:
        frac = (all_in <= threshold).mean()
        print(f'    <= {threshold:>2}: {frac:.1%}')

if __name__ == '__main__':
    print("Loading MNIST...")
    mnist_images, mnist_labels = load_mnist()

    rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

    print("\nRunning SGD baseline...")
    t0 = time.time()
    run_sgd = make_run_fn(mnist_images, mnist_labels, use_adam=False)
    Mi_sgd, Mo_sgd = run_sgd(rngs)
    jax.block_until_ready((Mi_sgd, Mo_sgd))
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nRunning Adam...")
    t0 = time.time()
    run_adam = make_run_fn(mnist_images, mnist_labels, use_adam=True)
    Mi_adam, Mo_adam = run_adam(rngs)
    jax.block_until_ready((Mi_adam, Mo_adam))
    print(f"  Done in {time.time()-t0:.1f}s")

    analyze(Mi_sgd, Mo_sgd, "SGD Baseline")
    analyze(Mi_adam, Mo_adam, "Adam")
