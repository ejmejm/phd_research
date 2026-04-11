"""Quick test: very small budgets at 100in/task partial 64 conns.

Same setup as 06 baseline: 3x turnover, 0.5% batch, 1M steps, spp~1666.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

N_TASKS = 2
N_SEEDS = 20
BASE_SEED = 42
EMA_DECAY = 0.999
MAX_GEN_PER_CYCLE = 5
LR = 1e-3

IN_PER_TASK = 100
OUT_PER_TASK = 10
INPUT_DIM = IN_PER_TASK * N_TASKS
OUTPUT_DIM = OUT_PER_TASK * N_TASKS

_tgt_key = jax.random.key(999)
HYPER = [jax.random.normal(jax.random.split(_tgt_key)[i], (OUT_PER_TASK, IN_PER_TASK)) * 0.1
         for i in range(N_TASKS)]

def sample_fn(key):
    k1, k2 = jax.random.split(key)
    xs = [jax.random.normal(k1, (IN_PER_TASK,)), jax.random.normal(k2, (IN_PER_TASK,))]
    labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(HYPER, xs)])
    return jnp.concatenate(xs), labels

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

def gen_partial(Wi, Wo, Mi, Mo, key, *, nhid, idim, odim, maxc):
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(nhid), nhid))
    has = jnp.any(~active)
    k1, k2 = jax.random.split(key)
    nc = min(maxc, idim)
    noise = jax.random.uniform(k1, (idim,))
    perm = jnp.argsort(noise)
    im = jnp.zeros(idim, dtype=jnp.int32).at[perm[:nc]].set(1)
    bound = jnp.sqrt(3.0 / nc)
    w = jax.random.uniform(k2, (idim,), minval=-bound, maxval=bound) * im
    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(im), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(1), Mo)
    return Wi, Wo, Mi, Mo

def make_run_fn(nhid, budget, spp, n_prunes, ppe):
    gfn = partial(gen_partial, nhid=nhid, idim=INPUT_DIM, odim=OUTPUT_DIM, maxc=64)

    def run_one(rng):
        Wi = jnp.zeros((nhid, INPUT_DIM))
        Wo = jnp.zeros((OUTPUT_DIM, nhid))
        Mi = jnp.zeros((nhid, INPUT_DIM), dtype=jnp.int32)
        Mo = jnp.zeros((OUTPUT_DIM, nhid), dtype=jnp.int32)
        Ui = jnp.zeros((nhid, INPUT_DIM))
        Uo = jnp.zeros((OUTPUT_DIM, nhid))

        rng, gk = jax.random.split(rng)
        Wi, Wo, Mi, Mo = gfn(Wi, Wo, Mi, Mo, gk)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, Ui, Uo, rng = carry
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

            rng, pk = jax.random.split(rng)
            Mi, Mo = prune_n(Ui, Uo, Mi, Mo, ppe, pk)
            Mi = cleanup_dead(Mi, Mo)
            Ui, Uo = reset_utility_to_median(Ui, Uo, Mi, Mo)

            def gen_one(carry, _):
                Wi, Wo, Mi, Mo, rng = carry
                nc = Mi.sum() + Mo.sum()
                need = nc < budget
                rng, gk = jax.random.split(rng)
                Wg, Wog, Mg, Mog = gfn(Wi, Wo, Mi, Mo, gk)
                Wi = jnp.where(need, Wg, Wi)
                Wo = jnp.where(need, Wog, Wo)
                Mi = jnp.where(need, Mg, Mi)
                Mo = jnp.where(need, Mog, Mo)
                return (Wi, Wo, Mi, Mo, rng), None
            (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
                gen_one, (Wi, Wo, Mi, Mo, rng), None, length=MAX_GEN_PER_CYCLE)

            return (Wi, Wo, Mi, Mo, Ui, Uo, rng), None

        (Wi, Wo, Mi, Mo, Ui, Uo, _), _ = jax.lax.scan(
            prune_cycle, (Wi, Wo, Mi, Mo, Ui, Uo, rng), None, length=n_prunes)
        return Mi, Mo

    @jax.jit
    def run_all(rngs):
        return jax.vmap(run_one)(rngs)
    return run_all

def compute_purities(all_Mi, all_Mo, nhid):
    all_Mi, all_Mo = np.array(all_Mi), np.array(all_Mo)
    ns = all_Mi.shape[0]
    purs, ents = np.zeros(ns), np.zeros(ns)
    for s in range(ns):
        Mi, Mo = all_Mi[s], all_Mo[s]
        ups, ues = [], []
        for k in range(nhid):
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

def run_one(name, budget, nhid):
    ppe = max(1, round(0.005 * budget))
    total_pruned = round(3.0 * budget)
    n_prunes = max(1, total_pruned // ppe)
    spp = max(100, 1_000_000 // n_prunes)
    n_prunes = 1_000_000 // spp
    actual_turn = (n_prunes * ppe) / budget
    conns_per_unit = 64 + OUTPUT_DIM
    approx_units = budget // conns_per_unit

    print(f'\n{"=" * 60}')
    print(f'{name}')
    print(f'  budget={budget} nhid={nhid} ~units={approx_units}')
    print(f'  ppe={ppe} events={n_prunes} spp={spp} turn={actual_turn:.1f}x')

    run_fn = make_run_fn(nhid, budget, spp, n_prunes, ppe)
    rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

    t0 = time.time()
    print(f'  Running ...', end=' ', flush=True)
    all_Mi, all_Mo = run_fn(rngs)
    jax.block_until_ready((all_Mi, all_Mo))
    print(f'{time.time() - t0:.1f}s')

    purs, ents = compute_purities(all_Mi, all_Mo, nhid)
    mp, sp = purs.mean(), 1.96 * purs.std() / np.sqrt(N_SEEDS)
    me, se = ents.mean(), 1.96 * ents.std() / np.sqrt(N_SEEDS)
    print(f'  >>> Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}')
    return mp, sp, me, se

if __name__ == '__main__':
    configs = [
        ('budget=84 (~1 unit)',   84,   5),
        ('budget=100',           100,   5),
        ('budget=168 (~2 units)', 168,  5),
        ('budget=200',           200,   5),
        ('budget=400',           400,  10),
        ('budget=1000',         1000,  20),
    ]
    results = {}
    for name, budget, nhid in configs:
        results[name] = run_one(name, budget, nhid)

    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    for name, (mp, sp, me, se) in results.items():
        print(f'  {name:<25} Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}')
