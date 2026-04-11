"""Scaling progression with proper pruning.

Steps 1-5 (full connectivity): 3x turnover, 0.5% batch, ~1M steps.
Steps 6-8 (partial connectivity): 3x turnover, 0.5% batch, spp=8000
  (more training between prunes, per findings from 06).

Uses EMA utility (decay=0.999) with median-reset after each prune event.
20 seeds, vmapped. Reports purity, entropy, and loss with 95% CI.
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
MAX_GEN_PER_CYCLE = 3

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

# ══════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════

def gen_full(Wi, Wo, Mi, Mo, key, *, nhid, idim, odim):
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(nhid), nhid))
    has = jnp.any(~active)
    bound = jnp.sqrt(3.0 / idim)
    w = jax.random.uniform(key, (idim,), minval=-bound, maxval=bound)
    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(1), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(1), Mo)
    return Wi, Wo, Mi, Mo

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

# ══════════════════════════════════════════════════════════════════════
# PRUNING PARAMETERS
# ══════════════════════════════════════════════════════════════════════

def compute_prune_params(budget, turnover=3.0, batch_frac=0.005,
                         spp=None, total_steps=None):
    """Compute prune params. Specify either spp or total_steps."""
    ppe = max(1, round(batch_frac * budget))
    total_pruned = round(turnover * budget)
    n_prunes = max(1, total_pruned // ppe)
    if spp is not None:
        # Fixed spp, total_steps derived
        pass
    else:
        assert total_steps is not None
        spp = max(100, total_steps // n_prunes)
        n_prunes = total_steps // spp
    actual_turn = (n_prunes * ppe) / budget
    actual_steps = n_prunes * spp
    return ppe, n_prunes, spp, actual_turn, actual_steps

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT BUILDER
# ══════════════════════════════════════════════════════════════════════

N_EVAL = 200  # samples for loss evaluation

def make_run_fn(sample_fn, loss_type, gfn, nhid, ninit, idim, odim,
                budget, spp, n_prunes, ppe, lr, opc):
    if loss_type == 'mse':
        def lossfn(Wi, Wo, Mi, Mo, x, y):
            yh, _ = forward(Wi, Wo, Mi, Mo, x)
            return jnp.mean((yh - y) ** 2)
    else:
        def lossfn(Wi, Wo, Mi, Mo, x, y):
            raw, _ = forward(Wi, Wo, Mi, Mo, x)
            logits = raw.reshape(N_TASKS, opc)
            lp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.sum(jax.nn.one_hot(y, opc) * lp, axis=-1))

    def run_one(rng):
        Wi = jnp.zeros((nhid, idim))
        Wo = jnp.zeros((odim, nhid))
        Mi = jnp.zeros((nhid, idim), dtype=jnp.int32)
        Mo = jnp.zeros((odim, nhid), dtype=jnp.int32)
        Ui = jnp.zeros((nhid, idim))
        Uo = jnp.zeros((odim, nhid))

        def init_body(carry, _):
            Wi, Wo, Mi, Mo, rng = carry
            rng, gk = jax.random.split(rng)
            Wi, Wo, Mi, Mo = gfn(Wi, Wo, Mi, Mo, gk)
            return (Wi, Wo, Mi, Mo, rng), None
        (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
            init_body, (Wi, Wo, Mi, Mo, rng), None, length=ninit)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, Ui, Uo, rng = carry
            rng, tk = jax.random.split(rng)
            tkeys = jax.random.split(tk, spp)

            def train_body(carry, key):
                Wi, Wo, Ui, Uo = carry
                x, y = sample_fn(key)
                gi, go = jax.grad(lossfn, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
                Wi = Wi - lr * gi * Mi
                Wo = Wo - lr * go * Mo
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

        (Wi, Wo, Mi, Mo, Ui, Uo, rng), _ = jax.lax.scan(
            prune_cycle, (Wi, Wo, Mi, Mo, Ui, Uo, rng), None, length=n_prunes)

        # Eval loss
        rng, ek = jax.random.split(rng)
        eval_keys = jax.random.split(ek, N_EVAL)
        def one_loss(k):
            x, y = sample_fn(k)
            return lossfn(Wi, Wo, Mi, Mo, x, y)
        final_loss = jax.vmap(one_loss)(eval_keys).mean()

        return Mi, Mo, final_loss

    @jax.jit
    def run_all(rngs):
        return jax.vmap(run_one)(rngs)
    return run_all

# ══════════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════════

def compute_purities(all_Mi, all_Mo, ipt, nhid):
    all_Mi, all_Mo = np.array(all_Mi), np.array(all_Mo)
    ns = all_Mi.shape[0]
    purs, ents = np.zeros(ns), np.zeros(ns)
    for s in range(ns):
        Mi, Mo = all_Mi[s], all_Mo[s]
        ups, ues = [], []
        for k in range(nhid):
            if not (Mi[k].any() or Mo[:, k].any()):
                continue
            ic = [int(Mi[k, t * ipt:(t + 1) * ipt].sum()) for t in range(N_TASKS)]
            ti = sum(ic)
            if ti == 0:
                continue
            ups.append(max(ic) / ti)
            ues.append(sum(-c / ti * np.log2(c / ti + 1e-15) for c in ic if c > 0))
        purs[s] = np.mean(ups) if ups else 0.0
        ents[s] = np.mean(ues) if ues else 1.0
    return purs, ents

def ci(arr):
    return 1.96 * arr.std() / np.sqrt(len(arr))

# ══════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(name, *, in_per_task, out_per_task, n_hidden_max,
                   n_hidden_init, connection_budget, sample_fn, loss_type,
                   max_conns=None, lr=3e-3, spp=None, total_steps=1_000_000):
    idim = in_per_task * N_TASKS
    odim = out_per_task * N_TASKS
    use_partial = max_conns is not None and max_conns < idim

    if use_partial:
        gfn = partial(gen_partial, nhid=n_hidden_max, idim=idim, odim=odim, maxc=max_conns)
    else:
        gfn = partial(gen_full, nhid=n_hidden_max, idim=idim, odim=odim)

    ppe, n_prunes, spp_actual, actual_turn, actual_steps = compute_prune_params(
        connection_budget, spp=spp,
        total_steps=total_steps if spp is None else None)

    print(f'\n{"=" * 70}')
    print(f'{name}')
    print(f'  in={idim} out={odim} nhid={n_hidden_max} budget={connection_budget} '
          f'maxc={max_conns or "all"}')
    print(f'  ppe={ppe} events={n_prunes} spp={spp_actual} '
          f'turn={actual_turn:.1f}x steps={actual_steps / 1e6:.1f}M')

    run_fn = make_run_fn(sample_fn, loss_type, gfn, n_hidden_max, n_hidden_init,
                         idim, odim, connection_budget, spp_actual, n_prunes, ppe,
                         lr, out_per_task)

    rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

    t0 = time.time()
    print(f'  Compiling + running {N_SEEDS} seeds ...', end=' ', flush=True)
    all_Mi, all_Mo, all_loss = run_fn(rngs)
    jax.block_until_ready((all_Mi, all_Mo, all_loss))
    print(f'{time.time() - t0:.1f}s')

    purs, ents = compute_purities(all_Mi, all_Mo, in_per_task, n_hidden_max)
    losses = np.array(all_loss)
    mp, me, ml = purs.mean(), ents.mean(), losses.mean()
    sp, se, sl = ci(purs), ci(ents), ci(losses)

    print(f'  >>> Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}'
          f'  Loss={ml:.4f}+/-{sl:.4f}')
    return mp, sp, me, se, ml, sl

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

def main():
    rng = jax.random.key(0)
    results = {}

    # ── 1. Baseline toy (2in/2out, MSE, linear) ──
    rng, k = jax.random.split(rng)
    As1 = [jax.random.rademacher(jax.random.split(k)[i], (2, 2)).astype(jnp.float32)
           for i in range(N_TASKS)]
    def sample_1(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))]
        return jnp.concatenate(xs), jnp.concatenate([A @ x for A, x in zip(As1, xs)])

    results['1_baseline'] = run_experiment(
        '1. Baseline toy (2in/2out, linear, MSE)',
        in_per_task=2, out_per_task=2, n_hidden_max=10, n_hidden_init=1,
        connection_budget=20, sample_fn=sample_1, loss_type='mse')

    # ── 2. Noisy targets ──
    rng, k = jax.random.split(rng)
    As2 = [jax.random.rademacher(jax.random.split(k)[i], (2, 2)).astype(jnp.float32)
           for i in range(N_TASKS)]
    def sample_2(key):
        k1, k2, kn = jax.random.split(key, 3)
        xs = [jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))]
        y = jnp.concatenate([A @ x for A, x in zip(As2, xs)])
        return jnp.concatenate(xs), y + jax.random.normal(kn, (4,)) * 0.5

    results['2_noisy'] = run_experiment(
        '2. Noisy targets (sigma=0.5)',
        in_per_task=2, out_per_task=2, n_hidden_max=10, n_hidden_init=1,
        connection_budget=20, sample_fn=sample_2, loss_type='mse')

    # ── 3. Classification (2 classes, CE) ──
    rng, k = jax.random.split(rng)
    ws3 = [jax.random.normal(jax.random.split(k)[i], (2,)) for i in range(N_TASKS)]
    def sample_3(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))]
        labels = jnp.array([(ws3[t] @ xs[t] > 0).astype(jnp.int32) for t in range(N_TASKS)])
        return jnp.concatenate(xs), labels

    results['3_classify_2'] = run_experiment(
        '3. Classification (2in/2class, CE)',
        in_per_task=2, out_per_task=2, n_hidden_max=10, n_hidden_init=1,
        connection_budget=20, sample_fn=sample_3, loss_type='ce')

    # ── 4. 10 classes ──
    rng, k = jax.random.split(rng)
    hyp4 = [jax.random.normal(jax.random.split(k)[i], (10, 2)) for i in range(N_TASKS)]
    def sample_4(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))]
        labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(hyp4, xs)])
        return jnp.concatenate(xs), labels

    results['4_10class'] = run_experiment(
        '4. 10 classes (2in/10class, CE)',
        in_per_task=2, out_per_task=10, n_hidden_max=10, n_hidden_init=1,
        connection_budget=60, sample_fn=sample_4, loss_type='ce')

    # ── 5. 100 inputs/task, fully connected ──
    rng, k = jax.random.split(rng)
    hyp5 = [jax.random.normal(jax.random.split(k)[i], (10, 100)) * 0.1
            for i in range(N_TASKS)]
    def sample_5(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (100,)), jax.random.normal(k2, (100,))]
        labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(hyp5, xs)])
        return jnp.concatenate(xs), labels

    results['5_100in_full'] = run_experiment(
        '5. 100in/task, fully connected',
        in_per_task=100, out_per_task=10, n_hidden_max=30, n_hidden_init=1,
        connection_budget=5500, sample_fn=sample_5, loss_type='ce', lr=1e-3)

    # ── 6. 100 inputs/task, partial 64 conns (spp=8000) ──
    results['6_100in_partial'] = run_experiment(
        '6. 100in/task, partial 64 conns',
        in_per_task=100, out_per_task=10, n_hidden_max=60, n_hidden_init=1,
        connection_budget=5000, sample_fn=sample_5, loss_type='ce',
        lr=1e-3, max_conns=64, spp=8000)

    # ── 7. 784 inputs/task, partial (spp=8000) ──
    rng, k = jax.random.split(rng)
    hyp7 = [jax.random.normal(jax.random.split(k)[i], (10, 784)) * jnp.sqrt(1.0 / 784)
            for i in range(N_TASKS)]
    def sample_7(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (784,)), jax.random.normal(k2, (784,))]
        labels = jnp.array([jnp.argmax(h @ x) for h, x in zip(hyp7, xs)])
        return jnp.concatenate(xs), labels

    results['7_784in_partial'] = run_experiment(
        '7. 784in/task, partial 64 conns',
        in_per_task=784, out_per_task=10, n_hidden_max=80, n_hidden_init=1,
        connection_budget=5000, sample_fn=sample_7, loss_type='ce',
        lr=1e-3, max_conns=64, spp=8000)

    # ── 8. Nonlinear teacher (spp=8000) ──
    rng, k = jax.random.split(rng)
    teachers = []
    for i in range(N_TASKS):
        k = jax.random.split(k)[0]
        k1, k2 = jax.random.split(k)
        W1 = jax.random.normal(k1, (64, 784)) * jnp.sqrt(2.0 / 784)
        W2 = jax.random.normal(k2, (10, 64)) * jnp.sqrt(2.0 / 64)
        teachers.append((W1, W2))
    def sample_8(key):
        k1, k2 = jax.random.split(key)
        xs = [jax.random.normal(k1, (784,)), jax.random.normal(k2, (784,))]
        labels = []
        for (W1, W2), x in zip(teachers, xs):
            labels.append(jnp.argmax(W2 @ jax.nn.leaky_relu(W1 @ x)))
        return jnp.concatenate(xs), jnp.array(labels)

    results['8_784in_mlp'] = run_experiment(
        '8. 784in/task, nonlinear teacher',
        in_per_task=784, out_per_task=10, n_hidden_max=80, n_hidden_init=1,
        connection_budget=5000, sample_fn=sample_8, loss_type='ce',
        lr=1e-3, max_conns=64, spp=8000)

    # ── Summary ──
    print(f'\n{"=" * 85}')
    print(f'SUMMARY ({N_SEEDS} seeds, 3x turnover, 0.5% batch)')
    print(f'{"=" * 85}')
    print(f'{"Experiment":<30} {"Purity":>14} {"Entropy":>14} {"Loss":>16}')
    print('-' * 85)
    for name, (mp, sp, me, se, ml, sl) in results.items():
        flag = ' <--' if mp < 0.8 else ''
        print(f'{name:<30} {mp:.3f}+/-{sp:.3f}   {me:.3f}+/-{se:.3f}'
              f'   {ml:.4f}+/-{sl:.4f}{flag}')

if __name__ == '__main__':
    main()
