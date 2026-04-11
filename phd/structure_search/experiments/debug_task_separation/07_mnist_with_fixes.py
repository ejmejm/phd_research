"""MNIST with proper pruning: 3x turnover, 0.5% batch, spp=8000.

Re-tests the algorithmic variants from script 02 but with the pruning
fixes from scripts 04-06. Uses real MNIST data loaded as a JAX array
for vmapped execution. 20 seeds. Reports purity, entropy, and loss.
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

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_mnist():
    from data import load_dataset
    images, labels, _, _ = load_dataset('mnist', split='train')
    return jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)

# ══════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════

def forward(Wi, Wo, Mi, Mo, x):
    h = jax.nn.leaky_relu((Wi * Mi) @ x)
    return (Wo * Mo) @ h, h

def loss_ce(Wi, Wo, Mi, Mo, x, y):
    raw, _ = forward(Wi, Wo, Mi, Mo, x)
    logits = raw.reshape(N_TASKS, NUM_CLASSES)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, NUM_CLASSES) * lp, axis=-1))

def compute_utility(Wi, Wo, Mi, Mo, x, use_propagated):
    _, h = forward(Wi, Wo, Mi, Mo, x)
    U_in_c = jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi
    U_out = jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo
    parent = U_out.sum(axis=0)
    raw = U_in_c.sum(axis=1)
    sc = jnp.where(jnp.abs(raw) > 1e-10, parent / raw, 1.0)
    U_in_p = U_in_c * sc[:, None] * Mi
    U_in = jnp.where(use_propagated, U_in_p, U_in_c)
    return U_in, U_out

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

def gen_unit(Wi, Wo, Mi, Mo, key, *, single_output=False):
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has = jnp.any(~active)
    k1, k2, k3 = jax.random.split(key, 3)
    nc = min(MAX_CONNS, INPUT_DIM)
    noise = jax.random.uniform(k1, (INPUT_DIM,))
    perm = jnp.argsort(noise)
    im = jnp.zeros(INPUT_DIM, dtype=jnp.int32).at[perm[:nc]].set(1)
    bound = jnp.sqrt(3.0 / nc)
    w = jax.random.uniform(k2, (INPUT_DIM,), minval=-bound, maxval=bound) * im
    if single_output:
        oi = jax.random.randint(k3, (), 0, OUTPUT_DIM)
        om = jnp.zeros(OUTPUT_DIM, dtype=jnp.int32).at[oi].set(1)
    else:
        om = jnp.ones(OUTPUT_DIM, dtype=jnp.int32)
    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(im), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(om), Mo)
    return Wi, Wo, Mi, Mo

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT BUILDER
# ══════════════════════════════════════════════════════════════════════

def compute_prune_params():
    ppe = max(1, round(0.005 * BUDGET))
    total_pruned = round(3.0 * BUDGET)
    n_prunes = max(1, total_pruned // ppe)
    actual_turn = (n_prunes * ppe) / BUDGET
    return ppe, n_prunes, actual_turn

def make_run_fn(mnist_images, mnist_labels, *,
                use_propagated=False, single_output=False,
                use_adam=False, permute_period=0, n_hidden_init=1):
    ppe, n_prunes, _ = compute_prune_params()
    gfn = partial(gen_unit, single_output=single_output)

    if use_adam:
        opt = optax.adam(LR)

    def make_sample(key):
        k1, k2 = jax.random.split(key)
        idx1 = jax.random.randint(k1, (), 0, mnist_images.shape[0])
        idx2 = jax.random.randint(k2, (), 0, mnist_images.shape[0])
        x = jnp.concatenate([mnist_images[idx1], mnist_images[idx2]])
        y = jnp.array([mnist_labels[idx1], mnist_labels[idx2]])
        return x, y

    use_prop_arr = jnp.array(use_propagated)

    def run_one(rng):
        Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
        Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
        Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)
        Ui = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Uo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))

        # Init units
        def init_body(carry, _):
            Wi, Wo, Mi, Mo, rng = carry
            rng, gk = jax.random.split(rng)
            Wi, Wo, Mi, Mo = gfn(Wi, Wo, Mi, Mo, gk)
            return (Wi, Wo, Mi, Mo, rng), None
        (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
            init_body, (Wi, Wo, Mi, Mo, rng), None, length=n_hidden_init)

        if use_adam:
            opt_state = opt.init((Wi, Wo))
        else:
            opt_state = jnp.array(0.0)  # dummy

        step = jnp.array(0, dtype=jnp.int32)
        perm0 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        perm1 = jnp.arange(NUM_CLASSES, dtype=jnp.int32)

        def prune_cycle(carry, _):
            Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng, step, perm0, perm1 = carry

            rng, tk, pk = jax.random.split(rng, 3)
            data_keys = jax.random.split(tk, SPP)
            perm_keys = jax.random.split(pk, SPP)

            def train_body(carry, inputs):
                Wi, Wo, Ui, Uo, opt_state, step, perm0, perm1 = carry
                dkey, pkey = inputs

                x, y_raw = make_sample(dkey)
                y = jnp.array([perm0[y_raw[0]], perm1[y_raw[1]]])

                gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
                if use_adam:
                    updates, opt_state = opt.update((gi, go), opt_state, (Wi, Wo))
                    Wi = Wi + updates[0] * Mi
                    Wo = Wo + updates[1] * Mo
                else:
                    Wi = Wi - LR * gi * Mi
                    Wo = Wo - LR * go * Mo

                ui, uo = compute_utility(Wi, Wo, Mi, Mo, x, use_prop_arr)
                Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
                Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo

                # Permutation
                step = step + 1
                should_perm = (permute_period > 0) & (step % permute_period == 0)
                pk1, pk2 = jax.random.split(pkey)
                which = jax.random.randint(pk1, (), 0, 2)
                new_perm = jax.random.permutation(pk2, NUM_CLASSES).astype(jnp.int32)
                perm0 = jnp.where(should_perm & (which == 0), new_perm, perm0)
                perm1 = jnp.where(should_perm & (which == 1), new_perm, perm1)

                return (Wi, Wo, Ui, Uo, opt_state, step, perm0, perm1), None

            (Wi, Wo, Ui, Uo, opt_state, step, perm0, perm1), _ = jax.lax.scan(
                train_body,
                (Wi, Wo, Ui, Uo, opt_state, step, perm0, perm1),
                (data_keys, perm_keys))

            rng, prk = jax.random.split(rng)
            Mi, Mo = prune_n(Ui, Uo, Mi, Mo, ppe, prk)
            Mi = cleanup_dead(Mi, Mo)
            Ui, Uo = reset_utility_to_median(Ui, Uo, Mi, Mo)

            def gen_one(carry, _):
                Wi, Wo, Mi, Mo, rng = carry
                nc = Mi.sum() + Mo.sum()
                need = nc < BUDGET
                rng, gk = jax.random.split(rng)
                Wg, Wog, Mg, Mog = gfn(Wi, Wo, Mi, Mo, gk)
                Wi = jnp.where(need, Wg, Wi)
                Wo = jnp.where(need, Wog, Wo)
                Mi = jnp.where(need, Mg, Mi)
                Mo = jnp.where(need, Mog, Mo)
                return (Wi, Wo, Mi, Mo, rng), None
            (Wi, Wo, Mi, Mo, rng), _ = jax.lax.scan(
                gen_one, (Wi, Wo, Mi, Mo, rng), None, length=MAX_GEN_PER_CYCLE)

            return (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng, step, perm0, perm1), None

        (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng, step, perm0, perm1), _ = jax.lax.scan(
            prune_cycle,
            (Wi, Wo, Mi, Mo, Ui, Uo, opt_state, rng, step, perm0, perm1),
            None, length=n_prunes)

        # Eval loss (stationary — identity permutations)
        rng, ek = jax.random.split(rng)
        eval_keys = jax.random.split(ek, N_EVAL)
        def one_loss(k):
            x, y = make_sample(k)
            return loss_ce(Wi, Wo, Mi, Mo, x, y)
        final_loss = jax.vmap(one_loss)(eval_keys).mean()

        return Mi, Mo, final_loss

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
            ic = [int(Mi[k, t * INPUT_PER_TASK:(t + 1) * INPUT_PER_TASK].sum())
                  for t in range(N_TASKS)]
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
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ('1. Baseline (contrib, all_out, SGD, stat)',
     {}),
    ('2. Propagated utility',
     dict(use_propagated=True)),
    ('3. Single output gen',
     dict(single_output=True)),
    ('4. Adam optimizer',
     dict(use_adam=True)),
    ('5. Non-stationary (permute=2000)',
     dict(permute_period=2000)),
    ('6. Prop + single + Adam + nonstat',
     dict(use_propagated=True, single_output=True, use_adam=True,
          permute_period=2000, n_hidden_init=0)),
]

def main():
    print("Loading MNIST...")
    mnist_images, mnist_labels = load_mnist()
    print(f"Loaded {mnist_images.shape[0]} images")

    ppe, n_prunes, actual_turn = compute_prune_params()
    total_steps = n_prunes * SPP
    print(f"Config: budget={BUDGET} maxc={MAX_CONNS} spp={SPP} ppe={ppe} "
          f"events={n_prunes} turn={actual_turn:.1f}x steps={total_steps/1e6:.1f}M")

    results = {}
    for name, kw in EXPERIMENTS:
        print(f'\n{"=" * 70}')
        print(f'{name}')
        print(f'  {kw or "defaults"}')

        run_fn = make_run_fn(mnist_images, mnist_labels, **kw)
        rngs = jax.random.split(jax.random.key(BASE_SEED), N_SEEDS)

        t0 = time.time()
        print(f'  Compiling + running {N_SEEDS} seeds ...', end=' ', flush=True)
        all_Mi, all_Mo, all_loss = run_fn(rngs)
        jax.block_until_ready((all_Mi, all_Mo, all_loss))
        print(f'{time.time() - t0:.1f}s')

        purs, ents = compute_purities(all_Mi, all_Mo)
        losses = np.array(all_loss)
        mp, me, ml = purs.mean(), ents.mean(), losses.mean()
        sp, se, sl = ci(purs), ci(ents), ci(losses)
        print(f'  >>> Purity={mp:.3f}+/-{sp:.3f}  Entropy={me:.3f}+/-{se:.3f}'
              f'  Loss={ml:.4f}+/-{sl:.4f}')
        results[name] = (mp, sp, me, se, ml, sl)

    print(f'\n{"=" * 90}')
    print(f'SUMMARY ({N_SEEDS} seeds, MNIST, spp={SPP}, 3x turnover)')
    print(f'{"=" * 90}')
    print(f'{"Experiment":<45} {"Purity":>14} {"Entropy":>14} {"Loss":>16}')
    print('-' * 90)
    for name, (mp, sp, me, se, ml, sl) in results.items():
        flag = ' ***' if mp > 0.9 else (' <--' if mp < 0.8 else '')
        print(f'{name:<45} {mp:.3f}+/-{sp:.3f}   {me:.3f}+/-{se:.3f}'
              f'   {ml:.4f}+/-{sl:.4f}{flag}')

if __name__ == '__main__':
    main()
