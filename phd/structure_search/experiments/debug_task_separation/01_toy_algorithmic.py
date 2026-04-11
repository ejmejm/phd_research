"""Debug task separation: incremental changes from working notebook -> failing script.

The toy notebook (outgoing_pruning_gen.ipynb) achieves task specialization via
pruning+generation. The full script (column_guided_search.py) with similar settings
fails (input_entropy stays ~0.35). This script identifies which change(s) break it
by applying ONE change at a time and measuring task purity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
import time

# ══════════════════════════════════════════════════════════════════
# CONSTANTS (matching toy notebook)
# ══════════════════════════════════════════════════════════════════
N_TASKS = 2
N_INPUTS_PER_TASK = 2
N_OUTPUTS_PER_TASK = 2
INPUT_DIM = N_INPUTS_PER_TASK * N_TASKS   # 4
OUTPUT_DIM = N_OUTPUTS_PER_TASK * N_TASKS  # 4
N_HIDDEN_MAX = 10
SEED = 2133470219

BLOCK_SIZE = 1000           # steps per JIT call
TRAIN_STEPS_PER_PRUNE = 5000
TOTAL_PRUNE_STEPS = 200
CONNECTION_BUDGET = 20

# ══════════════════════════════════════════════════════════════════
# CORE NETWORK
# ══════════════════════════════════════════════════════════════════

def make_targets(key):
    k1, k2 = jax.random.split(key)
    A1 = jax.random.rademacher(k1, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
    A2 = jax.random.rademacher(k2, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
    return A1, A2

def sample_data(key, A1, A2):
    k1, k2 = jax.random.split(key)
    x1 = jax.random.normal(k1, (N_INPUTS_PER_TASK,))
    x2 = jax.random.normal(k2, (N_INPUTS_PER_TASK,))
    return jnp.concatenate([x1, x2]), jnp.concatenate([A1 @ x1, A2 @ x2])

def forward(W_in, W_out, M_in, M_out, x):
    h = jax.nn.leaky_relu((W_in * M_in) @ x)
    return (W_out * M_out) @ h, h

def loss_mse(W_in, W_out, M_in, M_out, x, y):
    y_hat, _ = forward(W_in, W_out, M_in, M_out, x)
    return jnp.mean((y_hat - y) ** 2)

# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════

def compute_utility(W_in, W_out, M_in, M_out, x, use_propagated):
    """Contribution or propagated (backward-scaled) connection utility."""
    _, h = forward(W_in, W_out, M_in, M_out, x)
    # Contribution utility
    U_in_c = jnp.abs(x)[None, :] * jnp.abs(W_in) * M_in       # (H, I)
    U_out = jnp.abs(h)[None, :] * jnp.abs(W_out) * M_out       # (O, H)
    # Propagated: scale input utilities so they sum to unit's output utility
    parent = U_out.sum(axis=0)          # (H,) per-unit outgoing utility
    raw_sum = U_in_c.sum(axis=1)        # (H,) per-unit raw input utility sum
    scale = jnp.where(jnp.abs(raw_sum) > 1e-10, parent / raw_sum, 1.0)
    U_in_p = U_in_c * scale[:, None] * M_in
    U_in = jnp.where(use_propagated, U_in_p, U_in_c)
    return U_in, U_out

# ══════════════════════════════════════════════════════════════════
# PRUNING
# ══════════════════════════════════════════════════════════════════

def find_lowest_utility(U_in, U_out, M_in, M_out):
    u_in = jnp.where(M_in.reshape(-1) == 1, U_in.reshape(-1), jnp.inf)
    u_out = jnp.where(M_out.reshape(-1) == 1, U_out.reshape(-1), jnp.inf)
    all_u = jnp.concatenate([u_in, u_out])
    idx = jnp.argmin(all_u)
    n_in = M_in.size
    return idx >= n_in, jnp.where(idx >= n_in, idx - n_in, idx)

def prune_connection(M_in, M_out, is_output, local_idx):
    mi, mo = M_in.reshape(-1), M_out.reshape(-1)
    ii = jnp.where(is_output, 0, local_idx)
    oi = jnp.where(is_output, local_idx, 0)
    iv = jnp.where(is_output, mi[ii], 0)
    ov = jnp.where(is_output, 0, mo[oi])
    return mi.at[ii].set(iv).reshape(M_in.shape), mo.at[oi].set(ov).reshape(M_out.shape)

def cleanup_dead_units(M_in, M_out):
    return M_in * (M_out.sum(axis=0) > 0)[:, None]

# ══════════════════════════════════════════════════════════════════
# GENERATION VARIANTS
# ══════════════════════════════════════════════════════════════════

def _find_slot(M_in, M_out):
    active = M_in.sum(axis=1) + M_out.sum(axis=0)
    slot = jnp.argmin(jnp.where(active == 0, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has_slot = jnp.any(active == 0)
    return slot, has_slot

def generate_full(W_in, W_out, M_in, M_out, key):
    """Notebook default: all inputs connected, all outputs with weight=0."""
    slot, has = _find_slot(M_in, M_out)
    limit = jnp.sqrt(3.0 / INPUT_DIM)
    w = jax.random.uniform(key, (INPUT_DIM,), minval=-limit, maxval=limit)
    W_in = jnp.where(has, W_in.at[slot].set(w), W_in)
    W_out = jnp.where(has, W_out.at[:, slot].set(0.0), W_out)
    M_in = jnp.where(has, M_in.at[slot].set(1), M_in)
    M_out = jnp.where(has, M_out.at[:, slot].set(1), M_out)
    return W_in, W_out, M_in, M_out

def generate_single_output(W_in, W_out, M_in, M_out, key):
    """All inputs, 1 random output."""
    slot, has = _find_slot(M_in, M_out)
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(3.0 / INPUT_DIM)
    w = jax.random.uniform(k1, (INPUT_DIM,), minval=-limit, maxval=limit)
    oi = jax.random.randint(k2, (), 0, OUTPUT_DIM)
    mo = jnp.zeros(OUTPUT_DIM, dtype=jnp.int32).at[oi].set(1)
    W_in = jnp.where(has, W_in.at[slot].set(w), W_in)
    W_out = jnp.where(has, W_out.at[:, slot].set(0.0), W_out)
    M_in = jnp.where(has, M_in.at[slot].set(1), M_in)
    M_out = jnp.where(has, M_out.at[:, slot].set(mo), M_out)
    return W_in, W_out, M_in, M_out

def generate_partial_single(W_in, W_out, M_in, M_out, key):
    """Half random inputs, 1 random output (closest to script behavior)."""
    slot, has = _find_slot(M_in, M_out)
    k1, k2, k3 = jax.random.split(key, 3)
    n_conns = max(1, INPUT_DIM // 2)  # 2 out of 4
    limit = jnp.sqrt(3.0 / n_conns)
    w = jax.random.uniform(k1, (INPUT_DIM,), minval=-limit, maxval=limit)
    perm = jax.random.permutation(k2, INPUT_DIM)
    im = jnp.zeros(INPUT_DIM, dtype=jnp.int32).at[perm[:n_conns]].set(1)
    w = w * im
    oi = jax.random.randint(k3, (), 0, OUTPUT_DIM)
    mo = jnp.zeros(OUTPUT_DIM, dtype=jnp.int32).at[oi].set(1)
    W_in = jnp.where(has, W_in.at[slot].set(w), W_in)
    W_out = jnp.where(has, W_out.at[:, slot].set(0.0), W_out)
    M_in = jnp.where(has, M_in.at[slot].set(im), M_in)
    M_out = jnp.where(has, M_out.at[:, slot].set(mo), M_out)
    return W_in, W_out, M_in, M_out

# ══════════════════════════════════════════════════════════════════
# JIT-COMPILED TRAINING BLOCKS
# ══════════════════════════════════════════════════════════════════

_adam = optax.adam(1e-3)

@partial(jax.jit, static_argnames=('n_steps',))
def train_block_sgd(W_in, W_out, M_in, M_out, A1, A2, rng, use_prop, n_steps):
    def body(carry, key):
        Wi, Wo, Ui, Uo = carry
        x, y = sample_data(key, A1, A2)
        gi, go = jax.grad(loss_mse, argnums=(0, 1))(Wi, Wo, M_in, M_out, x, y)
        Wi = Wi - 3e-3 * gi * M_in
        Wo = Wo - 3e-3 * go * M_out
        ui, uo = compute_utility(Wi, Wo, M_in, M_out, x, use_prop)
        return (Wi, Wo, Ui + ui, Uo + uo), None
    keys = jax.random.split(rng, n_steps)
    z_in = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
    z_out = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
    (Wi, Wo, Ui, Uo), _ = jax.lax.scan(body, (W_in, W_out, z_in, z_out), keys)
    return Wi, Wo, Ui / n_steps, Uo / n_steps

@partial(jax.jit, static_argnames=('n_steps',))
def train_block_adam(W_in, W_out, M_in, M_out, A1, A2, rng, opt_st, use_prop, n_steps):
    def body(carry, key):
        Wi, Wo, ost, Ui, Uo = carry
        x, y = sample_data(key, A1, A2)
        gi, go = jax.grad(loss_mse, argnums=(0, 1))(Wi, Wo, M_in, M_out, x, y)
        upd, ost2 = _adam.update((gi, go), ost, (Wi, Wo))
        Wi = Wi + upd[0] * M_in
        Wo = Wo + upd[1] * M_out
        ui, uo = compute_utility(Wi, Wo, M_in, M_out, x, use_prop)
        return (Wi, Wo, ost2, Ui + ui, Uo + uo), None
    keys = jax.random.split(rng, n_steps)
    z_in = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
    z_out = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
    (Wi, Wo, ost, Ui, Uo), _ = jax.lax.scan(
        body, (W_in, W_out, opt_st, z_in, z_out), keys)
    return Wi, Wo, Ui / n_steps, Uo / n_steps, ost

# ══════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════

def measure_purity(M_in, M_out):
    """Per-unit task purity and entropy. Returns (in_purity, out_purity, in_entropy)."""
    M_in, M_out = np.array(M_in), np.array(M_out)
    rows = []
    for k in range(N_HIDDEN_MAX):
        if not (M_in[k].any() or M_out[:, k].any()):
            continue
        ic = [int(M_in[k, t * N_INPUTS_PER_TASK:(t + 1) * N_INPUTS_PER_TASK].sum())
              for t in range(N_TASKS)]
        oc = [int(M_out[t * N_OUTPUTS_PER_TASK:(t + 1) * N_OUTPUTS_PER_TASK, k].sum())
              for t in range(N_TASKS)]
        ti, to = sum(ic), sum(oc)
        ip = max(ic) / max(ti, 1)
        op = max(oc) / max(to, 1)
        ie = sum(-c / ti * np.log2(c / ti + 1e-15) for c in ic if c > 0) if ti > 0 else 0.0
        rows.append(dict(u=k, ic=ic, oc=oc, ip=ip, op=op, ie=ie))

    if not rows:
        print("  No active units!")
        return 0.0, 0.0, 1.0

    mip = np.mean([r['ip'] for r in rows])
    mop = np.mean([r['op'] for r in rows])
    mie = np.mean([r['ie'] for r in rows])

    for r in rows:
        ic_s = '/'.join(str(c) for c in r['ic'])
        oc_s = '/'.join(str(c) for c in r['oc'])
        print(f"  h{r['u']:2d}: in=[{ic_s}] pur={r['ip']:.2f} ent={r['ie']:.2f} "
              f"| out=[{oc_s}] pur={r['op']:.2f}")
    n_conns = int(M_in.sum() + M_out.sum())
    print(f'  Units: {len(rows)} | Conns: {n_conns}')
    print(f'  >>> Purity: in={mip:.3f} out={mop:.3f} | Entropy: in={mie:.3f}')
    return mip, mop, mie

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════

def run_experiment(name, *, utility='contribution', generation='full',
                   optimizer='sgd', n_hidden_init=1, permute_period=0,
                   seed=SEED):
    use_prop = jnp.array(utility == 'propagated')
    use_adam = (optimizer == 'adam')
    gen_fn = {'full': generate_full, 'single_output': generate_single_output,
              'partial_single': generate_partial_single}[generation]

    print(f'\n{"=" * 60}')
    print(f'{name}')
    print(f'  utility={utility} gen={generation} opt={optimizer} '
          f'init={n_hidden_init} permute={permute_period}')

    assert permute_period == 0 or permute_period % BLOCK_SIZE == 0

    rng = jax.random.key(seed)
    rng, kt, ki = jax.random.split(rng, 3)
    A1, A2 = make_targets(kt)

    # Init params
    Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
    Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
    Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
    Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)
    if n_hidden_init > 0:
        k1, k2 = jax.random.split(ki)
        Wi = Wi.at[:n_hidden_init].set(
            jax.random.normal(k1, (n_hidden_init, INPUT_DIM)) * jnp.sqrt(2.0 / INPUT_DIM))
        Wo = Wo.at[:, :n_hidden_init].set(
            jax.random.normal(k2, (OUTPUT_DIM, n_hidden_init)) * jnp.sqrt(2.0 / n_hidden_init))
        Mi = Mi.at[:n_hidden_init].set(1)
        Mo = Mo.at[:, :n_hidden_init].set(1)

    ost = _adam.init((Wi, Wo)) if use_adam else None

    t0 = time.time()
    step = 0
    n_blocks = TRAIN_STEPS_PER_PRUNE // BLOCK_SIZE

    for pi in range(TOTAL_PRUNE_STEPS):
        Ui_acc = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
        Uo_acc = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))

        for _ in range(n_blocks):
            rng, tk = jax.random.split(rng)
            if use_adam:
                Wi, Wo, Ui_b, Uo_b, ost = train_block_adam(
                    Wi, Wo, Mi, Mo, A1, A2, tk, ost, use_prop, BLOCK_SIZE)
            else:
                Wi, Wo, Ui_b, Uo_b = train_block_sgd(
                    Wi, Wo, Mi, Mo, A1, A2, tk, use_prop, BLOCK_SIZE)
            Ui_acc = Ui_acc + Ui_b
            Uo_acc = Uo_acc + Uo_b
            step += BLOCK_SIZE

            # Non-stationarity: permute one target matrix
            if permute_period > 0 and step % permute_period == 0:
                rng, pk = jax.random.split(rng)
                kw, kn = jax.random.split(pk)
                which = jax.random.randint(kw, (), 0, N_TASKS)
                A_new = jax.random.rademacher(
                    kn, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
                A1 = jnp.where(which == 0, A_new, A1)
                A2 = jnp.where(which == 1, A_new, A2)

        Ui = Ui_acc / n_blocks
        Uo = Uo_acc / n_blocks

        # Prune 1 connection
        is_out, loc = find_lowest_utility(Ui, Uo, Mi, Mo)
        Mi, Mo = prune_connection(Mi, Mo, is_out, loc)
        Mi = cleanup_dead_units(Mi, Mo)

        # Generate if below budget
        nc = int(Mi.sum() + Mo.sum())
        if nc < CONNECTION_BUDGET:
            rng, gk = jax.random.split(rng)
            Wi, Wo, Mi, Mo = gen_fn(Wi, Wo, Mi, Mo, gk)

        if (pi + 1) % 50 == 0:
            na = int(np.array((Mi.sum(axis=1) + Mo.sum(axis=0)) > 0).sum())
            nc2 = int(Mi.sum() + Mo.sum())
            print(f'  [{pi + 1}/{TOTAL_PRUNE_STEPS}] {nc2} conns, {na} units')

    print(f'  Done in {time.time() - t0:.1f}s')
    return measure_purity(Mi, Mo)

# ══════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ('1. Baseline (notebook)',
     {}),
    ('2. Propagated utility',
     dict(utility='propagated')),
    ('3. Empty init (0 hidden)',
     dict(n_hidden_init=0)),
    ('4. Single output gen',
     dict(generation='single_output')),
    ('5. Partial inputs + single output',
     dict(generation='partial_single')),
    ('6. Adam optimizer',
     dict(optimizer='adam')),
    ('7. Non-stationary (permute=100k)',
     dict(permute_period=100_000)),
    ('8. Propagated + single output',
     dict(utility='propagated', generation='single_output')),
    ('9. Propagated + Adam + empty',
     dict(utility='propagated', optimizer='adam', n_hidden_init=0)),
    ('10. All combined (script-like)',
     dict(utility='propagated', generation='partial_single',
          optimizer='adam', n_hidden_init=0, permute_period=100_000)),
]

if __name__ == '__main__':
    results = {}
    for name, kw in EXPERIMENTS:
        ip, op, ie = run_experiment(name, **kw)
        results[name] = (ip, op, ie)

    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    print(f'{"Experiment":<45} {"InPur":>6} {"OutPur":>7} {"InEnt":>6}')
    print('-' * 70)
    for name, (ip, op, ie) in results.items():
        flag = ' <--' if ip < 0.8 else ''
        print(f'{name:<45} {ip:>6.3f} {op:>7.3f} {ie:>6.3f}{flag}')
