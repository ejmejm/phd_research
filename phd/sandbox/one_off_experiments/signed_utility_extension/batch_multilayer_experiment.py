"""
Multi-Layer Signed Utility Experiment (Multi-Seed, Vmapped)
===========================================================

Same setup as multilayer_experiment.py but vmaps training over N_SEEDS
independent seeds, then plots means with 95% confidence intervals.

Configure N_SEEDS and MASTER_SEED at the top of the file.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from phd.jax_core.models import MLP
from phd.jax_core.optimizers import EqxOptimizer, optax_idbd
from phd.sandbox.one_off_experiments.signed_utility_extension.utility_functions import (
    contribution_utility, upgd_utility, si_utility,
    approach_a_utility, approach_b_utility, approach_c_utility,
    approach_e_utility, approach_f_utility,
    true_loo_utility,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# Constants
# ==============================================================================
N_TEACHER_INPUTS = 5
N_TEACHER_HIDDEN = 4
N_STUDENT_INPUTS = 20
N_STUDENT_HIDDEN = 16
N_RELEVANT = 5
N_STEPS = 50_000
DRIFT_FREQUENCY = 100
TRACE_DECAY = 0.999
SCAN_CHUNK = 5000

# Multi-seed config
N_SEEDS = 30
MASTER_SEED = 42

# Optimizer hyperparameters
SGD_LR = 0.01
AUTOSTEP_INIT_LR = 1.0
AUTOSTEP_META_LR = 0.005

COMPUTE_TRUE_LOO = True

# Utility method registry: (display_name, trace_key, budget_input_key, budget_hidden_key, fn, needs_updates)
UTILITY_METHODS = [
    ('Contribution', 'contribution_traces', None,              None,               contribution_utility, False),
    ('UPGD',         'upgd_traces',         'sum_input_upgd',  'sum_hidden_upgd',  upgd_utility,        False),
    ('SI',           'si_traces',           'sum_input_si',    'sum_hidden_si',    si_utility,          True),
    ('Approach A',   'approach_a_traces',   'sum_input_a',     'sum_hidden_a',     approach_a_utility,  False),
    ('Approach B',   'approach_b_traces',   'sum_input_b',     'sum_hidden_b',     approach_b_utility,  False),
    ('Approach C',   'approach_c_traces',   'sum_input_c',     'sum_hidden_c',     approach_c_utility,  False),
    ('Approach E',   'approach_e_traces',   'sum_input_e',     'sum_hidden_e',     approach_e_utility,  False),
    ('Approach F',   'approach_f_traces',   'sum_input_f',     'sum_hidden_f',     approach_f_utility,  False),
]


# ==============================================================================
# Teacher and data generation (pure JAX, vmappable)
# ==============================================================================

def init_teacher(key):
    """Initialize teacher network with random +/-1 weights."""
    k1, k2 = jax.random.split(key)
    W1 = jax.random.randint(k1, (N_TEACHER_HIDDEN, N_TEACHER_INPUTS), 0, 2).astype(jnp.float32) * 2 - 1
    W2 = jax.random.randint(k2, (1, N_TEACHER_HIDDEN), 0, 2).astype(jnp.float32) * 2 - 1
    return W1, W2


def _make_sign_schedule(key):
    """Generate sign flip schedule (pure JAX, vmappable)."""
    n_flips = (N_STEPS - 1) // DRIFT_FREQUENCY
    flip_indices = jax.random.randint(key, (n_flips,), 0, N_RELEVANT)

    def apply_flip(signs, idx):
        new_signs = signs * (1.0 - 2.0 * jax.nn.one_hot(idx, N_TEACHER_INPUTS))
        return new_signs, new_signs

    init_signs = jnp.ones(N_TEACHER_INPUTS)
    _, flipped_signs = jax.lax.scan(apply_flip, init_signs, flip_indices)
    all_signs = jnp.concatenate([init_signs[None], flipped_signs], axis=0)
    return jnp.repeat(all_signs, DRIFT_FREQUENCY, axis=0)[:N_STEPS]


def _precompute_data(key):
    """Precompute all (x, y*) pairs with sign drift. Pure JAX, vmappable."""
    teacher_key, data_key, sign_key = jax.random.split(key, 3)
    W1_teacher, W2_teacher = init_teacher(teacher_key)
    signs_schedule = _make_sign_schedule(sign_key)

    x_all = jax.random.normal(data_key, (N_STEPS, N_STUDENT_INPUTS))
    x_relevant = x_all[:, :N_TEACHER_INPUTS] * signs_schedule
    z_hidden = jax.vmap(lambda xr: W1_teacher @ xr)(x_relevant)
    a_hidden = (z_hidden > 0.0).astype(jnp.float32)
    y_star_all = jax.vmap(lambda a: (W2_teacher @ a).squeeze())(a_hidden)
    return x_all, y_star_all


# ==============================================================================
# Scanned training steps
# ==============================================================================

def _train_step_body(model, optimizer, x, y_star, ema, budget,
                     compute_pred_grads, compute_loo):
    """Core training step: compute utilities, update EMA traces, update model."""
    y_hat_arr, _ = model(x)
    y_hat = y_hat_arr.squeeze()
    mse = (y_star - y_hat) ** 2

    loss_grads = eqx.filter_grad(lambda m: (m(x)[0].squeeze() - y_star) ** 2)(model)

    # Pre-update utilities
    utilities = {}
    for _, trace_key, _, _, fn, needs_updates in UTILITY_METHODS:
        if not needs_updates:
            utilities[trace_key] = fn(model, x, y_star, y_hat, loss_grads, None)

    # Optimizer update
    if compute_pred_grads:
        pred_grads = eqx.filter_grad(lambda m: m(x)[0].squeeze())(model)
        updates, new_optimizer = optimizer.with_update((loss_grads, pred_grads), model)
    else:
        updates, new_optimizer = optimizer.with_update(loss_grads, model)

    # Post-update utilities
    for _, trace_key, _, _, fn, needs_updates in UTILITY_METHODS:
        if needs_updates:
            utilities[trace_key] = fn(model, x, y_star, y_hat, loss_grads, updates)

    if compute_loo:
        utilities['loo_traces'] = true_loo_utility(model, x, y_star, y_hat, loss_grads, None)

    # Update EMA input traces
    ema = {k: TRACE_DECAY * ema[k] + (1 - TRACE_DECAY) * utilities[k][0]
           if k in utilities else ema[k] for k in ema}

    # Update budget traces
    error_reduced = jnp.abs(y_star) - jnp.abs(y_star - y_hat)
    budget_updates = {'target_mag': jnp.abs(y_star), 'error_reduced': error_reduced}
    for _, _, bik, bhk, _, _ in UTILITY_METHODS:
        if bik is not None:
            trace_key = [tk for _, tk, ik, _, _, _ in UTILITY_METHODS if ik == bik][0]
            u_input, u_hidden = utilities[trace_key]
            budget_updates[bik] = jnp.sum(u_input)
            if bhk is not None:
                budget_updates[bhk] = jnp.sum(u_hidden)
    if compute_loo:
        budget_updates['sum_loo'] = jnp.sum(utilities['loo_traces'][0])
    budget = {k: TRACE_DECAY * budget[k] + (1 - TRACE_DECAY) * budget_updates[k]
              if k in budget_updates else budget[k] for k in budget}

    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optimizer, mse, ema, budget


def _make_scan_fn(compute_pred_grads, compute_loo):
    """Build a scan body for either SGD or Autostep."""
    def scan_fn(carry, step_data):
        model, optimizer, ema, budget = carry
        x, y_star = step_data

        model, optimizer, mse, ema, budget = _train_step_body(
            model, optimizer, x, y_star, ema, budget,
            compute_pred_grads, compute_loo)

        if compute_pred_grads:
            beta_leaves = jax.tree.leaves(optimizer.state.beta)
            step_sizes = jnp.exp(beta_leaves[0]).mean(axis=0)
        else:
            step_sizes = jnp.zeros(N_STUDENT_INPUTS)

        carry = (model, optimizer, ema, budget)
        outputs = (mse, step_sizes, ema, budget)
        return carry, outputs

    return scan_fn


# ==============================================================================
# Multi-seed training loop
# ==============================================================================

def _init_ema_budget():
    """Create initial EMA trace and budget dictionaries."""
    trace_keys = [tk for _, tk, _, _, _, _ in UTILITY_METHODS]
    if COMPUTE_TRUE_LOO:
        trace_keys.append('loo_traces')
    ema = {k: jnp.zeros(N_STUDENT_INPUTS) for k in trace_keys}

    budget = {'target_mag': jnp.float32(0.0), 'error_reduced': jnp.float32(0.0)}
    for _, _, ik, hk, _, _ in UTILITY_METHODS:
        if ik is not None:
            budget[ik] = jnp.float32(0.0)
        if hk is not None:
            budget[hk] = jnp.float32(0.0)
    if COMPUTE_TRUE_LOO:
        budget['sum_loo'] = jnp.float32(0.0)
    return ema, budget


def _stack_pytrees(*pytrees):
    """Stack N pytrees into a batched pytree (leading dim = N).

    All pytrees must have identical structure. Array leaves are stacked
    along a new leading axis; non-array leaves (encoded in treedef by
    equinox modules) are shared.
    """
    leaves_list = [jax.tree.leaves(pt) for pt in pytrees]
    treedef = jax.tree.structure(pytrees[0])
    stacked = [jnp.stack([ls[i] for ls in leaves_list])
               for i in range(len(leaves_list[0]))]
    return jax.tree.unflatten(treedef, stacked)


def run_experiment(optimizer_name, master_seed, n_seeds):
    """Run experiment with multiple seeds using vmap."""
    master_key = jax.random.PRNGKey(master_seed)
    all_keys = jax.random.split(master_key, n_seeds * 2)
    data_keys = all_keys[:n_seeds]
    model_keys = all_keys[n_seeds:]

    # Precompute data for all seeds (vmapped)
    print(f"  Precomputing data for {n_seeds} seeds...")
    x_batch, y_batch = jax.jit(jax.vmap(_precompute_data))(data_keys)
    # x_batch: (N_SEEDS, N_STEPS, N_STUDENT_INPUTS)
    # y_batch: (N_SEEDS, N_STEPS)

    # Initialize carries for each seed (Python loop, then stack)
    is_autostep = optimizer_name == 'autostep'
    carries = []
    for i in range(n_seeds):
        model = MLP(
            input_dim=N_STUDENT_INPUTS, output_dim=1, n_layers=2,
            hidden_dim=N_STUDENT_HIDDEN, weight_init_method='lecun_uniform',
            activation='sigmoid', key=model_keys[i],
        )
        if is_autostep:
            opt = optax_idbd(meta_lr=AUTOSTEP_META_LR, init_lr=AUTOSTEP_INIT_LR, autostep=True)
            optimizer = EqxOptimizer(opt, model, name='autostep')
        else:
            opt = optax.sgd(SGD_LR)
            optimizer = EqxOptimizer(opt, model, name='sgd')
        ema, budget = _init_ema_budget()
        carries.append((model, optimizer, ema, budget))

    batched_carry = _stack_pytrees(*carries)

    # Vmapped + scanned training
    scan_fn = _make_scan_fn(compute_pred_grads=is_autostep, compute_loo=COMPUTE_TRUE_LOO)

    @eqx.filter_jit
    def run_chunk(carry, data):
        return eqx.filter_vmap(
            lambda c, d: jax.lax.scan(scan_fn, c, d)
        )(carry, data)

    n_chunks = N_STEPS // SCAN_CHUNK
    assert N_STEPS % SCAN_CHUNK == 0

    all_outputs = []
    carry = batched_carry
    for chunk_idx in trange(n_chunks, desc=f"{optimizer_name} ({n_seeds} seeds)", unit="chunk"):
        sl = slice(chunk_idx * SCAN_CHUNK, (chunk_idx + 1) * SCAN_CHUNK)
        chunk_data = (x_batch[:, sl], y_batch[:, sl])
        carry, outputs = run_chunk(carry, chunk_data)
        all_outputs.append(jax.tree.map(np.array, outputs))

    # Concatenate along step axis (axis=1): (N_SEEDS, N_STEPS, ...)
    mse_hist = np.concatenate([o[0] for o in all_outputs], axis=1)
    ss_hist = np.concatenate([o[1] for o in all_outputs], axis=1)
    ema_hist = {k: np.concatenate([o[2][k] for o in all_outputs], axis=1)
                for k in all_outputs[0][2]}
    budget_hist = {k: np.concatenate([o[3][k] for o in all_outputs], axis=1)
                   for k in all_outputs[0][3]}

    return {
        'n_seeds': n_seeds,
        'mse_history': mse_hist,            # (N_SEEDS, N_STEPS)
        'step_size_history': ss_hist if is_autostep else None,
        **ema_hist,                          # each: (N_SEEDS, N_STEPS, N_INPUTS)
        **budget_hist,                       # each: (N_SEEDS, N_STEPS)
    }


# ==============================================================================
# Plotting helpers
# ==============================================================================

def _plot_ci(ax, data, label, color, smooth_window=0, linewidth=1.5, alpha=0.2):
    """Plot mean +/- 95% CI across seeds (axis 0). data: (n_seeds, n_steps)."""
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        data = np.array([np.convolve(data[s], kernel, mode='valid')
                         for s in range(data.shape[0])])
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
    steps = np.arange(len(mean))
    ax.plot(steps, mean, label=label, color=color, linewidth=linewidth)
    ax.fill_between(steps, mean - 1.96 * sem, mean + 1.96 * sem,
                    alpha=alpha, color=color)


def _rolling_cohens_d(traces, n_relevant, win):
    """Rolling Cohen's d for a single seed. traces: (N_STEPS, N_INPUTS)."""
    kernel = np.ones(win) / win
    rel = np.mean(traces[:, :n_relevant], axis=1)
    irr = np.mean(traces[:, n_relevant:], axis=1)
    rel_roll = np.convolve(rel, kernel, mode='valid')
    irr_roll = np.convolve(irr, kernel, mode='valid')
    rel_var_roll = np.convolve(np.var(traces[:, :n_relevant], axis=1), kernel, mode='valid')
    irr_var_roll = np.convolve(np.var(traces[:, n_relevant:], axis=1), kernel, mode='valid')
    pool_std = np.sqrt((rel_var_roll + irr_var_roll) / 2)
    return (rel_roll - irr_roll) / (pool_std + 1e-10)


# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(sgd_results, autostep_results):
    """Generate all figures with mean +/- 95% CI across seeds."""
    n_seeds = sgd_results['n_seeds']

    # ---- Figure 1: Learning curves + step sizes ----
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))

    for results, name, color in [(sgd_results, 'SGD', 'tab:blue'),
                                  (autostep_results, 'Autostep', 'tab:orange')]:
        _plot_ci(ax1a, results['mse_history'], name, color, smooth_window=500)
    ax1a.set_xlabel('Step')
    ax1a.set_ylabel('MSE (smoothed)')
    ax1a.set_title(f'Learning Curves (mean ± 95% CI, {n_seeds} seeds)')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    ss = autostep_results['step_size_history']  # (N_SEEDS, N_STEPS, N_INPUTS)
    rel_ss = np.mean(ss[:, :, :N_RELEVANT], axis=2)  # (N_SEEDS, N_STEPS)
    irr_ss = np.mean(ss[:, :, N_RELEVANT:], axis=2)
    _plot_ci(ax1b, rel_ss, 'Relevant (0-4)', 'blue')
    _plot_ci(ax1b, irr_ss, 'Irrelevant (5-19)', 'red')
    ax1b.set_xlabel('Step')
    ax1b.set_ylabel('Step Size (exp(beta), avg over hidden)')
    ax1b.set_title('Autostep Per-Input Step Sizes')
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(SCRIPT_DIR / 'fig1_learning_curves.png', dpi=150)
    print("Saved fig1_learning_curves.png")

    # ---- Figure 2: Utility traces (group means ± CI) ----
    utility_names = [name for name, _, _, _, _, _ in UTILITY_METHODS]
    utility_keys = [tk for _, tk, _, _, _, _ in UTILITY_METHODS]
    if COMPUTE_TRUE_LOO:
        utility_names.append('True LOO')
        utility_keys.append('loo_traces')
    n_cols = len(utility_names)
    fig2, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))

    for row, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        for col, (u_key, u_name) in enumerate(zip(utility_keys, utility_names)):
            ax = axes[row, col]
            traces = results[u_key]  # (N_SEEDS, N_STEPS, N_INPUTS)
            rel = np.mean(traces[:, :, :N_RELEVANT], axis=2)  # (N_SEEDS, N_STEPS)
            irr = np.mean(traces[:, :, N_RELEVANT:], axis=2)
            _plot_ci(ax, rel, 'Relevant', 'blue')
            _plot_ci(ax, irr, 'Irrelevant', 'red')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_title(f'{u_name} ({opt_name})')
            ax.set_xlabel('Step')
            ax.set_ylabel('Mean Utility')
            ax.grid(True, alpha=0.3)

    axes[0, -1].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(SCRIPT_DIR / 'fig2_utility_traces.png', dpi=150)
    print("Saved fig2_utility_traces.png")

    # ---- Figure 3: Utility budget per method ----
    budget_methods = [(name, ik, hk) for name, _, ik, hk, _, _ in UTILITY_METHODS
                      if ik is not None]
    if COMPUTE_TRUE_LOO:
        budget_methods.append(('True LOO', 'sum_loo', None))
    n_budget_cols = len(budget_methods)
    fig3, axes3 = plt.subplots(2, n_budget_cols, figsize=(6 * n_budget_cols, 9))

    for row, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        for col, (method_name, input_key, hidden_key) in enumerate(budget_methods):
            ax = axes3[row, col]
            _plot_ci(ax, results['target_mag'], '|y*|', 'black')
            _plot_ci(ax, results['error_reduced'], 'error reduced', 'tab:green')
            if hidden_key is not None:
                _plot_ci(ax, results[hidden_key], 'Σ U_hidden', 'tab:orange')
            _plot_ci(ax, results[input_key], 'Σ U_input', 'tab:blue')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_title(f'{method_name} ({opt_name})')
            ax.set_xlabel('Step')
            ax.set_ylabel('EMA Trace')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(SCRIPT_DIR / 'fig3_utility_budget.png', dpi=150)
    print("Saved fig3_utility_budget.png")

    # ---- Figure 4: Separation (Cohen's d) mean ± CI ----
    sep_win = 100
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    for col, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        ax = axes4[col]
        for u_key, u_name in zip(utility_keys, utility_names):
            traces = results[u_key]  # (N_SEEDS, N_STEPS, N_INPUTS)
            d_per_seed = np.array([
                _rolling_cohens_d(traces[s], N_RELEVANT, sep_win)
                for s in range(n_seeds)
            ])  # (N_SEEDS, N_STEPS - sep_win + 1)
            steps = np.arange(sep_win - 1, sep_win - 1 + d_per_seed.shape[1])
            mean_d = np.mean(d_per_seed, axis=0)
            sem_d = np.std(d_per_seed, axis=0) / np.sqrt(n_seeds)
            ax.plot(steps, mean_d, label=u_name, linewidth=1.2)
            ax.fill_between(steps, mean_d - 1.96 * sem_d, mean_d + 1.96 * sem_d,
                            alpha=0.15)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
                   label='d=1')
        ax.set_title(f'Separation over time ({opt_name})')
        ax.set_xlabel('Step')
        ax.set_ylabel(f"Cohen's d (rolling {sep_win})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(SCRIPT_DIR / 'fig4_separation.png', dpi=150)
    print("Saved fig4_separation.png")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("Running multi-layer signed utility experiment (multi-seed)...")
    print(f"  Teacher: {N_TEACHER_INPUTS} -> {N_TEACHER_HIDDEN} (sigmoid) -> 1")
    print(f"  Student: {N_STUDENT_INPUTS} -> {N_STUDENT_HIDDEN} (sigmoid) -> 1")
    print(f"  Steps: {N_STEPS}, drift every {DRIFT_FREQUENCY} steps")
    print(f"  Seeds: {N_SEEDS} (master seed: {MASTER_SEED})")
    print()

    print("Running SGD...")
    sgd_results = run_experiment('sgd', MASTER_SEED, N_SEEDS)

    print("\nRunning Autostep...")
    autostep_results = run_experiment('autostep', MASTER_SEED, N_SEEDS)

    print("\nPlotting results...")
    plot_results(sgd_results, autostep_results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print(f"Summary ({N_SEEDS} seeds)")
    print("=" * 60)
    last_5k = slice(-5000, None)

    summary_methods = [(name, tk) for name, tk, _, _, _, _ in UTILITY_METHODS]
    if COMPUTE_TRUE_LOO:
        summary_methods.append(('True LOO', 'loo_traces'))

    for opt_name, results in [('SGD', sgd_results), ('Autostep', autostep_results)]:
        n_seeds = results['n_seeds']
        mse_per_seed = np.mean(results['mse_history'][:, last_5k], axis=1)
        print(f"\n{opt_name} -- Final MSE (last 5k): "
              f"{np.mean(mse_per_seed):.4f} ± {np.std(mse_per_seed):.4f}")

        for u_name, u_key in summary_methods:
            traces = results[u_key]  # (N_SEEDS, N_STEPS, N_INPUTS)

            # Per-seed Cohen's d on last 5k steps
            d_per_seed = []
            sep_steps = []
            for s in range(n_seeds):
                t = traces[s]
                last_t = t[last_5k]
                rel_mean = np.mean(last_t[:, :N_RELEVANT])
                irr_mean = np.mean(last_t[:, N_RELEVANT:])
                rel_std = np.std(last_t[:, :N_RELEVANT])
                irr_std = np.std(last_t[:, N_RELEVANT:])
                pooled = np.sqrt((rel_std**2 + irr_std**2) / 2)
                d_per_seed.append((rel_mean - irr_mean) / (pooled + 1e-10))

                # First step where rolling d > 1
                d_roll = _rolling_cohens_d(t, N_RELEVANT, 100)
                idx = np.where(d_roll > 1.0)[0]
                sep_steps.append(int(idx[0]) + 100 if len(idx) > 0 else None)

            d_arr = np.array(d_per_seed)
            valid_sep = [s for s in sep_steps if s is not None]
            if valid_sep:
                sep_str = (f"{np.mean(valid_sep):.0f} ± {np.std(valid_sep):.0f} "
                           f"({len(valid_sep)}/{n_seeds})")
            else:
                sep_str = "never"
            print(f"  {u_name}: d = {np.mean(d_arr):.2f} ± {np.std(d_arr):.2f}, "
                  f"d>1 @ step {sep_str}")

    print("\nDone.")
