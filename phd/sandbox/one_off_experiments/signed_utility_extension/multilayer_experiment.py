"""
Multi-Layer Signed Utility Experiment
=====================================

Compares utility methods for identifying relevant vs irrelevant input features
in a nonlinear teacher-student setup.

Problem Setup
-------------
A nonlinear teacher network (5 -> 4 hidden sigmoid -> 1 linear output) generates targets.
A larger student network (20 -> 16 hidden sigmoid -> 1 linear output) learns online. Only
inputs 0-4 carry signal; inputs 5-19 are irrelevant noise. Teacher input weight signs flip
periodically to create nonstationarity.

Utility Methods
---------------
1. **Contribution** (CBP analog): |x_i| * sum_j |W1[j,i]|. Always positive.
2. **UPGD** (Elsayed & Mahmood 2023): -(dL/da_i) * a_i. First-order Taylor approximation.
3. **SI** (Zenke et al. 2017): sum_j (-dL/dW1[j,i]) * dW1[j,i]. Parameter loss-decrease.
4. **Approach A** (Proportional Redistribution): inherits parent's sign.
5. **Approach B** (Recursive Signed): applies LOO formula recursively with pseudo-error.
6. **Approach C** (Sign-Conserving): signed normalization with Approach A fallback.
7. **Approach E** (Calibrated Pseudo-Error): calibrates pseudo-error so scores sum to target.
8. **Approach F** (Capped Signed): C's signed normalization with magnitude cap instead of fallback.
9. **Approach G** (Target Propagation): real pre-activation targets via f^{-1}, signed normalization + cap.
10. **Approach H** (Coherence-Weighted): blends signed/absolute decomposition via coherence β = |z_j|/Σ|c_k|.
11. **True LOO**: 20 extra forward passes per step (optional).
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
    approach_e_utility, approach_f_utility, approach_g_utility, approach_h_utility,
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
SEED = np.random.randint(0, 1000000)
SCAN_CHUNK = 5000  # Steps per scan chunk (for progress updates)

# Optimizer hyperparameters
SGD_LR = 0.01
AUTOSTEP_INIT_LR = 1.0
AUTOSTEP_META_LR = 0.005

# Whether to compute true leave-one-out utilities (20 extra forward passes per step)
COMPUTE_TRUE_LOO = True

# Utility method registry: (display_name, trace_key, budget_input_key, budget_hidden_key, fn, needs_updates)
# budget keys are None for methods that don't track budget.
# needs_updates=True means the method requires optimizer updates (computed after optimizer step).
UTILITY_METHODS = [
    ('Contribution', 'contribution_traces', None,              None,               contribution_utility, False),
    ('UPGD',         'upgd_traces',         'sum_input_upgd',  'sum_hidden_upgd',  upgd_utility,        False),
    ('SI',           'si_traces',           'sum_input_si',    'sum_hidden_si',    si_utility,          True),
    ('Approach A',   'approach_a_traces',   'sum_input_a',     'sum_hidden_a',     approach_a_utility,  False),
    ('Approach B',   'approach_b_traces',   'sum_input_b',     'sum_hidden_b',     approach_b_utility,  False),
    ('Approach C',   'approach_c_traces',   'sum_input_c',     'sum_hidden_c',     approach_c_utility,  False),
    ('Approach E',   'approach_e_traces',   'sum_input_e',     'sum_hidden_e',     approach_e_utility,  False),
    ('Approach F',   'approach_f_traces',   'sum_input_f',     'sum_hidden_f',     approach_f_utility,  False),
    ('Approach G',   'approach_g_traces',   'sum_input_g',     'sum_hidden_g',     approach_g_utility,  False),
    ('Approach H',   'approach_h_traces',   'sum_input_h',     'sum_hidden_h',     approach_h_utility,  False),
]


# ==============================================================================
# Teacher functions
# ==============================================================================

def init_teacher(key):
    """Initialize teacher network with random +/-1 weights."""
    k1, k2 = jax.random.split(key)
    W1 = jax.random.randint(k1, (N_TEACHER_HIDDEN, N_TEACHER_INPUTS), 0, 2).astype(jnp.float32) * 2 - 1
    W2 = jax.random.randint(k2, (1, N_TEACHER_HIDDEN), 0, 2).astype(jnp.float32) * 2 - 1
    return W1, W2


# ==============================================================================
# Scanned training steps
# ==============================================================================

def _train_step_body(model, optimizer, x, y_star, ema, budget,
                     compute_pred_grads, compute_loo):
    """Core training step: compute utilities, update EMA traces, update model."""
    y_hat_arr, _ = model(x)
    y_hat = y_hat_arr.squeeze()
    mse = (y_star - y_hat) ** 2

    # Loss gradients (needed for optimizer, UPGD, and SI)
    loss_grads = eqx.filter_grad(lambda m: (m(x)[0].squeeze() - y_star) ** 2)(model)

    # Pre-update utilities (don't need optimizer updates)
    utilities = {}
    for _, trace_key, _, _, fn, needs_updates in UTILITY_METHODS:
        if not needs_updates:
            utilities[trace_key] = fn(model, x, y_star, y_hat, loss_grads, None)

    # Optimizer update (needed for SI and model update)
    if compute_pred_grads:
        pred_grads = eqx.filter_grad(lambda m: m(x)[0].squeeze())(model)
        updates, new_optimizer = optimizer.with_update((loss_grads, pred_grads), model)
    else:
        updates, new_optimizer = optimizer.with_update(loss_grads, model)

    # Post-update utilities (need optimizer updates)
    for _, trace_key, _, _, fn, needs_updates in UTILITY_METHODS:
        if needs_updates:
            utilities[trace_key] = fn(model, x, y_star, y_hat, loss_grads, updates)

    # True LOO (optional)
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
            # Find the trace_key for this method to look up its utilities
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

        # Extract first-layer step sizes (zeros for SGD, actual for Autostep)
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
# Training loop
# ==============================================================================

def _precompute_data(seed):
    """Pre-compute all (x, y*) pairs with sign drift."""
    key = jax.random.PRNGKey(seed)
    key, teacher_key, data_key = jax.random.split(key, 3)

    W1_teacher, W2_teacher = init_teacher(teacher_key)

    # Sign schedule
    drift_rng = np.random.RandomState(seed + 1)
    signs_np = np.ones(N_TEACHER_INPUTS)
    signs_schedule = np.zeros((N_STEPS, N_TEACHER_INPUTS))
    for step in range(N_STEPS):
        if step > 0 and step % DRIFT_FREQUENCY == 0:
            idx = drift_rng.randint(N_RELEVANT)
            signs_np[idx] *= -1
        signs_schedule[step] = signs_np.copy()
    signs_schedule = jnp.array(signs_schedule)

    # All inputs and targets
    x_all = jax.random.normal(data_key, (N_STEPS, N_STUDENT_INPUTS))
    x_relevant = x_all[:, :N_TEACHER_INPUTS] * signs_schedule
    z_hidden = jax.vmap(lambda xr: W1_teacher @ xr)(x_relevant)
    a_hidden = (z_hidden > 0.0).astype(jnp.float32)
    y_star_all = jax.vmap(lambda a: (W2_teacher @ a).squeeze())(a_hidden)

    return key, x_all, y_star_all


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


def run_experiment(optimizer_name, seed):
    """Run the tracking experiment with the given optimizer."""
    key, x_all, y_star_all = _precompute_data(seed)
    _, model_key = jax.random.split(key)

    model = MLP(
        input_dim=N_STUDENT_INPUTS, output_dim=1, n_layers=2,
        hidden_dim=N_STUDENT_HIDDEN, weight_init_method='lecun_uniform',
        activation='sigmoid', key=model_key,
    )

    is_autostep = optimizer_name == 'autostep'
    if optimizer_name == 'sgd':
        opt = optax.sgd(SGD_LR)
        optimizer = EqxOptimizer(opt, model, name='sgd')
    elif is_autostep:
        opt = optax_idbd(meta_lr=AUTOSTEP_META_LR, init_lr=AUTOSTEP_INIT_LR, autostep=True)
        optimizer = EqxOptimizer(opt, model, name='autostep')
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    ema, budget = _init_ema_budget()

    scan_fn = _make_scan_fn(compute_pred_grads=is_autostep, compute_loo=COMPUTE_TRUE_LOO)
    init_carry = (model, optimizer, ema, budget)

    @eqx.filter_jit
    def run_scan_chunk(carry, data):
        return jax.lax.scan(scan_fn, carry, data)

    n_chunks = N_STEPS // SCAN_CHUNK
    assert N_STEPS % SCAN_CHUNK == 0

    all_outputs = []
    carry = init_carry
    for chunk_idx in trange(n_chunks, desc=optimizer_name, unit="chunk"):
        chunk_slice = slice(chunk_idx * SCAN_CHUNK, (chunk_idx + 1) * SCAN_CHUNK)
        chunk_data = (x_all[chunk_slice], y_star_all[chunk_slice])
        carry, outputs = run_scan_chunk(carry, chunk_data)
        all_outputs.append(jax.tree.map(np.array, outputs))

    # Concatenate outputs: (mse, step_sizes, ema_dict, budget_dict)
    mse_hist = np.concatenate([o[0] for o in all_outputs])
    ss_hist = np.concatenate([o[1] for o in all_outputs])
    ema_hist = {k: np.concatenate([o[2][k] for o in all_outputs]) for k in all_outputs[0][2]}
    budget_hist = {k: np.concatenate([o[3][k] for o in all_outputs]) for k in all_outputs[0][3]}

    return {
        'mse_history': mse_hist,
        'step_size_history': ss_hist if is_autostep else None,
        **ema_hist,
        **budget_hist,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(sgd_results, autostep_results):
    """Generate all figures from the experiment results."""
    # ---- Figure 1: Learning curves + step sizes ----
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))

    window = 500
    for results, name in [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]:
        mse = results['mse_history']
        smoothed = np.convolve(mse, np.ones(window) / window, mode='valid')
        ax1a.plot(smoothed, label=name, linewidth=1.5)
    ax1a.set_xlabel('Step')
    ax1a.set_ylabel('MSE (smoothed)')
    ax1a.set_title('Learning Curves: SGD vs Autostep')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    ss = autostep_results['step_size_history']
    for i in range(N_STUDENT_INPUTS):
        color = 'blue' if i < N_RELEVANT else 'red'
        alpha = 0.5 if i < N_RELEVANT else 0.2
        ax1b.plot(ss[:, i], color=color, alpha=alpha, linewidth=0.8)
    ax1b.plot([], [], color='blue', linewidth=2, label='Relevant (0-4)')
    ax1b.plot([], [], color='red', linewidth=2, label='Irrelevant (5-19)')
    ax1b.set_xlabel('Step')
    ax1b.set_ylabel('Step Size (exp(beta), avg over hidden)')
    ax1b.set_title('Autostep Per-Input Step Sizes')
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(SCRIPT_DIR / 'fig1_learning_curves.png', dpi=150)
    print("Saved fig1_learning_curves.png")

    # ---- Figure 2: Input-level utility grid ----
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
            traces = results[u_key]
            for i in range(N_STUDENT_INPUTS):
                color = 'blue' if i < N_RELEVANT else 'red'
                alpha = 0.5 if i < N_RELEVANT else 0.2
                ax.plot(traces[:, i], color=color, alpha=alpha, linewidth=0.8)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_title(f'{u_name} ({opt_name})')
            ax.set_xlabel('Step')
            ax.set_ylabel('Utility Trace')
            ax.grid(True, alpha=0.3)

    axes[0, -1].plot([], [], color='blue', linewidth=2, label='Relevant (0-4)')
    axes[0, -1].plot([], [], color='red', linewidth=2, label='Irrelevant (5-19)')
    axes[0, -1].legend()

    fig2.tight_layout()
    fig2.savefig(SCRIPT_DIR / 'fig2_utility_traces.png', dpi=150)
    print("Saved fig2_utility_traces.png")

    # ---- Figure 3: Utility budget per method ----
    budget_methods = [(name, ik, hk) for name, _, ik, hk, _, _ in UTILITY_METHODS if ik is not None]
    if COMPUTE_TRUE_LOO:
        budget_methods.append(('True LOO', 'sum_loo', None))
    n_budget_cols = len(budget_methods)
    fig3, axes3 = plt.subplots(2, n_budget_cols, figsize=(6 * n_budget_cols, 9))

    for row, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        for col, (method_name, input_key, hidden_key) in enumerate(budget_methods):
            ax = axes3[row, col]
            ax.plot(results['target_mag'], label='|y*|', linewidth=1.5, color='black')
            ax.plot(results['error_reduced'], label='error reduced', linewidth=1.5, color='tab:green')
            if hidden_key is not None:
                ax.plot(results[hidden_key], label='Σ U_hidden', linewidth=1.5, color='tab:orange')
            ax.plot(results[input_key], label='Σ U_input', linewidth=1.5, color='tab:blue')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_title(f'{method_name} ({opt_name})')
            ax.set_xlabel('Step')
            ax.set_ylabel('EMA Trace')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(SCRIPT_DIR / 'fig3_utility_budget.png', dpi=150)
    print("Saved fig3_utility_budget.png")

    # ---- Figure 4: Separation (Cohen's d) over time ----
    sep_win = 100
    kernel = np.ones(sep_win) / sep_win
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    for col, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        ax = axes4[col]
        for u_key, u_name in zip(utility_keys, utility_names):
            traces = results[u_key]
            rel = np.mean(traces[:, :N_RELEVANT], axis=1)
            irr = np.mean(traces[:, N_RELEVANT:], axis=1)
            rel_roll = np.convolve(rel, kernel, mode='valid')
            irr_roll = np.convolve(irr, kernel, mode='valid')
            rel_var_roll = np.convolve(np.var(traces[:, :N_RELEVANT], axis=1), kernel, mode='valid')
            irr_var_roll = np.convolve(np.var(traces[:, N_RELEVANT:], axis=1), kernel, mode='valid')
            pool_std = np.sqrt((rel_var_roll + irr_var_roll) / 2)
            d_roll = (rel_roll - irr_roll) / (pool_std + 1e-10)
            steps = np.arange(sep_win - 1, sep_win - 1 + len(d_roll))
            ax.plot(steps, d_roll, label=u_name, linewidth=1.2)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='d=1')
        ax.set_title(f'Separation over time ({opt_name})')
        ax.set_xlabel('Step')
        ax.set_ylabel("Cohen's d (rolling {})".format(sep_win))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(SCRIPT_DIR / 'fig4_separation.png', dpi=150)
    print("Saved fig4_separation.png")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("Running multi-layer signed utility experiment...")
    print(f"  Teacher: {N_TEACHER_INPUTS} -> {N_TEACHER_HIDDEN} (sigmoid) -> 1")
    print(f"  Student: {N_STUDENT_INPUTS} -> {N_STUDENT_HIDDEN} (sigmoid) -> 1")
    print(f"  Steps: {N_STEPS}, drift every {DRIFT_FREQUENCY} steps")
    print()

    print("Running SGD...")
    sgd_results = run_experiment('sgd', SEED)

    print("\nRunning Autostep...")
    autostep_results = run_experiment('autostep', SEED)

    print("\nPlotting results...")
    plot_results(sgd_results, autostep_results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    last_5k = slice(-5000, None)

    summary_methods = [(name, tk) for name, tk, _, _, _, _ in UTILITY_METHODS]
    if COMPUTE_TRUE_LOO:
        summary_methods.append(('True LOO', 'loo_traces'))

    for name, results in [('SGD', sgd_results), ('Autostep', autostep_results)]:
        mse = np.mean(results['mse_history'][last_5k])
        print(f"\n{name} -- Final MSE (last 5k): {mse:.4f}")
        for u_name, u_key in summary_methods:
            traces = results[u_key]
            last_traces = traces[last_5k]
            rel_mean = np.mean(last_traces[:, :N_RELEVANT])
            irr_mean = np.mean(last_traces[:, N_RELEVANT:])
            gap = rel_mean - irr_mean
            rel_std = np.std(last_traces[:, :N_RELEVANT])
            irr_std = np.std(last_traces[:, N_RELEVANT:])
            pooled_std = np.sqrt((rel_std**2 + irr_std**2) / 2)
            d = gap / (pooled_std + 1e-10)
            # Speed of separation: first step where rolling d > 1.0
            win = 100
            if len(traces) >= win:
                kernel = np.ones(win) / win
                rel_roll = np.convolve(np.mean(traces[:, :N_RELEVANT], axis=1), kernel, mode='valid')
                irr_roll = np.convolve(np.mean(traces[:, N_RELEVANT:], axis=1), kernel, mode='valid')
                rel_std_roll = np.convolve(np.var(traces[:, :N_RELEVANT], axis=1), kernel, mode='valid') ** 0.5
                irr_std_roll = np.convolve(np.var(traces[:, N_RELEVANT:], axis=1), kernel, mode='valid') ** 0.5
                pool_roll = np.sqrt((rel_std_roll**2 + irr_std_roll**2) / 2)
                d_roll = (rel_roll - irr_roll) / (pool_roll + 1e-10)
                sep_idx = np.where(d_roll > 1.0)[0]
                sep_step = int(sep_idx[0]) + win if len(sep_idx) > 0 else None
            else:
                sep_step = None
            sep_str = f"{sep_step}" if sep_step is not None else "never"
            print(f"  {u_name}: rel={rel_mean:.5f}, irr={irr_mean:.5f}, "
                  f"gap={gap:.5f}, d={d:.2f}, d>1 @ step {sep_str}")

    print("\nDone.")
