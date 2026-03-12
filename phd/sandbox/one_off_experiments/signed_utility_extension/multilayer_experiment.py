"""
Multi-Layer Signed Utility Experiment
=====================================

Tests whether three candidate algorithms for extending signed utility to multi-layer
networks can correctly identify relevant vs irrelevant input features when utility must
propagate backward through a hidden layer with nonlinear activations.

Problem Setup
-------------
A nonlinear teacher network (5 -> 4 hidden sigmoid -> 1 linear output) generates targets.
A larger student network (20 -> 16 hidden sigmoid -> 1 linear output) learns online. Only
inputs 0-4 carry signal; inputs 5-19 are irrelevant noise. Teacher input weight signs flip
every 20 steps to create nonstationarity.

The nonlinear teacher forces the hidden layer to do real computational work. Sign drifts
create transient periods where some hidden units may be harmful (their learned representation
is temporarily wrong), which is where Approach B should outperform A.

Utility Methods
---------------
1. **Contribution Utility** (CBP analog): |x_i| * sum_j |W1[j,i]|. Always positive.
   Measures feature activity weighted by connection strength, but cannot distinguish helpful
   from harmful features.

2. **Approach A** (Proportional Redistribution): Computes signed utility at the output layer,
   then distributes each hidden unit's utility to its input children proportionally by
   |W1[j,k]*x_k|. All children inherit the parent's sign -- if a parent is harmful, all its
   children appear harmful regardless of their actual role.

3. **Approach B** (Recursive Signed Utility): Applies the signed utility formula recursively
   at each layer using a pseudo-error e_j = |U_j| / f'(z_j). Correctly distinguishes children
   that cause harm from those that fight it. Should produce the cleanest separation between
   relevant and irrelevant inputs.

Hypotheses
----------
- All three methods should assign higher utility to relevant inputs (0-4) than irrelevant
  ones (5-19) on average.
- Approach B should show the cleanest signed separation, with relevant inputs consistently
  positive and irrelevant inputs near zero.
- During sign-drift transients, Approach A may incorrectly assign negative utility to helpful
  children of temporarily-harmful hidden units; Approach B should handle this correctly.
- Autostep should adapt step sizes to give larger learning rates to relevant inputs.

Future Experiments
------------------
Experiment 2 -- XOR + Distractors:
  Target: XOR(x_1 > 0, x_2 > 0), embedded in 20-dim input space. Requires nonlinearity
  to solve; utility must credit hidden units that compute XOR. Ground truth known at both
  input level (x_1, x_2 matter) and hidden level.

Experiment 3 -- Planted Dead/Harmful Units:
  Construct network with known helpful, useless, and harmful hidden units (freeze some).
  Most controlled test of Approach B's sign-correctness advantage over A. Directly tests
  the harmful-feature case from the worked examples in multilayer_signed_utility.md.
"""

from functools import partial

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

SCRIPT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# Constants
# ==============================================================================
N_TEACHER_INPUTS = 5
N_TEACHER_HIDDEN = 4
N_STUDENT_INPUTS = 20
N_STUDENT_HIDDEN = 16
N_RELEVANT = 5
N_STEPS = 100_000
DRIFT_FREQUENCY = 50
TRACE_DECAY = 0.999
SEED = np.random.randint(0, 1000000)
SCAN_CHUNK = 5000  # Steps per scan chunk (for progress updates)

# Optimizer hyperparameters
SGD_LR = 0.01
AUTOSTEP_INIT_LR = 1.0
AUTOSTEP_META_LR = 0.005

# ==============================================================================
# Teacher functions
# ==============================================================================

def init_teacher(key):
    """Initialize teacher network as plain JAX arrays.

    Returns (W1, W2) where:
      W1: (N_TEACHER_HIDDEN, N_TEACHER_INPUTS) -- hidden layer weights
      W2: (1, N_TEACHER_HIDDEN) -- output layer weights
    """
    k1, k2 = jax.random.split(key)
    W1 = jax.random.randint(k1, (N_TEACHER_HIDDEN, N_TEACHER_INPUTS), 0, 2).astype(jnp.float32) * 2 - 1
    W2 = jax.random.randint(k2, (1, N_TEACHER_HIDDEN), 0, 2).astype(jnp.float32) * 2 - 1
    return W1, W2


def teacher_forward(W1, W2, x_relevant):
    """Forward pass through teacher: linear -> LTU -> linear."""
    z = W1 @ x_relevant
    a = (z > 0.0).astype(jnp.float32)  # linear threshold unit
    return (W2 @ a).squeeze()


def generate_step(key, W1, W2, signs):
    """Generate one (x, y*) pair from the teacher with current signs.

    Args:
        key: PRNG key
        W1, W2: teacher weights
        signs: current sign array for teacher inputs, shape (N_TEACHER_INPUTS,)

    Returns:
        x_full: full student input, shape (N_STUDENT_INPUTS,)
        y_star: teacher target, scalar
    """
    x_full = jax.random.normal(key, (N_STUDENT_INPUTS,))
    x_relevant = x_full[:N_TEACHER_INPUTS] * signs
    y_star = teacher_forward(W1, W2, x_relevant)
    return x_full, y_star


# ==============================================================================
# Utility functions
# ==============================================================================

def compute_contribution_utility(model, x):
    """Contribution utility (CBP analog): |x_i| * sum_j |W1[j,i]|.

    Args:
        model: MLP student model
        x: input vector, shape (N_STUDENT_INPUTS,)

    Returns:
        utility per input, shape (N_STUDENT_INPUTS,)
    """
    W1 = model.layers[0].weight  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    return jnp.abs(x) * jnp.sum(jnp.abs(W1), axis=0)


def compute_utility_approach_a(model, x, y_star, y_hat):
    """Approach A: Proportional Redistribution.

    Output layer: U_j = |e + c_j| - |e| where e = y* - y_hat, c_j = w_out[j] * a_j
    Propagate: U_{k<-j} = U_j * |W1[j,k]*x_k| / sum_m |W1[j,m]*x_m|
    Per input: U_k = sum_j U_{k<-j}

    Args:
        model: MLP student model
        x: input vector, shape (N_STUDENT_INPUTS,)
        y_star: teacher target, scalar
        y_hat: student prediction, scalar

    Returns:
        utility per input, shape (N_STUDENT_INPUTS,)
    """
    W1 = model.layers[0].weight  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    w_out = model.layers[1].weight.squeeze(0)  # (N_STUDENT_HIDDEN,)

    # Hidden activations
    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)  # (N_STUDENT_HIDDEN,)

    # Output-layer signed utility per hidden unit
    e = y_star - y_hat  # scalar error
    c_j = w_out * a_hidden  # (N_STUDENT_HIDDEN,)
    U_hidden = jnp.abs(e + c_j) - jnp.abs(e)  # (N_STUDENT_HIDDEN,)

    # Propagate to inputs: proportional redistribution
    # contrib[j, k] = |W1[j,k] * x[k]|
    contrib = jnp.abs(W1 * x[None, :])  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    contrib_sum = jnp.sum(contrib, axis=1, keepdims=True)  # (N_STUDENT_HIDDEN, 1)
    # Avoid division by zero
    contrib_sum = jnp.maximum(contrib_sum, 1e-10)
    frac = contrib / contrib_sum  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)

    # U_{k<-j} = U_j * frac[j, k], then sum over j
    U_input = jnp.sum(U_hidden[:, None] * frac, axis=0)  # (N_STUDENT_INPUTS,)
    return U_input


def compute_utility_approach_b(model, x, y_star, y_hat):
    """Approach B: Recursive Signed Utility.

    Output layer: U_j = |e + c_j| - |e|
    Pseudo-error: e_j = |U_j| / f'(z_j) where f'(z) = sigmoid(z)*(1-sigmoid(z))
    Raw scores: s_{k->j} = |e_j + W1[j,k]*x_k| - |e_j|
    Normalize: U_{k<-j} = s_{k->j} * |U_j| / sum_m |s_{m->j}|
    Per input: U_k = sum_j U_{k<-j}

    Args:
        model: MLP student model
        x: input vector, shape (N_STUDENT_INPUTS,)
        y_star: teacher target, scalar
        y_hat: student prediction, scalar

    Returns:
        (U_input, U_hidden) where:
          U_input: utility per input, shape (N_STUDENT_INPUTS,)
          U_hidden: utility per hidden unit, shape (N_STUDENT_HIDDEN,)
    """
    W1 = model.layers[0].weight  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    w_out = model.layers[1].weight.squeeze(0)  # (N_STUDENT_HIDDEN,)

    # Hidden pre-activations and activations
    z_hidden = W1 @ x  # (N_STUDENT_HIDDEN,)
    a_hidden = jax.nn.sigmoid(z_hidden)  # (N_STUDENT_HIDDEN,)

    # Output-layer signed utility per hidden unit
    e = y_star - y_hat
    c_j = w_out * a_hidden
    U_hidden = jnp.abs(e + c_j) - jnp.abs(e)  # (N_STUDENT_HIDDEN,)

    # Sigmoid derivative: f'(z) = a*(1-a)
    f_prime = a_hidden * (1.0 - a_hidden)  # (N_STUDENT_HIDDEN,)
    # Clip derivative away from zero to avoid huge pseudo-errors
    f_prime_safe = jnp.maximum(f_prime, 1e-6)

    # Pseudo-error per hidden unit (always positive)
    e_j = jnp.abs(U_hidden) / f_prime_safe  # (N_STUDENT_HIDDEN,)

    # Raw signed utility scores: s_{k->j} = |e_j + W1[j,k]*x_k| - |e_j|
    contributions = W1 * x[None, :]  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])
    # shape: (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)

    # Normalize: U_{k<-j} = s_{k->j} * |U_j| / sum_m |s_{m->j}|
    s_abs_sum = jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True)  # (N_STUDENT_HIDDEN, 1)
    s_abs_sum = jnp.maximum(s_abs_sum, 1e-10)
    U_from_j = s_raw * jnp.abs(U_hidden[:, None]) / s_abs_sum
    # shape: (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)

    # Per input: sum over hidden units
    U_input = jnp.sum(U_from_j, axis=0)  # (N_STUDENT_INPUTS,)
    return U_input, U_hidden


# ==============================================================================
# Scanned training steps
# ==============================================================================

def _train_step_body(model, optimizer, x, y_star,
                     ema_c, ema_a, ema_b,
                     ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced,
                     compute_pred_grads):
    """Core training step logic, shared by SGD and Autostep scan bodies."""
    # Forward pass
    y_hat_arr, _ = model(x)
    y_hat = y_hat_arr.squeeze()
    mse = (y_star - y_hat) ** 2

    # Compute utilities
    u_contribution = compute_contribution_utility(model, x)
    u_approach_a = compute_utility_approach_a(model, x, y_star, y_hat)
    u_approach_b, u_hidden_b = compute_utility_approach_b(model, x, y_star, y_hat)

    # Update EMA traces
    ema_c = TRACE_DECAY * ema_c + (1 - TRACE_DECAY) * u_contribution
    ema_a = TRACE_DECAY * ema_a + (1 - TRACE_DECAY) * u_approach_a
    ema_b = TRACE_DECAY * ema_b + (1 - TRACE_DECAY) * u_approach_b

    # Budget traces: |y*|, error reduced, sum of input/hidden utilities
    error_reduced = jnp.abs(y_star) - jnp.abs(y_star - y_hat)
    ema_target_mag = TRACE_DECAY * ema_target_mag + (1 - TRACE_DECAY) * jnp.abs(y_star)
    ema_sum_input_b = TRACE_DECAY * ema_sum_input_b + (1 - TRACE_DECAY) * jnp.sum(u_approach_b)
    ema_sum_hidden_b = TRACE_DECAY * ema_sum_hidden_b + (1 - TRACE_DECAY) * jnp.sum(u_hidden_b)
    ema_error_reduced = TRACE_DECAY * ema_error_reduced + (1 - TRACE_DECAY) * error_reduced

    # Compute gradients and update
    loss_grads = eqx.filter_grad(lambda m: (m(x)[0].squeeze() - y_star) ** 2)(model)
    if compute_pred_grads:
        pred_grads = eqx.filter_grad(lambda m: m(x)[0].squeeze())(model)
        updates, new_optimizer = optimizer.with_update((loss_grads, pred_grads), model)
    else:
        updates, new_optimizer = optimizer.with_update(loss_grads, model)
    new_model = eqx.apply_updates(model, updates)

    return (new_model, new_optimizer, mse,
            ema_c, ema_a, ema_b,
            ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced)


def _make_scan_fn(compute_pred_grads):
    """Build a scan body for either SGD or Autostep."""
    def scan_fn(carry, step_data):
        (model, optimizer, ema_c, ema_a, ema_b,
         ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced) = carry
        x, y_star = step_data

        (model, optimizer, mse, ema_c, ema_a, ema_b,
         ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced) = _train_step_body(
            model, optimizer, x, y_star,
            ema_c, ema_a, ema_b,
            ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced,
            compute_pred_grads)

        # Extract first-layer step sizes (zeros for SGD, actual for Autostep)
        if compute_pred_grads:
            beta_leaves = jax.tree.leaves(optimizer.state.beta)
            step_sizes = jnp.exp(beta_leaves[0]).mean(axis=0)
        else:
            step_sizes = jnp.zeros(N_STUDENT_INPUTS)

        carry = (model, optimizer, ema_c, ema_a, ema_b,
                 ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced)
        outputs = (mse, ema_c, ema_a, ema_b, step_sizes,
                   ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced)
        return carry, outputs

    return scan_fn


# ==============================================================================
# Training loop
# ==============================================================================

def _precompute_data(seed):
    """Pre-compute all (x, y*) pairs with sign drift using numpy/JAX."""
    key = jax.random.PRNGKey(seed)
    key, teacher_key, data_key = jax.random.split(key, 3)

    # Teacher
    W1_teacher, W2_teacher = init_teacher(teacher_key)

    # Pre-compute sign schedule
    drift_rng = np.random.RandomState(seed + 1)
    signs_np = np.ones(N_TEACHER_INPUTS)
    signs_schedule = np.zeros((N_STEPS, N_TEACHER_INPUTS))
    for step in range(N_STEPS):
        if step > 0 and step % DRIFT_FREQUENCY == 0:
            idx = drift_rng.randint(N_RELEVANT)
            signs_np[idx] *= -1
        signs_schedule[step] = signs_np.copy()
    signs_schedule = jnp.array(signs_schedule)

    # Pre-generate all inputs
    x_all = jax.random.normal(data_key, (N_STEPS, N_STUDENT_INPUTS))

    # Vectorized teacher forward: apply signs then compute targets
    x_relevant = x_all[:, :N_TEACHER_INPUTS] * signs_schedule  # (N_STEPS, 5)
    z_hidden = jax.vmap(lambda xr: W1_teacher @ xr)(x_relevant)  # (N_STEPS, 4)
    a_hidden = (z_hidden > 0.0).astype(jnp.float32)  # LTU, (N_STEPS, 4)
    y_star_all = jax.vmap(lambda a: (W2_teacher @ a).squeeze())(a_hidden)  # (N_STEPS,)

    return key, x_all, y_star_all


def run_experiment(optimizer_name, seed):
    """Run the tracking experiment with the given optimizer.

    Args:
        optimizer_name: 'sgd' or 'autostep'
        seed: random seed

    Returns:
        dict with keys: mse_history, contribution_traces, approach_a_traces,
        approach_b_traces, step_size_history (None for SGD)
    """
    key, x_all, y_star_all = _precompute_data(seed)
    _, model_key = jax.random.split(key)

    # Initialize student
    model = MLP(
        input_dim=N_STUDENT_INPUTS,
        output_dim=1,
        n_layers=2,
        hidden_dim=N_STUDENT_HIDDEN,
        weight_init_method='lecun_uniform',
        activation='sigmoid',
        key=model_key,
    )

    # Initialize optimizer
    is_autostep = optimizer_name == 'autostep'
    if optimizer_name == 'sgd':
        opt = optax.sgd(SGD_LR)
        optimizer = EqxOptimizer(opt, model, name='sgd')
    elif optimizer_name == 'autostep':
        opt = optax_idbd(
            meta_lr=AUTOSTEP_META_LR,
            init_lr=AUTOSTEP_INIT_LR,
            autostep=True,
        )
        optimizer = EqxOptimizer(opt, model, name='autostep')
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # EMA accumulators
    ema_c = jnp.zeros(N_STUDENT_INPUTS)
    ema_a = jnp.zeros(N_STUDENT_INPUTS)
    ema_b = jnp.zeros(N_STUDENT_INPUTS)
    ema_target_mag = jnp.float32(0.0)
    ema_sum_input_b = jnp.float32(0.0)
    ema_sum_hidden_b = jnp.float32(0.0)
    ema_error_reduced = jnp.float32(0.0)

    # Build and run the scan
    scan_fn = _make_scan_fn(compute_pred_grads=is_autostep)
    init_carry = (model, optimizer, ema_c, ema_a, ema_b,
                  ema_target_mag, ema_sum_input_b, ema_sum_hidden_b, ema_error_reduced)
    step_data = (x_all, y_star_all)

    @eqx.filter_jit
    def run_scan_chunk(carry, data):
        return jax.lax.scan(scan_fn, carry, data)

    # Run in chunks for progress updates
    n_chunks = N_STEPS // SCAN_CHUNK
    assert N_STEPS % SCAN_CHUNK == 0, f"N_STEPS ({N_STEPS}) must be divisible by SCAN_CHUNK ({SCAN_CHUNK})"

    all_outputs = []
    carry = init_carry
    for chunk_idx in trange(n_chunks, desc=optimizer_name, unit="chunk"):
        chunk_slice = slice(chunk_idx * SCAN_CHUNK, (chunk_idx + 1) * SCAN_CHUNK)
        chunk_data = (x_all[chunk_slice], y_star_all[chunk_slice])
        carry, outputs = run_scan_chunk(carry, chunk_data)
        all_outputs.append(jax.tree.map(np.array, outputs))

    # Concatenate chunk outputs
    mse_hist = np.concatenate([o[0] for o in all_outputs])
    contrib_hist = np.concatenate([o[1] for o in all_outputs])
    app_a_hist = np.concatenate([o[2] for o in all_outputs])
    app_b_hist = np.concatenate([o[3] for o in all_outputs])
    ss_hist = np.concatenate([o[4] for o in all_outputs])
    target_mag_hist = np.concatenate([o[5] for o in all_outputs])
    sum_input_b_hist = np.concatenate([o[6] for o in all_outputs])
    sum_hidden_b_hist = np.concatenate([o[7] for o in all_outputs])
    error_reduced_hist = np.concatenate([o[8] for o in all_outputs])

    return {
        'mse_history': mse_hist,
        'contribution_traces': contrib_hist,
        'approach_a_traces': app_a_hist,
        'approach_b_traces': app_b_hist,
        'step_size_history': ss_hist if is_autostep else None,
        'target_mag': target_mag_hist,
        'sum_input_b': sum_input_b_hist,
        'sum_hidden_b': sum_hidden_b_hist,
        'error_reduced': error_reduced_hist,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(sgd_results, autostep_results):
    """Generate all figures from the experiment results."""
    # ---- Figure 1: Learning curves + step sizes ----
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))

    # 1a: Smoothed MSE
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

    # 1b: Per-input step sizes (Autostep only)
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
    fig1.savefig(
        SCRIPT_DIR / 'fig1_learning_curves.png',
        dpi=150,
    )
    print("Saved fig1_learning_curves.png")

    # ---- Figure 2: Input-level utility (2x3 grid) ----
    fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
    utility_names = ['Contribution', 'Approach A', 'Approach B']
    utility_keys = ['contribution_traces', 'approach_a_traces', 'approach_b_traces']

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

    # Legend on last subplot
    axes[0, 2].plot([], [], color='blue', linewidth=2, label='Relevant (0-4)')
    axes[0, 2].plot([], [], color='red', linewidth=2, label='Irrelevant (5-19)')
    axes[0, 2].legend()

    fig2.tight_layout()
    fig2.savefig(
        SCRIPT_DIR / 'fig2_utility_traces.png',
        dpi=150,
    )
    print("Saved fig2_utility_traces.png")

    # ---- Figure 3: Utility budget (target mag vs sum of utilities) ----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, opt_name) in zip(
        [ax3a, ax3b], [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        ax.plot(results['target_mag'], label='|y*| (target magnitude)', linewidth=1.5, color='black')
        ax.plot(results['error_reduced'], label='|y*| - |error| (error reduced)', linewidth=1.5, color='tab:green')
        ax.plot(results['sum_hidden_b'], label='Σ U_hidden (Approach B)', linewidth=1.5, color='tab:orange')
        ax.plot(results['sum_input_b'], label='Σ U_input (Approach B)', linewidth=1.5, color='tab:blue')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title(f'Utility Budget ({opt_name})')
        ax.set_xlabel('Step')
        ax.set_ylabel('EMA Trace')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(SCRIPT_DIR / 'fig3_utility_budget.png', dpi=150)
    print("Saved fig3_utility_budget.png")


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
    for name, results in [('SGD', sgd_results), ('Autostep', autostep_results)]:
        mse = np.mean(results['mse_history'][last_5k])
        print(f"\n{name} -- Final MSE (last 5k): {mse:.4f}")
        for u_name, u_key in [
            ('Contribution', 'contribution_traces'),
            ('Approach A', 'approach_a_traces'),
            ('Approach B', 'approach_b_traces'),
        ]:
            traces = results[u_key][last_5k]
            rel_mean = np.mean(traces[:, :N_RELEVANT])
            irr_mean = np.mean(traces[:, N_RELEVANT:])
            print(f"  {u_name}: relevant={rel_mean:.5f}, irrelevant={irr_mean:.5f}, "
                  f"ratio={rel_mean / (abs(irr_mean) + 1e-10):.1f}x")

    print("\nDone.")
