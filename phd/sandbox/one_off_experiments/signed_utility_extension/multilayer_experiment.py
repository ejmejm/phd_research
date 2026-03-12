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

4. **UPGD** (first-order Taylor, Elsayed & Mahmood 2023): -(∂L/∂a_i) · a_i.
   First-order approximation of the true leave-one-out utility. Signed, but uses
   local gradient information rather than the counterfactual |e+c|-|e| formulation.

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
        (U_input, U_hidden) where:
          U_input: utility per input, shape (N_STUDENT_INPUTS,)
          U_hidden: utility per hidden unit, shape (N_STUDENT_HIDDEN,)
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
    return U_input, U_hidden


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


ALIGNMENT_THRESHOLD = 0.1  # fallback to Approach A when |Σs| / Σ|s| < threshold


# ==============================================================================
# Approach E: Calibrated Pseudo-Error (analytical, no iterations)
# ==============================================================================

def _find_e_nonneg(contributions, target):
    """Find e >= 0 such that g(e) = Σ(|e + c_k| - e) = target.

    g is piecewise linear with breakpoints at |c_k| for each negative c_k.
    Between breakpoints, g has constant slope, so we solve a linear equation
    in the correct segment. No iterations needed.

    g(0) = Σ|c_k|, g(∞) → Σc_k = z.
    Feasible when z <= target <= Σ|c_k|.

    Args:
        contributions: shape (K,)
        target: scalar

    Returns:
        (e, feasible) where e >= 0 is the solution and feasible is a bool.
    """
    K = contributions.shape[0]

    P = jnp.sum(jnp.maximum(contributions, 0.0))    # sum of positive contributions
    neg_mags = jnp.maximum(-contributions, 0.0)       # |c_k| for negatives, 0 for positives
    N = jnp.sum(neg_mags)                              # sum of negative magnitudes
    z = P - N                                          # Σc_k

    # Feasibility: z <= target <= P + N
    feasible_range = (target >= z - 1e-6) & (target <= P + N + 1e-6)

    # Sort negative magnitudes ascending, padding positives with large sentinel
    padded = jnp.where(contributions < 0, -contributions, jnp.float32(1e30))
    sorted_b = jnp.sort(padded)
    # sorted_b = [b_0, b_1, ..., b_{m-1}, 1e30, ..., 1e30]
    # where b_i are actual negative magnitudes in ascending order

    # Cumulative sums (clip sentinel values to avoid overflow)
    cum = jnp.cumsum(jnp.minimum(sorted_b, 1e20))
    B = jnp.concatenate([jnp.array([0.0]), cum[:-1]])  # B[i] = sum of first i breakpoints

    # Count of actual negative contributions
    m = jnp.sum((contributions < 0).astype(jnp.float32))

    # In segment i (i breakpoints crossed), g(e) = P + N - 2*B[i] - 2*(m-i)*e
    # Solving: e = (P + N - 2*B[i] - target) / (2*(m - i))
    remaining = jnp.maximum(m - jnp.arange(K, dtype=jnp.float32), 0.0)
    numer = P + N - 2.0 * B - target
    safe_denom = jnp.maximum(2.0 * remaining, 1.0)
    e_cand = numer / safe_denom

    # Segment i covers [left[i], right[i])
    left_bounds = jnp.concatenate([jnp.array([0.0]), sorted_b[:-1]])
    right_bounds = sorted_b

    eps = 1e-6
    valid = ((e_cand >= left_bounds - eps) &
             (e_cand <= right_bounds + eps) &
             (e_cand >= -eps) &
             (remaining > 0.5))

    # Select first valid segment (argmax on bools returns first True, 0 if none)
    first_idx = jnp.argmax(valid)
    e_segment = jnp.maximum(e_cand[first_idx], 0.0)
    segment_ok = valid[first_idx]

    # Edge case: target ≈ z (asymptotic, all negatives crossed, g = z)
    asymptotic_ok = (jnp.abs(target - z) < eps)
    last_bp_idx = jnp.maximum(m.astype(jnp.int32) - 1, 0)
    e_asymptotic = jnp.where(m > 0.5, sorted_b[last_bp_idx] + 1.0, 0.0)

    e_out = jnp.where(segment_ok, e_segment,
                       jnp.where(asymptotic_ok, e_asymptotic, 0.0))
    feasible = (segment_ok | asymptotic_ok) & feasible_range

    return e_out, feasible


def _calibrate_pseudo_error(contributions, target):
    """Find e such that g(e) = Σ(|e + c_k| - |e|) = target.

    For target >= z = Σc_k: search e >= 0 (g decreases from Σ|c_k| to z).
    For target < z: search e <= 0 by negating contributions
    (g_c(-d) = g_{-c}(d), so find d >= 0 for negated c, return e = -d).

    Args:
        contributions: shape (K,)
        target: scalar

    Returns:
        (e, feasible) where feasible indicates whether calibration succeeded.
    """
    z = jnp.sum(contributions)
    search_positive = (target >= z)

    # When searching e <= 0, negate contributions: g_c(-d) = g_{-c}(d)
    c = jnp.where(search_positive, contributions, -contributions)
    d, feas = _find_e_nonneg(c, target)
    e = jnp.where(search_positive, d, -d)
    return e, feas


def compute_utility_approach_e(model, x, y_star, y_hat):
    """Approach E: Calibrated Pseudo-Error.

    Instead of computing raw scores and normalizing (which blows up when children
    cancel), find the pseudo-error e such that the raw scores naturally sum to the
    target utility. No normalization needed.

    g(e) = Σ(|e + c_k| - |e|) is piecewise linear and monotonic, so the solution
    is found analytically by locating the correct linear segment.

    Output layer: find e_out such that Σ U_j = E_reduced.
    Hidden layer: for each unit j, find e_j such that Σ U_{k←j} = U_j.
    Falls back to Approach A when calibration is infeasible.

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
    W1 = model.layers[0].weight       # (H, N)
    w_out = model.layers[1].weight.squeeze(0)  # (H,)

    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)

    # === Output layer: calibrate e_out ===
    e_true = y_star - y_hat
    c_out = w_out * a_hidden                      # (H,)
    E_reduced = jnp.abs(y_star) - jnp.abs(e_true)

    e_out, out_feasible = _calibrate_pseudo_error(c_out, E_reduced)
    # When feasible (always, by triangle inequality): raw scores sum to E_reduced
    # When infeasible (shouldn't happen): fall back to standard LOO + rescaling
    U_hidden_calib = jnp.abs(e_out + c_out) - jnp.abs(e_out)

    # Fallback: standard LOO rescaled (same as C's output layer)
    u_raw = jnp.abs(e_true + c_out) - jnp.abs(e_true)
    u_raw_sum = jnp.sum(u_raw)
    scale = jnp.where(jnp.abs(u_raw_sum) > 1e-10, E_reduced / u_raw_sum, 1.0)
    U_hidden_fallback = u_raw * scale

    U_hidden = jnp.where(out_feasible, U_hidden_calib, U_hidden_fallback)

    # === Hidden layer: calibrate e_j for each hidden unit ===
    contributions = W1 * x[None, :]   # (H, N)

    # vmap calibration over hidden units
    e_j, feasible_j = jax.vmap(_calibrate_pseudo_error)(contributions, U_hidden)
    # e_j: (H,), feasible_j: (H,)

    # Raw scores from calibrated pseudo-errors (no normalization)
    U_from_j_calib = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])
    # shape: (H, N)

    # Fallback to Approach A for infeasible units
    abs_contrib = jnp.abs(contributions)
    abs_contrib_sum = jnp.maximum(jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10)
    U_from_j_fallback = U_hidden[:, None] * abs_contrib / abs_contrib_sum

    use_calib = feasible_j[:, None].astype(jnp.float32)
    U_from_j = use_calib * U_from_j_calib + (1.0 - use_calib) * U_from_j_fallback

    U_input = jnp.sum(U_from_j, axis=0)  # (N,)
    return U_input, U_hidden


def compute_utility_upgd(model, x, loss_grads):
    """UPGD first-order feature utility (Elsayed & Mahmood 2023).

    u_j ≈ -(∂L/∂a_j) · a_j

    Uses the already-computed loss gradients (w.r.t. weights) to extract
    activation gradients without an extra backward pass:
      grad_W1 = outer(∂L/∂z, x), so ∂L/∂z = grad_W1 @ x / ||x||²
      ∂L/∂x = W1.T @ ∂L/∂z
      ∂L/∂a = ∂L/∂z / f'(z)

    Args:
        model: MLP student model
        x: input vector, shape (N_STUDENT_INPUTS,)
        loss_grads: gradient of loss w.r.t. model parameters (same pytree as model)

    Returns:
        (U_input, U_hidden) where:
          U_input: utility per input, shape (N_STUDENT_INPUTS,)
          U_hidden: utility per hidden unit, shape (N_STUDENT_HIDDEN,)
    """
    W1 = model.layers[0].weight          # (H, N)
    grad_W1 = loss_grads.layers[0].weight  # (H, N) = outer(∂L/∂z, x)

    # Extract ∂L/∂z from weight gradient: grad_W1 = outer(dL_dz, x)
    x_norm_sq = jnp.sum(x * x)
    dL_dz = (grad_W1 @ x) / jnp.maximum(x_norm_sq, 1e-10)  # (H,)

    # Input utility: -(∂L/∂x_i) · x_i
    dL_dx = W1.T @ dL_dz  # (N,)
    U_input = -dL_dx * x  # (N,)

    # Hidden utility: -(∂L/∂a_j) · a_j
    a_hidden = jax.nn.sigmoid(W1 @ x)  # (H,)
    f_prime = a_hidden * (1.0 - a_hidden)
    dL_da = dL_dz / jnp.maximum(f_prime, 1e-6)  # (H,)
    U_hidden = -dL_da * a_hidden  # (H,)

    return U_input, U_hidden


def compute_utility_approach_c(model, x, y_star, y_hat):
    """Approach C: Signed-Conserving Redistribution.

    Same raw scores as Approach B, but normalized to preserve signed sum:
      U_{k<-j} = s_{k->j} * U_j / Σ_m s_{m->j}
    Falls back to Approach A when children nearly cancel (Σs ≈ 0).
    Output-layer utilities are rescaled so Σ U_j = error_reduced.

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

    # Output-layer: raw leave-one-out utilities, then rescale to sum to error_reduced
    e = y_star - y_hat
    c_j = w_out * a_hidden  # (N_STUDENT_HIDDEN,)
    u_raw = jnp.abs(e + c_j) - jnp.abs(e)  # (N_STUDENT_HIDDEN,)
    error_reduced = jnp.abs(y_star) - jnp.abs(e)
    u_raw_sum = jnp.sum(u_raw)
    # Rescale: U_j = u_j * E / Σu_j. When Σu_j ≈ 0, error_reduced ≈ 0 too, so just use raw.
    scale = jnp.where(jnp.abs(u_raw_sum) > 1e-10, error_reduced / u_raw_sum, 1.0)
    U_hidden = u_raw * scale  # (N_STUDENT_HIDDEN,)

    # Sigmoid derivative
    f_prime = a_hidden * (1.0 - a_hidden)
    f_prime_safe = jnp.maximum(f_prime, 1e-6)

    # Pseudo-error per hidden unit (always positive, same as B)
    e_j = jnp.abs(U_hidden) / f_prime_safe  # (N_STUDENT_HIDDEN,)

    # Raw signed utility scores (same as B)
    contributions = W1 * x[None, :]  # (N_STUDENT_HIDDEN, N_STUDENT_INPUTS)
    s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])

    # Signed normalization: U_{k<-j} = s_k * U_j / Σ s_k
    s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)  # (N_STUDENT_HIDDEN, 1)
    s_abs_sum = jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True)

    # Alignment ratio: how well-conditioned is the signed normalization?
    alignment = jnp.abs(s_signed_sum) / jnp.maximum(s_abs_sum, 1e-10)

    # Signed normalization (Approach C)
    s_signed_safe = jnp.where(jnp.abs(s_signed_sum) > 1e-10, s_signed_sum, 1.0)
    U_signed = s_raw * U_hidden[:, None] / s_signed_safe

    # Approach A fallback: U_{k<-j} = U_j * |c_k| / Σ|c_m|
    abs_contrib = jnp.abs(contributions)
    abs_contrib_sum = jnp.maximum(jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10)
    U_fallback = U_hidden[:, None] * abs_contrib / abs_contrib_sum

    # Blend: use signed normalization when well-conditioned, A otherwise
    use_signed = (alignment > ALIGNMENT_THRESHOLD).astype(jnp.float32)
    U_from_j = use_signed * U_signed + (1.0 - use_signed) * U_fallback

    U_input = jnp.sum(U_from_j, axis=0)  # (N_STUDENT_INPUTS,)
    return U_input, U_hidden


def compute_true_loo_utility(model, x, y_star):
    """True leave-one-out utility at the input level via extra forward passes.

    For each input k, zero it out, do a full forward pass, and compute:
      u_k = |y* - y_hat_without_k| - |y* - y_hat|

    Uses vmap over inputs to parallelize the 20 forward passes.

    Args:
        model: MLP student model
        x: input vector, shape (N_STUDENT_INPUTS,)
        y_star: teacher target, scalar

    Returns:
        utility per input, shape (N_STUDENT_INPUTS,)
    """
    y_hat = model(x)[0].squeeze()
    base_error = jnp.abs(y_star - y_hat)

    # Build all masked inputs at once: (N_STUDENT_INPUTS, N_STUDENT_INPUTS)
    # Row k has x with element k zeroed out
    mask = 1.0 - jnp.eye(N_STUDENT_INPUTS)  # (N, N)
    x_masked_all = mask * x[None, :]  # broadcast: (N, N)

    # vmap the forward pass over the batch of masked inputs
    def forward_one(x_masked):
        return model(x_masked)[0].squeeze()

    y_hat_masked = jax.vmap(forward_one)(x_masked_all)  # (N_STUDENT_INPUTS,)
    errors_without = jnp.abs(y_star - y_hat_masked)  # (N_STUDENT_INPUTS,)

    return errors_without - base_error


# ==============================================================================
# Scanned training steps
# ==============================================================================

def _train_step_body(model, optimizer, x, y_star,
                     ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
                     ema_target_mag, ema_error_reduced,
                     ema_sum_input_a, ema_sum_hidden_a,
                     ema_sum_input_b, ema_sum_hidden_b,
                     ema_sum_input_c, ema_sum_hidden_c,
                     ema_sum_input_e, ema_sum_hidden_e,
                     ema_sum_input_upgd, ema_sum_hidden_upgd,
                     ema_sum_loo,
                     compute_pred_grads, compute_loo):
    """Core training step logic, shared by SGD and Autostep scan bodies."""
    # Forward pass
    y_hat_arr, _ = model(x)
    y_hat = y_hat_arr.squeeze()
    mse = (y_star - y_hat) ** 2

    # Compute loss gradients (needed for optimizer update AND UPGD utility)
    loss_grads = eqx.filter_grad(lambda m: (m(x)[0].squeeze() - y_star) ** 2)(model)

    # Compute utilities
    u_contribution = compute_contribution_utility(model, x)
    u_approach_a, u_hidden_a = compute_utility_approach_a(model, x, y_star, y_hat)
    u_approach_b, u_hidden_b = compute_utility_approach_b(model, x, y_star, y_hat)
    u_approach_c, u_hidden_c = compute_utility_approach_c(model, x, y_star, y_hat)
    u_approach_e, u_hidden_e = compute_utility_approach_e(model, x, y_star, y_hat)
    u_upgd, u_hidden_upgd = compute_utility_upgd(model, x, loss_grads)

    # True LOO (optional, 20 extra forward passes)
    if compute_loo:
        u_loo = compute_true_loo_utility(model, x, y_star)
        ema_loo = TRACE_DECAY * ema_loo + (1 - TRACE_DECAY) * u_loo
        ema_sum_loo = TRACE_DECAY * ema_sum_loo + (1 - TRACE_DECAY) * jnp.sum(u_loo)

    # Update EMA traces
    ema_contrib = TRACE_DECAY * ema_contrib + (1 - TRACE_DECAY) * u_contribution
    ema_a = TRACE_DECAY * ema_a + (1 - TRACE_DECAY) * u_approach_a
    ema_b = TRACE_DECAY * ema_b + (1 - TRACE_DECAY) * u_approach_b
    ema_c = TRACE_DECAY * ema_c + (1 - TRACE_DECAY) * u_approach_c
    ema_e = TRACE_DECAY * ema_e + (1 - TRACE_DECAY) * u_approach_e
    ema_upgd = TRACE_DECAY * ema_upgd + (1 - TRACE_DECAY) * u_upgd

    # Budget traces: |y*|, error reduced, sum of input/hidden utilities
    error_reduced = jnp.abs(y_star) - jnp.abs(y_star - y_hat)
    ema_target_mag = TRACE_DECAY * ema_target_mag + (1 - TRACE_DECAY) * jnp.abs(y_star)
    ema_error_reduced = TRACE_DECAY * ema_error_reduced + (1 - TRACE_DECAY) * error_reduced
    ema_sum_input_a = TRACE_DECAY * ema_sum_input_a + (1 - TRACE_DECAY) * jnp.sum(u_approach_a)
    ema_sum_hidden_a = TRACE_DECAY * ema_sum_hidden_a + (1 - TRACE_DECAY) * jnp.sum(u_hidden_a)
    ema_sum_input_b = TRACE_DECAY * ema_sum_input_b + (1 - TRACE_DECAY) * jnp.sum(u_approach_b)
    ema_sum_hidden_b = TRACE_DECAY * ema_sum_hidden_b + (1 - TRACE_DECAY) * jnp.sum(u_hidden_b)
    ema_sum_input_c = TRACE_DECAY * ema_sum_input_c + (1 - TRACE_DECAY) * jnp.sum(u_approach_c)
    ema_sum_hidden_c = TRACE_DECAY * ema_sum_hidden_c + (1 - TRACE_DECAY) * jnp.sum(u_hidden_c)
    ema_sum_input_e = TRACE_DECAY * ema_sum_input_e + (1 - TRACE_DECAY) * jnp.sum(u_approach_e)
    ema_sum_hidden_e = TRACE_DECAY * ema_sum_hidden_e + (1 - TRACE_DECAY) * jnp.sum(u_hidden_e)
    ema_sum_input_upgd = TRACE_DECAY * ema_sum_input_upgd + (1 - TRACE_DECAY) * jnp.sum(u_upgd)
    ema_sum_hidden_upgd = TRACE_DECAY * ema_sum_hidden_upgd + (1 - TRACE_DECAY) * jnp.sum(u_hidden_upgd)

    # Update model
    if compute_pred_grads:
        pred_grads = eqx.filter_grad(lambda m: m(x)[0].squeeze())(model)
        updates, new_optimizer = optimizer.with_update((loss_grads, pred_grads), model)
    else:
        updates, new_optimizer = optimizer.with_update(loss_grads, model)
    new_model = eqx.apply_updates(model, updates)

    return (new_model, new_optimizer, mse,
            ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
            ema_target_mag, ema_error_reduced,
            ema_sum_input_a, ema_sum_hidden_a,
            ema_sum_input_b, ema_sum_hidden_b,
            ema_sum_input_c, ema_sum_hidden_c,
            ema_sum_input_e, ema_sum_hidden_e,
            ema_sum_input_upgd, ema_sum_hidden_upgd,
            ema_sum_loo)


def _make_scan_fn(compute_pred_grads, compute_loo):
    """Build a scan body for either SGD or Autostep."""
    def scan_fn(carry, step_data):
        (model, optimizer, ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
         ema_target_mag, ema_error_reduced,
         ema_sum_input_a, ema_sum_hidden_a,
         ema_sum_input_b, ema_sum_hidden_b,
         ema_sum_input_c, ema_sum_hidden_c,
         ema_sum_input_e, ema_sum_hidden_e,
         ema_sum_input_upgd, ema_sum_hidden_upgd,
         ema_sum_loo) = carry
        x, y_star = step_data

        (model, optimizer, mse, ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
         ema_target_mag, ema_error_reduced,
         ema_sum_input_a, ema_sum_hidden_a,
         ema_sum_input_b, ema_sum_hidden_b,
         ema_sum_input_c, ema_sum_hidden_c,
         ema_sum_input_e, ema_sum_hidden_e,
         ema_sum_input_upgd, ema_sum_hidden_upgd,
         ema_sum_loo) = _train_step_body(
            model, optimizer, x, y_star,
            ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
            ema_target_mag, ema_error_reduced,
            ema_sum_input_a, ema_sum_hidden_a,
            ema_sum_input_b, ema_sum_hidden_b,
            ema_sum_input_c, ema_sum_hidden_c,
            ema_sum_input_e, ema_sum_hidden_e,
            ema_sum_input_upgd, ema_sum_hidden_upgd,
            ema_sum_loo,
            compute_pred_grads, compute_loo)

        # Extract first-layer step sizes (zeros for SGD, actual for Autostep)
        if compute_pred_grads:
            beta_leaves = jax.tree.leaves(optimizer.state.beta)
            step_sizes = jnp.exp(beta_leaves[0]).mean(axis=0)
        else:
            step_sizes = jnp.zeros(N_STUDENT_INPUTS)

        carry = (model, optimizer, ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
                 ema_target_mag, ema_error_reduced,
                 ema_sum_input_a, ema_sum_hidden_a,
                 ema_sum_input_b, ema_sum_hidden_b,
                 ema_sum_input_c, ema_sum_hidden_c,
                 ema_sum_input_e, ema_sum_hidden_e,
                 ema_sum_input_upgd, ema_sum_hidden_upgd,
                 ema_sum_loo)
        outputs = (mse, ema_contrib, ema_a, ema_b, step_sizes, ema_c, ema_e, ema_upgd, ema_loo,
                   ema_target_mag, ema_error_reduced,
                   ema_sum_input_a, ema_sum_hidden_a,
                   ema_sum_input_b, ema_sum_hidden_b,
                   ema_sum_input_c, ema_sum_hidden_c,
                   ema_sum_input_e, ema_sum_hidden_e,
                   ema_sum_input_upgd, ema_sum_hidden_upgd,
                   ema_sum_loo)
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
    ema_contrib = jnp.zeros(N_STUDENT_INPUTS)
    ema_a = jnp.zeros(N_STUDENT_INPUTS)
    ema_b = jnp.zeros(N_STUDENT_INPUTS)
    ema_c = jnp.zeros(N_STUDENT_INPUTS)
    ema_e = jnp.zeros(N_STUDENT_INPUTS)
    ema_upgd = jnp.zeros(N_STUDENT_INPUTS)
    ema_loo = jnp.zeros(N_STUDENT_INPUTS)
    ema_target_mag = jnp.float32(0.0)
    ema_error_reduced = jnp.float32(0.0)
    ema_sum_input_a = jnp.float32(0.0)
    ema_sum_hidden_a = jnp.float32(0.0)
    ema_sum_input_b = jnp.float32(0.0)
    ema_sum_hidden_b = jnp.float32(0.0)
    ema_sum_input_c = jnp.float32(0.0)
    ema_sum_hidden_c = jnp.float32(0.0)
    ema_sum_input_e = jnp.float32(0.0)
    ema_sum_hidden_e = jnp.float32(0.0)
    ema_sum_input_upgd = jnp.float32(0.0)
    ema_sum_hidden_upgd = jnp.float32(0.0)
    ema_sum_loo = jnp.float32(0.0)

    # Build and run the scan
    scan_fn = _make_scan_fn(compute_pred_grads=is_autostep, compute_loo=COMPUTE_TRUE_LOO)
    init_carry = (model, optimizer, ema_contrib, ema_a, ema_b, ema_c, ema_e, ema_upgd, ema_loo,
                  ema_target_mag, ema_error_reduced,
                  ema_sum_input_a, ema_sum_hidden_a,
                  ema_sum_input_b, ema_sum_hidden_b,
                  ema_sum_input_c, ema_sum_hidden_c,
                  ema_sum_input_e, ema_sum_hidden_e,
                  ema_sum_input_upgd, ema_sum_hidden_upgd,
                  ema_sum_loo)
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
    app_c_hist = np.concatenate([o[5] for o in all_outputs])
    app_e_hist = np.concatenate([o[6] for o in all_outputs])
    upgd_hist = np.concatenate([o[7] for o in all_outputs])
    loo_hist = np.concatenate([o[8] for o in all_outputs])
    target_mag_hist = np.concatenate([o[9] for o in all_outputs])
    error_reduced_hist = np.concatenate([o[10] for o in all_outputs])
    sum_input_a_hist = np.concatenate([o[11] for o in all_outputs])
    sum_hidden_a_hist = np.concatenate([o[12] for o in all_outputs])
    sum_input_b_hist = np.concatenate([o[13] for o in all_outputs])
    sum_hidden_b_hist = np.concatenate([o[14] for o in all_outputs])
    sum_input_c_hist = np.concatenate([o[15] for o in all_outputs])
    sum_hidden_c_hist = np.concatenate([o[16] for o in all_outputs])
    sum_input_e_hist = np.concatenate([o[17] for o in all_outputs])
    sum_hidden_e_hist = np.concatenate([o[18] for o in all_outputs])
    sum_input_upgd_hist = np.concatenate([o[19] for o in all_outputs])
    sum_hidden_upgd_hist = np.concatenate([o[20] for o in all_outputs])
    sum_loo_hist = np.concatenate([o[21] for o in all_outputs])

    return {
        'mse_history': mse_hist,
        'contribution_traces': contrib_hist,
        'approach_a_traces': app_a_hist,
        'approach_b_traces': app_b_hist,
        'approach_c_traces': app_c_hist,
        'approach_e_traces': app_e_hist,
        'upgd_traces': upgd_hist,
        'loo_traces': loo_hist,
        'step_size_history': ss_hist if is_autostep else None,
        'target_mag': target_mag_hist,
        'error_reduced': error_reduced_hist,
        'sum_input_a': sum_input_a_hist,
        'sum_hidden_a': sum_hidden_a_hist,
        'sum_input_b': sum_input_b_hist,
        'sum_hidden_b': sum_hidden_b_hist,
        'sum_input_c': sum_input_c_hist,
        'sum_hidden_c': sum_hidden_c_hist,
        'sum_input_e': sum_input_e_hist,
        'sum_hidden_e': sum_hidden_e_hist,
        'sum_input_upgd': sum_input_upgd_hist,
        'sum_hidden_upgd': sum_hidden_upgd_hist,
        'sum_loo': sum_loo_hist,
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

    # ---- Figure 2: Input-level utility grid ----
    utility_names = ['Contribution', 'Approach A', 'Approach B', 'Approach C', 'Approach E', 'UPGD']
    utility_keys = ['contribution_traces', 'approach_a_traces', 'approach_b_traces', 'approach_c_traces', 'approach_e_traces', 'upgd_traces']
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

    # Legend on last subplot
    axes[0, -1].plot([], [], color='blue', linewidth=2, label='Relevant (0-4)')
    axes[0, -1].plot([], [], color='red', linewidth=2, label='Irrelevant (5-19)')
    axes[0, -1].legend()

    fig2.tight_layout()
    fig2.savefig(
        SCRIPT_DIR / 'fig2_utility_traces.png',
        dpi=150,
    )
    print("Saved fig2_utility_traces.png")

    # ---- Figure 3: Utility budget per method ----
    # Each method gets: (name, input_sum_key, hidden_sum_key_or_None)
    method_budget_keys = [
        ('Approach A', 'sum_input_a', 'sum_hidden_a'),
        ('Approach B', 'sum_input_b', 'sum_hidden_b'),
        ('Approach C', 'sum_input_c', 'sum_hidden_c'),
        ('Approach E', 'sum_input_e', 'sum_hidden_e'),
        ('UPGD', 'sum_input_upgd', 'sum_hidden_upgd'),
    ]
    if COMPUTE_TRUE_LOO:
        method_budget_keys.append(('True LOO', 'sum_loo', None))
    n_budget_cols = len(method_budget_keys)
    fig3, axes3 = plt.subplots(2, n_budget_cols, figsize=(6 * n_budget_cols, 9))

    for row, (results, opt_name) in enumerate(
        [(sgd_results, 'SGD'), (autostep_results, 'Autostep')]
    ):
        for col, (method_name, input_key, hidden_key) in enumerate(method_budget_keys):
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
    sep_win = 1000
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
    for name, results in [('SGD', sgd_results), ('Autostep', autostep_results)]:
        mse = np.mean(results['mse_history'][last_5k])
        print(f"\n{name} -- Final MSE (last 5k): {mse:.4f}")
        summary_methods = [
            ('Contribution', 'contribution_traces'),
            ('Approach A', 'approach_a_traces'),
            ('Approach B', 'approach_b_traces'),
            ('Approach C', 'approach_c_traces'),
            ('Approach E', 'approach_e_traces'),
            ('UPGD', 'upgd_traces'),
        ]
        if COMPUTE_TRUE_LOO:
            summary_methods.append(('True LOO', 'loo_traces'))
        for u_name, u_key in summary_methods:
            traces = results[u_key]  # (N_STEPS, N_INPUT)
            last_traces = traces[last_5k]
            rel_mean = np.mean(last_traces[:, :N_RELEVANT])
            irr_mean = np.mean(last_traces[:, N_RELEVANT:])
            gap = rel_mean - irr_mean
            # Cohen's d: separation quality normalized by spread
            rel_std = np.std(last_traces[:, :N_RELEVANT])
            irr_std = np.std(last_traces[:, N_RELEVANT:])
            pooled_std = np.sqrt((rel_std**2 + irr_std**2) / 2)
            d = gap / (pooled_std + 1e-10)
            # Speed of separation: first step where rolling d > 1.0
            win = 1000
            if len(traces) >= win:
                rel_roll = np.convolve(np.mean(traces[:, :N_RELEVANT], axis=1), np.ones(win)/win, mode='valid')
                irr_roll = np.convolve(np.mean(traces[:, N_RELEVANT:], axis=1), np.ones(win)/win, mode='valid')
                rel_std_roll = np.convolve(np.var(traces[:, :N_RELEVANT], axis=1), np.ones(win)/win, mode='valid') ** 0.5
                irr_std_roll = np.convolve(np.var(traces[:, N_RELEVANT:], axis=1), np.ones(win)/win, mode='valid') ** 0.5
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
