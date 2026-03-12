"""
Utility functions for measuring per-feature importance in multi-layer networks.

All functions share a uniform interface:
    fn(model, x, y_star, y_hat, loss_grads, updates) -> (U_input, U_hidden)

Arguments not needed by a given method are ignored.
- model: equinox MLP with model.layers[0].weight (H, N) and model.layers[1].weight (1, H)
- x: input vector (N,)
- y_star: scalar target
- y_hat: scalar prediction
- loss_grads: gradient of loss w.r.t. model parameters (same pytree as model)
- updates: optimizer parameter updates (same pytree as model)

Returns:
- U_input: per-input utility (N,)
- U_hidden: per-hidden-unit utility (H,)
"""

import jax
import jax.numpy as jnp

ALIGNMENT_THRESHOLD = 0.1  # Approach C fallback threshold


# ==============================================================================
# Helpers
# ==============================================================================

def _output_layer_loo(w_out, a_hidden, y_star, y_hat):
    """Output-layer leave-one-out: U_j = |e + c_j| - |e|."""
    e = y_star - y_hat
    c_j = w_out * a_hidden
    U_hidden = jnp.abs(e + c_j) - jnp.abs(e)
    return e, c_j, U_hidden


def _approach_a_redistribution(W1, x, U_hidden):
    """Proportional redistribution: U_{k<-j} = U_j * |W1[j,k]*x_k| / Σ|W1[j,m]*x_m|."""
    contrib = jnp.abs(W1 * x[None, :])
    frac = contrib / jnp.maximum(jnp.sum(contrib, axis=1, keepdims=True), 1e-10)
    return jnp.sum(U_hidden[:, None] * frac, axis=0)


# ==============================================================================
# Approach E calibration internals
# ==============================================================================

def _find_e_nonneg(contributions, target):
    """Find e >= 0 such that g(e) = Σ(|e + c_k| - e) = target.

    g is piecewise linear with breakpoints at |c_k| for each negative c_k.
    Solves analytically by locating the correct linear segment.
    """
    K = contributions.shape[0]

    P = jnp.sum(jnp.maximum(contributions, 0.0))
    neg_mags = jnp.maximum(-contributions, 0.0)
    N = jnp.sum(neg_mags)
    z = P - N

    feasible_range = (target >= z - 1e-6) & (target <= P + N + 1e-6)

    # Sort negative magnitudes ascending, padding positives with large sentinel
    padded = jnp.where(contributions < 0, -contributions, jnp.float32(1e30))
    sorted_b = jnp.sort(padded)

    cum = jnp.cumsum(jnp.minimum(sorted_b, 1e20))
    B = jnp.concatenate([jnp.array([0.0]), cum[:-1]])

    m = jnp.sum((contributions < 0).astype(jnp.float32))

    # In segment i: g(e) = P + N - 2*B[i] - 2*(m-i)*e
    remaining = jnp.maximum(m - jnp.arange(K, dtype=jnp.float32), 0.0)
    numer = P + N - 2.0 * B - target
    safe_denom = jnp.maximum(2.0 * remaining, 1.0)
    e_cand = numer / safe_denom

    left_bounds = jnp.concatenate([jnp.array([0.0]), sorted_b[:-1]])
    right_bounds = sorted_b

    eps = 1e-6
    valid = ((e_cand >= left_bounds - eps) &
             (e_cand <= right_bounds + eps) &
             (e_cand >= -eps) &
             (remaining > 0.5))

    first_idx = jnp.argmax(valid)
    e_segment = jnp.maximum(e_cand[first_idx], 0.0)
    segment_ok = valid[first_idx]

    # Edge case: target ≈ z (asymptotic)
    asymptotic_ok = (jnp.abs(target - z) < eps)
    last_bp_idx = jnp.maximum(m.astype(jnp.int32) - 1, 0)
    e_asymptotic = jnp.where(m > 0.5, sorted_b[last_bp_idx] + 1.0, 0.0)

    e_out = jnp.where(segment_ok, e_segment,
                       jnp.where(asymptotic_ok, e_asymptotic, 0.0))
    feasible = (segment_ok | asymptotic_ok) & feasible_range

    return e_out, feasible


def _calibrate_pseudo_error(contributions, target):
    """Find e such that g(e) = Σ(|e + c_k| - |e|) = target.

    For target >= Σc_k: search e >= 0. Otherwise: negate contributions and search e >= 0.
    """
    z = jnp.sum(contributions)
    search_positive = (target >= z)
    c = jnp.where(search_positive, contributions, -contributions)
    d, feas = _find_e_nonneg(c, target)
    e = jnp.where(search_positive, d, -d)
    return e, feas


# ==============================================================================
# Utility functions (uniform interface)
# ==============================================================================

def contribution_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Contribution utility (CBP analog): |x_i| * Σ_j |W1[j,i]|. Always positive."""
    W1 = model.layers[0].weight
    n_hidden = W1.shape[0]
    U_input = jnp.abs(x) * jnp.sum(jnp.abs(W1), axis=0)
    U_hidden = jnp.zeros(n_hidden)
    return U_input, U_hidden


def upgd_utility(model, x, y_star, y_hat, loss_grads, updates):
    """UPGD first-order Taylor (Elsayed & Mahmood 2023): u_j = -(dL/da_j) * a_j.

    Extracts activation gradients from weight gradients:
    grad_W1 = outer(dL/dz, x), so dL/dz = grad_W1 @ x / ||x||².
    """
    W1 = model.layers[0].weight
    grad_W1 = loss_grads.layers[0].weight

    x_norm_sq = jnp.sum(x * x)
    dL_dz = (grad_W1 @ x) / jnp.maximum(x_norm_sq, 1e-10)

    dL_dx = W1.T @ dL_dz
    U_input = -dL_dx * x

    a_hidden = jax.nn.sigmoid(W1 @ x)
    f_prime = jnp.maximum(a_hidden * (1.0 - a_hidden), 1e-6)
    U_hidden = -(dL_dz / f_prime) * a_hidden

    return U_input, U_hidden


def si_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Synaptic Intelligence (Zenke et al. 2017): omega_k = (-dL/dtheta_k) * delta_theta_k.

    Per-feature: sum over weights connected to each input.
    """
    U_input = jnp.sum(-loss_grads.layers[0].weight * updates.layers[0].weight, axis=0)
    U_hidden = -loss_grads.layers[1].weight.squeeze(0) * updates.layers[1].weight.squeeze(0)
    return U_input, U_hidden


def approach_a_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Approach A: Proportional Redistribution.

    Distributes each hidden unit's utility to inputs proportionally by |W1[j,k]*x_k|.
    All children inherit the parent's sign.
    """
    W1 = model.layers[0].weight
    w_out = model.layers[1].weight.squeeze(0)
    a_hidden = jax.nn.sigmoid(W1 @ x)

    _, _, U_hidden = _output_layer_loo(w_out, a_hidden, y_star, y_hat)
    U_input = _approach_a_redistribution(W1, x, U_hidden)
    return U_input, U_hidden


def approach_b_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Approach B: Recursive Signed Utility.

    Pseudo-error e_j = |U_j| / f'(z_j), then applies LOO formula at hidden layer.
    Normalized so Σ|U_{k<-j}| = |U_j|.
    """
    W1 = model.layers[0].weight
    w_out = model.layers[1].weight.squeeze(0)
    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)

    _, _, U_hidden = _output_layer_loo(w_out, a_hidden, y_star, y_hat)

    f_prime = jnp.maximum(a_hidden * (1.0 - a_hidden), 1e-6)
    e_j = jnp.abs(U_hidden) / f_prime

    contributions = W1 * x[None, :]
    s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])
    s_abs_sum = jnp.maximum(jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True), 1e-10)
    U_from_j = s_raw * jnp.abs(U_hidden[:, None]) / s_abs_sum

    U_input = jnp.sum(U_from_j, axis=0)
    return U_input, U_hidden


def approach_c_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Approach C: Signed-Conserving Redistribution.

    Same raw scores as B, but normalized to preserve signed sum: U_{k<-j} = s_k * U_j / Σs_k.
    Falls back to Approach A when children nearly cancel.
    Output-layer utilities rescaled so Σ U_j = error_reduced.
    """
    W1 = model.layers[0].weight
    w_out = model.layers[1].weight.squeeze(0)
    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)

    # Output layer: rescale LOO utilities to sum to error_reduced
    e = y_star - y_hat
    c_j = w_out * a_hidden
    u_raw = jnp.abs(e + c_j) - jnp.abs(e)
    error_reduced = jnp.abs(y_star) - jnp.abs(e)
    u_raw_sum = jnp.sum(u_raw)
    scale = jnp.where(jnp.abs(u_raw_sum) > 1e-10, error_reduced / u_raw_sum, 1.0)
    U_hidden = u_raw * scale

    f_prime = jnp.maximum(a_hidden * (1.0 - a_hidden), 1e-6)
    e_j = jnp.abs(U_hidden) / f_prime

    contributions = W1 * x[None, :]
    s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])

    # Signed normalization with alignment-based fallback
    s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)
    s_abs_sum = jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True)
    alignment = jnp.abs(s_signed_sum) / jnp.maximum(s_abs_sum, 1e-10)

    s_signed_safe = jnp.where(jnp.abs(s_signed_sum) > 1e-10, s_signed_sum, 1.0)
    U_signed = s_raw * U_hidden[:, None] / s_signed_safe

    abs_contrib = jnp.abs(contributions)
    U_fallback = U_hidden[:, None] * abs_contrib / jnp.maximum(jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10)

    use_signed = (alignment > ALIGNMENT_THRESHOLD).astype(jnp.float32)
    U_from_j = use_signed * U_signed + (1.0 - use_signed) * U_fallback

    U_input = jnp.sum(U_from_j, axis=0)
    return U_input, U_hidden


def approach_e_utility(model, x, y_star, y_hat, loss_grads, updates):
    """Approach E: Calibrated Pseudo-Error.

    Finds pseudo-error e such that raw LOO scores naturally sum to the target utility.
    No normalization needed. Falls back to Approach A when infeasible.
    """
    W1 = model.layers[0].weight
    w_out = model.layers[1].weight.squeeze(0)
    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)

    # Output layer: calibrate e_out so scores sum to E_reduced
    e_true = y_star - y_hat
    c_out = w_out * a_hidden
    E_reduced = jnp.abs(y_star) - jnp.abs(e_true)

    e_out, out_feasible = _calibrate_pseudo_error(c_out, E_reduced)
    U_hidden_calib = jnp.abs(e_out + c_out) - jnp.abs(e_out)

    # Fallback: standard LOO rescaled
    u_raw = jnp.abs(e_true + c_out) - jnp.abs(e_true)
    u_raw_sum = jnp.sum(u_raw)
    scale = jnp.where(jnp.abs(u_raw_sum) > 1e-10, E_reduced / u_raw_sum, 1.0)
    U_hidden_fallback = u_raw * scale

    U_hidden = jnp.where(out_feasible, U_hidden_calib, U_hidden_fallback)

    # Hidden layer: calibrate e_j for each unit
    contributions = W1 * x[None, :]
    e_j, feasible_j = jax.vmap(_calibrate_pseudo_error)(contributions, U_hidden)
    U_from_j_calib = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])

    # Fallback to Approach A for infeasible units
    abs_contrib = jnp.abs(contributions)
    U_from_j_fallback = U_hidden[:, None] * abs_contrib / jnp.maximum(jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10)

    use_calib = feasible_j[:, None].astype(jnp.float32)
    U_from_j = use_calib * U_from_j_calib + (1.0 - use_calib) * U_from_j_fallback

    U_input = jnp.sum(U_from_j, axis=0)
    return U_input, U_hidden


def true_loo_utility(model, x, y_star, y_hat, loss_grads, updates):
    """True leave-one-out utility via N extra forward passes (vmapped)."""
    n_inputs = x.shape[0]
    base_error = jnp.abs(y_star - y_hat)

    mask = 1.0 - jnp.eye(n_inputs)
    x_masked_all = mask * x[None, :]

    y_hat_masked = jax.vmap(lambda xm: model(xm)[0].squeeze())(x_masked_all)
    errors_without = jnp.abs(y_star - y_hat_masked)

    U_input = errors_without - base_error
    U_hidden = jnp.zeros(model.layers[0].weight.shape[0])
    return U_input, U_hidden
