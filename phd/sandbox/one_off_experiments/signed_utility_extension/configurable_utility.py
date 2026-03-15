"""Configurable utility function for multi-layer networks with arbitrary depth
and monotonically increasing activation functions (sigmoid, tanh, relu).

Decomposes the existing utility approaches (A-M) into independent configuration choices:
- child_score_method: how children's scores are computed (loo, proportional, coherence)
- output_normalization: how output-layer LOO scores relate to error_reduced
- pseudo_error: how the hidden-layer pseudo-error is derived
- hidden_normalization: how hidden-layer LOO scores relate to parent utility
- cap: whether to clip |child| <= |parent| after normalization

See UtilityConfig for the full configuration space and enumerate_valid_configs()
for all 115 valid combinations.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ==============================================================================
# Activation functions and inverses
# ==============================================================================

_LARGE = 1e6
_EPS = 1e-6


def _sigmoid_inverse(a):
    """Inverse sigmoid (logit) with ±1e6 fallback for out-of-range values."""
    in_range = (a > _EPS) & (a < 1.0 - _EPS)
    a_safe = jnp.clip(a, _EPS, 1.0 - _EPS)
    z_normal = jnp.log(a_safe / (1.0 - a_safe))
    z_fallback = jnp.where(a <= _EPS, -_LARGE, _LARGE)
    return jnp.where(in_range, z_normal, z_fallback)


def _tanh_inverse(a):
    """Inverse tanh (arctanh) with ±1e6 fallback for out-of-range values."""
    in_range = (a > -1.0 + _EPS) & (a < 1.0 - _EPS)
    a_safe = jnp.clip(a, -1.0 + _EPS, 1.0 - _EPS)
    z_normal = jnp.arctanh(a_safe)
    z_fallback = jnp.where(a <= -1.0 + _EPS, -_LARGE, _LARGE)
    return jnp.where(in_range, z_normal, z_fallback)


def _relu_inverse(a):
    """Inverse ReLU: identity for a >= 0, -1e6 for negative (unreachable) targets."""
    return jnp.where(a >= 0, a, -_LARGE)


ACTIVATIONS = {
    'sigmoid': jax.nn.sigmoid,
    'tanh': jnp.tanh,
    'relu': jax.nn.relu,
}

ACTIVATION_INVERSES = {
    'sigmoid': _sigmoid_inverse,
    'tanh': _tanh_inverse,
    'relu': _relu_inverse,
}


def _resolve_activation(activation):
    """Resolve activation to (fn, inverse_fn) tuple.

    Args:
        activation: 'sigmoid', 'tanh', 'relu', or a (fn, inverse_fn) tuple.
    """
    if isinstance(activation, str):
        return ACTIVATIONS[activation], ACTIVATION_INVERSES[activation]
    return activation


# ==============================================================================
# Configuration
# ==============================================================================

CHILD_SCORE_METHODS = ("loo", "proportional", "coherence")
NORMALIZATION_METHODS = ("none", "absolute", "signed", "overflow_only", "absorption")
PSEUDO_ERROR_METHODS = ("abs_derivative", "signed_derivative", "target_prop")


@dataclass(frozen=True)
class UtilityConfig:
    """Configuration for the configurable utility function.

    Existing approach mapping (all sigmoid, 2-layer):
        A: proportional, none, -, -, F
        B: loo, none, abs_derivative, absolute, F
        C: loo, signed, abs_derivative, signed, F  (approx — C uses alignment threshold fallback)
        F: loo, signed, abs_derivative, signed, T
        G: loo, none, target_prop, signed, T
        H: loo, none, abs_derivative, absorption, F
        I: coherence, none, -, -, F
        J: loo, signed, signed_derivative, overflow_only, F
        K: loo, signed, target_prop, overflow_only, F
        L: loo, none, signed_derivative, absolute, F
        M: loo, none, target_prop, absolute, F
    """
    child_score_method: str = "loo"
    output_normalization: str = "none"
    pseudo_error: str = "abs_derivative"
    hidden_normalization: str = "absolute"
    cap: bool = False


def is_valid_config(config):
    """Check whether a UtilityConfig represents a valid combination."""
    if config.child_score_method not in CHILD_SCORE_METHODS:
        return False
    if config.output_normalization not in NORMALIZATION_METHODS:
        return False
    if config.child_score_method in ("proportional", "coherence"):
        return True
    if config.pseudo_error not in PSEUDO_ERROR_METHODS:
        return False
    if config.hidden_normalization not in NORMALIZATION_METHODS:
        return False
    if config.cap and config.hidden_normalization in ("none", "absolute", "absorption"):
        return False
    return True


def enumerate_valid_configs():
    """Generate all valid UtilityConfig instances (115 total)."""
    configs = []
    for output_norm in NORMALIZATION_METHODS:
        for child_score in ("proportional", "coherence"):
            configs.append(UtilityConfig(
                child_score_method=child_score,
                output_normalization=output_norm,
            ))
        for pseudo in PSEUDO_ERROR_METHODS:
            for hidden_norm in NORMALIZATION_METHODS:
                if hidden_norm in ("none", "absolute", "absorption"):
                    configs.append(UtilityConfig(
                        child_score_method="loo",
                        output_normalization=output_norm,
                        pseudo_error=pseudo,
                        hidden_normalization=hidden_norm,
                        cap=False,
                    ))
                else:
                    for cap in (False, True):
                        configs.append(UtilityConfig(
                            child_score_method="loo",
                            output_normalization=output_norm,
                            pseudo_error=pseudo,
                            hidden_normalization=hidden_norm,
                            cap=cap,
                        ))
    return configs


# ==============================================================================
# Helpers
# ==============================================================================

def _output_layer_loo(w_out, a_last, y_star, y_hat):
    """Output-layer leave-one-out: U_j = |e + c_j| - |e|.

    Supports both scalar and vector outputs:
    - Scalar: w_out (H,), y_star scalar, y_hat scalar → u_raw (H,)
    - Vector: w_out (D, H), y_star (D,), y_hat (D,) → u_raw (H,) summed over D
    """
    e = y_star - y_hat
    if w_out.ndim == 1:
        # Scalar output: original behavior
        c_j = w_out * a_last
        u_raw = jnp.abs(e + c_j) - jnp.abs(e)
    else:
        # Multi-output: w_out is (D, H), e is (D,)
        c_j = w_out * a_last[None, :]  # (D, H)
        u_raw_per_d = jnp.abs(e[:, None] + c_j) - jnp.abs(e[:, None])  # (D, H)
        u_raw = jnp.sum(u_raw_per_d, axis=0)  # (H,)
    return e, u_raw


def _proportional_redistribution(W, a_below, U):
    """Proportional redistribution: U_{k<-j} = U_j * |c_k| / Σ|c_m|."""
    contrib = jnp.abs(W * a_below[None, :])
    frac = contrib / jnp.maximum(jnp.sum(contrib, axis=1, keepdims=True), 1e-10)
    return jnp.sum(U[:, None] * frac, axis=0)


def _target_prop_pseudo_error(w_eff, U, a, z, activation_inverse):
    """Target propagation pseudo-error using explicit activation inverse.

    Computes a_target = a + U/w_eff, then e_j = f^{-1}(a_target) - z.
    When w_eff ≈ 0, returns 0 (unit has no outgoing influence).

    Args:
        w_eff: effective weight connecting each unit to the layer above.
            For the last hidden layer: the output weight per unit.
            For intermediate layers: column sum of the weight matrix above.
        U: utility of each unit (H,)
        a: activation of each unit (H,)
        z: pre-activation of each unit (H,)
        activation_inverse: inverse of the activation function
    """
    w_safe = jnp.where(jnp.abs(w_eff) > 1e-6, w_eff, jnp.ones_like(w_eff))
    delta_a = jnp.where(jnp.abs(w_eff) > 1e-6, U / w_safe, 0.0)
    a_target = a + delta_a
    z_target = activation_inverse(a_target)
    return z_target - z


def _normalize_output(u_raw, parent, method):
    """Apply normalization to 1D output-layer scores."""
    if method == "none":
        return u_raw
    elif method == "absolute":
        abs_sum = jnp.maximum(jnp.sum(jnp.abs(u_raw)), 1e-10)
        return u_raw * jnp.abs(parent) / abs_sum
    elif method == "signed":
        signed_sum = jnp.sum(u_raw)
        scale = jnp.where(jnp.abs(signed_sum) > 1e-10, parent / signed_sum, 1.0)
        return u_raw * scale
    elif method == "overflow_only":
        signed_sum = jnp.sum(u_raw)
        signed_abs = jnp.abs(signed_sum)
        needs_norm = signed_abs > jnp.abs(parent)
        safe_sum = jnp.where(signed_abs > 1e-10, signed_sum, 1.0)
        normed = u_raw * parent / safe_sum
        return jnp.where(needs_norm, normed, u_raw)
    elif method == "absorption":
        abs_sum = jnp.sum(jnp.abs(u_raw))
        scale = jnp.minimum(1.0, jnp.abs(parent) / jnp.maximum(abs_sum, 1e-10))
        return u_raw * scale


def _normalize_hidden(s_raw, U_parent, contributions, method):
    """Apply normalization to 2D hidden-layer scores (H, N)."""
    if method == "none":
        return s_raw

    elif method == "absolute":
        s_abs_sum = jnp.maximum(jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True), 1e-10)
        return s_raw * jnp.abs(U_parent[:, None]) / s_abs_sum

    elif method == "signed":
        s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)
        can_normalize = jnp.abs(s_signed_sum) > 1e-10
        s_signed_safe = jnp.where(can_normalize, s_signed_sum, 1.0)
        U_signed = s_raw * U_parent[:, None] / s_signed_safe
        # Proportional fallback when Σs ≈ 0
        abs_contrib = jnp.abs(contributions)
        U_fallback = U_parent[:, None] * abs_contrib / jnp.maximum(
            jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10
        )
        use_signed = can_normalize.astype(jnp.float32)
        return use_signed * U_signed + (1.0 - use_signed) * U_fallback

    elif method == "overflow_only":
        s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)
        s_signed_abs = jnp.abs(s_signed_sum)
        needs_norm = s_signed_abs > jnp.abs(U_parent[:, None])
        s_signed_safe = jnp.where(s_signed_abs > 1e-10, s_signed_sum, 1.0)
        U_normed = s_raw * U_parent[:, None] / s_signed_safe
        return jnp.where(needs_norm, U_normed, s_raw)

    elif method == "absorption":
        s_abs_sum = jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True)
        scale = jnp.minimum(1.0, jnp.abs(U_parent[:, None]) / jnp.maximum(s_abs_sum, 1e-10))
        return s_raw * scale


# ==============================================================================
# Main function
# ==============================================================================

def configurable_utility(model, x, y_star, y_hat, loss_grads, updates, config,
                         activation='sigmoid', masks=None):
    """Configurable utility function for arbitrary-depth networks.

    Args:
        model: equinox MLP (no bias). Hidden layers use the specified activation;
            output layer is linear.
        x: input vector (N,)
        y_star: scalar or vector target (D,)
        y_hat: scalar or vector prediction (D,)
        loss_grads: unused (kept for interface compatibility)
        updates: unused (kept for interface compatibility)
        config: UtilityConfig
        activation: 'sigmoid', 'tanh', 'relu', or (fn, inverse_fn) tuple
        masks: optional tuple of (H_l,) arrays, one per hidden layer.
            1 = active, 0 = pruned. Applied after activation. Default None = no masking.

    Returns:
        U_input: per-input utility (N,)
        hidden_utilities: tuple of per-hidden-unit utilities, ordered first to last layer.
            For a 2-layer network (1 hidden layer), this is (U_hidden,).
    """
    act_fn, act_inv = _resolve_activation(activation)
    n_layers = len(model.layers)

    # --- Forward pass: compute all layer activations ---
    activations = [x]
    pre_activations = []
    for l in range(n_layers - 1):  # hidden layers
        z = model.layers[l].weight @ activations[-1]
        a = act_fn(z)
        if masks is not None:
            a = a * masks[l]
        pre_activations.append(z)
        activations.append(a)

    # --- Output layer LOO ---
    w_out = model.layers[-1].weight
    if w_out.shape[0] == 1:
        w_out = w_out.squeeze(0)  # (H,) for single output
    e, u_raw = _output_layer_loo(w_out, activations[-1], y_star, y_hat)
    error_reduced = jnp.sum(jnp.abs(y_star)) - jnp.sum(jnp.abs(e))

    U_current = _normalize_output(u_raw, error_reduced, config.output_normalization)
    layer_utilities = [U_current]

    # --- Backward through hidden layers ---
    n_hidden = n_layers - 1
    for l in range(n_hidden - 1, -1, -1):
        W = model.layers[l].weight       # (H_l, H_{l-1}) or (H_l, N) for l=0
        a_below = activations[l]          # inputs to this layer
        z_l = pre_activations[l]          # pre-activations of this layer
        a_l = activations[l + 1]          # activations of this layer

        if config.child_score_method == "proportional":
            U_next = _proportional_redistribution(W, a_below, U_current)

        elif config.child_score_method == "coherence":
            contributions = W * a_below[None, :]
            z_j = jnp.sum(contributions, axis=1, keepdims=True)
            Sigma = jnp.sum(jnp.abs(contributions), axis=1, keepdims=True)
            Sigma_safe = jnp.maximum(Sigma, 1e-10)
            beta = jnp.abs(z_j) / Sigma_safe
            sign_z = jnp.sign(z_j)
            weights = (sign_z * contributions + (1.0 - beta) * jnp.abs(contributions)) / Sigma_safe
            U_from_j = U_current[:, None] * weights
            U_next = jnp.sum(U_from_j, axis=0)

        else:  # loo
            # Pseudo-error
            if config.pseudo_error in ("abs_derivative", "signed_derivative"):
                f_prime = jax.vmap(jax.grad(act_fn))(z_l)
                f_prime = jnp.maximum(f_prime, 1e-6)
                if config.pseudo_error == "abs_derivative":
                    e_j = jnp.abs(U_current) / f_prime
                else:
                    e_j = U_current / f_prime
            else:  # target_prop
                if l == n_hidden - 1:
                    # Last hidden layer: column sum of output weight matrix
                    w_eff = jnp.sum(model.layers[-1].weight, axis=0)
                else:
                    # Intermediate layer: column sum of weight matrix above
                    w_eff = jnp.sum(model.layers[l + 1].weight, axis=0)
                e_j = _target_prop_pseudo_error(w_eff, U_current, a_l, z_l, act_inv)

            # LOO scores
            contributions = W * a_below[None, :]
            s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])

            # Normalization
            U_from_j = _normalize_hidden(s_raw, U_current, contributions,
                                         config.hidden_normalization)

            # Post-normalization cap
            if config.cap:
                U_from_j = jnp.clip(U_from_j,
                                    -jnp.abs(U_current[:, None]),
                                    jnp.abs(U_current[:, None]))

            U_next = jnp.sum(U_from_j, axis=0)

        U_current = U_next
        layer_utilities.append(U_current)

    U_input = layer_utilities[-1]
    hidden_utilities = tuple(reversed(layer_utilities[:-1]))
    return U_input, hidden_utilities


def make_utility_fn(config, activation='sigmoid'):
    """Factory: returns a standard-interface utility function from a config.

    The returned function has signature:
        fn(model, x, y_star, y_hat, loss_grads, updates, masks=None) -> (U_input, hidden_utilities)
    """
    def fn(model, x, y_star, y_hat, loss_grads, updates, masks=None):
        return configurable_utility(model, x, y_star, y_hat, loss_grads, updates,
                                    config, activation, masks=masks)
    fn.__name__ = (
        f"configurable_"
        f"{config.child_score_method}_"
        f"{config.output_normalization}_"
        f"{config.pseudo_error}_"
        f"{config.hidden_normalization}_"
        f"cap{config.cap}"
    )
    return fn


# ==============================================================================
# Baseline utility functions (generalized for arbitrary depth/activation)
# ==============================================================================

def contribution_utility(model, x, y_star, y_hat, loss_grads, updates,
                         activation='sigmoid', masks=None):
    """CBP-style contribution utility for arbitrary-depth networks.

    Per-unit utility = |activation| * Σ|outgoing weights|.
    For inputs: U_i = |x_i| * Σ_j |W_0[j,i]|.
    For hidden unit j in layer l: U_j = |a_l[j]| * Σ_p |W_{l+1}[p,j]|.
    Masked units naturally get 0 utility (their activation is zeroed).
    """
    act_fn, _ = _resolve_activation(activation)
    n_layers = len(model.layers)

    # Forward pass
    activations = [x]
    for l in range(n_layers - 1):
        z = model.layers[l].weight @ activations[-1]
        a = act_fn(z)
        if masks is not None:
            a = a * masks[l]
        activations.append(a)

    # Input utility: |x| * Σ|outgoing weights from each input|
    U_input = jnp.abs(x) * jnp.sum(jnp.abs(model.layers[0].weight), axis=0)

    # Hidden layer utilities: |a| * Σ|outgoing weights from each unit|
    hidden_utilities = []
    for l in range(n_layers - 1):
        a_l = activations[l + 1]
        W_above = model.layers[l + 1].weight
        U_l = jnp.abs(a_l) * jnp.sum(jnp.abs(W_above), axis=0)
        hidden_utilities.append(U_l)

    return U_input, tuple(hidden_utilities)


def upgd_utility(model, x, y_star, y_hat, loss_grads, updates,
                 activation='sigmoid', masks=None):
    """UPGD first-order Taylor utility for arbitrary-depth networks.

    u_j = -(dL/da_j) * a_j (Elsayed & Mahmood 2023).
    Extracts dL/dz at each layer from weight gradients:
        grad_W_l = outer(dL/dz_l, a_{l-1}), so dL/dz_l = grad_W_l @ a_{l-1} / ||a_{l-1}||².
    Then dL/da_l = dL/dz_l / f'(z_l).
    Masked units naturally get 0 utility (their activation is zeroed).
    """
    act_fn, _ = _resolve_activation(activation)
    n_layers = len(model.layers)

    # Forward pass
    activations = [x]
    pre_activations = []
    for l in range(n_layers - 1):
        z = model.layers[l].weight @ activations[-1]
        a = act_fn(z)
        if masks is not None:
            a = a * masks[l]
        pre_activations.append(z)
        activations.append(a)

    # Hidden layer utilities
    hidden_utilities = []
    for l in range(n_layers - 1):
        grad_W = loss_grads.layers[l].weight
        a_below = activations[l]
        a_below_norm_sq = jnp.sum(a_below * a_below)
        dL_dz = (grad_W @ a_below) / jnp.maximum(a_below_norm_sq, 1e-10)

        f_prime = jax.vmap(jax.grad(act_fn))(pre_activations[l])
        f_prime = jnp.maximum(f_prime, 1e-6)
        U_l = -(dL_dz / f_prime) * activations[l + 1]
        hidden_utilities.append(U_l)

    # Input utility: dL/dx = W_0.T @ dL/dz_0, U = -(dL/dx) * x
    grad_W0 = loss_grads.layers[0].weight
    x_norm_sq = jnp.sum(x * x)
    dL_dz0 = (grad_W0 @ x) / jnp.maximum(x_norm_sq, 1e-10)
    dL_dx = model.layers[0].weight.T @ dL_dz0
    U_input = -dL_dx * x

    return U_input, tuple(hidden_utilities)


def si_utility(model, x, y_star, y_hat, loss_grads, updates,
               activation='sigmoid', masks=None):
    """Synaptic Intelligence utility for arbitrary-depth networks.

    omega_k = (-dL/dtheta_k) * delta_theta_k (Zenke et al. 2017).
    Per-unit utility = sum of omega over outgoing weights from that unit.
    Activation parameter is unused (SI depends only on weight gradients/updates).
    Masks are applied to zero out utilities of pruned units.
    """
    n_layers = len(model.layers)

    # Input utility: sum omega over outgoing weights from each input
    U_input = jnp.sum(-loss_grads.layers[0].weight * updates.layers[0].weight, axis=0)

    # Hidden layer utilities: sum omega over outgoing weights from each unit
    hidden_utilities = []
    for l in range(n_layers - 1):
        grad_W = loss_grads.layers[l + 1].weight
        upd_W = updates.layers[l + 1].weight
        U_l = jnp.sum(-grad_W * upd_W, axis=0)
        if masks is not None:
            U_l = U_l * masks[l]
        hidden_utilities.append(U_l)

    return U_input, tuple(hidden_utilities)


APPROACH_CONFIGS = {
    "A": UtilityConfig(
        child_score_method="proportional",
        output_normalization="none",
    ),
    "B": UtilityConfig(
        child_score_method="loo",
        output_normalization="none",
        pseudo_error="abs_derivative",
        hidden_normalization="absolute",
        cap=False,
    ),
    "C": UtilityConfig(  # Approximate: C uses alignment-threshold fallback, we use zero-sum fallback
        child_score_method="loo",
        output_normalization="signed",
        pseudo_error="abs_derivative",
        hidden_normalization="signed",
        cap=False,
    ),
    "F": UtilityConfig(
        child_score_method="loo",
        output_normalization="signed",
        pseudo_error="abs_derivative",
        hidden_normalization="signed",
        cap=True,
    ),
    "G": UtilityConfig(
        child_score_method="loo",
        output_normalization="none",
        pseudo_error="target_prop",
        hidden_normalization="signed",
        cap=True,
    ),
    "H": UtilityConfig(
        child_score_method="loo",
        output_normalization="none",
        pseudo_error="abs_derivative",
        hidden_normalization="absorption",
        cap=False,
    ),
    "I": UtilityConfig(
        child_score_method="coherence",
        output_normalization="none",
    ),
    "J": UtilityConfig(
        child_score_method="loo",
        output_normalization="signed",
        pseudo_error="signed_derivative",
        hidden_normalization="overflow_only",
        cap=False,
    ),
    "K": UtilityConfig(
        child_score_method="loo",
        output_normalization="signed",
        pseudo_error="target_prop",
        hidden_normalization="overflow_only",
        cap=False,
    ),
    "L": UtilityConfig(
        child_score_method="loo",
        output_normalization="none",
        pseudo_error="signed_derivative",
        hidden_normalization="absolute",
        cap=False,
    ),
    "M": UtilityConfig(
        child_score_method="loo",
        output_normalization="none",
        pseudo_error="target_prop",
        hidden_normalization="absolute",
        cap=False,
    ),
}


def config_for_approach(name):
    """Return the UtilityConfig matching a named existing approach (A-M)."""
    return APPROACH_CONFIGS[name.upper()]
