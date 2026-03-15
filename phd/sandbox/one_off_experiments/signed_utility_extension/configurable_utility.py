"""Configurable utility function for measuring per-feature importance in multi-layer networks.

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
from itertools import product

import jax
import jax.numpy as jnp

from utility_functions import (
    _output_layer_loo,
    _approach_a_redistribution,
    _target_prop_pseudo_error,
)

CHILD_SCORE_METHODS = ("loo", "proportional", "coherence")
NORMALIZATION_METHODS = ("none", "absolute", "signed", "overflow_only", "absorption")
PSEUDO_ERROR_METHODS = ("abs_derivative", "signed_derivative", "target_prop")


@dataclass(frozen=True)
class UtilityConfig:
    """Configuration for the configurable utility function.

    Existing approach mapping:
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


def configurable_utility(model, x, y_star, y_hat, loss_grads, updates, config):
    """Configurable utility function covering approaches A-M.

    Same interface as the individual utility functions, plus a config parameter.
    """
    W1 = model.layers[0].weight  # (H, N)
    w_out = model.layers[1].weight.squeeze(0)  # (H,)
    z_hidden = W1 @ x
    a_hidden = jax.nn.sigmoid(z_hidden)

    # --- Output layer: LOO scores ---
    e, c_j, u_raw = _output_layer_loo(w_out, a_hidden, y_star, y_hat)
    error_reduced = jnp.abs(y_star) - jnp.abs(e)

    # --- Output normalization ---
    if config.output_normalization == "none":
        U_hidden = u_raw
    elif config.output_normalization == "absolute":
        abs_sum = jnp.maximum(jnp.sum(jnp.abs(u_raw)), 1e-10)
        U_hidden = u_raw * jnp.abs(error_reduced) / abs_sum
    elif config.output_normalization == "signed":
        signed_sum = jnp.sum(u_raw)
        scale = jnp.where(jnp.abs(signed_sum) > 1e-10, error_reduced / signed_sum, 1.0)
        U_hidden = u_raw * scale
    elif config.output_normalization == "overflow_only":
        signed_sum = jnp.sum(u_raw)
        signed_abs = jnp.abs(signed_sum)
        needs_norm = signed_abs > jnp.abs(error_reduced)
        safe_sum = jnp.where(signed_abs > 1e-10, signed_sum, 1.0)
        normed = u_raw * error_reduced / safe_sum
        U_hidden = jnp.where(needs_norm, normed, u_raw)
    elif config.output_normalization == "absorption":
        abs_sum = jnp.sum(jnp.abs(u_raw))
        scale = jnp.minimum(1.0, jnp.abs(error_reduced) / jnp.maximum(abs_sum, 1e-10))
        U_hidden = u_raw * scale

    # --- Hidden layer redistribution ---
    if config.child_score_method == "proportional":
        U_input = _approach_a_redistribution(W1, x, U_hidden)
        return U_input, U_hidden

    if config.child_score_method == "coherence":
        contributions = W1 * x[None, :]
        z_j = jnp.sum(contributions, axis=1, keepdims=True)
        Sigma = jnp.sum(jnp.abs(contributions), axis=1, keepdims=True)
        Sigma_safe = jnp.maximum(Sigma, 1e-10)
        beta = jnp.abs(z_j) / Sigma_safe
        sign_z = jnp.sign(z_j)
        weights = (sign_z * contributions + (1.0 - beta) * jnp.abs(contributions)) / Sigma_safe
        U_from_j = U_hidden[:, None] * weights
        U_input = jnp.sum(U_from_j, axis=0)
        return U_input, U_hidden

    # --- LOO child scores ---
    # Pseudo-error
    if config.pseudo_error == "abs_derivative":
        f_prime = jnp.maximum(a_hidden * (1.0 - a_hidden), 1e-6)
        e_j = jnp.abs(U_hidden) / f_prime
    elif config.pseudo_error == "signed_derivative":
        f_prime = jnp.maximum(a_hidden * (1.0 - a_hidden), 1e-6)
        e_j = U_hidden / f_prime
    elif config.pseudo_error == "target_prop":
        e_j = _target_prop_pseudo_error(w_out, U_hidden, a_hidden, z_hidden)

    # LOO scores
    contributions = W1 * x[None, :]  # (H, N)
    s_raw = jnp.abs(e_j[:, None] + contributions) - jnp.abs(e_j[:, None])

    # --- Hidden normalization ---
    if config.hidden_normalization == "none":
        U_from_j = s_raw

    elif config.hidden_normalization == "absolute":
        s_abs_sum = jnp.maximum(jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True), 1e-10)
        U_from_j = s_raw * jnp.abs(U_hidden[:, None]) / s_abs_sum

    elif config.hidden_normalization == "signed":
        s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)
        can_normalize = jnp.abs(s_signed_sum) > 1e-10
        s_signed_safe = jnp.where(can_normalize, s_signed_sum, 1.0)
        U_signed = s_raw * U_hidden[:, None] / s_signed_safe
        # Proportional fallback when Σs ≈ 0
        abs_contrib = jnp.abs(contributions)
        U_fallback = U_hidden[:, None] * abs_contrib / jnp.maximum(
            jnp.sum(abs_contrib, axis=1, keepdims=True), 1e-10
        )
        use_signed = can_normalize.astype(jnp.float32)
        U_from_j = use_signed * U_signed + (1.0 - use_signed) * U_fallback

    elif config.hidden_normalization == "overflow_only":
        s_signed_sum = jnp.sum(s_raw, axis=1, keepdims=True)
        s_signed_abs = jnp.abs(s_signed_sum)
        needs_norm = s_signed_abs > jnp.abs(U_hidden[:, None])
        s_signed_safe = jnp.where(s_signed_abs > 1e-10, s_signed_sum, 1.0)
        U_normed = s_raw * U_hidden[:, None] / s_signed_safe
        U_from_j = jnp.where(needs_norm, U_normed, s_raw)

    elif config.hidden_normalization == "absorption":
        s_abs_sum = jnp.sum(jnp.abs(s_raw), axis=1, keepdims=True)
        scale = jnp.minimum(1.0, jnp.abs(U_hidden[:, None]) / jnp.maximum(s_abs_sum, 1e-10))
        U_from_j = s_raw * scale

    # Post-normalization cap
    if config.cap:
        U_from_j = jnp.clip(U_from_j, -jnp.abs(U_hidden[:, None]), jnp.abs(U_hidden[:, None]))

    U_input = jnp.sum(U_from_j, axis=0)
    return U_input, U_hidden


def make_utility_fn(config):
    """Factory: returns a standard-interface utility function from a config."""
    def fn(model, x, y_star, y_hat, loss_grads, updates):
        return configurable_utility(model, x, y_star, y_hat, loss_grads, updates, config)
    fn.__name__ = (
        f"configurable_"
        f"{config.child_score_method}_"
        f"{config.output_normalization}_"
        f"{config.pseudo_error}_"
        f"{config.hidden_normalization}_"
        f"cap{config.cap}"
    )
    return fn


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
