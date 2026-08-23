"""Connection maturity stages for the multi-GEOFF experiments.

Implements the plan's stage-tracking rule: every connection is **nascent** when created, enters
**growth** once its weight clears a jitter threshold, and becomes **mature** once its Autostep
gradient trace ``h`` crosses zero while in growth. A unit's stage is the most advanced stage among
its outgoing connections, so a hidden unit matures when its first outgoing connection does.

Written for Experiment 2, which validates the rule against ground-truth useful / useless labels,
and imported unchanged by Experiment 3's maturity-gated generation. Everything here is a pure
function over the same ``{'w1': ..., 'w2': ...}`` dicts that `masked_mlp` uses, with static shapes
and no data-dependent branching, so it runs inside ``jax.lax.scan``.

Design notes
------------
* Two different quantities are called ``v`` in this codebase and they must not be confused.
  Autostep's ``IDBDState.v`` is a **per-connection** normalizer living in the optimizer state; the
  ``v`` here is the plan's **per-destination-unit** squared-error trace. This module only owns the
  latter, and indexes it by layer name — ``v['w1']`` is a trace per *hidden* unit (the destinations
  of ``w1``), ``v['w2']`` one per *output* unit.
* Every ``x²`` is the squared value of the connection's source unit (`param_inputs`), the Variant B
  convention Experiment 1 settled on.
* The per-unit error ``delta_u`` is ``-dL/dz_u`` at the unit's *pre-activation* ``z_u``. That is the
  convention under which ``w <- w + alpha * delta_u * x`` is exactly the gradient step, i.e.
  ``loss_grads == -delta_u * x`` elementwise. Since the loss is squared error *summed* over
  outputs, an output unit's ``delta`` is ``2 * (y - out)``; the factor 2 is carried consistently by
  every gradient in these experiments. `perturbed_forward` is what exposes those pre-activations.
"""

from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int

from phd.jax_core.models import ACTIVATION_MAP
from phd.jax_core.optimizers.idbd import IDBDState

from masked_mlp import Masks, MaskedTwoLayerMLP, Weights, reset_idbd_state_at


# Connection stages. Ordered so that `max` over a unit's outgoing connections gives its stage.
NASCENT = 0
GROWTH = 1
MATURE = 2
N_STAGES = 3
STAGE_NAMES = ('nascent', 'growth', 'mature')

# Trace time constant of the per-unit v and c traces. A constant per the plan, not a
# hyperparameter.
TRACE_K = 5.0

# Jitter-threshold factors. `h` is jitter-sized when h² < alpha*v/4, i.e. |h| < sqrt(alpha*v)/2, so
# the h factor is 1/2; the w threshold is 2x that, on the bias-corrected v.
H_JITTER_FACTOR = 0.5
W_JITTER_FACTOR = 1.0
# The growth stage is unreachable until the v trace is at least half occupied, otherwise v starts
# at 0 and every connection would clear a zero threshold on the first step.
MIN_OCCUPANCY = 0.5

# Sentinel for "this connection is never reset" / "this transition never happened".
NEVER = -1


UnitValues = Dict[str, Float[Array, '...']]
Stages = Dict[str, Int[Array, '...']]


class StageState(eqx.Module):
    """Per-connection stages plus the per-destination-unit traces they are computed from.

    Attributes:
        stage: Per-layer stage of every connection, shaped like the masks
        v: Per-destination-unit trace of the squared per-unit error
        c: Per-destination-unit occupancy of that trace (its bias-correction term)
    """

    stage: Stages
    v: UnitValues
    c: UnitValues


def init_stage_state(masks: Masks) -> StageState:
    """All connections nascent, both traces empty.

    Args:
        masks: Connectivity masks, ``{'w1': (n_hidden, n_features), 'w2': (n_outputs, n_hidden)}``

    Returns:
        The initial `StageState`
    """
    return StageState(
        stage={k: jnp.full(m.shape, NASCENT, dtype=jnp.int8) for k, m in masks.items()},
        v={k: jnp.zeros(m.shape[0]) for k, m in masks.items()},
        c={k: jnp.zeros(m.shape[0]) for k, m in masks.items()},
    )


# ------------------------------------------------------------------------- per-unit errors
def zero_perturbations(masks: Masks) -> UnitValues:
    """Zero perturbation for every hidden and output unit, one entry per layer's destinations."""
    return {k: jnp.zeros(m.shape[0]) for k, m in masks.items()}


def perturbed_forward(
    model: MaskedTwoLayerMLP,
    x: Float[Array, 'n_features'],
    perturb: UnitValues,
) -> Tuple[Float[Array, 'n_outputs'], Weights]:
    """`MaskedTwoLayerMLP.__call__` with an additive perturbation at each pre-activation.

    Differentiating a loss built on this w.r.t. `perturb` gives ``dL/dz`` for every hidden and
    output unit, which is the only thing the plain forward pass does not expose. At zero
    perturbation it is numerically identical to the model's own ``__call__``, so the weight
    gradients are unaffected — the perturbations exist purely as differentiation handles.

    Args:
        model: The masked MLP
        x: A single input sample
        perturb: Per-layer perturbation of the destination units' pre-activations

    Returns:
        Tuple of the output and the per-weight source values (Autostep's `param_inputs`)
    """
    activation_fn = ACTIVATION_MAP[model.activation]
    hidden_pre = (model.weights['w1'] * model.masks['w1']) @ x + perturb['w1']
    hidden = activation_fn(hidden_pre)
    output = (model.weights['w2'] * model.masks['w2']) @ hidden + perturb['w2']
    return output, {'w1': x[None, :], 'w2': hidden[None, :]}


def deltas_from_perturbation_grads(perturb_grads: UnitValues) -> UnitValues:
    """Per-unit errors from the gradients w.r.t. the pre-activation perturbations.

    ``delta_u = -dL/dz_u``, the sign convention that makes ``w <- w + alpha * delta_u * x`` the
    gradient step (see the module docstring).
    """
    return {k: -g for k, g in perturb_grads.items()}


# ----------------------------------------------------------------------------- thresholds
def h_jitter_threshold(alpha: Weights, v: UnitValues) -> Weights:
    """``J_h = sqrt(alpha_i * v_u) / 2``, i.e. the point below which ``h_i^2 < alpha_i v_u / 4``.

    Shared by the maturity rule and Experiment 3's prune rule, which both turn on the same test.
    """
    return {k: H_JITTER_FACTOR * jnp.sqrt(alpha[k] * v[k][:, None]) for k in alpha}


def jitter_thresholds(
    alpha: Weights, v: UnitValues, c: UnitValues,
) -> Tuple[Weights, Weights]:
    """The h and w jitter thresholds of every connection.

    ``J_h = sqrt(alpha_i * v_u) / 2`` uses the raw trace; ``J_w = sqrt(alpha_i * v_u / c_u)`` is
    2x that on the bias-corrected trace. Both are per connection through ``alpha`` and per
    destination unit through ``v`` and ``c``, so the unit values broadcast over the incoming axis.

    Args:
        alpha: Per-connection step-sizes
        v: Per-destination-unit squared-error trace
        c: Per-destination-unit trace occupancy

    Returns:
        Tuple of the per-connection ``(J_h, J_w)``
    """
    j_h = h_jitter_threshold(alpha, v)
    # `c` is 0 before the trace has seen anything; the MIN_OCCUPANCY gate blocks growth over that
    # whole period, so the +eps only keeps the intermediate finite.
    j_w = {k: W_JITTER_FACTOR * jnp.sqrt(alpha[k] * v[k][:, None] / (c[k][:, None] + 1e-30))
           for k in alpha}
    return j_h, j_w


# --------------------------------------------------------------------------- stage update
def update_stage_state(
    state: StageState,
    masks: Masks,
    alpha: Weights,
    param_inputs: Weights,
    deltas: UnitValues,
    weights: Weights,
    h_prev: Weights,
    h: Weights,
) -> Tuple[StageState, Weights, Weights]:
    """One step of the plan's stage-tracking rule, run after the normal training update.

    Per destination unit u, with incoming weights w, source values x and per-unit error delta::

        v <- v + (1/k) * sum_i alpha_i x_i^2 * (delta_u^2 - v)
        c <- c + (1/k) * sum_i alpha_i x_i^2 * (1 - c)

    then, per incoming connection i, ``nascent -> growth`` when the trace is occupied enough and
    the weight clears its jitter threshold, and ``growth -> mature`` when h crosses zero. Maturity
    is absorbing until the connection is reset (`reset_connections_at`) or pruned.

    The ``sum_i alpha_i x_i^2`` coefficient is masked, so inactive positions never contribute. It
    is the same quantity Autostep clips its effective-step-size normalizer on, and that clip holds
    it at or below 1, so ``(1/k) * sum`` never exceeds ``1/k`` and both traces are stable.

    Args:
        state: Current `StageState`
        masks: Connectivity masks
        alpha: Per-connection step-sizes *after* this step's Autostep update
        param_inputs: Per-weight source values, ``(1, in_features)`` per layer
        deltas: Per-destination-unit errors (see `deltas_from_perturbation_grads`)
        weights: Per-connection weights after this step's update
        h_prev: Autostep's gradient trace before this step's update
        h: Autostep's gradient trace after it

    Returns:
        Tuple of the new `StageState` and the ``(J_h, J_w)`` thresholds used this step, which the
        callers record for the diagnostic plots
    """
    # sum_i alpha_i x_i^2 over each destination unit's existing incoming connections.
    weighted_alpha = {
        k: jnp.sum(masks[k] * alpha[k] * jnp.square(param_inputs[k]), axis=-1) for k in masks
    }

    v = {k: state.v[k] + weighted_alpha[k] / TRACE_K * (jnp.square(deltas[k]) - state.v[k])
         for k in masks}
    c = {k: state.c[k] + weighted_alpha[k] / TRACE_K * (1.0 - state.c[k]) for k in masks}

    j_h, j_w = jitter_thresholds(alpha, v, c)

    stage = {}
    for k in masks:
        to_growth = ((state.stage[k] == NASCENT)
                     & (c[k][:, None] > MIN_OCCUPANCY)
                     & (jnp.abs(weights[k]) > j_w[k]))
        # `else if` in the pseudocode: a connection cannot skip nascent -> mature in one step.
        to_mature = (state.stage[k] == GROWTH) & (h_prev[k] * h[k] < 0)
        new_stage = jnp.where(
            to_growth, GROWTH, jnp.where(to_mature, MATURE, state.stage[k])).astype(jnp.int8)
        # Inactive positions have no connection and are held at nascent so they never contaminate
        # a unit-level stage.
        stage[k] = jnp.where(masks[k] > 0, new_stage, jnp.int8(NASCENT))

    return StageState(stage=stage, v=v, c=c), j_h, j_w


def prune_test(
    alpha: Weights,
    v: UnitValues,
    h: Weights,
    w_prev: Weights,
    w: Weights,
) -> Dict[str, Array]:
    """Experiment 3's prune condition: the weight crossed zero this step and h is small.

    ``h_i^2 < alpha_i v_u / 4`` is the same test as ``|h_i| < J_h``, so the prune rule and the
    maturity rule are two readings of the same jitter threshold — maturity asks whether ``h``
    crossed zero, pruning asks whether ``w`` did while ``h`` was too small to matter.

    Nothing here prunes anything; the caller decides what to do with the mask. Experiment 2 only
    records when it first fires, so the rule can be read against ground-truth labels before
    Experiment 3 acts on it.

    Args:
        alpha: Per-connection step-sizes after this step's Autostep update
        v: Per-destination-unit squared-error trace, after this step's update
        h: Autostep's gradient trace after this step's update
        w_prev: Per-connection weights before this step's update
        w: Per-connection weights after it

    Returns:
        A per-layer boolean mask of the connections the rule would prune this step
    """
    j_h = h_jitter_threshold(alpha, v)
    return {k: (w_prev[k] * w[k] < 0) & (jnp.abs(h[k]) < j_h[k]) for k in w}


def hidden_unit_stages(
    stage: Stages, masks: Masks, layer: str = 'w2',
) -> Int[Array, 'n_hidden']:
    """Stage of each hidden unit: the most advanced stage among its outgoing connections.

    A hidden unit with no outgoing connection at all reads as `NASCENT`. Input units are mature
    from the first step and are not represented here.

    Args:
        stage: Per-connection stages
        masks: Connectivity masks
        layer: The layer whose *columns* are the units in question (``'w2'`` for hidden units)

    Returns:
        One stage per unit
    """
    active = masks[layer] > 0
    return jnp.max(jnp.where(active, stage[layer], NASCENT), axis=0)


# ---------------------------------------------------------------------------- resetting
def reset_connections_at(
    model: MaskedTwoLayerMLP,
    opt_state: IDBDState,
    stage_state: StageState,
    select: Dict[str, Array],
) -> Tuple[MaskedTwoLayerMLP, IDBDState, StageState]:
    """Make the selected connections look exactly like freshly generated ones.

    Weight to 0, stage back to nascent, and the Autostep state (step-size, gradient trace h,
    per-connection normalizer v) back to its initial value via `masked_mlp.reset_idbd_state_at`.

    The per-*unit* v and c traces are deliberately left alone: they aggregate over a unit's whole
    incoming fan-in and have no per-connection component to reset, and clearing them would blind
    the thresholds of every *other* connection into the same unit.

    Args:
        model: The masked MLP
        opt_state: Current Autostep state
        stage_state: Current `StageState`
        select: Per-layer boolean (or 0/1) masks of the positions to reset

    Returns:
        Tuple of the new model, optimizer state and stage state
    """
    picked = {k: s > 0 for k, s in select.items()}
    model = eqx.tree_at(
        lambda m: m.weights, model,
        {k: jnp.where(picked[k], 0.0, w) for k, w in model.weights.items()})
    opt_state = reset_idbd_state_at(opt_state, select)
    stage_state = eqx.tree_at(
        lambda s: s.stage, stage_state,
        {k: jnp.where(picked[k], jnp.int8(NASCENT), st) for k, st in stage_state.stage.items()})
    return model, opt_state, stage_state


def sample_reset_schedule(
    masks: Masks,
    key: random.PRNGKey,
    fraction: float,
    window: Tuple[int, int],
) -> Dict[str, Int[Array, '...']]:
    """Draw a per-connection reset step, or `NEVER` for connections that are not reset.

    Each existing connection is selected independently with probability `fraction` and given a step
    drawn uniformly from `window`. Independent selection rather than sampling a fixed count keeps
    this a two-line branchless draw and gives every group the same expected coverage, which is all
    the statistics need.

    Args:
        masks: Connectivity masks
        key: PRNG key
        fraction: Probability that a given connection is reset
        window: Inclusive-exclusive ``(first_step, last_step)`` the reset steps are drawn from

    Returns:
        A per-layer int32 array of reset steps, `NEVER` where no reset is scheduled
    """
    lo, hi = window
    assert 0 <= fraction <= 1, f"fraction must be a probability, got {fraction}!"
    assert lo < hi, f"Empty reset window: {window}!"

    schedule = {}
    # Folded in by position in the sorted key order, not by `hash(k)`: string hashing is salted per
    # process, which would make the schedule differ between runs of the same notebook.
    for i, k in enumerate(sorted(masks)):
        mask = masks[k]
        pick_key, step_key = random.split(random.fold_in(key, i))
        picked = (mask > 0) & (random.uniform(pick_key, mask.shape) < fraction)
        steps = random.randint(step_key, mask.shape, lo, hi)
        schedule[k] = jnp.where(picked, steps, NEVER).astype(jnp.int32)
    return schedule
