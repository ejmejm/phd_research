"""Tests for the connection maturity stages used by multi-GEOFF Experiments 2 and 3.

Three properties the stage machinery rests on:

* `perturbed_forward` is the plain forward pass at zero perturbation, and the per-unit errors it
  exposes satisfy ``loss_grads == -delta_u * x``. That identity is the whole justification for
  calling ``-dL/dz`` "the per-unit error" in the plan's pseudocode.
* The stage transitions fire exactly when the plan says they do, and not otherwise.
* A reset makes a connection indistinguishable from a fresh one, and touches nothing else.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phd.jax_core.optimizers.idbd import optax_idbd
from phd.jax_core.utils import tree_replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multi_geoff_exploration'))
from masked_mlp import (  # noqa: E402
    MaskedTwoLayerMLP, block_sparse_masks, contaminate_masks, slot_ids, thirds_group_ids,
)
from connection_stages import (  # noqa: E402
    GROWTH, MATURE, MIN_OCCUPANCY, NASCENT, NEVER, TRACE_K, StageState,
    deltas_from_perturbation_grads, hidden_unit_stages, init_stage_state, jitter_thresholds,
    perturbed_forward, prune_test, reset_connections_at, sample_reset_schedule,
    update_stage_state, zero_perturbations,
)

N_TASKS = 2
N_FEATURES_PER_TASK = 20
N_OUTPUTS_PER_TASK = 10
N_HIDDEN = 64
INIT_LR = 1e-5
META_LR = 0.005

LAYERS = ('w1', 'w2')
INPUT_TASK_IDS = slot_ids(N_TASKS, N_FEATURES_PER_TASK)
OUTPUT_TASK_IDS = slot_ids(N_TASKS, N_OUTPUTS_PER_TASK)


def _model(key=0):
    base_masks, hidden_task_ids = block_sparse_masks(
        N_TASKS, N_FEATURES_PER_TASK, N_OUTPUTS_PER_TASK, N_HIDDEN)
    masks = contaminate_masks(
        base_masks, hidden_task_ids, thirds_group_ids(hidden_task_ids, N_TASKS),
        INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    return MaskedTwoLayerMLP(masks, key=jax.random.PRNGKey(key)), masks


def _sample(key=1):
    k_x, k_y = jax.random.split(jax.random.PRNGKey(key))
    x = jax.random.uniform(k_x, (N_TASKS * N_FEATURES_PER_TASK,), minval=-1.0, maxval=1.0)
    y = jax.random.normal(k_y, (N_TASKS * N_OUTPUTS_PER_TASK,))
    return x, y


# ------------------------------------------------------------------ the forward pass / deltas
def test_perturbed_forward_matches_plain_forward():
    """At zero perturbation it must be the model's own forward pass, output and `param_inputs`."""
    model, masks = _model()
    x, _ = _sample()

    out_ref, inputs_ref = model(x)
    out, param_inputs = perturbed_forward(model, x, zero_perturbations(masks))

    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=0, atol=0)
    for layer in LAYERS:
        np.testing.assert_allclose(
            np.asarray(param_inputs[layer]), np.asarray(inputs_ref[layer]), rtol=0, atol=0)


def test_unit_deltas_reproduce_the_loss_gradients():
    """``loss_grads == -delta_u * x`` on every existing connection.

    This is the identity that lets the plan write the network update as one linear-LMS problem per
    unit, and it is what fixes the sign and the factor-2 convention of `delta_u`.
    """
    model, masks = _model()
    x, y = _sample()

    def loss_fn(weights, perturb):
        out, param_inputs = perturbed_forward(tree_replace(model, weights=weights), x, perturb)
        return jnp.sum((out - y) ** 2), param_inputs

    (_, param_inputs), (grads, perturb_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True)(model.weights, zero_perturbations(masks))
    deltas = deltas_from_perturbation_grads(perturb_grads)

    for layer in LAYERS:
        reconstructed = -deltas[layer][:, None] * param_inputs[layer]
        np.testing.assert_allclose(
            np.asarray(grads[layer] * masks[layer]),
            np.asarray(reconstructed * masks[layer]), rtol=1e-5, atol=1e-6)


def test_output_unit_delta_is_twice_the_prediction_error():
    """The loss is squared error summed over outputs, so ``delta_o = 2 * (y_o - out_o)``."""
    model, masks = _model()
    x, y = _sample()

    def loss_fn(perturb):
        out, _ = perturbed_forward(model, x, perturb)
        return jnp.sum((out - y) ** 2)

    deltas = deltas_from_perturbation_grads(jax.grad(loss_fn)(zero_perturbations(masks)))
    out, _ = model(x)
    np.testing.assert_allclose(
        np.asarray(deltas['w2']), 2.0 * np.asarray(y - out), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------- stage transitions
def _tiny_state(stage_values, v=4.0, c=1.0):
    """A 2-destination-unit, 2-source-unit single-layer setup with everything pinned by hand."""
    masks = {'w1': jnp.ones((2, 2))}
    state = StageState(
        stage={'w1': jnp.asarray(stage_values, dtype=jnp.int8)},
        v={'w1': jnp.full((2,), v)},
        c={'w1': jnp.full((2,), c)},
    )
    return masks, state


def _tiny_update(stage_values, weights, h_prev, h, alpha=1e-4, v=4.0, c=1.0):
    masks, state = _tiny_state(stage_values, v=v, c=c)
    new_state, _, j_w = update_stage_state(
        state, masks,
        alpha={'w1': jnp.full((2, 2), alpha)},
        # Zero source values freeze the traces, so v and c stay exactly where they were set.
        param_inputs={'w1': jnp.zeros((1, 2))},
        deltas={'w1': jnp.zeros(2)},
        weights={'w1': jnp.asarray(weights)},
        h_prev={'w1': jnp.asarray(h_prev)},
        h={'w1': jnp.asarray(h)},
    )
    return np.asarray(new_state.stage['w1']), np.asarray(j_w['w1'])


def test_nascent_enters_growth_only_above_the_w_threshold():
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    stage, j_w = _tiny_update([[NASCENT, NASCENT], [NASCENT, NASCENT]],
                              weights=zeros, h_prev=zeros, h=zeros)
    threshold = float(j_w[0, 0])
    assert np.all(stage == NASCENT), "A zero weight must not clear the jitter threshold!"

    above = 1.01 * threshold
    below = 0.99 * threshold
    stage, _ = _tiny_update([[NASCENT, NASCENT], [NASCENT, NASCENT]],
                            weights=[[above, below], [-above, -below]],
                            h_prev=zeros, h=zeros)
    np.testing.assert_array_equal(stage, [[GROWTH, NASCENT], [GROWTH, NASCENT]])


def test_growth_is_blocked_until_the_trace_is_occupied():
    """`c` below `MIN_OCCUPANCY` must hold every connection at nascent however large the weight."""
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    big = [[1e3, 1e3], [1e3, 1e3]]
    stage, _ = _tiny_update([[NASCENT, NASCENT], [NASCENT, NASCENT]], weights=big,
                            h_prev=zeros, h=zeros, c=MIN_OCCUPANCY - 0.01)
    assert np.all(stage == NASCENT)

    stage, _ = _tiny_update([[NASCENT, NASCENT], [NASCENT, NASCENT]], weights=big,
                            h_prev=zeros, h=zeros, c=MIN_OCCUPANCY + 0.01)
    assert np.all(stage == GROWTH)


def test_growth_matures_exactly_on_an_h_sign_flip():
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    stage, _ = _tiny_update(
        [[GROWTH, GROWTH], [GROWTH, NASCENT]],
        weights=zeros,
        h_prev=[[1.0, 1.0], [-1.0, 1.0]],
        h=[[-1.0, 1.0], [2.0, -1.0]],
    )
    # Both flip directions mature; h staying positive does not, and a nascent connection cannot
    # mature no matter what h does.
    np.testing.assert_array_equal(stage, [[MATURE, GROWTH], [MATURE, NASCENT]])

    # h landing exactly on zero is not a crossing. This matters because a reset sets h to 0, so
    # the step after a reset must not count as one.
    stage, _ = _tiny_update([[GROWTH, GROWTH], [GROWTH, GROWTH]], weights=zeros,
                            h_prev=[[1.0, -1.0], [0.0, 0.0]], h=[[0.0, 0.0], [1.0, -1.0]])
    assert np.all(stage == GROWTH)


def test_maturity_is_absorbing():
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    stage, _ = _tiny_update([[MATURE, MATURE], [MATURE, MATURE]], weights=zeros,
                            h_prev=[[1.0, 1.0], [1.0, 1.0]], h=[[-1.0, -1.0], [-1.0, -1.0]])
    assert np.all(stage == MATURE)


def test_inactive_positions_never_leave_nascent():
    masks = {'w1': jnp.asarray([[1.0, 0.0], [0.0, 0.0]])}
    state = StageState(
        stage={'w1': jnp.full((2, 2), NASCENT, dtype=jnp.int8)},
        v={'w1': jnp.full((2,), 4.0)},
        c={'w1': jnp.ones(2)},
    )
    new_state, _, _ = update_stage_state(
        state, masks,
        alpha={'w1': jnp.full((2, 2), 1e-4)},
        param_inputs={'w1': jnp.zeros((1, 2))},
        deltas={'w1': jnp.zeros(2)},
        weights={'w1': jnp.full((2, 2), 1e3)},
        h_prev={'w1': jnp.zeros((2, 2))},
        h={'w1': jnp.zeros((2, 2))},
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.stage['w1']), [[GROWTH, NASCENT], [NASCENT, NASCENT]])


def test_traces_follow_the_pseudocode():
    """One step of the v and c recursions against the closed form, with masking applied."""
    masks = {'w1': jnp.asarray([[1.0, 0.0]])}   # the second source is not connected
    state = StageState(
        stage={'w1': jnp.full((1, 2), NASCENT, dtype=jnp.int8)},
        v={'w1': jnp.zeros(1)},
        c={'w1': jnp.zeros(1)},
    )
    alpha, x, delta = 0.25, 2.0, 3.0
    new_state, _, _ = update_stage_state(
        state, masks,
        alpha={'w1': jnp.full((1, 2), alpha)},
        param_inputs={'w1': jnp.asarray([[x, 100.0]])},   # the masked source must not contribute
        deltas={'w1': jnp.asarray([delta])},
        weights={'w1': jnp.zeros((1, 2))},
        h_prev={'w1': jnp.zeros((1, 2))},
        h={'w1': jnp.zeros((1, 2))},
    )
    coeff = alpha * x ** 2 / TRACE_K
    np.testing.assert_allclose(float(new_state.v['w1'][0]), coeff * delta ** 2, rtol=1e-6)
    np.testing.assert_allclose(float(new_state.c['w1'][0]), coeff, rtol=1e-6)


def test_jitter_thresholds_are_the_documented_multiples():
    alpha = {'w1': jnp.asarray([[4e-4]])}
    v = {'w1': jnp.asarray([9.0])}
    c = {'w1': jnp.asarray([0.25])}
    j_h, j_w = jitter_thresholds(alpha, v, c)
    np.testing.assert_allclose(float(j_h['w1'][0, 0]), np.sqrt(4e-4 * 9.0) / 2, rtol=1e-6)
    np.testing.assert_allclose(float(j_w['w1'][0, 0]), np.sqrt(4e-4 * 9.0 / 0.25), rtol=1e-6)
    # J_w is 2x J_h up to the bias correction, which is the relationship the plan specifies.
    np.testing.assert_allclose(
        float(j_w['w1'][0, 0]) * np.sqrt(0.25), 2 * float(j_h['w1'][0, 0]), rtol=1e-6)


def test_prune_test_needs_both_a_zero_crossing_and_a_small_h():
    """Experiment 3's rule: the weight crossed zero *and* ``h^2 < alpha v / 4``."""
    alpha = {'w1': jnp.full((1, 4), 1e-4)}
    v = {'w1': jnp.asarray([4.0])}
    j_h = float(np.sqrt(1e-4 * 4.0) / 2)
    small, large = 0.5 * j_h, 1.5 * j_h

    fired = prune_test(
        alpha, v,
        h={'w1': jnp.asarray([[small, large, small, large]])},
        #      crossed   crossed   stayed    stayed
        w_prev={'w1': jnp.asarray([[1.0, 1.0, 1.0, 1.0]])},
        w={'w1': jnp.asarray([[-1.0, -1.0, 1.0, 1.0]])},
    )
    np.testing.assert_array_equal(np.asarray(fired['w1']), [[True, False, False, False]])

    # A weight landing exactly on zero has not crossed it, which is what keeps the step after a
    # reset (w = 0, h = 0) from firing the rule immediately.
    fired = prune_test(
        alpha, v,
        h={'w1': jnp.zeros((1, 4))},
        w_prev={'w1': jnp.asarray([[0.0, 0.0, 1.0, -1.0]])},
        w={'w1': jnp.asarray([[1.0, -1.0, 0.0, 0.0]])},
    )
    assert not np.any(np.asarray(fired['w1']))


def test_prune_test_agrees_with_the_squared_form():
    """``|h| < J_h`` must be exactly the plan's ``h^2 < alpha v / 4``."""
    key = jax.random.PRNGKey(0)
    alpha = {'w1': jax.random.uniform(key, (3, 5), minval=1e-6, maxval=1e-2)}
    v = {'w1': jnp.asarray([0.5, 2.0, 8.0])}
    h = {'w1': jax.random.normal(jax.random.PRNGKey(1), (3, 5)) * 1e-2}
    crossed = {'w1': jnp.full((3, 5), -1.0)}    # every weight crosses, isolating the h test

    fired = prune_test(alpha, v, h, crossed, {'w1': jnp.ones((3, 5))})
    expected = np.square(np.asarray(h['w1'])) < np.asarray(alpha['w1']) * np.asarray(
        v['w1'])[:, None] / 4
    np.testing.assert_array_equal(np.asarray(fired['w1']), expected)


def test_hidden_unit_stage_is_the_most_advanced_outgoing_connection():
    masks = {'w2': jnp.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])}
    stage = {'w2': jnp.asarray([[GROWTH, NASCENT, MATURE], [MATURE, MATURE, MATURE]],
                               dtype=jnp.int8)}
    # Unit 0 has a mature outgoing connection; unit 1's only *existing* one is nascent; unit 2 has
    # none at all, and its inactive positions must not count.
    np.testing.assert_array_equal(
        np.asarray(hidden_unit_stages(stage, masks)), [MATURE, NASCENT, NASCENT])


# ------------------------------------------------------------------------------- resetting
def test_reset_restores_the_initial_condition_and_nothing_else():
    model, masks = _model()
    x, y = _sample()
    optimizer = optax_idbd(init_lr=INIT_LR, meta_lr=META_LR, autostep=True,
                           version='squared_inputs')
    opt_state = optimizer.init(model.weights)
    stage_state = init_stage_state(masks)

    # Train a little so h, v and the step-sizes are genuinely away from their initial values.
    for _ in range(20):
        def loss_fn(weights):
            out, param_inputs = tree_replace(model, weights=weights)(x)
            return jnp.sum((out - y) ** 2), param_inputs

        (_, param_inputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(model.weights)
        updates, opt_state = optimizer.update(
            (grads, None, param_inputs), opt_state, model.weights)
        model = tree_replace(
            model, weights={k: model.weights[k] + updates[k] for k in updates})
    stage_state = tree_replace(
        stage_state, stage={k: jnp.full(m.shape, MATURE, dtype=jnp.int8)
                            for k, m in masks.items()})

    select = {k: jnp.zeros(m.shape).at[0, 0].set(1.0) for k, m in masks.items()}
    before = jax.tree.map(np.asarray, (model.weights, opt_state.beta, opt_state.h, opt_state.v))
    new_model, new_opt_state, new_stage_state = reset_connections_at(
        model, opt_state, stage_state, select)

    for layer in LAYERS:
        assert float(new_model.weights[layer][0, 0]) == 0.0
        assert float(new_opt_state.h[layer][0, 0]) == 0.0
        assert float(new_opt_state.v[layer][0, 0]) == 0.0
        np.testing.assert_allclose(
            float(jnp.exp(new_opt_state.beta[layer][0, 0])), INIT_LR, rtol=1e-6)
        assert int(new_stage_state.stage[layer][0, 0]) == NASCENT

        # Everything else is bit-identical, and every other stage is untouched.
        for old, new in zip(before, (new_model.weights, new_opt_state.beta,
                                     new_opt_state.h, new_opt_state.v)):
            np.testing.assert_array_equal(old[layer][1:], np.asarray(new[layer])[1:])
            np.testing.assert_array_equal(old[layer][0, 1:], np.asarray(new[layer])[0, 1:])
        assert np.all(np.asarray(new_stage_state.stage[layer])[1:] == MATURE)

    # The per-unit traces are shared across a unit's fan-in and must survive a reset.
    for layer in LAYERS:
        np.testing.assert_array_equal(
            np.asarray(new_stage_state.v[layer]), np.asarray(stage_state.v[layer]))


def test_reset_schedule_covers_only_active_connections_and_is_reproducible():
    _, masks = _model()
    window = (100, 200)
    schedule = sample_reset_schedule(masks, jax.random.PRNGKey(0), 0.4, window)
    again = sample_reset_schedule(masks, jax.random.PRNGKey(0), 0.4, window)

    for layer in LAYERS:
        sched = np.asarray(schedule[layer])
        np.testing.assert_array_equal(sched, np.asarray(again[layer]))
        picked = sched != NEVER
        assert np.all(np.asarray(masks[layer])[picked] > 0), \
            "An inactive position was scheduled for a reset!"
        assert np.all((sched[picked] >= window[0]) & (sched[picked] < window[1]))
        # Independent selection, so the coverage is binomial around the requested fraction.
        n_active = int(np.asarray(masks[layer]).sum())
        assert abs(picked.sum() / n_active - 0.4) < 0.05

    assert np.all(np.asarray(sample_reset_schedule(
        masks, jax.random.PRNGKey(0), 0.0, window)['w1']) == NEVER)
