"""Tests for the revised Experiment 3 network — biases, fixed +-1 feedback, zero-crossing pruning.

The properties pinned down are the ones whose violation would quietly invalidate a run rather than
crash it: the explicit backward pass is the autodiff gradient (with the feedback swapped in on the
hidden error path and nowhere else), the connection budget is conserved, only input connections are
ever pruned, generated connections are indistinguishable from initial ones, and the separation
metrics count paths the way the plan defines them.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phd.jax_core.utils import tree_replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multi_geoff_exploration'))
from masked_mlp import INACTIVE, USEFUL, USELESS, slot_ids  # noqa: E402
from dynamic_skip_mlp import (  # noqa: E402
    LAYERS, NO_TASK, DynamicAutostepMLP, SkipMaskedMLP, block_sparse_skip_masks, dense_skip_masks,
    group_quantiles, initial_structure, input_columns, n_connections, random_feedback,
    separation_metrics, structure_metrics, task_labels,
)

N_TASKS = 4
N_FEATURES_PER_TASK = 20
N_OUTPUTS_PER_TASK = 10
N_HIDDEN = 16
N_FEATURES = N_TASKS * N_FEATURES_PER_TASK
N_OUTPUTS = N_TASKS * N_OUTPUTS_PER_TASK
BUDGET = 1024
INIT_LR = 1e-8
META_LR = 0.02

INPUT_TASK_IDS = slot_ids(N_TASKS, N_FEATURES_PER_TASK)
OUTPUT_TASK_IDS = slot_ids(N_TASKS, N_OUTPUTS_PER_TASK)


def _structure(budget=BUDGET, seed=0):
    key = jax.random.key(seed)
    masks, hidden_output = initial_structure(N_FEATURES, N_HIDDEN, N_OUTPUTS, budget, key)
    return masks, random_feedback(masks, jax.random.fold_in(key, 1)), OUTPUT_TASK_IDS[hidden_output]


def _learner(masks, feedback=None, seed=0, init_lr=INIT_LR, labels=None, **kwargs):
    return DynamicAutostepMLP.init(masks, META_LR, jax.random.key(seed), init_lr=init_lr,
                                   feedback=feedback, labels=labels, **kwargs)


def _batch(n_steps, seed=0, scale=1.0):
    x_key, y_key = jax.random.split(jax.random.key(seed))
    return (jax.random.uniform(x_key, (n_steps, N_FEATURES), minval=-1.0, maxval=1.0),
            scale * jax.random.normal(y_key, (n_steps, N_OUTPUTS)))


def _run(learner, n_steps, seed=0, scale=1.0):
    xs, ys = _batch(n_steps, seed=seed, scale=scale)
    step = jax.jit(lambda lr, x, y: lr.step(x, y))
    infos = []
    for x, y in zip(xs, ys):
        learner, info = step(learner, x, y)
        infos.append(info)
    return learner, {k: np.array([float(i[k]) for i in infos]) for k in infos[0]}


def _hidden_block(masks):
    return masks['w2'][:, N_FEATURES:N_FEATURES + N_HIDDEN]


# ------------------------------------------------------------------------ initial structure
def test_initial_structure_spends_the_budget_exactly():
    for budget in (BUDGET, BUDGET + 1, BUDGET + N_HIDDEN + N_OUTPUTS - 1):
        masks, _, _ = _structure(budget=budget)
        assert n_connections(masks) == budget, f'budget {budget}: got {n_connections(masks)}'


def test_initial_structure_gives_each_hidden_unit_one_output_and_every_unit_a_bias():
    masks, feedback, hidden_task_ids = _structure()
    hidden_out = _hidden_block(masks)
    assert np.array_equal(np.asarray(hidden_out.sum(axis=0)), np.ones(N_HIDDEN))
    assert bool((masks['w1'][:, -1] == 1).all()) and bool((masks['w2'][:, -1] == 1).all())
    # The feedback is +-1 exactly on the hidden->output connections and 0 elsewhere.
    assert set(np.unique(np.asarray(feedback[hidden_out > 0]))) <= {-1.0, 1.0}
    assert bool((feedback[hidden_out == 0] == 0).all())
    # A hidden unit's task is the task of the output it feeds.
    fed = jnp.argmax(hidden_out, axis=0)
    assert np.array_equal(np.asarray(hidden_task_ids), np.asarray(OUTPUT_TASK_IDS[fed]))


def test_initial_structure_rejects_an_impossible_budget():
    for budget in (N_HIDDEN + N_HIDDEN + N_OUTPUTS - 1, 10 ** 6):
        try:
            initial_structure(N_FEATURES, N_HIDDEN, N_OUTPUTS, budget, jax.random.key(0))
        except AssertionError:
            continue
        raise AssertionError(f'budget {budget} should have been rejected')


# --------------------------------------------------------------------------- the network
def test_forward_matches_the_explicit_computation():
    masks, feedback, _ = _structure()
    model = SkipMaskedMLP(masks, activation='leaky_relu', feedback=feedback, key=jax.random.key(1))
    x = jax.random.normal(jax.random.key(2), (N_FEATURES,))
    out, hidden_pre, param_inputs = model(x)

    w1, w2 = model.weights['w1'] * masks['w1'], model.weights['w2'] * masks['w2']
    pre = w1[:, :N_FEATURES] @ x + w1[:, -1]
    hidden = jax.nn.leaky_relu(pre)
    expected = w2[:, :N_FEATURES] @ x + w2[:, N_FEATURES:-1] @ hidden + w2[:, -1]
    assert np.allclose(np.asarray(out), np.asarray(expected), atol=1e-5)
    assert np.allclose(np.asarray(hidden_pre), np.asarray(pre), atol=1e-6)
    assert np.allclose(np.asarray(param_inputs['w2'][0]),
                       np.asarray(jnp.concatenate([x, hidden, jnp.ones(1)])), atol=1e-6)


def test_manual_gradients_match_autodiff_without_feedback():
    masks, _, _ = _structure()
    learner = _learner(masks, feedback=None, weight_init='lecun', activation='leaky_relu')
    x, y = _batch(1, seed=3)
    _, grads, _ = learner.loss_and_grads(x[0], y[0])

    def loss_fn(weights):
        out, _, _ = tree_replace(learner.model, weights=weights)(x[0])
        return jnp.sum((out - y[0]) ** 2)

    auto = jax.grad(loss_fn)(learner.model.weights)
    for k in LAYERS:
        assert np.allclose(np.asarray(grads[k]), np.asarray(auto[k]), atol=1e-5), k


def test_feedback_replaces_only_the_hidden_error_path():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, weight_init='lecun', activation='leaky_relu')
    x, y = _batch(1, seed=4)
    _, grads, _ = learner.loss_and_grads(x[0], y[0])
    model = learner.model

    # The forward value is the plain one, and w2's gradient is the ordinary one.
    plain = tree_replace(model, feedback=None)
    out, hidden_pre, param_inputs = model(x[0])
    out_plain, _, _ = plain(x[0])
    assert np.allclose(np.asarray(out), np.asarray(out_plain), atol=1e-6)
    d_out = 2.0 * (out - y[0])
    assert np.allclose(np.asarray(grads['w2']),
                       np.asarray(d_out[:, None] * param_inputs['w2'] * masks['w2']), atol=1e-5)

    # w1's gradient is backprop with the feedback in place of the hidden->output weights, and it
    # differs from true backprop.
    d_pre = (feedback.T @ d_out) * jax.vmap(jax.grad(jax.nn.leaky_relu))(hidden_pre)
    expected = d_pre[:, None] * param_inputs['w1'] * masks['w1']
    assert np.allclose(np.asarray(grads['w1']), np.asarray(expected), atol=1e-5)
    _, plain_grads, _ = tree_replace(learner, model=plain).loss_and_grads(x[0], y[0])
    assert not np.allclose(np.asarray(grads['w1']), np.asarray(plain_grads['w1']), atol=1e-3)


def test_ltu_backward_uses_the_surrogate_gradient():
    """With every weight tiny the LTU pre-activations sit at ~0, where the step function's true
    derivative is 0 but the surrogate is 0.25 — the input layer must still get a gradient."""
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, activation='ltu')
    x, y = _batch(1, seed=5)
    _, grads, _ = learner.loss_and_grads(x[0], y[0])
    assert float(jnp.abs(grads['w1'][:, :N_FEATURES]).max()) > 0.0


# ---------------------------------------------------------------------- initialization
def test_experiment_initialization():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, bias_init_lr=2 ** -4)
    w, alpha = learner.model.weights, learner.alpha
    cols = input_columns(masks)
    for k in LAYERS:
        inputs = np.asarray(w[k][cols[k] & (masks[k] > 0)])
        assert inputs.size > 0 and bool((np.abs(inputs) < INIT_LR).all()) and bool((inputs != 0).all()), k
        assert bool((w[k][~cols[k]] == 0).all()), f'{k}: hidden->output weights and biases must be 0'
        assert bool((w[k][masks[k] == 0] == 0).all()), k
        assert np.allclose(np.asarray(alpha[k][:, :-1]), INIT_LR, rtol=1e-5), k
        assert np.allclose(np.asarray(alpha[k][:, -1]), 2 ** -4, rtol=1e-5), k


def test_reference_initialization_is_the_ordinary_one():
    masks, _, _ = _structure()
    learner = _learner(masks, feedback=None, weight_init='lecun')
    w = learner.model.weights
    assert learner.model.feedback is None
    assert float(jnp.abs(w['w2'][:, N_FEATURES:-1]).max()) > 1e-3     # hidden->output not zero
    assert bool((w['w1'][:, -1] == 0).all()) and bool((w['w2'][:, -1] == 0).all())


# ----------------------------------------------------------------------- static training
def test_masked_connections_stay_inert_and_a_static_run_never_changes_connectivity():
    # Inactive positions keep their initial step-size only while Autostep's row normalizer
    # sum(alpha x^2) stays below 1 — it rescales every step-size in the row otherwise, inactive ones
    # included. That holds trivially at the experiment's 1e-8; 1e-3 keeps it true here too while
    # still letting weights move enough to cross zero.
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=1e-3, l1=0.001)
    trained, info = _run(learner, 300)
    for k in LAYERS:
        assert np.array_equal(np.asarray(trained.model.masks[k]), np.asarray(masks[k])), k
        inactive = np.asarray(masks[k] == 0)
        assert bool((np.asarray(trained.model.weights[k])[inactive] == 0).all()), k
        assert np.allclose(np.asarray(trained.alpha[k])[inactive], 1e-3, rtol=1e-5), k
    assert info['n_pruned'].sum() == 0 and info['n_generated'].sum() == 0
    assert info['n_crossed'].sum() > 0, 'crossings should still be counted in a static run'


def test_biases_move_at_their_own_step_size_and_are_exempt_from_l1():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, bias_init_lr=2 ** -4, l1=0.5)
    x, y = _batch(1, seed=6, scale=1.0)
    new, _ = learner.step(x[0], y[0])
    # Output biases: a big step-size and a unit input, so they move by roughly alpha * 2 * error.
    moved = np.asarray(new.model.weights['w2'][:, -1] - learner.model.weights['w2'][:, -1])
    assert float(np.abs(moved).max()) > 1e-3
    # An L1 of 0.5 on a zero bias would show up as a 0.5 * alpha-sized pull if it were applied; the
    # bias update must equal the plain gradient step alone.
    out, _, _ = learner.model(x[0])
    plain = -float(2 ** -4) * 2.0 * np.asarray(out - y[0])
    assert np.allclose(moved, plain, rtol=1e-4, atol=1e-7)


# ------------------------------------------------------------------------- restructuring
def test_pruning_conserves_the_budget_and_never_touches_hidden_output_or_biases():
    masks, feedback, hidden_task_ids = _structure()
    labels = task_labels({k: jnp.ones_like(m) for k, m in masks.items()},
                                     hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    learner = _learner(masks, feedback=feedback, init_lr=0.05, l1=0.001, restructure=True,
                       gen_cap=256, labels=labels)
    trained, info = _run(learner, 300)
    assert info['n_pruned'].sum() > 0, 'the test needs some pruning to happen'
    assert info['n_pruned'].sum() == info['n_crossed'].sum()
    assert np.array_equal(info['n_crossed'], info['n_crossed_useful'] + info['n_crossed_useless'])
    # Exactly budget - deficit connections at every step: check the final state.
    assert n_connections(trained.model.masks) == BUDGET - int(info['deficit'][-1])
    assert info['deficit'][-1] <= 2, 'the deficit should be refilled within a few steps'
    # The hidden->output block and the bias columns never change.
    assert np.array_equal(np.asarray(_hidden_block(trained.model.masks)), np.asarray(_hidden_block(masks)))
    for k in LAYERS:
        assert bool((trained.model.masks[k][:, -1] == 1).all()), k
        assert set(np.unique(np.asarray(trained.model.masks[k]))) <= {0.0, 1.0}, k


def test_generated_connections_start_fresh_and_only_on_input_pathways():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=0.05, restructure=True, gen_cap=256)
    x, y = _batch(200, seed=7, scale=3.0)
    step = jax.jit(lambda lr, x, y: lr.step(x, y))
    for t in range(200):
        before = learner
        learner, info = step(learner, x[t], y[t])
        if int(info['n_generated']) > 0:
            break
    else:
        raise AssertionError('no generation happened in 200 steps')

    cols = input_columns(masks)
    for k in LAYERS:
        new = np.asarray((learner.model.masks[k] > 0) & (before.model.masks[k] == 0))
        if not new.any():
            continue
        assert bool(np.asarray(cols[k])[new].all()), f'{k}: generated connection outside the input columns'
        w = np.asarray(learner.model.weights[k])[new]
        assert bool((np.abs(w) < 0.05).all()) and bool((w != 0).all()), k
        assert np.allclose(np.asarray(learner.alpha[k])[new], 0.05, rtol=1e-5), k
        assert bool((np.asarray(learner.opt_state.h[k])[new] == 0).all()), k
        assert bool((np.asarray(learner.opt_state.v[k])[new] == 0).all()), k


def test_a_saturated_generation_cap_is_accounted_for_not_hidden():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=0.05, restructure=True, gen_cap=1)
    trained, info = _run(learner, 100)
    assert info['n_pruned'].sum() > info['n_generated'].sum()
    assert info['deficit'][-1] == info['n_pruned'].sum() - info['n_generated'].sum()
    assert n_connections(trained.model.masks) == BUDGET - int(info['deficit'][-1])


def test_step_scans_under_jit():
    masks, feedback, hidden_task_ids = _structure()
    labels = task_labels({k: jnp.ones_like(m) for k, m in masks.items()},
                                     hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    learner = _learner(masks, feedback=feedback, init_lr=0.05, l1=0.001, bias_init_lr=2 ** -4,
                       restructure=True, gen_cap=64, labels=labels)
    xs, ys = _batch(50, seed=8)

    @jax.jit
    def train(learner):
        def one(lr, xy):
            lr, info = lr.step(*xy)
            return lr, info['loss']
        return jax.lax.scan(one, learner, (xs, ys))

    trained, losses = train(learner)
    assert losses.shape == (50,) and bool(jnp.isfinite(losses).all())
    assert n_connections(trained.model.masks) == BUDGET - int(trained.deficit)


# ---------------------------------------------------------------------------- labels
def test_task_labels():
    masks, hidden_task_ids = block_sparse_skip_masks(
        N_TASKS, N_FEATURES_PER_TASK, N_OUTPUTS_PER_TASK, N_HIDDEN)
    labels = task_labels(masks, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    # Block-sparse: every active input->hidden connection is same-task; no skips; nothing else labelled.
    active = np.asarray(masks['w1'][:, :N_FEATURES] > 0)
    assert bool((np.asarray(labels['w1'][:, :N_FEATURES])[active] == USEFUL).all())
    assert bool((np.asarray(labels['w1'][:, :N_FEATURES])[~active] == INACTIVE).all())
    assert bool((np.asarray(labels['w1'][:, -1]) == INACTIVE).all())
    hidden_block = np.asarray(labels['w2'][:, N_FEATURES:N_FEATURES + N_HIDDEN])
    hidden_active = np.asarray(masks['w2'][:, N_FEATURES:N_FEATURES + N_HIDDEN] > 0)
    assert bool((hidden_block[hidden_active] == USEFUL).all()) and bool((hidden_block[~hidden_active] == INACTIVE).all())
    assert bool((np.asarray(labels['w2'][:, :N_FEATURES]) == INACTIVE).all())      # no skips
    assert bool((np.asarray(labels['w2'][:, -1]) == INACTIVE).all())
    # Dense: hidden units belong to no task, so nothing is labelled.
    dense = dense_skip_masks(N_FEATURES, N_HIDDEN, N_OUTPUTS)
    dense_labels = task_labels(dense, jnp.full((N_HIDDEN,), NO_TASK), INPUT_TASK_IDS,
                                           OUTPUT_TASK_IDS)
    assert all(bool((np.asarray(v) == INACTIVE).all()) for v in dense_labels.values())
    # Experiment 3: about 1 / n_tasks of the input connections are same-task.
    masks3, _, tasks3 = _structure()
    labels3 = task_labels(masks3, tasks3, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    counts = {g: sum(int((labels3[k] == g).sum()) for k in LAYERS) for g in (USEFUL, USELESS)}
    assert counts[USEFUL] + counts[USELESS] == BUDGET
    # The 64 initial hidden->output connections are same-task by definition; among the input
    # connections about 1 / n_tasks are.
    assert bool((np.asarray(labels3['w2'][:, N_FEATURES:N_FEATURES + N_HIDDEN])[np.asarray(_hidden_block(masks3) > 0)] == USEFUL).all())
    assert 0.15 < (counts[USEFUL] - N_HIDDEN) / (BUDGET - N_HIDDEN) < 0.35


def test_group_quantiles_follow_the_current_masks():
    masks3, _, tasks3 = _structure()
    labels = task_labels({k: jnp.ones_like(m) for k, m in masks3.items()}, tasks3,
                                     INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    values = {k: jnp.full(m.shape, 3.0) for k, m in masks3.items()}
    q = group_quantiles(values, labels, masks3, (0.5,))
    assert q.shape == (3, 2, 1)
    assert np.allclose(np.asarray(q[:, 0]), 3.0)                 # same-task groups exist in all views
    assert np.allclose(np.asarray(q[:2, 1]), 3.0) and bool(jnp.isnan(q[2, 1]).all())   # no cross-task hidden->output yet
    empty = group_quantiles(values, labels, {k: jnp.zeros_like(m) for k, m in masks3.items()}, (0.5,))
    assert bool(jnp.isnan(empty).all())


# ---------------------------------------------------------------------------- metrics
def test_separation_of_the_reference_structures():
    masks, _ = block_sparse_skip_masks(N_TASKS, N_FEATURES_PER_TASK, N_OUTPUTS_PER_TASK, N_HIDDEN)
    model = SkipMaskedMLP(masks, key=jax.random.key(0))
    sep = separation_metrics(model, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    assert np.isclose(float(sep['connectivity_separation']), 1.0)
    assert np.isclose(float(sep['signal_separation']), 1.0, atol=1e-3)     # float32 sums
    dense = SkipMaskedMLP(dense_skip_masks(N_FEATURES, N_HIDDEN, N_OUTPUTS), key=jax.random.key(0))
    sep = separation_metrics(dense, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    assert np.isclose(float(sep['connectivity_separation']), 1.0 / N_TASKS)


def test_separation_counts_paths_not_inputs():
    """Output 0 (task 0): one skip from a task-0 input, plus hidden unit 0 reading two task-1 inputs.
    Three paths, one same-task -> 1/3. Biases are not paths."""
    n_hidden, n_features, n_outputs = 1, N_FEATURES, N_OUTPUTS
    w1 = jnp.zeros((n_hidden, n_features + 1)).at[0, -1].set(1.0)
    w1 = w1.at[0, N_FEATURES_PER_TASK].set(1.0).at[0, N_FEATURES_PER_TASK + 1].set(1.0)
    w2 = jnp.zeros((n_outputs, n_features + n_hidden + 1)).at[:, -1].set(1.0)
    w2 = w2.at[0, 0].set(1.0).at[0, n_features].set(1.0)
    model = SkipMaskedMLP({'w1': w1, 'w2': w2}, key=jax.random.key(0))
    model = tree_replace(model, weights={k: jnp.ones_like(v) for k, v in model.weights.items()})
    sep = separation_metrics(model, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    assert np.isclose(float(sep['connectivity_separation']), 1.0 / 3.0)
    assert np.isclose(float(sep['signal_separation']), 1.0 / 3.0)
    metrics = structure_metrics(model, INPUT_TASK_IDS, OUTPUT_TASK_IDS, n_tasks=N_TASKS)
    assert float(metrics['n_connections']) == 4.0
    assert float(metrics['hidden_incoming']) == 2.0 and float(metrics['hidden_outgoing']) == 1.0
    assert np.isclose(float(metrics['hidden_input_purity']), 1.0)


# ------------------------------------------------------------- 3c: feedback switch, hidden sources
def _feedback_expected(learner, x, y):
    """w1's gradient with the fixed feedback carrying the error back."""
    model = learner.model
    out, hidden_pre, param_inputs = model(x)
    d_pre = (model.feedback.T @ (2.0 * (out - y))) * jax.vmap(jax.grad(jax.nn.leaky_relu))(hidden_pre)
    return d_pre[:, None] * param_inputs['w1'] * model.masks['w1']


def _backprop_expected(learner, x, y):
    """w1's gradient with the actual hidden->output weights carrying the error back."""
    model = learner.model
    out, hidden_pre, param_inputs = model(x)
    hidden = model.weights['w2'][:, model.hidden_block] * model.masks['w2'][:, model.hidden_block]
    d_pre = (hidden.T @ (2.0 * (out - y))) * jax.vmap(jax.grad(jax.nn.leaky_relu))(hidden_pre)
    return d_pre[:, None] * param_inputs['w1'] * model.masks['w1']


def test_feedback_is_used_for_feedback_steps_then_the_actual_weights():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=0.05, activation='leaky_relu',
                       weight_init='lecun', feedback_steps=5)
    x, y = _batch(1, seed=11)
    assert bool(learner.feedback_active) and not bool(learner.hidden_sources_active)
    _, grads, _ = learner.loss_and_grads(x[0], y[0])
    assert np.allclose(np.asarray(grads['w1']), np.asarray(_feedback_expected(learner, x[0], y[0])), atol=1e-5)

    trained, _ = _run(learner, 5)
    assert int(trained.step_count) == 5 and not bool(trained.feedback_active)
    _, grads, _ = trained.loss_and_grads(x[0], y[0])
    assert np.allclose(np.asarray(grads['w1']), np.asarray(_backprop_expected(trained, x[0], y[0])), atol=1e-5)
    assert not np.allclose(np.asarray(grads['w1']), np.asarray(_feedback_expected(trained, x[0], y[0])), atol=1e-3)


def test_hidden_output_generation_waits_for_the_switch_and_respects_protection():
    masks, feedback, hidden_task_ids = _structure()
    labels = task_labels({k: jnp.ones_like(m) for k, m in masks.items()},
                         hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    learner = _learner(masks, feedback=feedback, init_lr=0.05, l1=0.001, restructure=True,
                       gen_cap=256, labels=labels, feedback_steps=40, hidden_sources=True)
    initial = np.asarray(_hidden_block(masks) > 0)

    # Before the switch: plenty of restructuring, but the hidden->output block is untouched.
    before, info = _run(learner, 40)
    assert info['n_generated'].sum() > 0
    assert np.array_equal(np.asarray(_hidden_block(before.model.masks) > 0), initial)
    assert info['n_crossed_hidden_out'].sum() == 0

    # After it: hidden->output connections get generated, some of those get pruned, the protected
    # ones never do, and the budget still holds.
    after, info = _run(before, 400, seed=12, scale=3.0)
    final = np.asarray(_hidden_block(after.model.masks) > 0)
    assert bool(final[initial].all()), 'a protected connection was pruned'
    assert final.sum() > initial.sum(), 'no hidden->output connection was generated'
    assert info['n_crossed_hidden_out'].sum() > 0, 'no generated hidden->output connection was pruned'
    assert info['n_crossed'].sum() == info['n_pruned'].sum()
    assert n_connections(after.model.masks) == BUDGET - int(info['deficit'][-1])
    for k in LAYERS:
        assert set(np.unique(np.asarray(after.model.masks[k]))) <= {0.0, 1.0}, k
    # A hidden->output connection is exactly 0 at the step it is generated, with fresh Autostep state.
    xs, ys = _batch(200, seed=13, scale=3.0)
    step = jax.jit(lambda lr, x, y: lr.step(x, y))
    learner = before
    for t in range(200):
        prev, (learner, _) = learner, step(learner, xs[t], ys[t])
        new = np.asarray((_hidden_block(learner.model.masks) > 0) & (_hidden_block(prev.model.masks) == 0))
        if new.any():
            block = slice(N_FEATURES, N_FEATURES + N_HIDDEN)
            assert bool((np.asarray(learner.model.weights['w2'][:, block])[new] == 0).all())
            assert np.allclose(np.asarray(learner.alpha['w2'][:, block])[new], 0.05, rtol=1e-5)
            assert bool((np.asarray(learner.opt_state.h['w2'][:, block])[new] == 0).all())
            break
    else:
        raise AssertionError('no hidden->output connection was generated in 200 steps')


def test_without_hidden_sources_generation_never_touches_the_hidden_block():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=0.05, restructure=True, gen_cap=256,
                       feedback_steps=5)
    trained, info = _run(learner, 200, scale=3.0)
    assert info['n_generated'].sum() > 0
    assert np.array_equal(np.asarray(_hidden_block(trained.model.masks)), np.asarray(_hidden_block(masks)))


def test_input_weight_scale_is_separate_from_the_step_size():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=1e-8, input_weight_scale=1e-3)
    cols = input_columns(masks)
    for k in LAYERS:
        inputs = np.abs(np.asarray(learner.model.weights[k][cols[k] & (masks[k] > 0)]))
        assert bool((inputs < 1e-3).all()) and float(inputs.max()) > 1e-4, k
        assert np.allclose(np.asarray(learner.alpha[k][:, :-1]), 1e-8, rtol=1e-5), k
    assert learner.input_weight_scale == 1e-3
    default = _learner(masks, feedback=feedback, init_lr=1e-8)
    assert default.input_weight_scale == 1e-8


# ------------------------------------------------------------------- same-destination regeneration
def _in_degree(masks):
    return np.concatenate([np.asarray(masks['w1'][:, :-1].sum(axis=1)),
                           np.asarray(masks['w2'][:, :-1].sum(axis=1))])


def test_same_destination_replaces_on_the_unit_that_lost_the_connection():
    masks, feedback, _ = _structure()
    learner = _learner(masks, feedback=feedback, init_lr=0.05, restructure=True,
                       regeneration='same_destination', feedback_steps=40, hidden_sources=True)
    model = learner.model
    # Pretend hidden unit 0 and output unit 3 each just lost two input connections.
    crossed = {k: jnp.zeros(m.shape, bool) for k, m in masks.items()}
    lost_h = np.flatnonzero(np.asarray(masks['w1'][0, :N_FEATURES]))[:2]
    lost_o = np.flatnonzero(np.asarray(masks['w2'][3, :N_FEATURES]))[:2]
    crossed['w1'] = crossed['w1'].at[0, lost_h].set(True)
    crossed['w2'] = crossed['w2'].at[3, lost_o].set(True)
    pruned = {k: jnp.where(crossed[k], 0.0, masks[k]) for k in LAYERS}

    generated, deficit, row_deficit = learner._propose_same_destination(
        model, pruned, crossed, jax.random.key(0))
    g1, g2 = np.asarray(generated['w1']), np.asarray(generated['w2'])
    assert g1.sum() == 2 and g1[0].sum() == 2, 'hidden unit 0 must get exactly its two replacements'
    assert g2.sum() == 2 and g2[3].sum() == 2, 'output unit 3 must get exactly its two replacements'
    assert int(deficit) == 0 and bool((np.asarray(row_deficit) == 0).all())
    # New sources are free positions, never the ones just pruned, and (before the switch) inputs only.
    assert bool((g1[np.asarray(pruned['w1']) > 0] == 0).all()) and g1[0, lost_h].sum() == 0
    assert bool((g2[np.asarray(pruned['w2']) > 0] == 0).all()) and g2[3, lost_o].sum() == 0
    assert g2[:, N_FEATURES:].sum() == 0

    # After the switch an output unit may draw a hidden unit as its source.
    switched = tree_replace(learner, step_count=jnp.array(40, jnp.int32))
    hidden_picks = 0
    for seed in range(30):
        generated, _, _ = switched._propose_same_destination(model, pruned, crossed, jax.random.key(seed))
        g2 = np.asarray(generated['w2'])
        assert g2[3].sum() == 2 and g2.sum() == 2
        hidden_picks += g2[3, N_FEATURES:N_FEATURES + N_HIDDEN].sum()
    assert hidden_picks > 0, 'hidden sources should be drawn sometimes once allowed'


def test_same_destination_keeps_every_units_in_degree_fixed_through_a_run():
    masks, feedback, hidden_task_ids = _structure()
    labels = task_labels({k: jnp.ones_like(m) for k, m in masks.items()},
                         hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    learner = _learner(masks, feedback=feedback, init_lr=0.05, l1=0.001, restructure=True,
                       regeneration='same_destination', labels=labels, feedback_steps=40,
                       hidden_sources=True)
    initial = _in_degree(masks)
    protected = np.asarray(_hidden_block(masks) > 0)
    xs, ys = _batch(300, seed=15, scale=3.0)
    step = jax.jit(lambda lr, x, y: lr.step(x, y))
    n_pruned = n_generated = 0
    for t in range(300):
        learner, info = step(learner, xs[t], ys[t])
        assert np.array_equal(_in_degree(learner.model.masks), initial), f'in-degree changed at step {t}'
        assert int(info['deficit']) == 0
        n_pruned += int(info['n_pruned'])
        n_generated += int(info['n_generated'])
    assert n_pruned > 0 and n_pruned == n_generated
    assert n_connections(learner.model.masks) == BUDGET
    final = np.asarray(_hidden_block(learner.model.masks) > 0)
    assert bool(final[protected].all()), 'a protected connection was pruned'
    assert final.sum() > protected.sum(), 'no hidden->output connection was generated after the switch'
