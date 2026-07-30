"""Tests for the multi-GEOFF masked MLP and Autostep's `squared_inputs` mode.

The two properties the multi-GEOFF experiments actually lean on:

* `squared_inputs` (Variant B) is a genuine port of the historic torch mode — in the linear case
  its curvature term *is* the squared prediction gradient, so it must coincide exactly with
  `prediction_grads` on a single linear layer. Any wiring mistake breaks that identity.
* Masked-out connections are inert under Autostep, which is what makes online connectivity
  changes cheap: nothing about a connection moves until its mask is flipped.
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phd.jax_core.optimizers.idbd import optax_idbd
from phd.jax_core.utils import tree_replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multi_geoff_exploration'))
from masked_mlp import (  # noqa: E402
    INACTIVE, USEFUL, USELESS, MaskedTwoLayerMLP, add_random_cross_task_connections,
    added_connection_labels, block_sparse_masks, connection_labels, contaminate_masks,
    masked_quantiles, reset_idbd_state_at, slot_ids, subsample_indices, thirds_group_ids,
)

N_TASKS = 2
N_FEATURES_PER_TASK = 20
N_OUTPUTS_PER_TASK = 10
N_HIDDEN = 64
INIT_LR = 1e-5
META_LR = 0.005

INPUT_TASK_IDS = slot_ids(N_TASKS, N_FEATURES_PER_TASK)
OUTPUT_TASK_IDS = slot_ids(N_TASKS, N_OUTPUTS_PER_TASK)


def _structures():
    base_masks, hidden_task_ids = block_sparse_masks(
        N_TASKS, N_FEATURES_PER_TASK, N_OUTPUTS_PER_TASK, N_HIDDEN)
    return base_masks, hidden_task_ids


def _train_steps(model, optimizer, opt_state, x, y, version, n_steps):
    """Run `n_steps` online Autostep updates, returning the final model and optimizer state."""
    for _ in range(n_steps):
        def loss_fn(weights):
            out, param_inputs = tree_replace(model, weights=weights)(x)
            return jnp.sum((out - y) ** 2), param_inputs

        (_, param_inputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(model.weights)
        if version == 'prediction_grads':
            pred_grads = jax.grad(
                lambda w: jnp.sum(tree_replace(model, weights=w)(x)[0]))(model.weights)
            updates, opt_state = optimizer.update(
                (grads, pred_grads), opt_state, model.weights)
        else:
            updates, opt_state = optimizer.update(
                (grads, None, param_inputs), opt_state, model.weights)
        model = tree_replace(
            model, weights={k: model.weights[k] + updates[k] for k in updates})
    return model, opt_state


# ------------------------------------------------------------------ squared_inputs semantics
def test_squared_inputs_matches_prediction_grads_in_linear_case():
    """On a single linear layer the two curvature terms are the same quantity, so the two
    Autostep modes must agree bit for bit. This is the sharp test that Variant B is wired
    correctly: the historic torch mode used `inputs.pow(2)` with inputs broadcast over the
    output-row axis, and for a linear predictor the prediction gradient wrt w is exactly x."""
    w = jax.random.normal(jax.random.key(0), (1, 5))
    x = jax.random.normal(jax.random.key(1), (5,))
    y = 1.7

    def loss_grads(weights):
        error = jnp.sum(weights * x[None, :]) - y
        return 2.0 * error * x[None, :]

    cfg = dict(init_lr=INIT_LR, meta_lr=META_LR, autostep=True)
    opt_a = optax_idbd(version='prediction_grads', **cfg)
    opt_b = optax_idbd(version='squared_inputs', **cfg)
    state_a, state_b = opt_a.init(w), opt_b.init(w)
    w_a = w_b = w

    for _ in range(50):
        upd_a, state_a = opt_a.update((loss_grads(w_a), x[None, :]), state_a, w_a)
        upd_b, state_b = opt_b.update((loss_grads(w_b), None, x[None, :]), state_b, w_b)
        w_a, w_b = w_a + upd_a, w_b + upd_b

    assert jnp.array_equal(w_a, w_b), 'Weights diverged between the two modes'
    assert jnp.array_equal(state_a.beta, state_b.beta), 'Step-sizes diverged'
    assert jnp.array_equal(state_a.v, state_b.v), 'Normalizers diverged'
    print('PASS: test_squared_inputs_matches_prediction_grads_in_linear_case')


def test_squared_inputs_requires_param_inputs():
    """`squared_inputs` has no curvature term without `param_inputs`, so it must refuse the
    two-element update tuple rather than silently falling back."""
    w = jnp.ones((1, 4))
    opt = optax_idbd(init_lr=INIT_LR, meta_lr=META_LR, autostep=True, version='squared_inputs')
    state = opt.init(w)
    try:
        opt.update((jnp.ones((1, 4)), jnp.ones((1, 4))), state, w)
    except AssertionError:
        print('PASS: test_squared_inputs_requires_param_inputs')
        return
    raise AssertionError('Expected an AssertionError for the missing param_inputs')


def test_invalid_version_rejected():
    """A bad `version` should fail at init rather than as an UnboundLocalError mid-update."""
    try:
        optax_idbd(version='not_a_version').init(jnp.ones((1, 4)))
    except AssertionError:
        print('PASS: test_invalid_version_rejected')
        return
    raise AssertionError('Expected an AssertionError for the invalid version')


# ------------------------------------------------------------------------- the masked model
def test_forward_shapes_and_param_inputs():
    """The forward pass returns (output, param_inputs), with param_inputs holding each weight
    matrix's source-unit values shaped for broadcast over the output-row axis."""
    masks, _ = _structures()
    model = MaskedTwoLayerMLP(masks, key=jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (N_TASKS * N_FEATURES_PER_TASK,))
    out, param_inputs = model(x)

    assert out.shape == (N_TASKS * N_OUTPUTS_PER_TASK,), f'Got {out.shape}'
    assert param_inputs['w1'].shape == (1, N_TASKS * N_FEATURES_PER_TASK)
    assert param_inputs['w2'].shape == (1, N_HIDDEN)
    assert jnp.array_equal(param_inputs['w1'][0], x), 'w1 source values should be the input'
    print('PASS: test_forward_shapes_and_param_inputs')


def test_masked_connections_are_inert():
    """A masked-out connection has zero gradient and zero curvature, so under Autostep its
    weight, gradient trace, normalizer and step-size must all stay exactly where they started.
    This is what lets a connection be activated by flipping a mask and resetting one position."""
    masks, _ = _structures()
    x = jax.random.normal(jax.random.key(1), (N_TASKS * N_FEATURES_PER_TASK,))
    y = jax.random.normal(jax.random.key(2), (N_TASKS * N_OUTPUTS_PER_TASK,))

    for version in ('prediction_grads', 'squared_inputs'):
        model = MaskedTwoLayerMLP(masks, key=jax.random.key(0))
        optimizer = optax_idbd(init_lr=INIT_LR, meta_lr=META_LR, autostep=True, version=version)
        model, opt_state = _train_steps(
            model, optimizer, optimizer.init(model.weights), x, y, version, 20)

        for layer in ('w1', 'w2'):
            off = masks[layer] == 0
            assert off.any(), f'{layer} has no masked-out positions to check'
            assert jnp.all(model.weights[layer][off] == 0), f'{version}/{layer}: weight moved'
            assert jnp.all(opt_state.h[layer][off] == 0), f'{version}/{layer}: h moved'
            assert jnp.all(opt_state.v[layer][off] == 0), f'{version}/{layer}: v moved'
            assert jnp.allclose(opt_state.beta[layer][off], opt_state.init_beta), \
                f'{version}/{layer}: step-size moved'
            # The active connections must actually have moved, or the test is vacuous.
            assert not jnp.allclose(opt_state.beta[layer][~off], opt_state.init_beta), \
                f'{version}/{layer}: no active step-size moved at all'
    print('PASS: test_masked_connections_are_inert')


def test_reset_idbd_state_at():
    """Activating connections mid-training puts exactly those positions back to initial."""
    masks, hidden_task_ids = _structures()
    new_masks, added = add_random_cross_task_connections(
        masks, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS, jax.random.key(3))

    x = jax.random.normal(jax.random.key(1), (N_TASKS * N_FEATURES_PER_TASK,))
    y = jax.random.normal(jax.random.key(2), (N_TASKS * N_OUTPUTS_PER_TASK,))
    model = MaskedTwoLayerMLP(masks, key=jax.random.key(0))
    optimizer = optax_idbd(init_lr=INIT_LR, meta_lr=META_LR, autostep=True,
                           version='squared_inputs')
    model, opt_state = _train_steps(
        model, optimizer, optimizer.init(model.weights), x, y, 'squared_inputs', 20)

    reset = reset_idbd_state_at(opt_state, added)
    for layer in ('w1', 'w2'):
        on = added[layer] > 0
        assert jnp.allclose(reset.beta[layer][on], reset.init_beta), 'added beta not reset'
        assert jnp.all(reset.h[layer][on] == 0), 'added h not reset'
        assert jnp.all(reset.v[layer][on] == 0), 'added v not reset'
        # Everything else is untouched.
        assert jnp.array_equal(reset.beta[layer][~on], opt_state.beta[layer][~on]), \
            'reset leaked into pre-existing connections'
    print('PASS: test_reset_idbd_state_at')


# ------------------------------------------------------------------------- connectivity
def test_block_sparse_masks():
    """Every hidden unit connects to exactly its own task's inputs and outputs."""
    masks, hidden_task_ids = _structures()
    assert int(masks['w1'].sum()) == N_HIDDEN * N_FEATURES_PER_TASK
    assert int(masks['w2'].sum()) == N_HIDDEN * N_OUTPUTS_PER_TASK
    cross_in = hidden_task_ids[:, None] != INPUT_TASK_IDS[None, :]
    cross_out = OUTPUT_TASK_IDS[:, None] != hidden_task_ids[None, :]
    assert not jnp.any(masks['w1'] * cross_in), 'block-sparse w1 crosses tasks'
    assert not jnp.any(masks['w2'] * cross_out), 'block-sparse w2 crosses tasks'
    print('PASS: test_block_sparse_masks')


def test_thirds_group_ids_split_within_task():
    """Groups are split inside each task, and are as even as 64 / 2 / 3 allows."""
    _, hidden_task_ids = _structures()
    group_ids = thirds_group_ids(hidden_task_ids, N_TASKS)
    for task in range(N_TASKS):
        sizes = [int(((group_ids == g) & (hidden_task_ids == task)).sum()) for g in range(3)]
        assert sum(sizes) == N_HIDDEN // N_TASKS, f'task {task} sizes {sizes}'
        assert max(sizes) - min(sizes) <= 1, f'task {task} split unevenly: {sizes}'
    print('PASS: test_thirds_group_ids_split_within_task')


def test_contaminate_masks_adds_only_cross_task():
    """The 1a contamination adds other-task outputs to one third and other-task inputs to
    another, and leaves the last third alone."""
    masks, hidden_task_ids = _structures()
    group_ids = thirds_group_ids(hidden_task_ids, N_TASKS)
    contaminated = contaminate_masks(
        masks, hidden_task_ids, group_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)

    added_w1 = contaminated['w1'] - masks['w1']
    added_w2 = contaminated['w2'] - masks['w2']
    assert jnp.all(added_w1 >= 0) and jnp.all(added_w2 >= 0), 'contamination removed connections'

    # Only cross-task, and only for the intended group.
    for group, layer, added, rows_are_hidden in ((1, 'w1', added_w1, True),
                                                 (0, 'w2', added_w2, False)):
        in_group = (group_ids == group)
        touched = added.sum(axis=1 if rows_are_hidden else 0) > 0
        assert jnp.array_equal(touched, in_group), \
            f'{layer}: contamination touched the wrong hidden units'

    n_other_in = (N_TASKS - 1) * N_FEATURES_PER_TASK
    n_other_out = (N_TASKS - 1) * N_OUTPUTS_PER_TASK
    assert int(added_w1.sum()) == int((group_ids == 1).sum()) * n_other_in
    assert int(added_w2.sum()) == int((group_ids == 0).sum()) * n_other_out
    print('PASS: test_contaminate_masks_adds_only_cross_task')


def test_add_random_cross_task_connections_respects_ranges():
    """Per hidden unit, the injected counts stay in range, the endpoints are cross-task, and
    nothing pre-existing is re-added."""
    masks, hidden_task_ids = _structures()
    n_in_range, n_out_range = (1, 10), (1, 3)
    new_masks, added = add_random_cross_task_connections(
        masks, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS, jax.random.key(3),
        n_out_range=n_out_range, n_in_range=n_in_range)

    per_unit_in = np.asarray(added['w1'].sum(axis=1))
    per_unit_out = np.asarray(added['w2'].sum(axis=0))
    assert per_unit_in.min() >= n_in_range[0] and per_unit_in.max() <= n_in_range[1], \
        f'incoming counts {per_unit_in.min()}-{per_unit_in.max()} outside {n_in_range}'
    assert per_unit_out.min() >= n_out_range[0] and per_unit_out.max() <= n_out_range[1], \
        f'outgoing counts {per_unit_out.min()}-{per_unit_out.max()} outside {n_out_range}'

    cross_in = hidden_task_ids[:, None] != INPUT_TASK_IDS[None, :]
    cross_out = OUTPUT_TASK_IDS[:, None] != hidden_task_ids[None, :]
    assert jnp.all((added['w1'] > 0) <= cross_in), 'injected a same-task incoming connection'
    assert jnp.all((added['w2'] > 0) <= cross_out), 'injected a same-task outgoing connection'
    for layer in ('w1', 'w2'):
        assert not jnp.any(added[layer] * masks[layer]), f'{layer}: re-added an existing connection'
        assert jnp.array_equal(new_masks[layer], jnp.maximum(masks[layer], added[layer]))
    print('PASS: test_add_random_cross_task_connections_respects_ranges')


def test_connection_labels():
    """Inactive / useful / useless partition every position, and useless means cross-task."""
    masks, hidden_task_ids = _structures()
    group_ids = thirds_group_ids(hidden_task_ids, N_TASKS)
    contaminated = contaminate_masks(
        masks, hidden_task_ids, group_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    labels = connection_labels(
        contaminated, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)

    for layer in ('w1', 'w2'):
        lab = labels[layer]
        assert lab.shape == masks[layer].shape
        counts = [int((lab == v).sum()) for v in (INACTIVE, USEFUL, USELESS)]
        assert sum(counts) == lab.size, 'labels do not partition the matrix'
        assert jnp.array_equal(lab == INACTIVE, contaminated[layer] == 0)
        # Every useless connection is active and crosses tasks.
        assert int((lab == USEFUL).sum()) == int(masks[layer].sum()), \
            'the block-sparse base should be exactly the useful set'
    print('PASS: test_connection_labels')


def test_added_connection_labels():
    """1b labels split original from newly added, over the post-injection structure."""
    masks, hidden_task_ids = _structures()
    new_masks, added = add_random_cross_task_connections(
        masks, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS, jax.random.key(3))
    labels = added_connection_labels(new_masks, added)
    for layer in ('w1', 'w2'):
        assert jnp.array_equal(labels[layer] == USELESS, added[layer] > 0)
        assert jnp.array_equal(labels[layer] == USEFUL, masks[layer] > 0)
        assert jnp.array_equal(labels[layer] == INACTIVE, new_masks[layer] == 0)
    print('PASS: test_added_connection_labels')


# ------------------------------------------------------------------------------ statistics
def test_masked_quantiles_matches_numpy():
    """Group quantiles must match a plain numpy computation over the selected positions. The
    jit-safe implementation sorts with `inf` padding rather than boolean-indexing, so this
    guards the index arithmetic."""
    values = jax.random.uniform(jax.random.key(0), (6, 7))
    labels = jax.random.randint(jax.random.key(1), (6, 7), INACTIVE, USELESS + 1)
    levels = (0.1, 0.25, 0.5, 0.75, 0.9)

    for group in (USEFUL, USELESS):
        got = np.asarray(masked_quantiles(values, labels, group, levels))
        selected = np.sort(np.asarray(values)[np.asarray(labels) == group])
        expected = [selected[int(np.clip(np.floor(q * (len(selected) - 1) + 0.5), 0,
                                        len(selected) - 1))] for q in levels]
        assert np.allclose(got, expected), f'group {group}: {got} != {expected}'
    print('PASS: test_masked_quantiles_matches_numpy')


def test_masked_quantiles_empty_group_is_nan():
    """An empty group reports NaN rather than a misleading number."""
    values = jnp.ones((3, 3))
    labels = jnp.full((3, 3), INACTIVE)
    assert jnp.all(jnp.isnan(masked_quantiles(values, labels, USEFUL, (0.5,))))
    print('PASS: test_masked_quantiles_empty_group_is_nan')


def test_subsample_indices():
    """Sampled positions are distinct and all belong to the requested group."""
    masks, hidden_task_ids = _structures()
    group_ids = thirds_group_ids(hidden_task_ids, N_TASKS)
    contaminated = contaminate_masks(
        masks, hidden_task_ids, group_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    labels = connection_labels(contaminated, hidden_task_ids, INPUT_TASK_IDS, OUTPUT_TASK_IDS)

    idxs = subsample_indices(labels['w1'], USELESS, 30, seed=0)
    assert idxs.shape == (30, 2)
    assert len({tuple(map(int, row)) for row in idxs}) == 30, 'sampled with replacement'
    assert jnp.all(labels['w1'][idxs[:, 0], idxs[:, 1]] == USELESS), 'sampled outside the group'
    assert subsample_indices(jnp.full((3, 3), INACTIVE), USEFUL, 5) is None
    print('PASS: test_subsample_indices')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nAll tests passed')
