"""Tests for Experiment 3's dynamic network — skip connections, pruning and generation.

Experiment 3 changes connectivity while training, so the properties worth pinning down are the ones
whose violation would quietly invalidate a run rather than crash it:

* the hand-written backward pass really is the autodiff gradient (and the gradient *mask* really
  does block the hidden layer's error, and only that);
* the connection budget is conserved exactly — every pruned connection is replaced, and a replaced
  connection is indistinguishable from a fresh one;
* generation only ever creates legal pathways, and under Variant 2 only from mature sources;
* the separation metrics count paths the way the plan defines them, checked against structures whose
  answer is known by hand.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phd.jax_core.utils import tree_replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multi_geoff_exploration'))
from masked_mlp import slot_ids  # noqa: E402
from dynamic_masked_mlp import (  # noqa: E402
    GROWTH, LAYERS, MATURE, NASCENT, DynamicAutostepMLP, SkipMaskedMLP, block_sparse_skip_masks,
    dense_skip_masks, initial_structure, n_connections, separation_metrics, structure_metrics,
    to_skip_layout,
)

N_TASKS = 4
N_FEATURES_PER_TASK = 20
N_OUTPUTS_PER_TASK = 10
N_HIDDEN = 16
N_FEATURES = N_TASKS * N_FEATURES_PER_TASK
N_OUTPUTS = N_TASKS * N_OUTPUTS_PER_TASK
BUDGET = 1024
INIT_LR = 1e-6
META_LR = 0.02

INPUT_TASK_IDS = slot_ids(N_TASKS, N_FEATURES_PER_TASK)
OUTPUT_TASK_IDS = slot_ids(N_TASKS, N_OUTPUTS_PER_TASK)


def _structure(budget=BUDGET, seed=0):
    return initial_structure(N_FEATURES, N_HIDDEN, N_OUTPUTS, budget, jax.random.key(seed))


def _learner(masks, protected=None, seed=0, **kwargs):
    kwargs.setdefault('init_lr', INIT_LR)
    return DynamicAutostepMLP.init(
        masks, META_LR, jax.random.key(seed), protected=protected, **kwargs)


def _batch(n_steps, seed=0, scale=1.0):
    """A stream of random samples, big enough that step-sizes and weights actually move."""
    x_key, y_key = jax.random.split(jax.random.key(seed))
    return (jax.random.uniform(x_key, (n_steps, N_FEATURES), minval=-1.0, maxval=1.0),
            scale * jax.random.normal(y_key, (n_steps, N_OUTPUTS)))


def _run(learner, n_steps, seed=0, scale=1.0):
    xs, ys = _batch(n_steps, seed=seed, scale=scale)
    infos = []
    for x, y in zip(xs, ys):
        learner, info = learner.step(x, y)
        infos.append(info)
    return learner, {k: np.array([float(i[k]) for i in infos]) for k in infos[0]}


# ------------------------------------------------------------------------ initial structure
def test_initial_structure_spends_the_budget_exactly():
    """Generation replaces pruned connections one-for-one, so the initial count *is* the budget for
    the whole run — a floor-divided budget would silently shrink the network."""
    for budget in (BUDGET, BUDGET + 1, BUDGET + N_HIDDEN + N_OUTPUTS - 1):
        masks, _ = _structure(budget=budget)
        assert n_connections(masks) == budget, f'budget {budget}: got {n_connections(masks)}'
    print('PASS: test_initial_structure_spends_the_budget_exactly')


def test_initial_structure_gives_each_hidden_unit_one_output():
    """Rule (1) of the initialization, which 3b's base version leans on: the hidden->output block
    holds exactly one connection per hidden unit and nothing else."""
    masks, protected = _structure()
    hidden_out = masks['w2'][:, N_FEATURES:]
    assert np.array_equal(np.asarray(hidden_out.sum(axis=0)), np.ones(N_HIDDEN)), \
        'a hidden unit has more or fewer than one outgoing connection'
    assert jnp.array_equal(protected[:, N_FEATURES:], hidden_out), \
        'the protected set is not the hidden->output block'
    assert not jnp.any(protected[:, :N_FEATURES]), 'skip connections must not be protected'

    # Rule (2): the input connections are spread as evenly as the budget allows, over hidden units
    # and output units alike.
    per_unit = np.concatenate([np.asarray(masks['w1'].sum(axis=1)),
                               np.asarray(masks['w2'][:, :N_FEATURES].sum(axis=1))])
    assert per_unit.max() - per_unit.min() <= 1, f'uneven input fan-in: {per_unit.min()}-{per_unit.max()}'
    print('PASS: test_initial_structure_gives_each_hidden_unit_one_output')


def test_initial_structure_rejects_an_impossible_budget():
    """Too small a budget must fail loudly rather than produce a network missing connections."""
    for budget in (N_HIDDEN, N_HIDDEN + N_OUTPUTS):
        try:
            _structure(budget=budget)
        except AssertionError:
            continue
        raise AssertionError(f'budget {budget} should have been rejected')
    print('PASS: test_initial_structure_rejects_an_impossible_budget')


# --------------------------------------------------------------------------- forward/backward
def test_forward_matches_the_explicit_three_block_computation():
    """The concatenated ``[inputs | hidden]`` output layer must be the same network as a two-layer
    MLP plus separate skip weights — the layout is bookkeeping, not a modelling choice."""
    masks, _ = _structure()
    model = SkipMaskedMLP(masks, key=jax.random.key(1))
    x = jax.random.uniform(jax.random.key(2), (N_FEATURES,), minval=-1.0, maxval=1.0)
    out, hidden_pre, param_inputs = model(x)

    hidden = model.activation_fn((model.weights['w1'] * masks['w1']) @ x)
    expected = ((model.skip_weights * masks['w2'][:, :N_FEATURES]) @ x
                + (model.hidden_out_weights * masks['w2'][:, N_FEATURES:]) @ hidden)
    assert jnp.allclose(out, expected), 'the skip and hidden contributions do not add up'
    assert jnp.allclose(param_inputs['w2'][0], jnp.concatenate([x, hidden])), \
        "param_inputs['w2'] must be the output layer's source values"
    assert jnp.allclose(param_inputs['w1'][0], x)
    assert jnp.allclose(model.activation_fn(hidden_pre), hidden)
    print('PASS: test_forward_matches_the_explicit_three_block_computation')


def test_manual_gradients_match_autodiff():
    """The hand-written backward pass is the whole reason per-unit errors are available, so it has
    to be the real gradient. With nothing protected, it must equal `jax.grad` exactly."""
    masks, _ = _structure()
    learner = _learner(masks)
    x, y = _batch(1)
    x, y = x[0], y[0]

    def loss_fn(weights):
        out, _, _ = tree_replace(learner.model, weights=weights)(x)
        return jnp.sum((out - y) ** 2)

    expected = jax.grad(loss_fn)(learner.model.weights)
    _, grads, _, _ = learner.loss_and_grads(x, y)
    for layer in LAYERS:
        assert jnp.allclose(grads[layer], expected[layer], atol=1e-6), f'{layer} gradient mismatch'
    print('PASS: test_manual_gradients_match_autodiff')


def test_gradient_mask_blocks_only_the_hidden_layer():
    """3b's base version stop-gradients every hidden->output connection except the initial ones.
    The masked connections must still get their own weight gradient — only the error flowing *into*
    the hidden layer is blocked, which is exactly a `stop_gradient` on the hidden activations."""
    masks, protected = _structure()
    # Give the output units some extra hidden sources, so there is something to block.
    extra = jnp.zeros_like(masks['w2']).at[0, N_FEATURES:].set(1.0)
    masks = {'w1': masks['w1'], 'w2': jnp.maximum(masks['w2'], extra)}
    learner = _learner(masks, protected=protected)
    x, y = _batch(1)
    x, y = x[0], y[0]

    model = learner.model
    keep = protected[:, N_FEATURES:]

    def loss_fn(weights):
        """The same network with the unprotected hidden->output paths stop-gradiented."""
        hidden = model.activation_fn((weights['w1'] * masks['w1']) @ x)
        w2 = weights['w2'] * masks['w2']
        out = (w2[:, :N_FEATURES] @ x
               + (w2[:, N_FEATURES:] * keep) @ hidden
               + (w2[:, N_FEATURES:] * (1 - keep)) @ jax.lax.stop_gradient(hidden))
        return jnp.sum((out - y) ** 2)

    expected = jax.grad(loss_fn)(model.weights)
    _, grads, _, _ = learner.loss_and_grads(x, y)
    for layer in LAYERS:
        assert jnp.allclose(grads[layer], expected[layer], atol=1e-6), f'{layer} gradient mismatch'

    # The blocked connections still learn, and the mask genuinely changed the first layer.
    blocked = (masks['w2'][:, N_FEATURES:] * (1 - keep)) > 0
    assert jnp.any(jnp.abs(grads['w2'][:, N_FEATURES:][blocked]) > 0), \
        'a stop-gradiented connection lost its own weight gradient'
    _, unmasked_grads, _, _ = _learner(masks).loss_and_grads(x, y)
    assert not jnp.allclose(grads['w1'], unmasked_grads['w1']), 'the gradient mask did nothing'
    print('PASS: test_gradient_mask_blocks_only_the_hidden_layer')


def test_delta_is_the_negated_error_signal():
    """delta_u is signed so the gradient is ``-delta_u * source``. Every threshold in the algorithm
    relates h (built from alpha * gradient) to v (built from delta^2), so a factor of two here would
    put the jitter thresholds out by 16x."""
    masks, _ = _structure()
    learner = _learner(masks)
    x, y = _batch(1)
    x, y = x[0], y[0]
    out, _, param_inputs = learner.model(x)
    _, grads, delta, _ = learner.loss_and_grads(x, y)

    assert jnp.allclose(delta['w2'], 2.0 * (y - out)), 'output delta is not -dL/d(out)'
    for layer, source in (('w1', param_inputs['w1'][0]), ('w2', param_inputs['w2'][0])):
        reconstructed = -delta[layer][:, None] * source[None, :] * masks[layer]
        assert jnp.allclose(grads[layer], reconstructed, atol=1e-6), \
            f'{layer}: gradient is not -delta * source'
    print('PASS: test_delta_is_the_negated_error_signal')


# -------------------------------------------------------------------------- static training
def test_masked_connections_stay_inert():
    """Without a mask flip nothing about an absent connection moves — the property that makes
    online connectivity change cheap, inherited from `masked_mlp`."""
    masks, _ = _structure()
    learner, _ = _run(_learner(masks), 30)
    for layer in LAYERS:
        off = masks[layer] == 0
        assert off.any(), f'{layer} has no masked-out positions'
        assert jnp.all(learner.model.weights[layer][off] == 0), f'{layer}: weight moved'
        assert jnp.all(learner.opt_state.h[layer][off] == 0), f'{layer}: h moved'
        assert jnp.all(learner.opt_state.v[layer][off] == 0), f'{layer}: Autostep v moved'
        assert jnp.allclose(learner.alpha[layer][off], INIT_LR), f'{layer}: step-size moved'
        assert not jnp.allclose(learner.alpha[layer][~off], INIT_LR), \
            f'{layer}: no active step-size moved at all, so the test is vacuous'
    print('PASS: test_masked_connections_stay_inert')


def test_static_run_never_changes_connectivity():
    """3a and the baselines must hold their structure exactly, including the stage bookkeeping,
    which is tracked but must have no side effects."""
    masks, _ = _structure()
    learner, info = _run(_learner(masks, restructure=False), 30)
    for layer in LAYERS:
        assert jnp.array_equal(learner.model.masks[layer], masks[layer]), f'{layer} mask changed'
    assert info['n_pruned'].sum() == 0 and info['n_generated'].sum() == 0
    print('PASS: test_static_run_never_changes_connectivity')


def test_per_unit_traces_are_shared_across_a_units_incoming_weights():
    """Experiment 2's traces v and c are per *unit*, over all of its incoming weights — which is the
    reason an output unit's skip and hidden weights live in one row of `w2`. This checks that the
    imported stage machinery is being fed this file's wider layout correctly: one trace per hidden
    and output unit, and c in [0, 1] because Autostep clips the effective step-size at 1."""
    masks, _ = _structure()
    learner, _ = _run(_learner(masks), 20)
    state = learner.stage_state
    assert state.v['w1'].shape == (N_HIDDEN,), state.v['w1'].shape
    assert state.v['w2'].shape == (N_OUTPUTS,), state.v['w2'].shape
    for layer in LAYERS:
        c = np.asarray(state.c[layer])
        assert np.all(c >= 0) and np.all(c <= 1), f'{layer}: occupancy left [0, 1]: {c.min()}, {c.max()}'
        assert np.all(np.asarray(state.v[layer]) >= 0), f'{layer}: negative squared error'
        assert np.all(c > 0), f'{layer}: a unit never accumulated any occupancy'
    print('PASS: test_per_unit_traces_are_shared_across_a_units_incoming_weights')


# ------------------------------------------------------------------------ prune and generate
def _forced_pruning_learner(seed=0, protect=False, **kwargs):
    """A learner whose connections really do get pruned within a few hundred steps.

    Pruning needs a weight to cross zero with a jitter-sized h, which at a 1e-6 initial step-size
    takes millions of steps in the real experiment. A large step-size and a large L1 coefficient
    reproduce the same dynamics in a test-sized number of steps: L1 drags dominated weights into a
    limit cycle around 0 (Experiment 1's finding), so they cross zero constantly.

    `protect` selects 3b's base version (initial hidden->output connections exempt from pruning and
    the only ones carrying gradient) over Variant 1's plain everything-is-prunable setup.
    """
    masks, protected = _structure()
    learner = _learner(masks, seed=seed, protected=protected if protect else None,
                       init_lr=0.01, l1=0.5, restructure=True, gen_cap=256, **kwargs)
    return learner, masks, protected


def test_pruning_conserves_the_connection_budget():
    """The invariant the whole design rests on: connections are replaced one-for-one, so the number
    of connections is exactly ``budget - deficit`` at every step, and the deficit — the handful of
    proposals per step that collide on the same pathway — stays near zero instead of accumulating."""
    learner, masks, _ = _forced_pruning_learner()
    budget = n_connections(masks)
    xs, ys = _batch(200, scale=3.0)
    pruned = deficits = 0
    for x, y in zip(xs, ys):
        learner, info = learner.step(x, y)
        assert n_connections(learner.model.masks) == budget - int(info['deficit']), \
            'connection count does not match budget minus deficit'
        pruned += int(info['n_pruned'])
        deficits += int(info['deficit'])

    assert pruned > 0, 'nothing was pruned, so the test is vacuous'
    # Colliding proposals are refilled on later steps, so the deficit is transient noise, not a leak.
    assert deficits / 200 < 0.01 * budget, f'generation is falling behind: mean deficit {deficits / 200}'
    for layer in LAYERS:
        assert jnp.all(learner.model.masks[layer] <= 1.0), f'{layer}: a mask entry exceeded 1'
        assert not jnp.array_equal(learner.model.masks[layer], masks[layer]), \
            f'{layer}: connectivity never changed'
    print('PASS: test_pruning_conserves_the_connection_budget')


def test_a_saturated_generation_cap_is_accounted_for_not_hidden():
    """`gen_cap` bounds the work per step, so a step that prunes more than the cap cannot refill
    immediately. The shortfall must show up in `deficit` — connection count is exactly
    ``budget - deficit`` at every step — rather than quietly shrinking the network."""
    masks, protected = _structure()
    learner = _learner(masks, protected=protected, init_lr=0.01, l1=0.5, restructure=True,
                       gen_cap=4)
    budget = n_connections(masks)
    xs, ys = _batch(60, scale=3.0)
    for x, y in zip(xs, ys):
        learner, info = learner.step(x, y)
        assert n_connections(learner.model.masks) == budget - int(info['deficit']), \
            'connection count does not match budget minus deficit'
    assert int(info['deficit']) > 0, 'the cap never saturated, so the test is vacuous'
    print('PASS: test_a_saturated_generation_cap_is_accounted_for_not_hidden')


def test_generated_connections_start_from_scratch():
    """A generated connection must be indistinguishable from a never-used one: weight 0, h 0,
    Autostep's normalizer 0, step-size at `init_lr`, stage nascent. Otherwise a recycled slot
    inherits the step-size of the connection that just failed there."""
    learner, _, _ = _forced_pruning_learner()
    xs, ys = _batch(100, scale=3.0)
    n_checked = 0
    for x, y in zip(xs, ys):
        before = {k: learner.model.masks[k] for k in LAYERS}
        learner, info = learner.step(x, y)
        for layer in LAYERS:
            created = (learner.model.masks[layer] > 0) & (before[layer] == 0)
            pruned = (learner.model.masks[layer] == 0) & (before[layer] > 0)
            for name, positions in (('generated', created), ('pruned', pruned)):
                if not bool(jnp.any(positions)):
                    continue
                assert jnp.all(learner.model.weights[layer][positions] == 0), \
                    f'{layer}: {name} weight not zeroed'
                assert jnp.all(learner.opt_state.h[layer][positions] == 0), f'{layer}: {name} h not reset'
                assert jnp.all(learner.opt_state.v[layer][positions] == 0), \
                    f'{layer}: {name} normalizer not reset'
                assert jnp.allclose(learner.alpha[layer][positions], 0.01), \
                    f'{layer}: {name} step-size not back at init_lr'
                assert jnp.all(learner.stages[layer][positions] == NASCENT), \
                    f'{layer}: {name} stage not reset'
                n_checked += int(jnp.sum(positions))
    assert n_checked > 0, 'nothing was pruned or generated, so the test is vacuous'
    print('PASS: test_generated_connections_start_from_scratch')


def test_generation_only_creates_legal_pathways():
    """Sources must come from earlier layers and must not already be connected: hidden units draw
    from inputs only, output units from inputs and hidden units."""
    learner, masks, _ = _forced_pruning_learner(seed=1)
    trained, info = _run(learner, 200, seed=1, scale=3.0)
    assert info['n_generated'].sum() > 0, 'nothing was generated, so the test is vacuous'
    assert trained.model.masks['w1'].shape == masks['w1'].shape
    for layer in LAYERS:
        values = np.unique(np.asarray(trained.model.masks[layer]))
        assert set(values.tolist()) <= {0.0, 1.0}, f'{layer}: mask is not 0/1: {values}'
    print('PASS: test_generation_only_creates_legal_pathways')


def test_protected_connections_are_never_pruned():
    """3b's base version keeps the initial 1:1 hidden->output connections, because they are the only
    ones carrying gradient into the hidden layer — pruning one would orphan a hidden unit."""
    learner, masks, protected = _forced_pruning_learner(protect=True)
    trained, info = _run(learner, 200, scale=3.0)
    assert info['n_pruned'].sum() > 0, 'nothing was pruned, so the test is vacuous'
    kept = protected > 0
    assert jnp.all(trained.model.masks['w2'][kept] == 1.0), 'a protected connection was pruned'
    # Without the exemption the same run really does prune some of them, so this is not a no-op.
    unprotected, _, _ = _forced_pruning_learner()
    plain, _ = _run(unprotected, 200, scale=3.0)
    assert not jnp.all(plain.model.masks['w2'][kept] == 1.0), \
        'nothing prunes those positions even unprotected, so the test is vacuous'
    print('PASS: test_protected_connections_are_never_pruned')


def test_a_hidden_unit_with_no_outgoing_connections_counts_as_mature():
    """Maturity is defined over a unit's outgoing connections, so a unit with none has no stage to
    read. It counts as mature, or Variant 2 could never pick it as a source, it could never get an
    outgoing connection, and the state would be absorbing — while generation kept feeding it
    incoming connections that no error signal can ever reach."""
    masks, _ = _structure()
    # Silence one hidden unit's single outgoing connection, leaving its incoming ones in place.
    orphan = int(jnp.argmax(masks['w2'][:, N_FEATURES:].sum(axis=0) > 0))
    masks = {'w1': masks['w1'],
             'w2': masks['w2'].at[:, N_FEATURES + orphan].set(0.0)}
    learner = _learner(masks)

    mature = learner.hidden_mature
    assert bool(mature[orphan]), 'a unit with no outgoing connections is not counted as mature'
    others = jnp.arange(mature.shape[0]) != orphan
    assert not bool(jnp.any(mature[others])), \
        'a unit with a nascent outgoing connection was counted as mature'
    print('PASS: test_a_hidden_unit_with_no_outgoing_connections_counts_as_mature')


def test_variant_2_generates_only_from_mature_sources():
    """Variant 2 gates generation on the source unit's maturity, checked per step.

    Whether a unit was a legal source at generation time is recoverable from the step's before and
    after states. A legal source either had a mature outgoing connection — maturity is absorbing and
    generation never prunes, so that connection is still there and still mature afterwards — or had
    none at all, in which case none of the outgoing connections it has afterwards were there when
    the gate was evaluated. Anything else (a surviving, non-mature outgoing connection and no mature
    one) was illegal.

    "None of them were there" is not the same as "created this step": a connection can be pruned and
    land on the same pathway again in one step, which a before/after mask diff cannot see. Both
    cases leave the position freshly reset, so the reset signature is what identifies them.

    An end-of-run check cannot see any of this: a unit is legal *precisely while* it has no outgoing
    connection, and that evidence disappears the moment generation gives it one.
    """
    learner, _, _ = _forced_pruning_learner(require_mature_source=True)
    xs, ys = _batch(200, scale=3.0)
    hidden = slice(N_FEATURES, None)
    n_checked = 0

    for x, y in zip(xs, ys):
        before = learner.model.masks['w2'][:, hidden] > 0
        learner, _ = learner.step(x, y)
        after = learner.model.masks['w2'][:, hidden] > 0

        created = after & ~before
        if not bool(jnp.any(created)):
            continue
        # Positions put back to their initial state this step: either created, or pruned and
        # refilled on the spot. Neither existed when the maturity gate was evaluated.
        recycled = (jnp.isclose(learner.alpha['w2'][:, hidden], 0.01)
                    & (learner.opt_state.h['w2'][:, hidden] == 0))
        survived = after & ~created & ~recycled
        legal = (jnp.any(after & (learner.stages['w2'][:, hidden] == MATURE), axis=0)
                 | ~jnp.any(survived, axis=0))
        used = jnp.any(created, axis=0)
        assert bool(jnp.all(used <= legal)), 'generated a connection from an immature hidden unit'
        n_checked += int(jnp.sum(used))

    assert n_checked > 0, 'no hidden->output connection was ever generated, so the test is vacuous'
    print('PASS: test_variant_2_generates_only_from_mature_sources')


# ----------------------------------------------------------------------------------- stages
def test_stages_progress_in_order_and_maturity_is_absorbing():
    """nascent -> growth -> mature, one transition per step, and mature never regresses (until the
    connection is pruned or reset, which the restructure path handles separately)."""
    masks, _ = _structure()
    learner = _learner(masks, init_lr=0.01, restructure=False)
    xs, ys = _batch(300, scale=3.0)
    seen_growth = seen_mature = False
    for x, y in zip(xs, ys):
        previous = {k: learner.stages[k] for k in LAYERS}
        learner, _ = learner.step(x, y)
        for layer in LAYERS:
            before, after = previous[layer], learner.stages[layer]
            assert jnp.all(after >= before), f'{layer}: a stage regressed'
            assert jnp.all(after - before <= 1), f'{layer}: skipped a stage in one step'
            assert jnp.all(after[before == MATURE] == MATURE), f'{layer}: maturity is not absorbing'
            assert jnp.all(after[masks[layer] == 0] == NASCENT), \
                f'{layer}: an inactive connection changed stage'
        seen_growth |= bool(jnp.any(learner.stages['w2'] == GROWTH))
        seen_mature |= bool(jnp.any(learner.stages['w2'] == MATURE))
    assert seen_growth and seen_mature, \
        f'no connection reached growth ({seen_growth}) / maturity ({seen_mature}), test is vacuous'
    print('PASS: test_stages_progress_in_order_and_maturity_is_absorbing')


def test_growth_requires_an_occupied_trace():
    """The w jitter threshold divides by the occupancy c, which starts at exactly 0, so the growth
    stage has to be unreachable until c passes 0.5 — otherwise every connection would enter growth
    on step one against a threshold of 0."""
    masks, _ = _structure()
    learner = _learner(masks, init_lr=0.01)
    x, y = _batch(1, scale=3.0)
    learner, _ = learner.step(x[0], y[0])
    for layer in LAYERS:
        assert jnp.all(learner.stages[layer] == NASCENT), \
            f'{layer}: a connection entered growth before its unit trace was occupied'
        assert jnp.all(jnp.isfinite(learner.stage_state.c[layer])), \
            f'{layer}: occupancy is not finite'
    print('PASS: test_growth_requires_an_occupied_trace')


# ---------------------------------------------------------------------------------- metrics
def test_separation_of_the_reference_structures():
    """Block-sparse is perfectly separated by construction; dense spreads every output's paths over
    all tasks equally, so it sits at 1 / n_tasks. Those two are the metric's calibration points."""
    block_masks, _ = block_sparse_skip_masks(
        N_TASKS, N_FEATURES_PER_TASK, N_OUTPUTS_PER_TASK, N_HIDDEN)
    model = SkipMaskedMLP(block_masks, key=jax.random.key(0))
    metrics = separation_metrics(model, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    # float32 sums over thousands of paths, so exact equality is not the right bar.
    assert np.isclose(float(metrics['connectivity_separation']), 1.0, atol=1e-4), metrics
    assert np.isclose(float(metrics['signal_separation']), 1.0, atol=1e-4), metrics

    dense = SkipMaskedMLP(dense_skip_masks(N_FEATURES, N_HIDDEN, N_OUTPUTS), key=jax.random.key(0))
    dense_metrics = separation_metrics(dense, INPUT_TASK_IDS, OUTPUT_TASK_IDS)
    assert np.isclose(float(dense_metrics['connectivity_separation']), 1.0 / N_TASKS, atol=1e-4), \
        dense_metrics
    print('PASS: test_separation_of_the_reference_structures')


def test_separation_counts_paths_not_inputs():
    """Hand-checked path counting on a three-input, two-hidden, one-output network: paths through a
    hidden unit are counted once per input reached, and a skip connection is a path of its own."""
    n_features, n_hidden, n_outputs = 3, 2, 1
    masks = {
        'w1': jnp.array([[1.0, 1.0, 0.0],     # hidden 0 <- inputs 0, 1  (tasks 0, 0)
                         [0.0, 0.0, 1.0]]),   # hidden 1 <- input 2      (task 1)
        # output 0 <- input 0 (skip, same task), hidden 0, hidden 1
        'w2': jnp.array([[1.0, 0.0, 0.0, 1.0, 1.0]]),
    }
    input_task_ids = jnp.array([0, 0, 1])
    output_task_ids = jnp.array([0])
    model = SkipMaskedMLP(masks, key=jax.random.key(0))
    # Paths: skip->input0 (same), h0->{input0, input1} (both same), h1->input2 (other) = 3/4.
    got = float(separation_metrics(model, input_task_ids, output_task_ids)['connectivity_separation'])
    assert np.isclose(got, 0.75), f'expected 0.75, got {got}'

    # Signal separation with unit weights on every path must agree with the unweighted count; the
    # weights below make the two hidden paths' products 2 and 3 respectively.
    weights = {'w1': jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]]),
               'w2': jnp.array([[1.0, 0.0, 0.0, 2.0, 1.0]])}
    model = tree_replace(model, weights=weights)
    # Same-task signal: skip 1 + h0 (2*1 + 2*1) = 5; other-task: h1 (1*3) = 3.
    got = float(separation_metrics(model, input_task_ids, output_task_ids)['signal_separation'])
    assert np.isclose(got, 5.0 / 8.0), f'expected {5 / 8}, got {got}'
    print('PASS: test_separation_counts_paths_not_inputs')


def test_structure_metrics_report_the_expected_counts():
    """The 3a measurements: fan-in / fan-out per hidden unit, and the stage fractions summing to 1
    over the active connections."""
    masks, _ = _structure()
    learner = _learner(masks)
    metrics = structure_metrics(learner.model, INPUT_TASK_IDS, OUTPUT_TASK_IDS, learner.stages)

    assert np.isclose(float(metrics['n_connections']), BUDGET)
    assert np.isclose(float(metrics['hidden_outgoing']), 1.0), 'one outgoing connection per hidden unit'
    assert np.isclose(float(metrics['hidden_incoming']),
                      float(masks['w1'].sum()) / N_HIDDEN)
    fractions = [float(metrics[f'fraction_{name}']) for name in ('nascent', 'growth', 'mature')]
    assert np.isclose(sum(fractions), 1.0), f'stage fractions do not sum to 1: {fractions}'
    assert np.isclose(fractions[0], 1.0), 'every connection starts nascent'
    print('PASS: test_structure_metrics_report_the_expected_counts')


def test_to_skip_layout_adds_no_skip_connections():
    """The baselines are plain two-layer networks lifted into this layout, so their skip block must
    be empty — otherwise dense and block-sparse would quietly get extra capacity."""
    lifted = to_skip_layout(
        {'w1': jnp.ones((N_HIDDEN, N_FEATURES)), 'w2': jnp.ones((N_OUTPUTS, N_HIDDEN))}, N_FEATURES)
    assert lifted['w2'].shape == (N_OUTPUTS, N_FEATURES + N_HIDDEN)
    assert not jnp.any(lifted['w2'][:, :N_FEATURES]), 'lifting introduced skip connections'
    assert jnp.all(lifted['w2'][:, N_FEATURES:] == 1.0)
    print('PASS: test_to_skip_layout_adds_no_skip_connections')


# ------------------------------------------------------------------------------- jit / scan
def test_step_scans_under_jit():
    """Everything above runs in a python loop; the experiments run inside `jax.lax.scan` under
    `jit`, which additionally requires every branch to be fixed-shape."""
    masks, protected = _structure()
    learner = _learner(masks, protected=protected, init_lr=0.01, l1=0.5, restructure=True,
                       gen_cap=256)
    xs, ys = _batch(50, scale=3.0)

    @jax.jit
    def run(learner, xs, ys):
        def body(learner, sample):
            return learner.step(*sample)
        return jax.lax.scan(body, learner, (xs, ys))

    trained, info = run(learner, xs, ys)
    assert int(info['n_pruned'].sum()) > 0, 'nothing was pruned, so the test is vacuous'
    assert n_connections(trained.model.masks) == n_connections(masks) - int(info['deficit'][-1]), \
        'budget drifted under jit'
    print('PASS: test_step_scans_under_jit')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nAll tests passed')
