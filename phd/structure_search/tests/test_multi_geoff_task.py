"""Tests for the MultiGEOFFTask testbed from phd/jax_core/tasks/multi_geoff.py.

The task had no coverage and no callers before the multi-GEOFF experiments, so these cover the
properties those experiments read results against: that the non-stationarity actually fires on
schedule and hits one slot at a time, that the teacher is genuinely block-diagonal, and that
`fraction_variance_explained` is calibrated the way the docstring claims (1 for an optimal
predictor, 0 for the mean predictor).
"""

import jax
import jax.numpy as jnp
import numpy as np

from phd.jax_core.tasks.multi_geoff import (
    MultiGEOFFTask, output_weight_scale, perturbation_period,
)


def _task(n_tasks=2, perturb_period=None, seed=0, **kwargs):
    return MultiGEOFFTask(n_tasks, perturb_period=perturb_period, seed=seed, **kwargs)


def test_perturbation_period_keeps_per_slot_rate_fixed():
    """A per-slot period of T over k slots means a global period of T/k."""
    assert perturbation_period(2000, 2) == 1000
    assert perturbation_period(2000, 8) == 250
    assert perturbation_period(1, 8) == 1, 'must never return a non-positive period'
    print('PASS: test_perturbation_period_keeps_per_slot_rate_fixed')


def test_shapes_and_slot_layout():
    """Inputs and outputs are slot-major with the advertised dimensions."""
    task = _task(n_tasks=3, n_features_per_task=5, n_outputs_per_task=4)
    assert (task.n_features, task.n_outputs) == (15, 12)
    assert np.array_equal(np.asarray(task.input_slot_ids), np.repeat([0, 1, 2], 5))
    assert np.array_equal(np.asarray(task.output_slot_ids), np.repeat([0, 1, 2], 4))

    task, (x, y) = task.generate_batch(7)
    assert x.shape == (7, 15) and y.shape == (7, 12)
    assert jnp.all((x >= -1.0) & (x <= 1.0)), 'inputs outside the default bounds'
    print('PASS: test_shapes_and_slot_layout')


def test_teacher_is_block_diagonal():
    """An output depends only on its own slot's inputs: changing another slot's inputs must
    leave it untouched. This is the property that makes cross-task connections useless."""
    task = _task(n_tasks=2, n_features_per_task=6, n_outputs_per_task=3)
    x = jax.random.uniform(jax.random.key(0), (1, task.n_features), minval=-1.0, maxval=1.0)
    base = task.forward(x)

    # Replace slot 1's inputs entirely; slot 0's outputs must not move.
    other = x.at[:, 6:].set(-x[:, 6:])
    changed = task.forward(other)
    assert jnp.allclose(base[:, :3], changed[:, :3]), 'slot 0 output depends on slot 1 inputs'
    print('PASS: test_teacher_is_block_diagonal')


def test_perturbation_fires_on_schedule():
    """With batch_size 1, the readout changes at exactly the multiples of `perturb_period`."""
    period = 100
    task = _task(perturb_period=period)
    previous = task.output_weights
    fired = []
    for step in range(1, 5 * period + 1):
        task, _ = task.generate_batch(1)
        if not bool(jnp.array_equal(task.output_weights, previous)):
            fired.append(step)
            previous = task.output_weights
    assert fired == [period * i for i in range(1, 6)], f'fired at {fired}'
    print('PASS: test_perturbation_fires_on_schedule')


def test_stationary_task_never_perturbs():
    """`perturb_period=None` is the stationary problem."""
    task = _task(perturb_period=None)
    initial = task.output_weights
    for _ in range(500):
        task, _ = task.generate_batch(1)
    assert jnp.array_equal(task.output_weights, initial)
    print('PASS: test_stationary_task_never_perturbs')


def test_perturbation_permutes_one_slot_only():
    """A perturbation permutes the columns of exactly one slot, preserving each row's weight
    multiset — so the signal variance is unchanged and only the feature labelling moves."""
    period = 10
    task = _task(n_tasks=4, perturb_period=period)
    before = np.asarray(task.output_weights)
    for _ in range(period):
        task, _ = task.generate_batch(1)
    after = np.asarray(task.output_weights)

    changed = [t for t in range(4) if not np.array_equal(before[t], after[t])]
    assert len(changed) == 1, f'{len(changed)} slots changed, expected exactly 1'
    slot = changed[0]
    # Every row of the changed slot is a permutation of its former self.
    for row_before, row_after in zip(before[slot], after[slot]):
        assert np.array_equal(np.sort(row_before), np.sort(row_after)), 'not a permutation'
    print('PASS: test_perturbation_permutes_one_slot_only')


def test_batch_size_may_not_exceed_perturb_period():
    """A batch longer than the period would silently drop perturbations, so it is rejected."""
    task = _task(perturb_period=4)
    try:
        task.generate_batch(5)
    except AssertionError:
        print('PASS: test_batch_size_may_not_exceed_perturb_period')
        return
    raise AssertionError('Expected an AssertionError for batch_size > perturb_period')


def _per_output_signal_variance(seed, n_samples=20_000, **kwargs):
    task = _task(n_tasks=2, n_hidden_per_task=64, noise_std=0.0, seed=seed, **kwargs)
    _, (x, _) = task.generate_batch(n_samples)
    return float(np.asarray(task.forward(x)).var(axis=0).mean())


def test_signal_variance_is_unit_per_output_in_expectation():
    """`output_weight_scale` gives unit signal variance per output, which is what makes the
    fraction-variance-explained metric scale-free.

    The guarantee is *in expectation over problem constructions*, not per draw: the readout signs
    are sampled, so a single teacher's per-output variance scatters around 1 by order 10%. The
    mean over seeds is what has to land on 1.
    """
    assert np.isclose(output_weight_scale(4), 1.0)
    variances = [_per_output_signal_variance(seed) for seed in range(8)]
    assert abs(np.mean(variances) - 1.0) < 0.05, \
        f'mean per-output signal variance {np.mean(variances):.3f} is not near 1'
    assert all(0.7 < v < 1.3 for v in variances), f'a single draw was extreme: {variances}'
    print('PASS: test_signal_variance_is_unit_per_output_in_expectation')


def test_fraction_variance_explained_calibration():
    """rho is 1 for an optimal predictor and 0 for the mean predictor.

    Two things worth being precise about, since experiments read results against these lines:

    * The predictor that sits at 0 is the *mean* predictor, not the zero predictor. The teacher's
      outputs have a nonzero mean (the LTUs fire half the time and a given output's readout
      weights do not sum to zero), so predicting 0 scores well below 0.
    * rho = 1 for an optimal predictor holds per draw, because its MSE is exactly the noise
      variance. rho = 0 for the mean predictor only holds *in expectation over teachers*, since
      the metric divides by an assumed unit signal variance that a single draw only approximates
      (see `test_signal_variance_is_unit_per_output_in_expectation`). A single seed can sit ~0.1
      away from 0, so the FVE-0 reference line on a plot is approximate per seed.
    """
    optimal_rhos, mean_rhos = [], []
    for seed in range(8):
        task = _task(n_tasks=2, n_hidden_per_task=64, noise_std=1.0, seed=seed)
        task, (x, y) = task.generate_batch(40_000)
        y = np.asarray(y)
        signal = np.asarray(task.forward(x))

        assert np.isclose(task.irreducible_mse, 1.0)
        optimal_rhos.append(float(task.fraction_variance_explained(((y - signal) ** 2).mean())))
        mean_rhos.append(float(task.fraction_variance_explained(
            ((y - y.mean(axis=0, keepdims=True)) ** 2).mean())))

    # Exact per draw: the optimal predictor's MSE is the noise variance by construction.
    assert all(abs(rho - 1.0) < 0.02 for rho in optimal_rhos), \
        f'optimal predictor should score ~1 on every draw: {optimal_rhos}'
    # Only exact in expectation over teachers.
    assert abs(np.mean(mean_rhos)) < 0.05, \
        f'mean predictor should score ~0 on average, got {np.mean(mean_rhos):.3f}'
    print('PASS: test_fraction_variance_explained_calibration')


def test_generate_batch_is_jit_and_scan_safe():
    """The task must run inside `jax.lax.scan`, which is how every experiment drives it."""
    task = _task(perturb_period=10)

    def step(task, _):
        task, (x, y) = task.generate_batch(1)
        return task, jnp.sum(y)

    final, sums = jax.jit(lambda t: jax.lax.scan(step, t, length=50))(task)
    assert sums.shape == (50,) and jnp.all(jnp.isfinite(sums))
    assert int(final.step) == 50, f'step counter is {int(final.step)}, expected 50'
    print('PASS: test_generate_batch_is_jit_and_scan_safe')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nAll tests passed')
