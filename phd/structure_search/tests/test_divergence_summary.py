"""A diverged run must complete, not fail.

When a hyperparameter blows up, the training loop stops early and exits 0 --
it has successfully established that the cell diverges. But the loss it reports
is inf/nan, and a sweep backend cannot store a non-finite objective: the trial
registers no result, the optimizer treats it as unfinished, and re-assigns it
(up to `retryAssignLimit`), burning a GPU-hour recomputing the same divergence
each time.

`finite_summary` substitutes a finite sentinel so the trial records a result.
These tests pin the three properties that makes it safe:

  1. every value it emits is finite and loggable,
  2. the sentinel sorts last under `minimize`, so a diverged cell can never win,
  3. real measurements pass through untouched -- including finite-but-huge
     losses, which are data rather than failures.
"""

import numpy as np
import pytest

from phd.structure_search.metrics import DIVERGENCE_LOSS_THRESHOLD, finite_summary


def _diverged_summary():
    """What run_config builds when training diverges: the loss aggregates are
    nan because the tail it averages contains nan."""
    return {
        'average_loss': float('nan'),
        'asymptotic_loss': float('nan'),
        'asymptotic_accuracy': float('nan'),
        'final_active_units': 1536.0,
        'final_active_connections': 101845.0,
        'diverged': 1.0,
    }


def test_all_values_are_finite_and_loggable():
    """Nothing non-finite survives -- that is the whole point."""
    out = finite_summary(_diverged_summary())
    bad = {k: v for k, v in out.items() if not np.isfinite(v)}
    assert not bad, f'non-finite values would still break the sweep: {bad}'


def test_diverged_loss_sorts_last_under_minimize():
    """The sentinel must lose to any plausible real loss, or a diverged cell
    could be picked as a sweep's winner."""
    out = finite_summary(_diverged_summary())
    # 32 tasks at chance is ~74; a bad-but-real run is far below the sentinel.
    for realistic in (2.2, 54.6, 85.5, 1e4):
        assert out['asymptotic_loss'] > realistic


def test_accuracy_uses_a_bounded_sentinel():
    """Accuracy is bounded in [0, 1], so a loss-shaped sentinel would be
    nonsense there; 0 is the natural worst."""
    out = finite_summary(_diverged_summary())
    assert out['asymptotic_accuracy'] == 0.0
    assert 0.0 <= out['asymptotic_accuracy'] <= 1.0


def test_diverged_flag_is_preserved():
    """The flag, not the sentinel, is what analysis keys on to mask the cell --
    so it has to survive untouched."""
    assert finite_summary(_diverged_summary())['diverged'] == 1.0


def test_healthy_summary_is_untouched():
    """No effect on a run that did not diverge."""
    healthy = {
        'average_loss': 58.4830, 'asymptotic_loss': 54.5876,
        'asymptotic_accuracy': 0.4292, 'final_active_units': 1536.0,
        'diverged': 0.0,
    }
    assert finite_summary(healthy) == healthy


def test_finite_but_huge_loss_passes_through():
    """A large finite loss is a real measurement, not a failure, and must not
    be flattened to the sentinel -- that would discard the distinction between
    'bad' and 'blew up'."""
    huge = 1e30
    out = finite_summary({'asymptotic_loss': huge, 'diverged': 1.0})
    assert out['asymptotic_loss'] == huge


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_every_non_finite_form_is_replaced(value):
    """-inf matters too: it would otherwise *win* a minimize objective."""
    out = finite_summary({'asymptotic_loss': value})
    assert np.isfinite(out['asymptotic_loss'])
    assert out['asymptotic_loss'] == DIVERGENCE_LOSS_THRESHOLD


def test_non_numeric_values_survive():
    """Summaries carry the occasional non-numeric entry; leave them alone."""
    out = finite_summary({'note': 'diverged at step 1024', 'asymptotic_loss': float('nan')})
    assert out['note'] == 'diverged at step 1024'
