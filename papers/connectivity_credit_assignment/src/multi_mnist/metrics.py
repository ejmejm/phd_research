"""Divergence, and how a diverged run is reported to a sweep backend."""

import numpy as np


# Losses at or above this are treated as divergence. Also the value substituted
# for a non-finite loss in `finite_summary` -- see there for why.
DIVERGENCE_LOSS_THRESHOLD = 1e6


def finite_summary(summary: dict, diverged_loss: float = DIVERGENCE_LOSS_THRESHOLD) -> dict:
    """Replace non-finite values in a run summary so a sweep can record them.

    A diverged run is a *successful* measurement of a bad hyperparameter: it
    stops early having established that the cell blows up, and exits 0. But the
    loss it reports is inf/nan, and a sweep backend cannot store that as an
    objective -- so the trial registers no result, the optimizer treats it as
    unfinished, and re-assigns it (up to `retryAssignLimit`), recomputing the
    same divergence several times over.

    Substituting a finite sentinel makes the trial record a result and complete.
    The sentinel sorts last under `minimize`, so it cannot win a sweep, and the
    accompanying `diverged` flag is what downstream analysis should key on --
    it should mask these cells rather than plot the sentinel as a real value.

    Only non-finite values are replaced. A finite-but-huge loss is a real
    measurement and is passed through unchanged.
    """
    out = {}
    for key, value in summary.items():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            # Accuracy is bounded in [0, 1], so the sentinel for a loss would be
            # nonsense there; 0 is the natural "worst".
            out[key] = 0.0 if 'accuracy' in key else diverged_loss
        else:
            out[key] = value
    return out
