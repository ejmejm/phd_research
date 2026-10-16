"""Shared content for the step-size analysis experiments.

This folder is a 1:1 split of ``step_size_analysis.ipynb`` (same directory) into
one file per experiment plus this shared module — the code is unchanged, just
reorganized so it is easier to work with outside a notebook. Everything used by
more than one experiment lives here: the problem generator, the LMS runner with
maturity indicators, the plotting helpers/colours, and the constants that later
experiments reuse from earlier ones.
"""

# Define problem
# Always two features
# Ideal weight of first feature is always 0.5, ideal weight of second feature is configurable
# Whether features are independent is configurable, when colinear, they are the same target feature with different amounts of noise
# Noise for the first feature is always 0.1, noise for the second feature is configurable
# Note that all of the three lines are like a small version of the feature sifting setup I'm already using with how features are generated with noise coefficients and such


# First experiment:
# Independent features, ideal weight of second feature with ideal weights of 0.1, 0.3, 0.5, 0.7, and 0.9
# The second feature should also have noise of 0.1
# Both features are trained with LMS with a step-size of 0.01 for 100k steps
# I want to plot several plots, vertically stacked, each for a different ideal weight of the second feature
# Each plot should include lines for:
#   - The current weights of both features
#   - A dashed line for the ideal weight of the second feature
#   - Traces of changes to the weights with varying decay rates (0.9, 0.99, 0.999)
#   - h value from IDBD (compute even though learning happens with a constant step-size)
#   - h value from IDBD3 (compute even though learning happens with a constant step-size)


# The goal of this experiment is to figure out which, if any, of the metrics I've mentioned would be sufficient to determine when a feature should be protected
# The indicators I'm considering are traces of changes to the weights, traces to changes to the step-size (when using idbd),
#   and the h value of IDBD or IDBD3 (when its sign flips) (which can be computed even when not using IDBD).

import numpy as np
import matplotlib.pyplot as plt


def make_problem_data(
    n_steps,
    target_weights,          # teacher weight on each feature's CLEAN signal (scalar or list)
    *,
    feature_noise=0.0,       # noise coefficient c per feature (scalar or list)
    target_noise_std=0.1,    # std of additive Gaussian noise on the target
    shared_signal=False,     # if True, all features are noisy views of ONE signal
    seed=0,
):
    """Generate (features, target) for the feature-sifting linear-regression probe.

    The target is a fixed "teacher network": ``y = sum_j target_weights[j] * s_j
    + noise``, built from the CLEAN signals ``s_j``. The learner instead sees the
    noisy features ``f_j = (1 - c_j) * s_j + c_j * U(-1, 1)``. The teacher weights
    are held fixed regardless of the feature noise; the learner's *optimal* weight
    on a noisy feature is attenuated away from the teacher weight (regression
    dilution -- see ``optimal_weight``), so the dashed reference in the plots is
    the computed optimum, not the teacher weight.

    Every knob that may vary across experiments is a parameter:

    - ``target_weights``: teacher weight per feature on its clean signal -> sets
      the number of features. With clean features (``c = 0``) this is also the
      learner's optimal weight.
    - ``feature_noise``: per-feature noise coefficient ``c`` in
      ``f = (1 - c) * s + c * U(-1, 1)``;  ``c = 0`` gives a clean feature.
    - ``target_noise_std``: std of the Gaussian noise added to the target.
    - ``shared_signal``: if True every feature is a noisy view of the *same*
      underlying signal (the colinear / "blocking" case); otherwise each feature
      has its own independent signal.

    Returns ``features`` (n_steps, n_feat) and ``y`` (n_steps,).
    """
    rng = np.random.default_rng(seed)
    target_weights = np.atleast_1d(np.asarray(target_weights, dtype=float))
    n_feat = len(target_weights)
    c = np.broadcast_to(np.asarray(feature_noise, dtype=float), (n_feat,))

    # underlying clean signals
    if shared_signal:
        s = rng.uniform(-1.0, 1.0, size=n_steps)
        signals = [s] * n_feat
    else:
        signals = [rng.uniform(-1.0, 1.0, size=n_steps) for _ in range(n_feat)]

    features = np.empty((n_steps, n_feat))
    y = rng.normal(0.0, target_noise_std, size=n_steps)
    for j in range(n_feat):
        feat_noise = rng.uniform(-1.0, 1.0, size=n_steps)
        features[:, j] = (1 - c[j]) * signals[j] + c[j] * feat_noise
        y = y + target_weights[j] * signals[j]   # teacher weight on the clean signal

    return features, y


def optimal_weight(target_weight, feature_noise):
    """Least-squares optimal weight on a single *independent* noisy feature.

    With ``f = (1 - c) * s + c * u`` (Var(s) == Var(u)) and target component
    ``target_weight * s``, the optimum is
        ``w* = Cov(f, y) / Var(f) = target_weight * (1 - c) / ((1 - c)**2 + c**2)``.
    The learner shrinks the weight (regression attenuation) as the feature gets
    noisier:  ``w* -> target_weight`` at ``c = 0`` and ``w* -> 0`` as ``c -> 1``.
    Assumes the feature's signal is independent of the other features' signals.
    """
    c = np.asarray(feature_noise, dtype=float)
    return target_weight * (1 - c) / ((1 - c) ** 2 + c ** 2)


# ---------------------------------------------------------------------------
# Constants shared across experiments (originally defined in the first
# experiment's cell and reused by the later ones).
# ---------------------------------------------------------------------------

STEP_SIZE = 0.01
TRACE_DECAYS = [0.9, 0.99, 0.999]


def run_lms_with_indicators(features, y, step_size=STEP_SIZE, trace_decays=TRACE_DECAYS,
                            w_init=None):
    """LMS over `features` (n_steps, n_feat); indicators are tracked for the
    feature under study, taken to be the LAST column. `step_size` is the constant
    LMS / IDBD step-size and the number of steps is set by len(features), so both
    are configurable per run. `w_init` defaults to zeros (a fresh weight); pass a
    vector to start e.g. an established weight at its optimum in the later blocking
    experiments."""
    n_steps, n_feat = features.shape
    tracked = n_feat - 1
    decays = np.asarray(trace_decays, dtype=float)
    n_dec = len(decays)

    w = np.zeros(n_feat) if w_init is None else np.asarray(w_init, dtype=float)
    h_idbd = 0.0
    h_idbd3 = 0.0
    dw_tr = np.zeros(n_dec)

    w_hist = np.empty((n_steps, n_feat))
    dw_trace_hist = np.empty((n_steps, n_dec))
    h_idbd_hist = np.empty(n_steps)
    h_idbd3_hist = np.empty(n_steps)

    for t in range(n_steps):
        f = features[t]
        error = float(w @ f) - y[t]              # y_hat - y  (project convention)

        fi = f[tracked]
        loss_grad = 2.0 * error * fi             # d(error**2)/dw_tracked
        decay_term = fi * fi                     # prediction-grad squared

        # IDBD / IDBD3 h-traces (observed only, constant step-size)
        keep = max(0.0, 1.0 - step_size * decay_term)
        h_idbd = h_idbd * keep + step_size * loss_grad
        h_idbd3 = h_idbd3 * keep + loss_grad

        # LMS weight update (constant step-size) for every weight
        dw = -step_size * (2.0 * error * f)
        w = w + dw

        # EMA traces of the tracked weight's changes
        dw_tr = decays * dw_tr + (1.0 - decays) * dw[tracked]

        w_hist[t] = w
        dw_trace_hist[t] = dw_tr
        h_idbd_hist[t] = h_idbd
        h_idbd3_hist[t] = h_idbd3

    return {'w': w_hist, 'dw_trace': dw_trace_hist,
            'h_idbd': h_idbd_hist, 'h_idbd3': h_idbd3_hist}


def _norm_peak(a):
    m = np.max(np.abs(a))
    return a / m if m > 0 else a

W_COLOR = 'black'
DW_COLORS = ['#3953e2', '#9923DC', '#029999']  # Brighter indigo, purple, dark aqua
H_IDBD3_COLOR = '#FF3300'  # slightly redder blood orange (stand-out)


# ---------------------------------------------------------------------------
# Blocking-setup constants (originally defined in experiments 3 and 4, reused
# by experiments 4 and 5).
# ---------------------------------------------------------------------------

W_TEACHER = 0.5                              # teacher weight on the shared signal
INCUMBENT_NOISE = 0.0                        # incumbent (learns from scratch) is clean
EST_NOISE_FIXED = 0.1                        # established-feature noise, held fixed in exps 4 & 5
