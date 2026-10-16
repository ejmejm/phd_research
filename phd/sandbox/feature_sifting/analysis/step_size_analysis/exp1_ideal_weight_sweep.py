"""First experiment: ideal-weight sweep (single clean feature, noisy target)."""

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    STEP_SIZE, TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    make_problem_data, optimal_weight, run_lms_with_indicators, _norm_peak,
)

# ---------------------------------------------------------------------------
# First experiment: a single feature with clean inputs (no feature noise) and a
# noisy target, learned from a zero weight with constant-step-size LMS, while we
# *observe* candidate "maturity" indicators for its weight.
#
# Learning is plain LMS; the IDBD / IDBD3 h-traces are computed alongside with
# the same constant step-size, exactly as the project optimisers do:
#   IDBD : h <- h * max(1 - step_size*f**2, 0) + step_size * loss_grad
#   IDBD3: h <- h * max(1 - step_size*f**2, 0) +             loss_grad
# with loss_grad = 2*error*f. They never feed back into the weights.
# (With a constant step-size the two share the same decay term and differ only by
#  the factor step_size, so h_idbd == step_size * h_idbd3 exactly -- they coincide
#  once normalized.)
# ---------------------------------------------------------------------------

N_STEPS = 10_000
IDEAL_WEIGHTS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]


results = {}
for w_ideal in IDEAL_WEIGHTS:
    # single clean feature (no feature noise), target noise = 0.1
    feats, targ = make_problem_data(N_STEPS, [w_ideal], feature_noise=0.0,
                                    target_noise_std=0.1, seed=0)
    results[w_ideal] = run_lms_with_indicators(feats, targ, step_size=STEP_SIZE)


# ---------------------------------------------------------------------------
# Reading the plots
#
# One panel per ideal weight $w^*$ (including $w^* = 0$ — a useless feature whose
# weight should stay at zero). A single feature with clean inputs is learned from
# a zero weight; all the noise lives in the target ($\sigma = 0.1$).
#
# **Left axis (true scale, black):** the weight (solid) and its ideal value
# (dashed). The grey vertical line marks where the weight first reaches 90% of
# its ideal — a rough "matured" reference (omitted for $w^* = 0$).
#
# **Right axis (linear, coloured):** the candidate maturity indicators, each
# normalized to unit peak so they are comparable (raw magnitudes differ by orders
# of magnitude). All are shown in the weight-change sign convention ($-h$), so
# each indicator is positive while the weight grows and collapses toward (and
# flips around) zero once the weight matures.
#
# The IDBD / IDBD3 $h$-traces use the project's exact recursion
# $h_t = h_{t-1}\,\max(1 - \alpha f^2,\,0) + (\alpha\ \text{or}\ 1)\cdot\text{loss\_grad}$,
# with $\alpha$ held constant. They share the *same* decay multiplier and their
# additive terms differ only by the constant factor $\alpha$, so
# $h^{\text{IDBD}}_t = \alpha\, h^{\text{IDBD3}}_t$ for all $t$ — i.e. they are
# exactly proportional and coincide after normalization. They only diverge once
# $\alpha$ is allowed to adapt (then the decay multipliers differ too), which is
# a later experiment.
# ---------------------------------------------------------------------------


# Only plot the first optimal weight (single plot)
plot_steps = np.arange(1, N_STEPS + 1)

w_ideal = IDEAL_WEIGHTS[-1]
res = results[w_ideal]
w_opt = optimal_weight(w_ideal, 0.0)   # clean features -> equals the teacher weight

fig, ax = plt.subplots(figsize=(11, 5))
# --- left axis: the weight (true scale, black) ---
ax.plot(plot_steps, res['w'][:, -1], color=W_COLOR, lw=2.0, label='weight')
ax.axhline(w_opt, color=W_COLOR, ls='--', lw=1.5, alpha=0.7, label='optimal weight')
ax.set_xscale('log')
ax.set_ylabel('weight', fontsize=16)
ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
ax.set_title(f'Weight Trend Indicators', fontsize=18)

# rough "matured" reference: weight first reaches 90% of its optimum (skip if 0)
if w_opt > 0:
    hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
    if len(hit):
        ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3)

# --- right axis: normalized maturity indicators (one colour each) ---
ax2 = ax.twinx()
for k, d in enumerate(TRACE_DECAYS):
    ax2.plot(plot_steps, _norm_peak(res['dw_trace'][:, k]),
             color=DW_COLORS[k], lw=1.1, alpha=0.9, label=f'$\Delta w$ trace ($\\beta = {d}$)')
ax2.plot(plot_steps, _norm_peak(-res['h_idbd3']), color=H_IDBD3_COLOR, lw=1.6,
         label='$-h$ (IDBD3)')
ax2.axhline(0.0, color='0.5', lw=0.6, alpha=0.3)
ax2.set_ylim(-1.15, 1.15)
ax2.set_ylabel('indicators\n(normalized)', fontsize=16)

ax.set_xlabel('step (log scale)', fontsize=16)

# Build legend with left and right handles/labels
hL, lL = ax.get_legend_handles_labels()
hR, lR = ax2.get_legend_handles_labels()
# Place the legend just above the axes, closer to the plot to reduce empty space
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.12),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))  # adjust so legend fits closely above plot
plt.show()


plot_steps = np.arange(1, N_STEPS + 1)
fig, axes = plt.subplots(len(IDEAL_WEIGHTS), 1,
                         figsize=(11, 3.6 * len(IDEAL_WEIGHTS)), sharex=True)
ax2_first = None
for ax, w_ideal in zip(axes, IDEAL_WEIGHTS[::-1]):
    res = results[w_ideal]
    w_opt = optimal_weight(w_ideal, 0.0)   # clean features -> equals the teacher weight

    # --- left axis: the weight (true scale, black) ---
    ax.plot(plot_steps, res['w'][:, -1], color=W_COLOR, lw=2.0, label='weight')
    ax.axhline(w_opt, color=W_COLOR, ls='--', lw=1.5, alpha=0.7,
               label='optimal weight')
    ax.set_xscale('log')
    ax.set_ylabel('weight', fontsize=16)
    ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
    ax.set_title(f'optimal weight = {w_opt:.3g}', fontsize=14)

    # rough "matured" reference: weight first reaches 90% of its optimum (skip if 0)
    if w_opt > 0:
        hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
        if len(hit):
            ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3)

    # --- right axis: normalized maturity indicators (one colour each) ---
    ax2 = ax.twinx()
    for k, d in enumerate(TRACE_DECAYS):
        ax2.plot(plot_steps, _norm_peak(res['dw_trace'][:, k]),
                 color=DW_COLORS[k], lw=1.1, alpha=0.9,
                 label=f'$\Delta w$ trace ($\\beta = {d}$)')
    ax2.plot(plot_steps, _norm_peak(-res['h_idbd3']), color=H_IDBD3_COLOR, lw=1.6,
             label='$-h$ (IDBD3)')
    ax2.axhline(0.0, color='0.5', lw=0.6, alpha=0.3)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_ylabel('indicators\n(normalized)', fontsize=16)
    if ax2_first is None:
        ax2_first = ax2

axes[-1].set_xlabel('step (log scale)', fontsize=16)
fig.suptitle('Weight Trend Indicators', fontsize=18)

# Build legend with left and right handles/labels
hL, lL = axes[0].get_legend_handles_labels()
hR, lR = ax2_first.get_legend_handles_labels()
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.02),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
