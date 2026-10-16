"""Second experiment: sweep the step-size (optimal weight fixed at $w^* = 0.001$).

Exactly the first experiment's single clean-feature / noisy-target setup, but the
optimal weight is now held fixed at $w^* = 0.001$ and the constant LMS / IDBD
step-size is swept instead. One panel per step-size; the indicators, axes,
normalization, and (fixed) weight y-range are unchanged from the first experiment.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    make_problem_data, optimal_weight, run_lms_with_indicators, _norm_peak,
)

# ---------------------------------------------------------------------------
# Second experiment: identical single clean-feature / noisy-target setup as the
# first experiment, but now the optimal weight is held FIXED at 0.001 and the
# constant LMS / IDBD step-size is what we sweep (instead of sweeping the ideal
# weight). Clean feature -> optimal weight == teacher weight, so target_weights
# = [0.001] gives an optimal weight of 0.001 exactly. Everything else (the run
# function, indicators, noise, n_steps) is unchanged from the first experiment.
# ---------------------------------------------------------------------------

OPT_WEIGHT_SS = 0.01
STEP_SIZES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
n_steps = 1_000_000

results_ss = {}
for ss in STEP_SIZES:
    # single clean feature (no feature noise), target noise = 0.1
    feats, targ = make_problem_data(n_steps, [OPT_WEIGHT_SS], feature_noise=0.0,
                                    target_noise_std=0.1, seed=0)
    results_ss[ss] = run_lms_with_indicators(feats, targ, step_size=ss)


plot_steps = np.arange(1, n_steps + 1)
w_opt = optimal_weight(OPT_WEIGHT_SS, 0.0)   # clean feature -> equals 0.001
fig, axes = plt.subplots(len(STEP_SIZES), 1,
                         figsize=(11, 3.6 * len(STEP_SIZES)), sharex=True)
ax2_first = None
for ax, ss in zip(axes, STEP_SIZES[::-1]):
    res = results_ss[ss]

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

    # --- left axis: the weight (true scale, black), now plotted after the right axis lines ---
    ax.plot(plot_steps, res['w'][:, -1], color=W_COLOR, lw=1.5, label='weight', zorder=10, alpha=0.9)
    ax.axhline(w_opt, color=W_COLOR, ls='--', lw=1.5, alpha=0.7,
               label='optimal weight', zorder=10)
    ax.set_xscale('log')
    ax.set_ylabel('weight', fontsize=16)
    ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
    ax.set_title(f'step-size = {ss:.3g}', fontsize=14)
    ax.set_zorder(ax2.get_zorder() + 1)   # put the weight's axes above the indicator axes
    ax.patch.set_visible(False)            # but keep ax transparent so ax2 shows through


    # rough "matured" reference: weight first reaches 90% of its optimum (skip if 0)
    if w_opt > 0:
        hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
        if len(hit):
            ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3, zorder=10)

axes[-1].set_xlabel('step (log scale)', fontsize=16)
fig.suptitle('Weight Trend Indicators (step-size sweep)', fontsize=18)

# Build legend with left and right handles/labels
hL, lL = axes[0].get_legend_handles_labels()
hR, lR = ax2_first.get_legend_handles_labels()
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.03),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
