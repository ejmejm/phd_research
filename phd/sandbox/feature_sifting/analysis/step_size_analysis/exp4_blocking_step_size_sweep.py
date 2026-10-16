"""Fourth experiment: blocking, sweeping the step-size (established noise fixed at 0.1).

The same jump we made from the first experiment to the second, now applied to the
blocking setup. The established feature's noise is held fixed at $c_{\\text{est}} =
0.1$ — so its weight starts at its solo optimum $w^\\star_{\\text{est}} =
\\text{optimal\\_weight}(0.5, 0.1) \\approx 0.549$ — and the constant LMS / IDBD
step-size is swept instead. One panel per step-size; the incumbent is still clean
and learned from a zero weight, so its dashed solo optimum stays at $0.5$. The
partially transparent established-weight line is kept from the third experiment.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    W_TEACHER, INCUMBENT_NOISE, EST_NOISE_FIXED,
    make_problem_data, optimal_weight, run_lms_with_indicators, _norm_peak,
)

# ---------------------------------------------------------------------------
# Fourth experiment: the step-size sweep of the blocking setup -- the same jump
# we made from the first experiment to the second, now applied to the third. We
# fix the established feature's noise at 0.1 (so its solo optimum, where its
# weight starts, is optimal_weight(0.5, 0.1) ~= 0.549) and sweep the constant
# LMS / IDBD step-size instead. The incumbent is still clean and learned from a
# zero weight, and the teacher weight on the shared signal is unchanged
# (W_TEACHER = 0.5, so the incumbent's dashed solo optimum stays at 0.5). One
# panel per step-size.
# ---------------------------------------------------------------------------

STEP_SIZES_BLOCK = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
N_STEPS_BLOCK_SS = 500_000

w_est0 = optimal_weight(W_TEACHER, EST_NOISE_FIXED)   # established starts at its solo optimum

results_block_ss = {}
for ss in STEP_SIZES_BLOCK:
    # column 0 = established feature (noise EST_NOISE_FIXED), column 1 = incumbent (clean, tracked)
    feats, targ = make_problem_data(
        N_STEPS_BLOCK_SS, [W_TEACHER, 0.0], feature_noise=[EST_NOISE_FIXED, INCUMBENT_NOISE],
        target_noise_std=0.1, shared_signal=True, seed=0,
    )
    results_block_ss[ss] = run_lms_with_indicators(
        feats, targ, step_size=ss, w_init=[w_est0, 0.0])


plot_steps = np.arange(1, N_STEPS_BLOCK_SS + 1)
w_opt = optimal_weight(W_TEACHER, INCUMBENT_NOISE)   # clean incumbent -> equals W_TEACHER
fig, axes = plt.subplots(len(STEP_SIZES_BLOCK), 1,
                         figsize=(11, 3.6 * len(STEP_SIZES_BLOCK)), sharex=True)
ax2_first = None
for ax, ss in zip(axes, STEP_SIZES_BLOCK[::-1]):
    res = results_block_ss[ss]

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

    # --- left axis: the weights (true scale, black) ---
    ax.plot(plot_steps, res['w'][:, -1], color=W_COLOR, lw=2.0,
            label='incumbent weight', zorder=10)
    # the established feature's weight: same scale, partially transparent
    ax.plot(plot_steps, res['w'][:, 0], color=W_COLOR, lw=2.0, alpha=0.3,
            label='established weight', zorder=10)
    ax.axhline(w_opt, color=W_COLOR, ls='--', lw=1.5, alpha=0.7,
               label='optimal weight', zorder=10)
    # ax.set_xscale('log')
    ax.set_ylabel('weight', fontsize=16)
    ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
    ax.set_title(f'step-size = {ss:.3g}', fontsize=14)
    ax.set_zorder(ax2.get_zorder() + 1)   # put the weight's axes above the indicator axes
    ax.patch.set_visible(False)            # but keep ax transparent so ax2 shows through

    # rough "matured" reference: incumbent first reaches 90% of its solo optimum
    if w_opt > 0:
        hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
        if len(hit):
            ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3, zorder=10)

axes[-1].set_xlabel('step (log scale)', fontsize=16)
fig.suptitle('Weight Trend Indicators (blocking, step-size sweep)', fontsize=18)

# Build legend with left and right handles/labels
hL, lL = axes[0].get_legend_handles_labels()
hR, lR = ax2_first.get_legend_handles_labels()
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.03),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
