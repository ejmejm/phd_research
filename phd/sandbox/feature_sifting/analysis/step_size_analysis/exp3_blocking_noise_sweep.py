"""Third experiment: blocking by an established feature.

Now there are **two colinear features** — two noisy views of the *same* signal
(`shared_signal=True`), differing only in their noise:

- **Established feature** — pre-trained: its weight starts at the value that is
  optimal when it is the *only* feature present, $w^*_{\\text{est}} =
  \\text{optimal\\_weight}(w^*, c_{\\text{est}})$. Its noise $c_{\\text{est}}$ is the
  swept variable (one panel each: $1.0, 0.5, 0.1, 0.01, 0.0$).
- **Incumbent feature** — the tracked feature from the earlier experiments: clean
  ($c = 0$), learned from a zero weight. It now has to compete with the
  established feature for the shared signal.

Because the established feature already explains part of the target, we expect it
to **partially block** the incumbent — the cleaner (lower-noise) the established
feature, the more it blocks, so the incumbent's weight settles further below its
solo optimum $w^*$ (the dashed reference).

Same plot as before, tracked on the **incumbent** feature, with one addition: a
**partially transparent** line for the established feature's weight.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    STEP_SIZE, TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    W_TEACHER, INCUMBENT_NOISE,
    make_problem_data, optimal_weight, run_lms_with_indicators, _norm_peak,
)

# ---------------------------------------------------------------------------
# Third experiment: a *blocking* setup. Two colinear features (shared_signal),
# i.e. two noisy views of the SAME underlying signal, differing only in noise:
#   - established feature (column 0): pre-trained, weight initialised at the
#     value that is optimal when it is the ONLY feature present,
#     optimal_weight(W_TEACHER, c_est). Its noise c_est is what we sweep.
#   - incumbent feature  (column 1, the tracked one): clean (noise 0), learned
#     from a zero weight -- exactly the feature studied before, but now it has to
#     compete with the established feature for the shared signal.
# Shared signal -> the teacher target is W_TEACHER * s + noise regardless of how
# the teacher weight is split across columns, so we put it all on column 0
# ([W_TEACHER, 0.0]); the incumbent's solo optimum is still
# optimal_weight(W_TEACHER, 0) == W_TEACHER (the dashed reference). How far the
# established feature blocks the incumbent below that reference is the thing to
# watch. Run function, step-size, indicators and noise are as in the first
# experiment; the step count gets its own value below.
# ---------------------------------------------------------------------------

EST_NOISES = [1.0, 0.5, 0.1, 0.01, 0.0]      # swept: noise of the established feature
N_STEPS_BLOCK = 500_000                      # training steps for this experiment

results_block = {}
for c_est in EST_NOISES:
    # column 0 = established feature (noise c_est), column 1 = incumbent (clean, tracked)
    feats, targ = make_problem_data(
        N_STEPS_BLOCK, [W_TEACHER, 0.0], feature_noise=[c_est, INCUMBENT_NOISE],
        target_noise_std=0.1, shared_signal=True, seed=0,
    )
    w_est0 = optimal_weight(W_TEACHER, c_est)   # established weight starts at its solo optimum
    results_block[c_est] = run_lms_with_indicators(
        feats, targ, step_size=STEP_SIZE, w_init=[w_est0, 0.0])


plot_steps = np.arange(1, N_STEPS_BLOCK + 1)
w_opt = optimal_weight(W_TEACHER, INCUMBENT_NOISE)   # clean incumbent -> equals W_TEACHER
fig, axes = plt.subplots(len(EST_NOISES), 1,
                         figsize=(11, 3.6 * len(EST_NOISES)), sharex=True)
ax2_first = None
for ax, c_est in zip(axes, EST_NOISES):
    res = results_block[c_est]

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
    ax.set_xscale('log')
    ax.set_ylabel('weight', fontsize=16)
    ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
    ax.set_title(f'established-feature noise = {c_est:.3g}', fontsize=14)
    ax.set_zorder(ax2.get_zorder() + 1)   # put the weight's axes above the indicator axes
    ax.patch.set_visible(False)            # but keep ax transparent so ax2 shows through

    # rough "matured" reference: incumbent first reaches 90% of its solo optimum
    if w_opt > 0:
        hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
        if len(hit):
            ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3, zorder=10)

axes[-1].set_xlabel('step (log scale)', fontsize=16)
fig.suptitle('Weight Trend Indicators (blocking by an established feature)', fontsize=18)

# Build legend with left and right handles/labels
hL, lL = axes[0].get_legend_handles_labels()
hR, lR = ax2_first.get_legend_handles_labels()
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.03),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
