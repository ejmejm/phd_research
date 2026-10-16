r"""Fifth experiment: blocking learned with IDBD3 (meta-step-size 0.003).

Exactly the fourth experiment's blocking setup (two colinear features, established
noise fixed at $c_{\text{est}} = 0.1$, clean incumbent learned from zero,
$W_{\text{teacher}} = 0.5$), but learning now uses the project's **IDBD3**
optimiser (`optax_idbd3`, prediction-grads version) instead of constant-step-size
LMS. Each weight carries its own adaptive step-size $\alpha = e^{\beta}$, updated
with meta-step-size $\eta = 0.003$; the **swept** quantity is now the *initial*
step-size $\alpha_0$ (one panel each).

Because IDBD3 adapts the step-size, the previously *observed* $-h$ indicator is now
the optimiser's own running $h$-trace. In this linear case the prediction gradient
is just the input, so the $h$-decay term is $x^2$, and per weight each step:

$$\beta \leftarrow \beta + \eta\, x\, h, \qquad \alpha = e^{\beta}, \qquad
h \leftarrow h\,[\,1 - \alpha x^2\,]_+ + 2\,\text{err}\,x, \qquad
w \leftarrow w - \alpha\,(2\,\text{err}\,x),$$

with $\beta_0 = \log \alpha_0$. The partially transparent established-weight line is
kept from the previous experiments.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    W_TEACHER, INCUMBENT_NOISE, EST_NOISE_FIXED,
    make_problem_data, optimal_weight, _norm_peak,
)

# ---------------------------------------------------------------------------
# Fifth experiment: the fourth experiment's blocking setup, but learned with the
# project's IDBD3 optimiser (optax_idbd3, prediction-grads version) instead of
# constant-step-size LMS. Each weight has its own adaptive step-size
# alpha = exp(beta), updated with meta-step-size META_LR_IDBD3; the swept quantity
# is now the INITIAL step-size (init_lr). The established / incumbent setup is
# unchanged from experiment 4 (W_TEACHER, INCUMBENT_NOISE, EST_NOISE_FIXED reused).
#
# Per weight, per step (prediction grads -> h-decay term is x**2):
#   loss_grad = 2*error*x ;  pred_grad = x ;  d = x**2
#   beta  <- beta + meta_lr * pred_grad * h          (project IDBD3 meta-update)
#   alpha = exp(beta)
#   h     <- h * max(1 - alpha*d, 0) + loss_grad     (IDBD3: no alpha on the add term)
#   w     <- w - alpha * loss_grad
# The h-trace is now the optimiser's own running h (it used to be observed-only).
# ---------------------------------------------------------------------------

META_LR_IDBD3 = 0.003
INIT_STEP_SIZES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]   # swept: the INITIAL step-size (alpha_0)
N_STEPS_IDBD3 = 500_000


def run_idbd3_with_indicators(features, y, meta_lr, init_lr,
                              trace_decays=TRACE_DECAYS, w_init=None):
    """Learn with IDBD3 (prediction-grads version, so the h-decay term is x**2)
    over `features` (n_steps, n_feat). Per-weight step-sizes alpha = exp(beta)
    adapt with meta-step-size `meta_lr`, all starting at `init_lr`. The h-trace and
    dw traces are tracked for the LAST column (the feature under study). Outputs
    mirror run_lms_with_indicators (plus 'alpha'); `w_init` defaults to zeros."""
    n_steps, n_feat = features.shape
    tracked = n_feat - 1
    decays = np.asarray(trace_decays, dtype=float)
    n_dec = len(decays)

    w = np.zeros(n_feat) if w_init is None else np.array(w_init, dtype=float)
    beta = np.full(n_feat, np.log(init_lr))     # per-weight log step-size
    h = np.zeros(n_feat)                          # per-weight IDBD3 gradient trace
    dw_tr = np.zeros(n_dec)

    w_hist = np.empty((n_steps, n_feat))
    dw_trace_hist = np.empty((n_steps, n_dec))
    h_idbd3_hist = np.empty(n_steps)
    alpha_hist = np.empty((n_steps, n_feat))

    for t in range(n_steps):
        f = features[t]
        error = float(w @ f) - y[t]              # y_hat - y  (project convention)
        loss_grad = 2.0 * error * f              # dL/dw      (per weight)
        pred_grad = f                            # dy_hat/dw = x   (prediction grads)
        decay_term = f * f                       # x**2

        # IDBD3 meta-update (prediction grad * h, project convention)
        beta = beta + meta_lr * pred_grad * h
        alpha = np.exp(beta)

        # IDBD3 h-trace: no alpha factor on the additive loss-grad term
        h = h * np.clip(1.0 - alpha * decay_term, 0.0, None) + loss_grad

        # adaptive-step-size weight update for every weight
        dw = -alpha * loss_grad
        w = w + dw

        # EMA traces of the tracked weight's changes
        dw_tr = decays * dw_tr + (1.0 - decays) * dw[tracked]

        w_hist[t] = w
        dw_trace_hist[t] = dw_tr
        h_idbd3_hist[t] = h[tracked]
        alpha_hist[t] = alpha

    return {'w': w_hist, 'dw_trace': dw_trace_hist,
            'h_idbd3': h_idbd3_hist, 'alpha': alpha_hist}


w_est0 = optimal_weight(W_TEACHER, EST_NOISE_FIXED)   # established starts at its solo optimum

results_idbd3 = {}
for ss in INIT_STEP_SIZES:
    # column 0 = established feature (noise EST_NOISE_FIXED), column 1 = incumbent (clean, tracked)
    feats, targ = make_problem_data(
        N_STEPS_IDBD3, [W_TEACHER, 0.0], feature_noise=[EST_NOISE_FIXED, INCUMBENT_NOISE],
        target_noise_std=0.1, shared_signal=True, seed=0,
    )
    results_idbd3[ss] = run_idbd3_with_indicators(
        feats, targ, meta_lr=META_LR_IDBD3, init_lr=ss, w_init=[w_est0, 0.0])


plot_steps = np.arange(1, N_STEPS_IDBD3 + 1)
w_opt = optimal_weight(W_TEACHER, INCUMBENT_NOISE)   # clean incumbent -> equals W_TEACHER
fig, axes = plt.subplots(len(INIT_STEP_SIZES), 1,
                         figsize=(11, 3.6 * len(INIT_STEP_SIZES)), sharex=True)
ax2_first = None
for ax, ss in zip(axes, INIT_STEP_SIZES[::-1]):
    res = results_idbd3[ss]

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
    ax.set_title(f'init step-size = {ss:.3g}', fontsize=14)
    ax.set_zorder(ax2.get_zorder() + 1)   # put the weight's axes above the indicator axes
    ax.patch.set_visible(False)            # but keep ax transparent so ax2 shows through

    # rough "matured" reference: incumbent first reaches 90% of its solo optimum
    if w_opt > 0:
        hit = np.where(res['w'][:, -1] >= 0.9 * w_opt)[0]
        if len(hit):
            ax.axvline(hit[0] + 1, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.3, zorder=10)

axes[-1].set_xlabel('step (log scale)', fontsize=16)
fig.suptitle(f'Weight Trend Indicators (blocking, IDBD3, meta-step-size = {META_LR_IDBD3})', fontsize=18)

# Build legend with left and right handles/labels
hL, lL = axes[0].get_legend_handles_labels()
hR, lR = ax2_first.get_legend_handles_labels()
legend = fig.legend(
    hL + hR, lL + lR, loc='upper center', ncol=4,
    fontsize=12, bbox_to_anchor=(0.5, 1.03),
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
