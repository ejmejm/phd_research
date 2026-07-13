r"""Sixth experiment: the *drift judge* — an adaptive-horizon protection indicator.

The earlier experiments established the gap: under **blocking** (a candidate that
is colinear with an already-established feature), the cheap maturity indicators —
the $\Delta w$ traces and, most sharply, the IDBD/IDBD3 $h$-flip — fire *too
early*. The incumbent blocks the candidate's gradient signal, so convergence
along the shared direction is slow and $h$ flips almost immediately, which in a
real generate-and-test learner would **prune a genuinely useful feature**.

This experiment closes that gap by turning the raw $h$-flip into a *statistical
test with a learned horizon* (the "drift judge", see ``algorithm.md`` / the
docstring block below). The judge keeps the step-size slaving of $h$ (so it needs
no retuning across step-sizes or noise) and adds one dimensionless, **globally
learned** patience multiplier ``P`` that stretches the horizon exactly when
blocking makes it necessary.

Unlike experiments 1–5, which only *observed* indicators, here the indicator
drives a real decision. Each "episode" generates a fresh candidate feature and
protects it while it learns (constant-step-size LMS, ``STEP_SIZE``); when the
judge returns a **converged verdict** protection ends and the candidate is
**kept** (tenured) or **pruned** by comparing $|w|$ to a utility bar. The problem
is then reset — weights back to their initial values — but the **algorithm's own
parameters persist** (the global patience prior ``P0`` and its running state).
We run episodes until ``P0`` converges (or a step budget is exhausted).

Why a mixed candidate pool. A strongly-blocked candidate (the corr $\approx0.99$,
10%-noise case that is our focus) never lands in the *veto* branch: its residual
gradient sits at the noise floor, so the judge protects it via the doubling
(inconclusive) branch and then keeps it. The *veto* pressure that the global
controller needs — a premature $h$-flip while genuine drift persists — comes from
the *easier* useful features, which flip early exactly when ``P0`` is too small.
So the pool is a realistic generate-and-test mix: strongly-blocked useful
features (KEEP, the focus), easier useful features (KEEP, the veto source that
pushes ``P0`` up), and near-duplicate junk features (PRUNE). ``P0`` then
converges to the pool's blocking difficulty, exactly as the algorithm intends:
"P0 tracks the blocking difficulty of the current candidate pool without any
explicit correlation measurement."

Three plots are produced:
  1. the patience prior ``P0`` over episodes (the value we run until it converges);
  2. cumulative keeps and prunes over episodes;
  3. the focus case (blocking, 10% noise) run with the *converged* ``P0``, drawn
     in the exact style of experiments 3–5 (weights + normalized indicators),
     annotated with where the naive $h$-flip would have pruned versus where the
     drift judge actually ends protection and keeps.

--- drift judge algorithm (verbatim summary of the spec) -------------------
Per protected feature the judge keeps EMA traces of g = delta * phi at two
timescales: a fast trend trace ``u`` (h-like, rate dilated by P) whose sign flips
are cheap event triggers, and slow mean/second-moment traces ``m``/``q`` (kappa
times slower). While a weight genuinely adapts, E[g] != 0, ``u`` holds its sign,
and nothing happens. When ``u`` flips, a three-way test on the slow mean decides:
drift still significant -> VETO (grow this feature's window, keep protecting);
CI entirely inside the indifference band -> CONVERGED VERDICT (end protection,
compare |w| to the utility bar); otherwise -> INCONCLUSIVE (grow window, keep
accumulating -> a doubling sequential test). A global controller nudges log P0 up
on vetoes and down on converged verdicts toward a target veto fraction r*.
"""

import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    STEP_SIZE, TRACE_DECAYS, W_COLOR, DW_COLORS, H_IDBD3_COLOR,
    W_TEACHER, EST_NOISE_FIXED,
    make_problem_data, optimal_weight, _norm_peak,
)

# ---------------------------------------------------------------------------
# Drift-judge hyperparameters (all dimensionless; the defaults from the spec).
# None need retuning across step-sizes / noise: that is absorbed by the
# alpha * phi^2 slaving of every rate.
# ---------------------------------------------------------------------------
KAPPA = 8.0        # slow/fast timescale ratio
Z_STAR = 2.5       # significance level (~1% two-sided false veto)
G_GROW = 2.0       # window growth factor per veto / inconclusive
EPS_W = 0.005      # indifference band: tolerated weight motion per adaptation-time
R_STAR = 0.075     # target veto fraction among decisive events
ETA = 0.05         # controller learning rate
P0_INIT = 1.0      # patience prior initial value
P_MAX = 1.0e6      # patience cap
NEFF_MIN = 30.0    # minimum effective sample count before a test may fire

# ---------------------------------------------------------------------------
# Experiment harness knobs.
# ---------------------------------------------------------------------------
UTILITY_BAR = 0.25          # keep iff |w_candidate| >= UTILITY_BAR at the verdict
MAX_EPISODE_STEPS = 200_000  # per-episode protection budget (steps)
MAX_EPISODES = 500           # outer stop if patience never settles
POOL_KINDS = ['blocked', 'indep', 'junk']
POOL_WEIGHTS = (0.5, 0.3, 0.2)
# convergence of P0: coefficient of variation over a trailing window
CONV_WINDOW = 80
CONV_COV = 0.04
CONV_MIN_EPISODES = 160


class DriftJudge:
    """Per-feature adaptive-horizon protection test (one candidate feature).

    ``step`` performs the ~6-flop per-step update and returns the event that
    fired this step, one of ``None`` / ``'veto'`` / ``'inconclusive'`` /
    ``'converged'``. A ``'converged'`` result means protection should end.
    """

    __slots__ = ('u', 'm', 'q', 'v', 'b', 's2', 'P', 't_last')

    def __init__(self, P0):
        self.u = 0.0
        self.m = 0.0
        self.q = 0.0
        self.v = 0.0
        self.b = 1.0
        self.s2 = 0.1        # small positive tracker of E[phi^2]
        self.P = P0          # this feature's patience (starts at the global prior)
        self.t_last = 0

    def step(self, g, alpha_j, phi_j, t):
        if phi_j == 0.0:
            return None      # lam == 0 anyway; skip

        P = self.P
        lam = alpha_j * phi_j * phi_j / P     # fast rate, alpha-slaved, P-dilated
        if lam > 0.5:
            lam = 0.5
        ls = lam / KAPPA                        # slow rate

        u_prev = self.u
        self.u += lam * (g - self.u)
        self.m += ls * (g - self.m)
        self.q += ls * (g * g - self.q)
        self.v = (1.0 - ls) * (1.0 - ls) * self.v + ls * ls   # exact variance factor
        self.b = (1.0 - ls) * self.b                           # bias remainder (b -> 0)
        self.s2 += 1e-3 * (phi_j * phi_j - self.s2)

        u = self.u
        if u_prev * u >= 0.0 or u_prev == 0.0 or u == 0.0:
            return None      # no sign flip (guard u == 0)

        omb = 1.0 - self.b
        n_eff = (omb * omb / self.v) if self.v > 0.0 else 0.0
        W = KAPPA * P / (alpha_j * self.s2)     # pacing: one slow window since last event
        if n_eff < NEFF_MIN or (t - self.t_last) < W:
            return None

        m_hat = self.m / omb
        q_hat = self.q / omb
        var = q_hat - m_hat * m_hat
        if var < 0.0:
            var = 0.0
        SE = math.sqrt(var * self.v) / omb      # calibrated std error of m_hat
        eps = EPS_W * self.s2                    # indifference band in gradient units

        if abs(m_hat) - Z_STAR * SE > eps:       # drift significantly outside band
            self.u = m_hat                        # re-anchor fast trace
            self.P = P * G_GROW                    # grow this feature's window
            self.t_last = t
            return 'veto'
        if abs(m_hat) + Z_STAR * SE < eps:       # CI entirely inside band
            return 'converged'
        self.P = P * G_GROW                        # inconclusive: grow window, keep going
        self.t_last = t
        return 'inconclusive'


class PatienceController:
    """Global controller adapting the patience prior P0 from decisive outcomes."""

    def __init__(self):
        self.log_P0 = math.log(P0_INIT)

    @property
    def P0(self):
        return min(max(math.exp(self.log_P0), 1.0), P_MAX)

    def event(self, veto):
        self.log_P0 += ETA * (veto - R_STAR)
        self.log_P0 = math.log(self.P0)      # keep log in sync with the clip


def make_candidate(kind, n_steps, seed):
    """Build one pool member: ``(features, y, w_est0, correct_action)``.

    Column 0 is always the established feature (held fixed at 10% noise, started
    at its solo optimum); column 1 is the tracked candidate, learned from zero.
    """
    if kind == 'blocked':
        # FOCUS: clean candidate sharing the signal with the 10%-noise established
        # feature (corr ~0.99). Joint optimum drives the candidate to ~0.5 -> KEEP.
        feats, y = make_problem_data(
            n_steps, [W_TEACHER, 0.0], feature_noise=[EST_NOISE_FIXED, 0.0],
            target_noise_std=0.1, shared_signal=True, seed=seed,
        )
        return feats, y, optimal_weight(W_TEACHER, EST_NOISE_FIXED), 'keep'
    if kind == 'indep':
        # Easier useful feature: independent of the established one. At a too-small
        # P0 its fast trace flips prematurely while it is still climbing -> VETO,
        # which is what pushes the global patience up. Correct action -> KEEP.
        feats, y = make_problem_data(
            n_steps, [0.0, W_TEACHER], feature_noise=[EST_NOISE_FIXED, 0.0],
            target_noise_std=0.1, shared_signal=False, seed=seed,
        )
        return feats, y, 0.0, 'keep'
    if kind == 'junk':
        # Near-duplicate of the established feature (perfect twin, joint optimum
        # ~0). The judge converges at |w| ~ 0 -> PRUNE.
        feats, y = make_problem_data(
            n_steps, [W_TEACHER, 0.0], feature_noise=[0.0, 0.0],
            target_noise_std=0.1, shared_signal=True, seed=seed,
        )
        return feats, y, optimal_weight(W_TEACHER, 0.0), 'prune'
    raise ValueError(kind)


def run_episode(P0, feats, y, w_est0, step_size=STEP_SIZE, record=False):
    """Protect the candidate (column 1) until the judge returns a verdict.

    Learning is constant-step-size LMS on both weights; the judge observes the
    candidate's gradient signal ``g = delta * phi`` each step. Returns a dict with
    the ``decision`` ('keep'/'prune'), the ``verdict_step``, veto/inconclusive
    counts, the final candidate weight, and (when ``record``) full per-step
    histories plus the naive IDBD3 h-trace for the experiment-3/4/5-style plot.
    """
    judge = DriftJudge(P0)
    f0 = feats[:, 0]
    f1 = feats[:, 1]
    n = len(y)
    w0 = float(w_est0)
    w1 = 0.0

    n_veto = 0
    n_incon = 0
    verdict_step = -1
    decision = None
    events = []

    if record:
        decays = np.asarray(TRACE_DECAYS, dtype=float)
        n_dec = len(decays)
        dw_tr = np.zeros(n_dec)
        h_idbd3 = 0.0
        w1_hist = np.empty(n)
        w0_hist = np.empty(n)
        u_hist = np.empty(n)
        P_hist = np.empty(n)
        dw_hist = np.empty((n, n_dec))
        h_hist = np.empty(n)

    for t in range(n):
        f0t = f0[t]
        f1t = f1[t]
        error = w0 * f0t + w1 * f1t - y[t]       # yhat - y
        g = -error * f1t                          # delta * phi  (delta = y - yhat)

        ev = judge.step(g, step_size, f1t, t)

        if record:
            loss_grad = 2.0 * error * f1t
            keep = max(0.0, 1.0 - step_size * f1t * f1t)
            h_idbd3 = h_idbd3 * keep + loss_grad
            dw1 = -step_size * loss_grad
            dw_tr = decays * dw_tr + (1.0 - decays) * dw1

        w0 -= step_size * 2.0 * error * f0t
        w1 -= step_size * 2.0 * error * f1t

        if record:
            w1_hist[t] = w1
            w0_hist[t] = w0
            u_hist[t] = judge.u
            P_hist[t] = judge.P
            dw_hist[t] = dw_tr
            h_hist[t] = h_idbd3

        if ev == 'veto':
            n_veto += 1
            events.append((t, 'veto'))
        elif ev == 'inconclusive':
            n_incon += 1
            events.append((t, 'inconclusive'))
        elif ev == 'converged':
            verdict_step = t
            decision = 'keep' if abs(w1) >= UTILITY_BAR else 'prune'
            events.append((t, 'converged'))
            break

    if verdict_step < 0:                           # budget exhausted: force a call
        verdict_step = n - 1
        decision = 'keep' if abs(w1) >= UTILITY_BAR else 'prune'

    out = dict(
        decision=decision,
        verdict_step=verdict_step,
        n_veto=n_veto,
        n_incon=n_incon,
        w1=w1,
        P_final=judge.P,
    )
    if record:
        sl = slice(0, verdict_step + 1)
        out['hist'] = dict(
            w1=w1_hist[sl], w0=w0_hist[sl], u=u_hist[sl], P=P_hist[sl],
            dw_trace=dw_hist[sl], h_idbd3=h_hist[sl], events=events,
        )
    return out


def calibrate():
    """Run episodes until the patience prior P0 converges (or the budget ends)."""
    controller = PatienceController()
    rng = np.random.default_rng(0)

    records = []          # per-episode dicts
    P0_series = []        # P0 after each episode's controller charges
    for ep in range(MAX_EPISODES):
        kind = str(rng.choice(POOL_KINDS, p=POOL_WEIGHTS))
        feats, y, w_est0, correct = make_candidate(kind, MAX_EPISODE_STEPS, seed=1000 + ep)
        res = run_episode(controller.P0, feats, y, w_est0)

        # controller: one veto=1 per veto event, one veto=0 for the converged verdict
        for _ in range(res['n_veto']):
            controller.event(1)
        controller.event(0)

        P0_now = controller.P0
        P0_series.append(P0_now)
        records.append(dict(
            episode=ep, kind=kind, decision=res['decision'], correct=correct,
            ok=res['decision'] == correct, verdict_step=res['verdict_step'],
            n_veto=res['n_veto'], w1=res['w1'], P0=P0_now,
        ))

        if ep + 1 >= CONV_MIN_EPISODES:
            tail = np.asarray(P0_series[-CONV_WINDOW:])
            if tail.std() / tail.mean() < CONV_COV:
                break

    P0_series = np.asarray(P0_series)
    converged = len(P0_series) < MAX_EPISODES
    P0_final = float(np.mean(P0_series[-CONV_WINDOW:]))
    return controller, records, P0_series, converged, P0_final


# ===========================================================================
# Run the calibration.
# ===========================================================================
controller, records, P0_series, converged, P0_final = calibrate()
n_ep = len(records)
keeps = [r for r in records if r['decision'] == 'keep']
prunes = [r for r in records if r['decision'] == 'prune']
accuracy = float(np.mean([r['ok'] for r in records]))
print(f'episodes run           : {n_ep}')
print(f'patience converged     : {converged}  (P0 -> {P0_final:.2f})')
print(f'keeps / prunes         : {len(keeps)} / {len(prunes)}')
print(f'decision accuracy      : {accuracy:.3f}')

# ---------------------------------------------------------------------------
# Plot 1 — the patience prior P0 over episodes (the value we run until it settles).
# ---------------------------------------------------------------------------
episodes = np.arange(1, n_ep + 1)
kind_color = {'blocked': '#3953e2', 'indep': '#029999', 'junk': '#FF3300'}

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(episodes, P0_series, color=W_COLOR, lw=2.0, label='patience prior $P_0$', zorder=5)
for kind in POOL_KINDS:
    xs = [r['episode'] + 1 for r in records if r['kind'] == kind]
    ys = [r['P0'] for r in records if r['kind'] == kind]
    ax.scatter(xs, ys, s=14, color=kind_color[kind], alpha=0.65,
               label=f'{kind} candidate', zorder=6)
ax.axhline(P0_final, color='0.4', ls='--', lw=1.5, alpha=0.8,
           label=f'converged $P_0 \\approx {P0_final:.2f}$')
if converged:
    ax.axvline(n_ep, color='0.55', ls=(0, (4, 3)), lw=1.0, alpha=0.5)
ax.set_xlabel('episode (fresh candidate feature)', fontsize=16)
ax.set_ylabel('patience prior $P_0$', fontsize=16)
ax.set_title('Adaptive patience over generate-and-test episodes', fontsize=18)
ax.legend(loc='lower right', fontsize=11, ncol=2)
fig.tight_layout()
fig.savefig('exp6_patience_over_time.png', dpi=130, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 2 — cumulative keeps and prunes over episodes.
# ---------------------------------------------------------------------------
keep_flag = np.array([1 if r['decision'] == 'keep' else 0 for r in records])
prune_flag = 1 - keep_flag
cum_keep = np.cumsum(keep_flag)
cum_prune = np.cumsum(prune_flag)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(episodes, cum_keep, color='#029999', lw=2.4, label='keeps (tenured)')
ax.plot(episodes, cum_prune, color='#FF3300', lw=2.4, label='prunes')
ax.fill_between(episodes, 0, cum_keep, color='#029999', alpha=0.12)
ax.fill_between(episodes, 0, cum_prune, color='#FF3300', alpha=0.12)
# outcome ticks along the top: correct (grey) vs wrong (black x)
wrong = [r['episode'] + 1 for r in records if not r['ok']]
top = max(cum_keep[-1], cum_prune[-1])
if wrong:
    ax.scatter(wrong, [top * 1.02] * len(wrong), marker='x', color='black',
               s=40, label='wrong decision', zorder=6)
ax.set_xlabel('episode (fresh candidate feature)', fontsize=16)
ax.set_ylabel('cumulative count', fontsize=16)
ax.set_title(f'Keeps vs prunes over time  (decision accuracy = {accuracy:.1%})',
             fontsize=18)
ax.legend(loc='upper left', fontsize=12)
fig.tight_layout()
fig.savefig('exp6_prunes_keeps_over_time.png', dpi=130, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 3 — the focus case (blocking, 10% noise) with the CONVERGED P0, drawn in
# the experiment 3/4/5 style, annotated with the naive h-flip (premature prune)
# versus the drift judge's actual verdict (protect-then-keep).
# ---------------------------------------------------------------------------
feats, y, w_est0, correct = make_candidate('blocked', MAX_EPISODE_STEPS, seed=7)
final = run_episode(P0_final, feats, y, w_est0, record=True)
hist = final['hist']
n_show = len(hist['w1'])
plot_steps = np.arange(1, n_show + 1)
w_opt = optimal_weight(W_TEACHER, 0.0)            # clean candidate solo optimum == 0.5

# naive IDBD3 indicator (as plotted before: -h) and its first premature flip
neg_h = -hist['h_idbd3']
naive_flip = None
for i in range(200, n_show):
    if neg_h[i - 1] > 0.0 and neg_h[i] <= 0.0:
        naive_flip = i + 1
        break

fig, ax = plt.subplots(figsize=(11, 6.0))

# --- right axis: normalized indicators (same colours / convention as before) ---
ax2 = ax.twinx()
for k, d in enumerate(TRACE_DECAYS):
    ax2.plot(plot_steps, _norm_peak(hist['dw_trace'][:, k]),
             color=DW_COLORS[k], lw=1.0, alpha=0.5,
             label=f'$\\Delta w$ trace ($\\beta = {d}$)')
ax2.plot(plot_steps, _norm_peak(neg_h), color=H_IDBD3_COLOR, lw=1.3, alpha=0.65,
         label='$-h$ (IDBD3, naive)')
ax2.plot(plot_steps, _norm_peak(-hist['u']), color='#9923DC', lw=1.3, alpha=0.6,
         label='$-u$ (drift-judge fast trace)')
ax2.axhline(0.0, color='0.5', lw=0.6, alpha=0.3)
ax2.set_ylim(-1.15, 1.15)
ax2.set_ylabel('indicators\n(normalized)', fontsize=16)

# --- left axis: the weights (true scale, black) ---
ax.plot(plot_steps, hist['w1'], color=W_COLOR, lw=2.6, label='candidate weight',
        zorder=10)
ax.plot(plot_steps, hist['w0'], color=W_COLOR, lw=2.0, alpha=0.3,
        label='established weight', zorder=10)
ax.axhline(w_opt, color=W_COLOR, ls='--', lw=1.5, alpha=0.7, label='optimal weight',
           zorder=10)
ax.axhline(UTILITY_BAR, color='0.45', ls=':', lw=1.5, alpha=0.8, label='utility bar',
           zorder=10)
ax.set_xscale('log')
ax.set_ylabel('weight', fontsize=16)
ax.set_ylim(-w_opt * 0.3, w_opt * 1.3)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

# annotations: naive premature prune vs drift-judge verdict
if naive_flip is not None:
    ax.axvline(naive_flip, color=H_IDBD3_COLOR, ls=(0, (4, 3)), lw=1.4, alpha=0.8,
               zorder=11, label='naive $h$-flip (would prune)')
verdict_x = final['verdict_step'] + 1
ax.axvline(verdict_x, color='#029999', ls='-', lw=1.8, alpha=0.85, zorder=11,
           label=f'drift-judge verdict → {final["decision"].upper()}')
for (t_ev, kind_ev) in hist['events']:
    if kind_ev in ('veto', 'inconclusive'):
        ax.axvline(t_ev + 1, color='0.6', ls=(0, (1, 3)), lw=0.9, alpha=0.5, zorder=9)

ax.set_xlabel('step (log scale)', fontsize=16)
ax.set_title(f'Drift judge on the blocking case (10% noise), converged $P_0 = {P0_final:.2f}$',
             fontsize=17, pad=12)

hL, lL = ax.get_legend_handles_labels()
hR, lR = ax2.get_legend_handles_labels()
fig.legend(hL + hR, lL + lR, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1.0))
fig.savefig('exp6_final_blocking_converged.png', dpi=130, bbox_inches='tight')

print('saved: exp6_patience_over_time.png, exp6_prunes_keeps_over_time.png, '
      'exp6_final_blocking_converged.png')
