r"""Sixth experiment: the *drift judge* — an adaptive-horizon protection indicator.

Same toy problem as experiments 3–5 (the blocking setup), now repeated as a
generate-and-test loop with a real prune/keep decision:

- **established feature** (column 0): fixed at 10% noise, started at its solo
  optimum $w^\star_{\text{est}} = \text{optimal\_weight}(0.5, 0.1) \approx 0.549$;
- **candidate feature** (column 1, tracked): a *perfect* (clean, $c = 0$) view of
  the same shared signal, learned from a zero weight — exactly the incumbent of
  experiments 3–5. It is genuinely useful (joint optimum drives it to $\approx
  0.5$ and the noisier established feature to $\approx 0$), so the correct action
  is always **KEEP**; the danger is *premature pruning* while it is still blocked.

Each episode protects the candidate under constant-step-size LMS while the drift
judge watches its gradient signal ``g = delta * phi``. When the judge returns a
**converged verdict** protection ends and the candidate is kept or pruned by
comparing $|w|$ to a utility bar; the problem is then reset (weights back to their
initial values) while the judge's **own parameters persist** (the global patience
prior ``P0`` and its controller state). We run episodes until ``P0`` converges.

The earlier experiments established the gap: under blocking the cheap IDBD/IDBD3
``h``-flip fires almost immediately (the incumbent blocks the candidate's gradient
signal, so ``h`` flips long before the weight matures), which would prune a
genuinely useful feature. The drift judge replaces that raw flip with a
statistical drift test that keeps the step-size slaving of ``h`` and adds one
dimensionless, globally learned patience multiplier ``P`` that stretches the
horizon exactly when blocking makes it necessary.

Two indicators are compared on every episode:
  * **naive** — the raw IDBD3 ``h``-flip of experiments 3–5 (stop protecting at the
    first sign flip; decide keep/prune from $|w|$ there);
  * **drift judge** — protect until a converged verdict, then decide.

Three plots are produced:
  1. the patience prior ``P0`` over episodes (plus the per-feature horizon the
     doubling test actually reaches), the value we run until it converges;
  2. cumulative keeps/prunes over episodes, drift judge vs the naive baseline;
  3. the blocking case run with the *converged* ``P0``, in the style of
     experiments 3–5 (weights + the two indicators; the $\Delta w$ traces are
     dropped), annotated with where the naive ``h``-flip would prune versus where
     the drift judge ends protection and keeps.

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
    STEP_SIZE, W_COLOR, H_IDBD3_COLOR,
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
MAX_EPISODES = 150           # outer stop if patience never settles
NAIVE_WARMUP = 10            # ignore the first few steps when finding the naive h-flip
# convergence of P0: coefficient of variation over a trailing window
CONV_WINDOW = 60
CONV_COV = 0.02
CONV_MIN_EPISODES = 120


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


def make_blocking_problem(n_steps, seed):
    """One episode of the exp 3-5 blocking setup.

    Column 0 = established feature (fixed 10% noise, started at its solo optimum);
    column 1 = a perfect (clean) candidate that is a noisy-free view of the same
    shared signal, learned from zero. Correct action is always KEEP.
    """
    feats, y = make_problem_data(
        n_steps, [W_TEACHER, 0.0], feature_noise=[EST_NOISE_FIXED, 0.0],
        target_noise_std=0.1, shared_signal=True, seed=seed,
    )
    w_est0 = optimal_weight(W_TEACHER, EST_NOISE_FIXED)
    return feats, y, w_est0


def run_episode(P0, feats, y, w_est0, step_size=STEP_SIZE, record=False):
    """Protect the candidate (column 1) until the judge returns a verdict.

    Both weights learn with constant-step-size LMS; the judge observes the
    candidate's gradient signal ``g = delta * phi`` each step. The naive IDBD3
    ``h``-trace of experiments 3-5 is tracked alongside so its (premature) first
    sign flip gives a baseline keep/prune decision on the same episode.
    """
    judge = DriftJudge(P0)
    f0 = feats[:, 0]
    f1 = feats[:, 1]
    n = len(y)
    w0 = float(w_est0)
    w1 = 0.0

    h_naive = 0.0
    naive_flip_step = -1
    naive_flip_w = None

    n_veto = 0
    n_incon = 0
    verdict_step = -1
    decision = None
    events = []

    if record:
        w1_hist = np.empty(n)
        w0_hist = np.empty(n)
        u_hist = np.empty(n)
        h_hist = np.empty(n)

    for t in range(n):
        f0t = f0[t]
        f1t = f1[t]
        error = w0 * f0t + w1 * f1t - y[t]       # yhat - y
        g = -error * f1t                          # delta * phi  (delta = y - yhat)

        ev = judge.step(g, step_size, f1t, t)

        # naive IDBD3 h-trace of exps 3-5 (observed only) + its first sign flip
        loss_grad = 2.0 * error * f1t
        keep = max(0.0, 1.0 - step_size * f1t * f1t)
        h_prev = h_naive
        h_naive = h_naive * keep + loss_grad
        if (naive_flip_step < 0 and t >= NAIVE_WARMUP
                and h_prev != 0.0 and h_naive * h_prev < 0.0):
            naive_flip_step = t
            naive_flip_w = w1

        w0 -= step_size * 2.0 * error * f0t
        w1 -= step_size * 2.0 * error * f1t

        if record:
            w1_hist[t] = w1
            w0_hist[t] = w0
            u_hist[t] = judge.u
            h_hist[t] = h_naive

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

    if naive_flip_w is None:                        # naive never flipped -> keep
        naive_decision = 'keep'
    else:
        naive_decision = 'keep' if abs(naive_flip_w) >= UTILITY_BAR else 'prune'

    out = dict(
        decision=decision,
        verdict_step=verdict_step,
        n_veto=n_veto,
        n_incon=n_incon,
        w1=w1,
        P_final=judge.P,
        naive_decision=naive_decision,
        naive_flip_step=naive_flip_step,
    )
    if record:
        sl = slice(0, verdict_step + 1)
        out['hist'] = dict(
            w1=w1_hist[sl], w0=w0_hist[sl], u=u_hist[sl], h_idbd3=h_hist[sl],
            events=events,
        )
    return out


def calibrate():
    """Run blocking episodes until the patience prior P0 converges (or budget)."""
    controller = PatienceController()

    records = []
    P0_series = []
    for ep in range(MAX_EPISODES):
        feats, y, w_est0 = make_blocking_problem(MAX_EPISODE_STEPS, seed=1000 + ep)
        res = run_episode(controller.P0, feats, y, w_est0)

        # controller: one veto=1 per veto event, one veto=0 for the converged verdict
        for _ in range(res['n_veto']):
            controller.event(1)
        controller.event(0)

        P0_series.append(controller.P0)
        records.append(dict(
            episode=ep, decision=res['decision'], verdict_step=res['verdict_step'],
            n_veto=res['n_veto'], w1=res['w1'], P0=controller.P0,
            P_final=res['P_final'], naive_decision=res['naive_decision'],
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
keeps = sum(r['decision'] == 'keep' for r in records)
prunes = sum(r['decision'] == 'prune' for r in records)
naive_prunes = sum(r['naive_decision'] == 'prune' for r in records)
print(f'episodes run              : {n_ep}')
print(f'patience prior P0         : {P0_final:.2f}  (converged: {converged})')
print(f'drift judge keeps / prunes: {keeps} / {prunes}')
print(f'naive h-flip keeps / prunes: {n_ep - naive_prunes} / {naive_prunes}')

episodes = np.arange(1, n_ep + 1)

# ---------------------------------------------------------------------------
# Plot 1 — the patience prior P0 over episodes (the value we run until it settles),
# with the per-feature horizon the doubling test actually reaches each episode.
# ---------------------------------------------------------------------------
P_reached = np.array([r['P_final'] for r in records], dtype=float)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(episodes, P0_series, color=W_COLOR, lw=2.4, label='patience prior $P_0$ (persists)')
ax.plot(episodes, P_reached, color='#3953e2', lw=1.4, alpha=0.55, marker='o', ms=3,
        label='per-feature horizon $P$ reached (doubling test)')
ax.axhline(P0_final, color='0.4', ls='--', lw=1.4, alpha=0.8,
           label=f'converged $P_0 \\approx {P0_final:.2f}$')
ax.set_yscale('log', base=2)
ax.set_xlabel('episode (fresh perfect candidate)', fontsize=16)
ax.set_ylabel('patience (log$_2$)', fontsize=16)
ax.set_title('Adaptive patience over blocking episodes', fontsize=18)
ax.legend(loc='center right', fontsize=11)
fig.tight_layout()
fig.savefig('exp6_patience_over_time.png', dpi=130, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 2 — cumulative keeps/prunes over episodes: drift judge vs naive h-flip.
# The candidate is always a perfect (useful) feature, so KEEP is always correct;
# the naive indicator prunes it prematurely under blocking, the drift judge keeps.
# ---------------------------------------------------------------------------
judge_keep = np.cumsum([r['decision'] == 'keep' for r in records])
judge_prune = np.cumsum([r['decision'] == 'prune' for r in records])
naive_prune = np.cumsum([r['naive_decision'] == 'prune' for r in records])

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(episodes, 0, judge_keep, color='#029999', alpha=0.15)
ax.plot(episodes, judge_keep, color='#029999', lw=3.4, zorder=4,
        label='kept — drift judge (correct)')
ax.plot(episodes, naive_prune, color=H_IDBD3_COLOR, lw=2.0, ls=(0, (6, 4)), zorder=6,
        label='pruned — naive $h$-flip (premature)')
ax.plot(episodes, judge_prune, color='#3953e2', lw=2.2, ls=':', zorder=5,
        label='pruned — drift judge')
ax.set_xlabel('episode (fresh perfect candidate)', fontsize=16)
ax.set_ylabel('cumulative count', fontsize=16)
ax.set_title('Keeps vs prunes over time (perfect blocked feature)', fontsize=18)
ax.legend(loc='upper left', fontsize=12)
fig.tight_layout()
fig.savefig('exp6_prunes_keeps_over_time.png', dpi=130, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 3 — the blocking case with the CONVERGED P0, drawn in the experiment 3/4/5
# style (weights + indicators), annotated with the naive h-flip (premature prune)
# versus the drift judge's actual verdict (protect-then-keep). The delta-w traces
# with different gammas are dropped.
# ---------------------------------------------------------------------------
feats, y, w_est0 = make_blocking_problem(MAX_EPISODE_STEPS, seed=7)
final = run_episode(P0_final, feats, y, w_est0, record=True)
hist = final['hist']
n_show = len(hist['w1'])
plot_steps = np.arange(1, n_show + 1)
w_opt = optimal_weight(W_TEACHER, 0.0)            # clean candidate solo optimum == 0.5
neg_h = -hist['h_idbd3']

fig, ax = plt.subplots(figsize=(11, 5.4))

# --- right axis: the two indicators (normalized to unit peak, -h convention) ---
ax2 = ax.twinx()
ax2.plot(plot_steps, _norm_peak(neg_h), color=H_IDBD3_COLOR, lw=1.5, alpha=0.8,
         label='$-h$ (IDBD3, naive)')
ax2.plot(plot_steps, _norm_peak(-hist['u']), color='#9923DC', lw=1.5, alpha=0.8,
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
if final['naive_flip_step'] >= 0:
    ax.axvline(final['naive_flip_step'] + 1, color=H_IDBD3_COLOR, ls=(0, (4, 3)),
               lw=1.4, alpha=0.85, zorder=11, label='naive $h$-flip (would prune)')
ax.axvline(final['verdict_step'] + 1, color='#029999', ls='-', lw=1.8, alpha=0.85,
           zorder=11, label=f'drift-judge verdict → {final["decision"].upper()}')
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
fig.tight_layout(rect=(0, 0.10, 1, 1.0))
fig.savefig('exp6_final_blocking_converged.png', dpi=130, bbox_inches='tight')

print('saved: exp6_patience_over_time.png, exp6_prunes_keeps_over_time.png, '
      'exp6_final_blocking_converged.png')
