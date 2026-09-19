r"""Sixth experiment: the *drift judge* — an adaptive-horizon protection indicator.

Same toy problem as experiments 3–5 (the blocking setup), now repeated as a
generate-and-test loop with a real prune/keep decision, so we can watch the
algorithm calibrate its patience over time:

- **established feature** (column 0): fixed at 10% noise, started at its solo
  optimum $w^\star_{\text{est}} = \text{optimal\_weight}(0.5, 0.1) \approx 0.549$;
- **candidate feature** (column 1, tracked): a *perfect* (clean, $c = 0$) view of
  the same shared signal, learned from a zero weight — exactly the incumbent of
  experiments 3–5. It is genuinely useful (joint optimum drives it to $\approx
  0.5$ and the established feature to $\approx 0$), so the correct action is
  always **KEEP**; the danger is premature pruning while it is still blocked.

Each episode protects the candidate under constant-step-size LMS while the drift
judge watches its gradient signal ``g = delta * phi``. When the judge returns a
**converged verdict** protection ends and the candidate is kept or pruned by
comparing $|w|$ to a utility bar; the problem is then reset (weights back to
their initial values) — **but the patience is NOT reset**. The very same blocking
problem is presented again and again and we watch the persistent patience adapt.

Patience persistence / calibration. The per-feature judge starts each episode at
the current global patience prior ``P0`` and, while resolving, grows its window by
the doubling test. The vanilla spec moves ``P0`` from *veto* vs *converged*
outcomes; but under strong blocking (corr $\approx 0.99$) the residual gradient
sits at the noise floor, so decisive events are *inconclusive* window-growths
rather than vetoes. We therefore drive the controller with the same
``log P0 += eta * (event - r*)`` rule but treat **any window-growth event**
(veto or inconclusive) as the "patience-too-short" signal and the converged
verdict as the "patience-long-enough" signal. ``P0`` then climbs from 1 and
converges to the horizon the blocking problem actually needs — with no reset
between features, exactly the "same thing again and again" calibration.

Plots produced:
  1. the patience prior ``P0`` over episodes (persists and adapts), plus the
     per-feature horizon the doubling test reaches — run until it converges;
  2. cumulative keeps/prunes over episodes, drift judge vs the naive $h$-flip;
  3. + 4. the within-episode drift-judge dynamics for the **first** and the
     **last** episode: the candidate weight, the slow-mean drift estimate
     $\hat m$ with its decision interval $\hat m \pm z^\star\,\mathrm{SE}$ against
     the indifference band $\pm\varepsilon$ (the actual keep/extend bounds), every
     extend decision (veto/inconclusive) and the final verdict, and the patience
     $P$ / effective-sample $n_{\text{eff}}$ growing through the episode.

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
on window-growth events and down on converged verdicts toward a target fraction r*.
"""

import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    STEP_SIZE, W_COLOR, H_IDBD3_COLOR,
    W_TEACHER, EST_NOISE_FIXED,
    make_problem_data, optimal_weight,
)

# ---------------------------------------------------------------------------
# Drift-judge hyperparameters (all dimensionless; the defaults from the spec).
# ---------------------------------------------------------------------------
KAPPA = 8.0        # slow/fast timescale ratio
Z_STAR = 2.5       # significance level (~1% two-sided false veto)
G_GROW = 2.0       # window growth factor per veto / inconclusive
EPS_W = 0.005      # indifference band: tolerated weight motion per adaptation-time
R_STAR = 0.075     # target window-growth fraction among decisive events
ETA = 0.05         # controller learning rate
P0_INIT = 1.0      # patience prior initial value
P_MAX = 1.0e6      # patience cap
NEFF_MIN = 30.0    # minimum effective sample count before a test may fire

# ---------------------------------------------------------------------------
# Experiment harness knobs.
# ---------------------------------------------------------------------------
UTILITY_BAR = 0.25          # keep iff |w_candidate| >= UTILITY_BAR at the verdict
MAX_EPISODE_STEPS = 200_000  # per-episode protection budget (steps)
MAX_EPISODES = 120           # outer stop if patience never settles
NAIVE_WARMUP = 10            # ignore the first few steps when finding the naive h-flip
# convergence of P0: coefficient of variation over a trailing window
CONV_WINDOW = 25
CONV_COV = 0.05
CONV_MIN_EPISODES = 80

EVENT_COLOR = {'veto': '#ff8c00', 'inconclusive': '0.55', 'converged': '#029999'}


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
    """Global, persistent patience prior P0 adapted from decisive outcomes.

    ``log P0`` moves up by ``eta * (1 - r*)`` for every window-growth event
    (veto or inconclusive: "patience was too short") and down by ``eta * r*`` for
    every converged verdict ("patience was long enough"). It is never reset.
    """

    def __init__(self):
        self.log_P0 = math.log(P0_INIT)

    @property
    def P0(self):
        return min(max(math.exp(self.log_P0), 1.0), P_MAX)

    def grow_event(self):
        self.log_P0 += ETA * (1.0 - R_STAR)
        self.log_P0 = math.log(self.P0)

    def settle_event(self):
        self.log_P0 += ETA * (0.0 - R_STAR)
        self.log_P0 = math.log(self.P0)


def make_blocking_problem(n_steps, seed):
    """One episode of the exp 3-5 blocking setup: established feature fixed at 10%
    noise (started at its solo optimum), candidate a perfect clean view of the
    same shared signal, learned from zero. Correct action is always KEEP."""
    feats, y = make_problem_data(
        n_steps, [W_TEACHER, 0.0], feature_noise=[EST_NOISE_FIXED, 0.0],
        target_noise_std=0.1, shared_signal=True, seed=seed,
    )
    return feats, y, optimal_weight(W_TEACHER, EST_NOISE_FIXED)


def run_episode(P0, feats, y, w_est0, step_size=STEP_SIZE, record=False):
    """Protect the candidate (column 1) until the judge returns a verdict.

    Both weights learn with constant-step-size LMS; the judge observes the
    candidate's gradient signal ``g = delta * phi`` each step. The naive IDBD3
    ``h``-trace of experiments 3-5 is tracked alongside so its (premature) first
    sign flip gives a baseline keep/prune decision on the same episode. When
    ``record`` is set, the judge's full test state is logged every step for the
    within-episode diagnostic plot.
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
        rec = {k: np.full(n, np.nan) for k in
               ('w1', 'w0', 'u', 'm_hat', 'se', 'eps', 'P', 'n_eff')}

    for t in range(n):
        f0t = f0[t]
        f1t = f1[t]
        error = w0 * f0t + w1 * f1t - y[t]       # yhat - y
        g = -error * f1t                          # delta * phi  (delta = y - yhat)

        ev = judge.step(g, step_size, f1t, t)

        loss_grad = 2.0 * error * f1t             # naive IDBD3 h-trace (exps 3-5)
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
            rec['w1'][t] = w1
            rec['w0'][t] = w0
            rec['u'][t] = judge.u
            rec['P'][t] = judge.P
            omb = 1.0 - judge.b
            if omb > 1e-9 and judge.v > 0.0:
                mh = judge.m / omb
                qh = judge.q / omb
                var = qh - mh * mh
                if var < 0.0:
                    var = 0.0
                rec['m_hat'][t] = mh
                rec['se'][t] = math.sqrt(var * judge.v) / omb
                rec['n_eff'][t] = omb * omb / judge.v
            rec['eps'][t] = EPS_W * judge.s2

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

    if naive_flip_w is None:
        naive_decision = 'keep'
    else:
        naive_decision = 'keep' if abs(naive_flip_w) >= UTILITY_BAR else 'prune'

    out = dict(
        decision=decision, verdict_step=verdict_step, n_veto=n_veto,
        n_incon=n_incon, w1=w1, P_final=judge.P,
        naive_decision=naive_decision, naive_flip_step=naive_flip_step,
    )
    if record:
        sl = slice(0, verdict_step + 1)
        out['hist'] = {k: v[sl] for k, v in rec.items()}
        out['hist']['events'] = events
    return out


def calibrate():
    """Repeat the SAME blocking problem; adapt the persistent patience prior."""
    controller = PatienceController()

    records = []
    P0_series = []
    for ep in range(MAX_EPISODES):
        start_P0 = controller.P0
        feats, y, w_est0 = make_blocking_problem(MAX_EPISODE_STEPS, seed=1000 + ep)
        res = run_episode(start_P0, feats, y, w_est0)

        for _ in range(res['n_veto'] + res['n_incon']):
            controller.grow_event()            # patience was too short -> push up
        controller.settle_event()              # converged -> gently trim down

        P0_series.append(start_P0)
        records.append(dict(
            episode=ep, seed=1000 + ep, start_P0=start_P0, decision=res['decision'],
            n_grow=res['n_veto'] + res['n_incon'], w1=res['w1'],
            P_final=res['P_final'], naive_decision=res['naive_decision'],
        ))

        if ep + 1 >= CONV_MIN_EPISODES:
            tail = np.asarray(P0_series[-CONV_WINDOW:])
            if tail.std() / tail.mean() < CONV_COV:
                break

    P0_series = np.asarray(P0_series)
    converged = len(P0_series) < MAX_EPISODES
    P0_final = float(np.mean(P0_series[-CONV_WINDOW:]))
    return records, P0_series, converged, P0_final


# ===========================================================================
# Run the calibration.
# ===========================================================================
records, P0_series, converged, P0_final = calibrate()
n_ep = len(records)
keeps = sum(r['decision'] == 'keep' for r in records)
prunes = sum(r['decision'] == 'prune' for r in records)
naive_prunes = sum(r['naive_decision'] == 'prune' for r in records)
print(f'episodes run               : {n_ep}')
print(f'patience prior P0          : {P0_final:.2f}  (converged: {converged})')
print(f'drift judge keeps / prunes : {keeps} / {prunes}')
print(f'naive h-flip keeps / prunes: {n_ep - naive_prunes} / {naive_prunes}')

episodes = np.arange(1, n_ep + 1)

# ---------------------------------------------------------------------------
# Plot 1 — the persistent patience prior P0 over episodes (the value we run until
# it converges), with the per-feature horizon the doubling test reaches.
# ---------------------------------------------------------------------------
P_reached = np.array([r['P_final'] for r in records], dtype=float)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(episodes, P_reached, color='#3953e2', lw=1.3, alpha=0.5, marker='o', ms=3,
        label='per-feature horizon $P$ reached (doubling test)')
ax.plot(episodes, P0_series, color=W_COLOR, lw=2.6,
        label='patience prior $P_0$ (persists & adapts)')
ax.axhline(P0_final, color='0.4', ls='--', lw=1.4, alpha=0.8,
           label=f'converged $P_0 \\approx {P0_final:.1f}$')
ax.set_yscale('log', base=2)
ax.set_xlabel('episode (same blocking problem, weights reset each time)', fontsize=15)
ax.set_ylabel('patience (log$_2$)', fontsize=16)
ax.set_title('Patience calibrating over repeated blocking episodes', fontsize=18)
ax.legend(loc='lower right', fontsize=11)
fig.tight_layout()
fig.savefig('exp6_patience_over_time.png', dpi=130, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Plot 2 — cumulative keeps/prunes over episodes: drift judge vs naive h-flip.
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
# Plots 3 & 4 — within-episode drift-judge dynamics for the first and last episode.
# ---------------------------------------------------------------------------
def episode_diagnostic(label, start_P0, seed, fname):
    feats, y, w_est0 = make_blocking_problem(MAX_EPISODE_STEPS, seed)
    res = run_episode(start_P0, feats, y, w_est0, record=True)
    h = res['hist']
    n_show = len(h['w1'])
    steps = np.arange(1, n_show + 1)
    w_opt = optimal_weight(W_TEACHER, 0.0)
    z = Z_STAR

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axA, axB, axC = axes

    def mark_events(ax):
        for (t_ev, kind_ev) in h['events']:
            ax.axvline(t_ev + 1, color=EVENT_COLOR[kind_ev],
                       ls='-' if kind_ev == 'converged' else (0, (3, 3)),
                       lw=1.6 if kind_ev == 'converged' else 1.0,
                       alpha=0.85 if kind_ev == 'converged' else 0.6, zorder=3)
        if res['naive_flip_step'] >= 0:
            ax.axvline(res['naive_flip_step'] + 1, color=H_IDBD3_COLOR,
                       ls=(0, (4, 3)), lw=1.4, alpha=0.8, zorder=3)

    # --- Panel A: weights ---
    axA.plot(steps, h['w1'], color=W_COLOR, lw=2.6, label='candidate weight', zorder=10)
    axA.plot(steps, h['w0'], color=W_COLOR, lw=2.0, alpha=0.3, label='established weight',
             zorder=10)
    axA.axhline(w_opt, color=W_COLOR, ls='--', lw=1.4, alpha=0.7, label='optimal weight')
    axA.axhline(UTILITY_BAR, color='0.45', ls=':', lw=1.5, alpha=0.9, label='utility bar')
    mark_events(axA)
    axA.set_ylabel('weight', fontsize=15)
    axA.set_ylim(-w_opt * 0.3, w_opt * 1.3)
    axA.legend(loc='center left', fontsize=10, ncol=2)

    # --- Panel B: the drift test (measure of when to prune + decision bounds) ---
    # mask the immature region where n_eff is tiny (SE explodes, no test can fire)
    immature = ~(h['n_eff'] >= 8.0)
    m_hat = np.where(immature, np.nan, h['m_hat'])
    se = np.where(immature, np.nan, h['se'])
    axB.fill_between(steps, -h['eps'], h['eps'], color='0.55', alpha=0.30,
                     label='indifference band $\\pm\\varepsilon$ (keep if inside)')
    axB.fill_between(steps, m_hat - z * se, m_hat + z * se,
                     color='#3953e2', alpha=0.20,
                     label='decision interval $\\hat m \\pm z^\\star\\,$SE')
    axB.plot(steps, m_hat, color='#3953e2', lw=1.8, label='drift estimate $\\hat m$')
    axB.plot(steps, h['u'], color='#9923DC', lw=1.0, alpha=0.55,
             label='fast trace $u$ (flips trigger a test)')
    axB.axhline(0.0, color='0.5', lw=0.6, alpha=0.4)
    mark_events(axB)
    eps_f = EPS_W * (1.0 / 3.0)
    axB.set_ylim(-6 * eps_f, 6 * eps_f)
    axB.set_ylabel('gradient drift', fontsize=15)
    axB.legend(loc='lower left', fontsize=9, ncol=2)

    # --- Panel C: patience and effective-sample count over the episode ---
    axC.plot(steps, h['P'], color='#3953e2', lw=2.0, label='patience $P$ (doubles per grow)')
    axC.axhline(start_P0, color='0.4', ls='--', lw=1.2, alpha=0.8,
                label=f'start $P_0 = {start_P0:.1f}$')
    axC.set_yscale('log', base=2)
    axC.set_ylabel('patience $P$ (log$_2$)', fontsize=14)
    axCr = axC.twinx()
    axCr.plot(steps, h['n_eff'], color='#029999', lw=1.4, alpha=0.8, label='$n_{eff}$')
    axCr.axhline(NEFF_MIN, color='#029999', ls=':', lw=1.2, alpha=0.7)
    axCr.set_yscale('log')
    axCr.set_ylabel('$n_{eff}$', fontsize=14, color='#029999')
    mark_events(axC)
    hL, lL = axC.get_legend_handles_labels()
    hR, lR = axCr.get_legend_handles_labels()
    axC.legend(hL + hR, lL + lR, loc='lower right', fontsize=9)

    axC.set_xscale('log')
    axC.set_xlabel('step (log scale)', fontsize=15)
    verdict = f"{res['decision'].upper()} at step {res['verdict_step'] + 1:,} " \
              f"(|w| = {abs(res['w1']):.2f})"
    n_grow = res['n_veto'] + res['n_incon']
    fig.suptitle(f'{label} episode — drift-judge dynamics (start $P_0={start_P0:.1f}$, '
                 f'{n_grow} extend events → {verdict})', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(fname, dpi=130, bbox_inches='tight')
    return res


first = records[0]
last = records[-1]
episode_diagnostic('First', first['start_P0'], first['seed'], 'exp6_episode_first.png')
episode_diagnostic('Last', last['start_P0'], last['seed'], 'exp6_episode_last.png')

print('saved: exp6_patience_over_time.png, exp6_prunes_keeps_over_time.png, '
      'exp6_episode_first.png, exp6_episode_last.png')
