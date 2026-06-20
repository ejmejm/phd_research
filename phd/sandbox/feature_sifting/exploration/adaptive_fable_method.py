# =============================================================================
# Adaptive Fable — reservoir-index pruning with adaptive measurement resolution.
#
# Paste this into its own method cell of feature_sifting_experiments.ipynb. It
# assumes the notebook globals: jax, jnp, eqx, tree_replace, N_LEARNER_FEATURES,
# MethodSpec, METHODS. It is fully self-contained (helpers prefixed `_afable_`),
# so it coexists with fable_method.py if both are pasted in.
#
# Identical to fable_method.py (the minimal version) except for the problem it
# fixes: a fixed EMA decay LAM floors the standard error of every estimate at
# ~ sqrt(Var(e*f) * LAM / 2). Once remaining utility gaps fall below that floor,
# tests can no longer resolve, churn becomes random among indistinguishable
# features, and excess loss plateaus at a LAM-determined (arbitrary) level. The
# fix sets the plateau by H instead — the economically principled resolution
# limit (~ sqrt(Var / H)). The four changes (each marked # CHANGED):
#
#   1. Unresolved candidates use growing windows: lam_j = 1/(age_j + 1), i.e.
#      plain sample means; error bars shrink like 1/sqrt(age) with no floor.
#      Resolved features revert to fixed LAM for tracking.
#   2. The timeout T_TO is adaptive: stretched when a test ends unresolved,
#      shrunk when it resolves early. Floored at the tracking memory 2/LAM,
#      capped at H. Self-tunes to where ~half of tests resolve in-window.
#   3. TAU (the test cost in the z* equation) is an EMA of measured durations
#      rather than the pinned 1/LAM, since growing windows unpin durations.
# =============================================================================


def _afable_record_mean(z, r_p, r_w, eps):
    """G(z) = weighted mean over the record R of max(0, P - z). Empty R -> 0."""
    return jnp.sum(r_w * jnp.maximum(r_p - z, 0.0)) / (jnp.sum(r_w) + eps)


def _afable_solve_z_star(r_p, r_w, horizon, tau, n_bisect, eps):
    """The bar z*, solving ``H * G(z) = TAU * z`` by fixed-iteration bisection on
    [0, max_P] (phi is non-increasing, phi(0) >= 0, phi(max_P) <= 0). Empty record
    (G(0) == 0) means no bar, z* = 0. TAU is now dynamic state, passed in."""
    g0 = _afable_record_mean(0.0, r_p, r_w, eps)
    hi0 = jnp.max(r_p)

    def phi(z):
        return horizon * _afable_record_mean(z, r_p, r_w, eps) - tau * z

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go_right = phi(mid) > 0.0
        return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (jnp.array(0.0), hi0))
    return jnp.where(g0 <= 0.0, 0.0, 0.5 * (lo + hi))


def _afable_resolve_and_record(r_p, r_w, r_ptr, tau, t_to, counted,
                               lcb, ucb, p, age, z_star,
                               r_lam, r_size, eta, lam, horizon):
    """Step 4 (record resolved tests) + the `finish` bookkeeping, as one sequential
    pass over the slots (slot order, matching the pseudocode's loop). Each resolved
    slot: pushes its quality into the ring buffer R (aging existing weights), folds
    its measured duration `age` into the TAU EMA, and nudges the adaptive timeout
    T_TO — stretched on a timeout, shrunk on an early resolution, clipped to
    [2/LAM, H]. The timeout threshold is read from the running T_TO, so later slots
    in the same step see earlier finishes' updates (a faithful within-step coupling)."""
    def body(j, carry):
        r_p, r_w, r_ptr, tau, t_to, counted = carry
        nc = ~counted[j]
        mature = age[j] >= 2.0                                   # never resolve on a single sample
        pass_j = nc & mature & (lcb[j] > z_star)                 # confirmed above bar -> p
        fail_j = nc & mature & (ucb[j] < z_star)                 # censored below bar -> ucb
        timeout_j = nc & (~pass_j) & (~fail_j) & (age[j] > t_to)  # ran out of patience -> p
        resolved_j = pass_j | fail_j | timeout_j
        value = jnp.where(fail_j, ucb[j], p[j])                  # pass & timeout both record p

        r_w = jnp.where(resolved_j, r_w * (1.0 - r_lam), r_w)    # age all existing entries
        r_p = r_p.at[r_ptr].set(jnp.where(resolved_j, value, r_p[r_ptr]))
        r_w = r_w.at[r_ptr].set(jnp.where(resolved_j, 1.0, r_w[r_ptr]))
        r_ptr = jnp.where(resolved_j, (r_ptr + 1) % r_size, r_ptr)

        tau = jnp.where(resolved_j, (1.0 - r_lam) * tau + r_lam * age[j], tau)
        grow = jnp.where(timeout_j, 1.0 + eta, 1.0 - eta)
        t_to = jnp.where(resolved_j, jnp.clip(t_to * grow, 2.0 / lam, horizon), t_to)
        counted = counted.at[j].set(counted[j] | resolved_j)
        return (r_p, r_w, r_ptr, tau, t_to, counted)

    return jax.lax.fori_loop(0, counted.shape[0], body,
                             (r_p, r_w, r_ptr, tau, t_to, counted))


class AdaptiveFable(eqx.Module):
    """Adaptive Fable — reservoir-index pruning with adaptive measurement resolution.
    Unresolved candidates use growing-window sample means (error bars shrink with age,
    no floor); the timeout and test-cost TAU self-tune. See the header comment."""

    # Static hyperparameters (changing any of these recompiles the jitted training loop).
    alpha: float = eqx.field(static=True)        # learner step-size (LMS)
    lam: float = eqx.field(static=True)          # tracking decay for RESOLVED features
    horizon: float = eqx.field(static=True)      # H: future steps valued; also the resolution limit
    beta: float = eqx.field(static=True)         # confidence-bound width, in std errors
    eta: float = eqx.field(static=True)          # timeout-controller gain
    r_lam: float = eqx.field(static=True)        # record aging rate
    r_size: int = eqx.field(static=True)         # record capacity
    n_bisect: int = eqx.field(static=True)       # bisection iterations for z*
    eps: float = eqx.field(static=True)

    # Per-feature state.
    v: jax.Array        # weights
    g: jax.Array        # (growing- or fixed-window) mean of e*f
    q: jax.Array        # mean of (e*f)^2
    m: jax.Array        # mean of f*f (starts at 0.25)
    age: jax.Array
    counted: jax.Array  # bool: resolved (tracking with fixed LAM) vs under test (growing window)
    e2_bar: jax.Array   # global EMA of squared error e^2 (scale for the small-n range term)

    # The record R of resolved-test qualities, plus the self-tuning scalars.
    r_p: jax.Array      # recorded utilities
    r_w: jax.Array      # their (aged) weights
    r_ptr: jax.Array    # next write index
    tau: jax.Array      # EMA of measured test durations (was the constant 1/LAM)  # CHANGED
    t_to: jax.Array     # adaptive timeout (was the constant A/LAM)                 # CHANGED

    prune_mask: jax.Array  # slots the task should regenerate next step (<=1 True)

    @classmethod
    def init(cls, hparams, key):
        n = N_LEARNER_FEATURES
        lam = hparams['lam']
        r_size = int(hparams.get('r_size', 128))
        timeout_a = hparams.get('timeout_a', 4.0)   # initial timeout = timeout_a / lam
        return cls(
            alpha=hparams['learning_rate'],
            lam=lam,
            horizon=hparams['horizon'],
            beta=hparams.get('beta', 2.0),
            eta=hparams.get('eta', 0.1),
            r_lam=hparams.get('r_lam', 0.01),
            r_size=r_size,
            n_bisect=int(hparams.get('n_bisect', 40)),
            eps=hparams.get('eps', 1e-8),
            v=jnp.zeros(n), g=jnp.zeros(n), q=jnp.zeros(n),
            m=jnp.full(n, 0.25), age=jnp.zeros(n),
            counted=jnp.zeros(n, dtype=bool), e2_bar=jnp.array(1.0),
            r_p=jnp.zeros(r_size), r_w=jnp.zeros(r_size), r_ptr=jnp.array(0, dtype=jnp.int32),
            tau=jnp.array(1.0 / lam),               # CHANGED: now state, EMA of durations
            t_to=jnp.array(timeout_a / lam),        # CHANGED: now state, adaptive
            prune_mask=jnp.zeros(n, dtype=bool),
        )

    def step(self, x, y):
        f = x
        e = y - jnp.sum(self.v * f)
        LAM = self.lam

        # 1. learn and measure ------------------------------------------------------
        v = self.v + self.alpha * e * f
        lam = jnp.where(self.counted, LAM, 1.0 / (self.age + 1.0))   # CHANGED: sample means under test
        ef = e * f
        g = (1.0 - lam) * self.g + lam * ef
        m = (1.0 - lam) * self.m + lam * (f * f)
        q = (1.0 - lam) * self.q + lam * ef ** 2
        e2_bar = (1.0 - LAM) * self.e2_bar + LAM * e ** 2          # global error scale (fixed LAM)
        age = self.age + 1.0

        # 2. potential utility with confidence bounds -------------------------------
        # Empirical-Bernstein width (Mnih et al. 2008): a Bessel-corrected variance term
        # plus a range term that keeps a young feature wide (a few sample-magnitudes at
        # n=1) and decays like 1/n, so nothing resolves on a single sample.
        n_eff = jnp.where(self.counted, 2.0 / LAM, age)             # CHANGED: no cap under test
        V = jnp.maximum(q - g * g, 0.0) * n_eff / jnp.maximum(n_eff - 1.0, 1.0)
        se = self.beta * (jnp.sqrt(V / n_eff) + 3.0 * jnp.sqrt(e2_bar * m) / n_eff)
        a = g + v * m
        lo_b, hi_b = a - se, a + se
        p = a * a / (m + self.eps)
        ucb = jnp.maximum(lo_b ** 2, hi_b ** 2) / (m + self.eps)
        lcb = jnp.where(lo_b * hi_b > 0.0,
                        jnp.minimum(lo_b ** 2, hi_b ** 2), 0.0) / (m + self.eps)

        # 3. the bar z* (from R and the measured TAU as they stand before this step) -
        z_star = _afable_solve_z_star(self.r_p, self.r_w, self.horizon, self.tau,
                                      self.n_bisect, self.eps)

        # 4. record tests that just resolved + self-tune TAU / T_TO ------------------
        r_p, r_w, r_ptr, tau, t_to, counted = _afable_resolve_and_record(
            self.r_p, self.r_w, self.r_ptr, self.tau, self.t_to, self.counted,
            lcb, ucb, p, age, z_star,
            self.r_lam, self.r_size, self.eta, LAM, self.horizon)

        # 5. prune at most one feature per step (utilities are coupled) --------------
        j = jnp.argmin(ucb)
        do_prune = ucb[j] < z_star
        prune_mask = jnp.zeros_like(self.prune_mask).at[j].set(do_prune)

        # respawn the pruned slot (task regenerates its feature next step).
        v = jnp.where(prune_mask, 0.0, v)
        g = jnp.where(prune_mask, 0.0, g)
        q = jnp.where(prune_mask, 0.0, q)
        age = jnp.where(prune_mask, 0.0, age)
        m = jnp.where(prune_mask, jnp.mean(m), m)
        counted = jnp.where(prune_mask, False, counted)

        return tree_replace(self, v=v, g=g, q=q, m=m, age=age, counted=counted, e2_bar=e2_bar,
                            r_p=r_p, r_w=r_w, r_ptr=r_ptr, tau=tau, t_to=t_to,
                            prune_mask=prune_mask), e ** 2


ADAPTIVE_FABLE_DEFAULTS = {
    'learning_rate': 0.05,   # ALPHA: LMS step-size
    'lam': 0.005,            # tracking decay for resolved features (memory ~ 1/lam)
    'horizon': 300.0,        # H: the real knob; also owns the resolution limit. On THIS task z*
                             # runs away above H~1e3 (measured TAU collapses, inflating z*); sweep low.
    'beta': 2.0,             # confidence-bound width, in standard errors
    'eta': 0.1,              # timeout-controller gain
    'timeout_a': 4.0,        # initial timeout = timeout_a / lam (self-tunes from here)
    'r_lam': 0.01,           # record aging rate
    'r_size': 128,           # record capacity
    'n_bisect': 40,          # bisection iterations for z*
}
METHODS['adaptive_fable'] = MethodSpec('adaptive_fable', 'Adaptive Fable',
                                       AdaptiveFable, ADAPTIVE_FABLE_DEFAULTS)


# --- Suggested run / sensitivity cell (paste into a new cell after the one above) ---
#
# ADAPTIVE_FABLE_GRID = {
#     'learning_rate': [2.0 ** e for e in range(-8, -1, 2)],   # 2^-8 .. 2^-2
#     'horizon': [100, 300, 1000, 3000],                       # owns the resolution limit; above
#                                                              # ~1e3 here z* runs away & over-prunes
# }
# run_method_sweep(METHODS['adaptive_fable'], ADAPTIVE_FABLE_GRID, N_SWEEP_SEEDS)
# cfgs, runs = load_sweep_data()
# af_summary = summarize_runs(cfgs, runs, 'adaptive_fable', swept_vars(ADAPTIVE_FABLE_GRID), FINAL_FRACTION, BASELINE_LOSS)
# plot_sensitivity(af_summary, swept_vars(ADAPTIVE_FABLE_GRID), 'Adaptive Fable')
# plt.show()
#
# --- Best-configuration learning curve (paste into a new cell after that) ---
#
# af_best = best_hparams(cfgs, runs, METHODS['adaptive_fable'], ADAPTIVE_FABLE_GRID)
# print('Adaptive Fable best:', af_best)
# run_method_sweep(METHODS['adaptive_fable'], af_best, N_CURVE_SEEDS, sweep_name='adaptive_fable_best')
# cfgs, runs = load_sweep_data()
# plt.figure(figsize=(7, 4))
# plot_learning_curve(runs, 'adaptive_fable_best', label='Adaptive Fable', color=get_color_palette(n=5)[4])
# plt.title('Adaptive Fable — best configuration (mean ± std over seeds)')
# plt.legend(); plt.tight_layout(); plt.show()
