# =============================================================================
# Fable — reservoir-index pruning by sequential confidence-bound tests.
#
# Paste this into the "## 4. Fable" method cell of feature_sifting_experiments.ipynb
# (replacing the placeholder). It assumes the notebook globals already exist:
#   jax, jnp, eqx, tree_replace, N_LEARNER_FEATURES, MethodSpec, METHODS.
# A suggested run / learning-curve cell is given (commented) at the bottom.
#
# One uniform loop, identical on every step (including t = 0): learn -> measure ->
# bound -> compute the bar z* -> record resolved tests -> prune at most the single
# worst feature below the bar. Per feature we keep a weight, three running averages,
# an age, and one bookkeeping bit; globally a small record R of how good fresh draws
# turned out to be. Cold start needs no special handling: an empty R gives z* = 0, so
# nothing is pruned until the timeout has fed enough resolved tests into R for the bar
# to lift off. Young features are never pruned prematurely because a small effective
# sample count gives them wide error bars and therefore a large utility upper bound.
# =============================================================================


def _fable_record_mean(z, r_p, r_w, eps):
    """G(z) = weighted mean over the record R of max(0, P - z). Empty R -> 0."""
    return jnp.sum(r_w * jnp.maximum(r_p - z, 0.0)) / (jnp.sum(r_w) + eps)


def _fable_solve_z_star(r_p, r_w, horizon, tau, n_bisect, eps):
    """The bar z*: it solves ``H * G(z) = TAU * z`` — the value at which the expected
    lifetime winnings (over horizon H) of one fresh redraw past z exactly pay for the
    TAU steps a test burns. phi(z) = H*G(z) - TAU*z is non-increasing with phi(0) >= 0
    and phi(max_P) <= 0, so a fixed-iteration bisection brackets the unique root. An
    empty / zero-mass record (G(0) == 0) means no bar at all, so z* = 0."""
    g0 = _fable_record_mean(0.0, r_p, r_w, eps)
    hi0 = jnp.max(r_p)  # recorded utilities are >= 0, so this upper-bounds the root

    def phi(z):
        return horizon * _fable_record_mean(z, r_p, r_w, eps) - tau * z

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go_right = phi(mid) > 0.0          # phi decreasing -> root lies to the right
        return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (jnp.array(0.0), hi0))
    return jnp.where(g0 <= 0.0, 0.0, 0.5 * (lo + hi))


def _fable_push_records(r_p, r_w, r_ptr, resolved, value, r_lam, r_size):
    """Push each resolved slot's measured quality into the ring buffer R, in slot order.
    Mirrors ``record(P)``: age every existing weight by (1 - r_lam), then write the new
    entry with weight 1.0 at the rolling pointer (overwriting the oldest once full)."""
    def body(j, carry):
        r_p, r_w, r_ptr = carry
        do = resolved[j]
        r_w = jnp.where(do, r_w * (1.0 - r_lam), r_w)          # age all existing entries
        r_p = r_p.at[r_ptr].set(jnp.where(do, value[j], r_p[r_ptr]))
        r_w = r_w.at[r_ptr].set(jnp.where(do, 1.0, r_w[r_ptr]))
        r_ptr = jnp.where(do, (r_ptr + 1) % r_size, r_ptr)
        return (r_p, r_w, r_ptr)

    return jax.lax.fori_loop(0, resolved.shape[0], body, (r_p, r_w, r_ptr))


class Fable(eqx.Module):
    """Fable — reservoir-index pruning. Each candidate slot runs a sequential test of
    "is my potential utility above the bar z*?". A slot is pruned (its task feature
    regenerated) only once its utility *upper* bound falls below z*; the resolved test
    outcomes feed a record R that sets z*. See the header comment for the full loop."""

    # Static hyperparameters (changing any of these recompiles the jitted training loop).
    alpha: float = eqx.field(static=True)        # learner step-size (LMS)
    lam: float = eqx.field(static=True)          # EMA decay; memory ~ 1/lam steps
    tau: float = eqx.field(static=True)          # test cost in steps (pinned to 1/lam)
    horizon: float = eqx.field(static=True)      # H: how many future steps we value
    beta: float = eqx.field(static=True)         # confidence-bound width, in std errors
    timeout_a: float = eqx.field(static=True)    # A: every test resolves within A/lam steps
    r_lam: float = eqx.field(static=True)        # record aging rate
    r_size: int = eqx.field(static=True)         # record capacity
    n_bisect: int = eqx.field(static=True)       # bisection iterations for z*
    eps: float = eqx.field(static=True)

    # Per-feature state (slot j holds one candidate; respawn is also how all slots start).
    v: jax.Array        # weights
    g: jax.Array        # EMA of e*f (doubles as the LMS gradient trace)
    q: jax.Array        # EMA of (e*f)^2
    m: jax.Array        # EMA of f*f (starts at 0.25)
    age: jax.Array
    counted: jax.Array  # bool: has this draw's quality been recorded into R yet?
    e2_bar: jax.Array   # global EMA of squared error e^2 (scale for the small-n range term)

    # The record R of resolved-test qualities, as a weighted ring buffer.
    r_p: jax.Array      # recorded utilities
    r_w: jax.Array      # their (aged) weights
    r_ptr: jax.Array    # next write index

    prune_mask: jax.Array  # slots the task should regenerate next step (<=1 True)

    @classmethod
    def init(cls, hparams, key):
        n = N_LEARNER_FEATURES
        lam = hparams['lam']
        tau = hparams.get('tau')
        tau = (1.0 / lam) if tau is None else tau
        r_size = int(hparams.get('r_size', 128))
        return cls(
            alpha=hparams['learning_rate'],
            lam=lam,
            tau=tau,
            horizon=hparams['horizon'],
            beta=hparams.get('beta', 2.0),
            timeout_a=hparams.get('timeout_a', 4.0),
            r_lam=hparams.get('r_lam', 0.01),
            r_size=r_size,
            n_bisect=int(hparams.get('n_bisect', 40)),
            eps=hparams.get('eps', 1e-8),
            v=jnp.zeros(n), g=jnp.zeros(n), q=jnp.zeros(n),
            m=jnp.full(n, 0.25), age=jnp.zeros(n),
            counted=jnp.zeros(n, dtype=bool), e2_bar=jnp.array(1.0),
            r_p=jnp.zeros(r_size), r_w=jnp.zeros(r_size), r_ptr=jnp.array(0, dtype=jnp.int32),
            prune_mask=jnp.zeros(n, dtype=bool),
        )

    def step(self, x, y):
        f = x                                   # each slot's frozen feature definition
        e = y - jnp.sum(self.v * f)             # prediction error (full residual)
        lam = self.lam

        # 1. learn and measure ------------------------------------------------------
        v = self.v + self.alpha * e * f         # LMS update; g below is its gradient trace
        ef = e * f
        g = (1.0 - lam) * self.g + lam * ef
        m = (1.0 - lam) * self.m + lam * (f * f)
        q = (1.0 - lam) * self.q + lam * ef ** 2
        e2_bar = (1.0 - lam) * self.e2_bar + lam * e ** 2   # global error scale
        age = self.age + 1.0

        # 2. potential utility with confidence bounds -------------------------------
        # Empirical-Bernstein width (Mnih et al. 2008): a Bessel-corrected variance term
        # plus a range term that keeps a young feature wide (a few sample-magnitudes at
        # n=1) and decays like 1/n, so nothing resolves on a single sample.
        n_eff = jnp.minimum(age, 2.0 / lam)
        V = jnp.maximum(q - g * g, 0.0) * n_eff / jnp.maximum(n_eff - 1.0, 1.0)
        se = self.beta * (jnp.sqrt(V / n_eff) + 3.0 * jnp.sqrt(e2_bar * m) / n_eff)
        a = g + v * m                           # effective target for this slot
        lo_b, hi_b = a - se, a + se
        p = a * a / (m + self.eps)              # point estimate of utility
        ucb = jnp.maximum(lo_b ** 2, hi_b ** 2) / (m + self.eps)
        lcb = jnp.where(lo_b * hi_b > 0.0,      # 0 when the interval straddles 0
                        jnp.minimum(lo_b ** 2, hi_b ** 2), 0.0) / (m + self.eps)

        # 3. the bar z* (from the record R as it stands before this step's records) -
        z_star = _fable_solve_z_star(self.r_p, self.r_w, self.horizon, self.tau,
                                     self.n_bisect, self.eps)

        # 4. record tests that just resolved ----------------------------------------
        not_counted = ~self.counted
        mature = age >= 2.0                                      # never resolve on a single sample
        pass_test = not_counted & mature & (lcb > z_star)        # confirmed above bar -> p
        fail_test = not_counted & mature & (ucb < z_star)        # censored below bar -> ucb
        timeout = (not_counted & ~pass_test & ~fail_test         # ran out of patience -> p
                   & (age > self.timeout_a / lam))
        resolved = pass_test | fail_test | timeout
        record_value = jnp.where(pass_test, p, jnp.where(fail_test, ucb, p))
        counted = self.counted | resolved
        r_p, r_w, r_ptr = _fable_push_records(
            self.r_p, self.r_w, self.r_ptr, resolved, record_value, self.r_lam, self.r_size)

        # 5. prune at most one feature per step (utilities are coupled) --------------
        j = jnp.argmin(ucb)
        do_prune = ucb[j] < z_star
        prune_mask = jnp.zeros_like(self.prune_mask).at[j].set(do_prune)

        # respawn the pruned slot: the task regenerates its feature next step, and we
        # reset its state. m carries over the mean scale (refines within ~1/lam steps).
        v = jnp.where(prune_mask, 0.0, v)
        g = jnp.where(prune_mask, 0.0, g)
        q = jnp.where(prune_mask, 0.0, q)
        age = jnp.where(prune_mask, 0.0, age)
        m = jnp.where(prune_mask, jnp.mean(m), m)
        counted = jnp.where(prune_mask, False, counted)

        return tree_replace(self, v=v, g=g, q=q, m=m, age=age, counted=counted, e2_bar=e2_bar,
                            r_p=r_p, r_w=r_w, r_ptr=r_ptr, prune_mask=prune_mask), e ** 2


FABLE_DEFAULTS = {
    'learning_rate': 0.05,   # ALPHA: LMS step-size
    'lam': 0.005,            # EMA decay (memory ~ 1/lam = 200 steps); tau defaults to 1/lam
    'horizon': 1000.0,       # H: the real knob. On THIS task z* runs away (over-prunes everything)
                             # above H~1e4 — keep H within the healthy regime; sweep it (grid below).
    'beta': 2.0,             # confidence-bound width, in standard errors
    'timeout_a': 4.0,        # every test resolves within A/lam = 800 steps
    'r_lam': 0.01,           # record aging rate
    'r_size': 128,           # record capacity
    'n_bisect': 40,          # bisection iterations for z*
}
METHODS['fable'] = MethodSpec('fable', 'Fable', Fable, FABLE_DEFAULTS)


# --- Suggested run / sensitivity cell (paste into a new cell after the one above) ---
#
# FABLE_GRID = {
#     'learning_rate': [2.0 ** e for e in range(-8, -1, 2)],   # 2^-8 .. 2^-2
#     'horizon': [100, 300, 1000, 3000, 10000],               # bar aggressiveness; above ~1e4 on
#                                                              # this task z* runs away & over-prunes
# }
# run_method_sweep(METHODS['fable'], FABLE_GRID, N_SWEEP_SEEDS)
# cfgs, runs = load_sweep_data()
# fable_summary = summarize_runs(cfgs, runs, 'fable', swept_vars(FABLE_GRID), FINAL_FRACTION, BASELINE_LOSS)
# plot_sensitivity(fable_summary, swept_vars(FABLE_GRID), 'Fable')
# plt.show()
#
# --- Best-configuration learning curve (paste into a new cell after that) ---
#
# fable_best = best_hparams(cfgs, runs, METHODS['fable'], FABLE_GRID)
# print('Fable best:', fable_best)
# run_method_sweep(METHODS['fable'], fable_best, N_CURVE_SEEDS, sweep_name='fable_best')
# cfgs, runs = load_sweep_data()
# plt.figure(figsize=(7, 4))
# plot_learning_curve(runs, 'fable_best', label='Fable', color=get_color_palette(n=4)[3])
# plt.title('Fable — best configuration (mean ± std over seeds)')
# plt.legend(); plt.tight_layout(); plt.show()
