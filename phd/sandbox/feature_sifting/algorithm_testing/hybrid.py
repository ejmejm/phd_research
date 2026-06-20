"""HybridV2 — Hybrid's sequential-test pruning, but with neither a ring buffer nor a
bisection. Both are too expensive for multi-layer nets with many units.

Identical per-feature logic to `hybrid.py` (Autostep/IDBD base learner, unbiased
centre a = ema(pr*phi), conservative width V = ema((e*phi)^2), Kish debiasing, L1
utility p = |w*|*ema|phi| with confidence bounds, prune-at-most-one-below-the-bar).
Only the bar machinery changes, via a two-timescale scalar approximation:

  * No record R / ring buffer. Resolved draws fold their exceedance over the bar,
    max(P_rec - z*, 0), into a single EMA G_hat (rate eta_G). A burst of k
    resolutions (e.g. the initial timeout) advances the EMA by k ticks toward the
    batch mean, so no per-slot loop is needed.
  * No bisection. The bar relaxes toward its fixed point z* = (H/tau)*G(z*) by a
    slow damped update z* <- (1-eta)*z* + eta*(H/tau)*G_hat. Keeping eta << eta_G
    (G_hat tracks the exceedance at the current bar faster than the bar moves) gives
    a stable two-timescale system that converges to the same z* the bisection found.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from phd.jax_core.optimizers.idbd import optax_idbd, IDBDState
from phd.jax_core.utils import tree_replace


DEFAULT_HYPERPARAMETERS = {
    'learning_rate': 0.05,           # Autostep initial step-size
    'meta_learning_rate': 0.0,       # Autostep meta step-size (0 => fixed-step LMS)
    'lam': 0.005,            # EMA decay (memory ~ 1/lam = 200 steps); tau defaults to 1/lam
    'horizon': 100_000.0,    # H: the real knob — how many future steps a redraw is valued over
    'beta': 1.96,            # confidence-bound width, in standard errors
    'kappa': 0.0,            # range-term weight; keeps young features wide
    'timeout_a': 4.0,        # every test resolves within A/lam = 800 steps
    'eta_G': 0.01,           # exceedance-EMA rate (~ 1 / old record size)
    'eta': 0.001,            # bar relaxation rate; SLOW (eta << eta_G) for two-timescale stability
}


def linear_loss_and_grads(w, x, y):
    """Squared-error loss of the linear prediction and its exact gradient wrt `w`."""
    y_hat = jnp.sum(w * x[None, :])
    error = y_hat - y
    return error ** 2, 2.0 * error * x[None, :]


def reset_opt_state(opt_state, prune_mask):
    """Reset the Autostep (IDBD) step-size and traces for pruned (regenerated) features."""
    pm = prune_mask[None, :]  # broadcast over the single output row -> (1, n_features)
    if isinstance(opt_state, IDBDState):
        return IDBDState(
            init_beta=opt_state.init_beta,
            beta=jnp.where(pm, opt_state.init_beta, opt_state.beta),  # step-size -> initial
            h=jnp.where(pm, 0.0, opt_state.h),
            v=jnp.where(pm, 0.0, opt_state.v),
        )
    return opt_state


class Hybrid(eqx.Module):
    """HybridV2 — Hybrid's reservoir-index pruning with a scalar bar (no ring buffer,
    no bisection). Each slot runs a sequential test of "is my L1 utility above the bar
    z*?"; it is pruned only once that utility's upper bound falls below z*. The bar is
    a slowly relaxed scalar fed by an EMA of resolved-test exceedances. See the module
    docstring for the full loop."""
    DEFAULTS: ClassVar[dict] = DEFAULT_HYPERPARAMETERS

    # Static hyperparameters (changing any of these recompiles the jitted training loop).
    optimizer: optax.GradientTransformation = eqx.field(static=True)  # Autostep/IDBD base learner
    lam: float = eqx.field(static=True)          # EMA decay; memory ~ 1/lam steps
    tau: float = eqx.field(static=True)          # test cost in steps (pinned to 1/lam)
    horizon: float = eqx.field(static=True)      # H: how many future steps we value
    beta: float = eqx.field(static=True)         # confidence-bound width, in std errors
    kappa: float = eqx.field(static=True)        # range-term weight; keeps young features wide
    timeout_a: float = eqx.field(static=True)    # A: every test resolves within A/lam steps
    eta_G: float = eqx.field(static=True)        # exceedance-EMA rate
    eta: float = eqx.field(static=True)          # bar relaxation rate (slow)
    eps: float = eqx.field(static=True)

    # Per-feature state (slot j holds one candidate; respawn is also how all slots start).
    w: jax.Array        # weights (1, n), Autostep-compatible
    opt_state: optax.OptState
    a: jax.Array        # ema(pr*phi)     unbiased correlation -> w* = a/m
    m: jax.Array        # ema(phi^2)
    q: jax.Array        # ema((e*phi)^2)  conservative variance scale (uses the prediction residual)
    xbar: jax.Array     # ema(|phi|)      maps |w*| to L1 utility
    age: jax.Array
    counted: jax.Array  # bool: has this draw's quality been folded into G_hat yet?

    # Reservoir-level scalars (replace the entire record R + bisection).
    z_star: jax.Array   # the bar; relaxes toward its fixed point
    G_hat: jax.Array    # EMA of exceedance max(P_rec - z*, 0) over resolved draws

    prune_mask: jax.Array  # slots the task should regenerate next step (<=1 True)

    @classmethod
    def init(cls, input_dim, hparams, key):
        n = input_dim
        lam = hparams['lam']
        tau = hparams.get('tau')
        tau = (1.0 / lam) if tau is None else tau
        optimizer = optax_idbd(init_lr=hparams['learning_rate'],
                               meta_lr=hparams['meta_learning_rate'], autostep=True)
        w = jnp.zeros((1, n))
        return cls(
            optimizer=optimizer,
            lam=lam,
            tau=tau,
            horizon=hparams['horizon'],
            beta=hparams.get('beta', 1.96),
            kappa=hparams.get('kappa', 0.0),
            timeout_a=hparams.get('timeout_a', 4.0),
            eta_G=hparams.get('eta_G', 0.01),
            eta=hparams.get('eta', 0.001),
            eps=hparams.get('eps', 1e-8),
            w=w, opt_state=optimizer.init(w),
            a=jnp.zeros(n), m=jnp.zeros(n), q=jnp.zeros(n),
            xbar=jnp.zeros(n), age=jnp.zeros(n),
            counted=jnp.zeros(n, dtype=bool),
            z_star=jnp.array(0.0), G_hat=jnp.array(0.0),
            prune_mask=jnp.zeros(n, dtype=bool),
        )

    def step(self, x, y):
        f = x                                   # each slot's frozen feature definition
        w_old = self.w[0]                       # (n,) pre-update weight
        e = y - jnp.sum(w_old * f)              # prediction error (full residual)
        lam = self.lam

        # 1. update EMAs (with the pre-update weight) and learn via Autostep ----------
        pr = e + w_old * f                      # partial residual incl. self (= y here)
        ex = e * f                              # residual x feature -> variance scale
        a = (1.0 - lam) * self.a + lam * (pr * f)
        m = (1.0 - lam) * self.m + lam * (f * f)
        q = (1.0 - lam) * self.q + lam * (ex * ex)
        xbar = (1.0 - lam) * self.xbar + lam * jnp.abs(f)
        age = self.age + 1.0

        loss, loss_grads = linear_loss_and_grads(self.w, x, y)
        # Autostep/IDBD consumes both the loss gradient and the prediction gradient (= x).
        updates, opt_state = self.optimizer.update((loss_grads, x[None, :]), self.opt_state, self.w)
        w = optax.apply_updates(self.w, updates)

        # 2. potential utility (L1) with confidence bounds --------------------------
        c = 1.0 - (1.0 - lam) ** age            # Kish debias factor
        a_hat, m_hat, q_hat, xbar_hat = a / c, m / c, q / c, xbar / c
        n_eff = (c * (2.0 - lam)) / (lam * (2.0 - c) + self.eps)   # Kish effective sample size
        V = q_hat                               # conservative: no mean^2 subtraction; e >= true residual early
        se = self.beta * (jnp.sqrt(V / n_eff) + self.kappa * jnp.sqrt(V) / n_eff)
        lo_a, hi_a = a_hat - se, a_hat + se
        inv_m = 1.0 / (m_hat + self.eps)
        w_hi = jnp.maximum(jnp.abs(lo_a), jnp.abs(hi_a)) * inv_m
        w_lo = jnp.where(lo_a * hi_a > 0.0,     # 0 when the interval straddles 0
                         jnp.minimum(jnp.abs(lo_a), jnp.abs(hi_a)), 0.0) * inv_m
        p = jnp.abs(a_hat) * inv_m * xbar_hat   # point estimate of L1 utility
        lcb = w_lo * xbar_hat                   # lower utility bound
        ucb = w_hi * xbar_hat                   # upper utility bound

        # 3. resolve not-yet-counted tests against the current bar, folding each
        #    resolved draw's exceedance over z* into the scalar EMA G_hat (no buffer) -
        z_star = self.z_star
        not_counted = ~self.counted
        pass_test = not_counted & (lcb > z_star)                 # confirmed above bar -> p
        fail_test = not_counted & (ucb < z_star)                 # censored below bar -> ucb
        timeout = (not_counted & ~pass_test & ~fail_test         # ran out of patience -> p
                   & (age > self.timeout_a / lam))
        resolved = pass_test | fail_test | timeout
        record_value = jnp.where(pass_test, p, jnp.where(fail_test, ucb, p))
        counted = self.counted | resolved

        # Fold the (possibly many) resolved exceedances into G_hat in one shot: advance
        # the EMA by n_res ticks toward the batch-mean exceedance. Below-bar draws give
        # max(.,0)=0, so aborted tests contribute nothing.
        exc = jnp.where(resolved, jnp.maximum(record_value - z_star, 0.0), 0.0)
        n_res = jnp.sum(resolved)
        mean_exc = jnp.sum(exc) / jnp.maximum(n_res, 1.0)
        decay = (1.0 - self.eta_G) ** n_res
        G_hat = decay * self.G_hat + (1.0 - decay) * mean_exc

        # 4. relax the bar toward its fixed point z* = (H/tau) * G(z*)  (no bisection) -
        z_star = (1.0 - self.eta) * z_star + self.eta * (self.horizon / self.tau) * G_hat

        # 5. prune at most one slot below the (updated) bar --------------------------
        j = jnp.argmin(ucb)
        do_prune = ucb[j] < z_star
        prune_mask = jnp.zeros_like(self.prune_mask).at[j].set(do_prune)

        # respawn the pruned slot: the task regenerates its feature next step, and we
        # reset all of its state (including the Autostep step-size) so the Kish
        # debiasing stays self-consistent. The reservoir scalars z*/G_hat persist.
        keep = ~prune_mask
        w = jnp.where(keep[None, :], w, 0.0)
        opt_state = reset_opt_state(opt_state, prune_mask)
        a = jnp.where(keep, a, 0.0)
        m = jnp.where(keep, m, 0.0)
        q = jnp.where(keep, q, 0.0)
        xbar = jnp.where(keep, xbar, 0.0)
        age = jnp.where(keep, age, 0.0)
        counted = jnp.where(keep, counted, False)

        return tree_replace(self, w=w, opt_state=opt_state, a=a, m=m, q=q, xbar=xbar,
                            age=age, counted=counted, z_star=z_star, G_hat=G_hat,
                            prune_mask=prune_mask), e ** 2
