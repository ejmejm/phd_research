```python
"""
Minimal reservoir-index pruning. One uniform loop: no warmup, no phases,
no feature classes. Per feature: a weight, three running averages, an age,
and one bookkeeping bit. Globally: a small record R of how good fresh
draws turned out to be.

Every step, identically (including t = 0):
  learn -> measure -> bound -> compute the bar -> record resolved tests
  -> prune at most the single worst feature below the bar.

Cold start needs no special handling. An empty R gives z* = 0, so nothing
is pruned at first; the timeout guarantees every initial feature's quality
is eventually recorded into R; once R has mass, z* lifts off and pruning
begins. Young features are never pruned prematurely because small n_eff
gives them wide error bars and therefore a large UCB.
"""

# constants -------------------------------------------------------------------
H     = 100_000          # horizon: how many future steps we value. The real knob.
LAM   = 0.005            # EMA decay; memory ~ 1/LAM steps
TAU   = 1 / LAM          # typical test duration (pinned by LAM), used as test cost
BETA  = 2.0              # confidence-bound width, in standard errors
A     = 4.0              # timeout: every test resolves within A/LAM steps
R_SIZE, R_LAM = 128, 0.01   # record capacity and aging
ALPHA = 0.05             # learner step-size (any online learner slots in)
EPS   = 1e-8

# state (slot j holds one feature; respawn() is also how all slots start) -------
v, g, q = zeros(N), zeros(N), zeros(N)
m       = full(N, 0.25)
age     = zeros(N)
counted = full(N, False)      # has this draw's quality been recorded in R?
R       = ring_buffer(R_SIZE) # entries (P, weight)


def step(t, x, y):
    f = features(x)                       # each slot's frozen feature definition
    e = y - v @ f

    # 1. learn and measure ------------------------------------------------------
    v += ALPHA * e * f                    # g doubles as this learner's gradient trace
    g[:] = (1 - LAM) * g + LAM * (e * f)
    m[:] = (1 - LAM) * m + LAM * (f * f)
    q[:] = (1 - LAM) * q + LAM * (e * f) ** 2
    age += 1

    # 2. potential utility with confidence bounds --------------------------------
    n_eff = minimum(age, 2 / LAM)
    se    = BETA * sqrt(maximum(q - g * g, 0) / n_eff)
    a     = g + v * m
    p     = a * a / (m + EPS)
    ucb   = maximum((a - se)**2, (a + se)**2) / (m + EPS)
    lcb   = where((a - se) * (a + se) > 0,
                  minimum((a - se)**2, (a + se)**2), 0) / (m + EPS)

    # 3. the bar ------------------------------------------------------------------
    #    z* solves H * G(z) = TAU * z : expected lifetime winnings of one redraw
    #    past z equal the test time a redraw burns. Empty R -> G = 0 -> z* = 0.
    G = lambda z: weighted_mean(max(0, P - z) for (P, w) in R)
    z_star = 0.0 if G(0) == 0 else bisect(lambda z: H * G(z) - TAU * z,
                                          lo=0.0, hi=max_P(R))

    # 4. record tests that just resolved -------------------------------------------
    for j in slots_where(~counted):
        if   lcb[j] > z_star:     record(p[j]);   counted[j] = True
        elif ucb[j] < z_star:     record(ucb[j]); counted[j] = True  # censored:
                                                                     # only "below
                                                                     # the bar" is known
        elif age[j] > A / LAM:    record(p[j]);   counted[j] = True  # timeout:
                                                                     # p ~ z*, either
                                                                     # decision is fine

    # 5. prune at most one feature per step (utilities are coupled) -----------------
    j = argmin(ucb)
    if ucb[j] < z_star:
        respawn(j)


def record(P):
    R.weights *= (1 - R_LAM)              # age old entries
    R.push(P, weight=1.0)


def respawn(j):
    # generator supplies a fresh feature definition for slot j
    v[j] = g[j] = q[j] = 0; age[j] = 0
    m[j] = mean(m)                        # scale carry-over; refines within 1/LAM
    counted[j] = False


# what was cut, and when it earns its place back ---------------------------------
# * refractory between prunes (a gap of ~2/LAM steps): worthwhile when features
#   heavily share inputs; gives a survivor's p time to rise before its redundant
#   partner is judged. The one-prune-per-step rule above is the minimal version.
# * exploration floor (respawn the worst slot if nothing is under test): insurance
#   against a stale R in nonstationary settings where fresh draws can improve
#   while incumbents still look fine.
# * soft prune (ramp v to 0 before respawn): only matters when drift leaves
#   stale large-weight features; here pruned features have small contributions
#   by construction, since ucb < z* bounds them.
# * adaptive test cost (EMA of measured durations) instead of TAU = 1/LAM:
#   mostly cosmetic, since the bound machinery pins durations to O(1/LAM).
```