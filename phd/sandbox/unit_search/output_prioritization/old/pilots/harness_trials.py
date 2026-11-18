# %% [markdown]
# # 7. A Needle That Moves
#
# > **Overview.** When one output changes, or a noise output starts to depend on features no LTU has learned, do new
# > LTUs find it among 100 outputs, and keep at it until it is learned?
#
# **Setup.** Notebook 4's learner, with 90 noise outputs. For the first 100k steps the target is notebook 4's; then every
# 50k steps one output changes, alternating between two kinds of change (4 of each):
# - **7a. An output changes:** one of the improvable outputs, at random, reads 2 new target LTUs instead, each reading
#   10 random inputs.
# - **7b. A noise output becomes a target:** one improvable output, at random, becomes noise of unit variance, and one
#   noise output, at random, reads 2 new target LTUs.
#
# The newest output reading new target LTUs is the *needle*. Parents drawn uniformly, by error, by notebook 2's progress
# (one period, half uniform), by error times explained fraction (notebook 4a), half as 4a and half by credited progress
# (notebook 4b), or by an oracle that knows the needle: every new LTU's parent is the needle (before the first change,
# one of the first 10). 5 seeds; 500k steps.
# %%
import os
from time import perf_counter
from typing import NamedTuple

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   # take GPU memory only as needed, so the GPU can be shared
import jax
jax.config.update('jax_platforms', 'cuda')
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from phd.jax_core.models import ltu
from phd.jax_core.optimizers.idbd import IDBDState, optax_idbd

%matplotlib inline
assert jax.default_backend() == 'gpu'
# %%
# --- Problem ---
N_INPUTS = 400
N_TARGETS = 8                        # target LTUs at first
N_SOURCES = 10                       # inputs each target LTU reads, drawn at random
N_OUTPUTS = 100
N_IMPROVABLE = 10                    # the first outputs, each reading target LTUs at first; the others are noise
OUTPUT_FAN_IN = 4                    # target LTUs each of them reads, drawn at random
NOISE_STD = 0.1                      # observation noise on outputs that read target LTUs: the RMSE floor
LINEAR_RMSE = np.sqrt(1 - 2 / np.pi + NOISE_STD ** 2)   # about the best linear learner's RMSE on such an output
N_CHANGES = 8                        # changes, one output at a time, alternating: an output changes; a noise output
FIRST_CHANGE, CHANGE_EVERY = 100_000, 50_000   # becomes a target (and an improvable output noise)
N_NEW = 2                            # new target LTUs the changed output reads
N_ALL_TARGETS = N_TARGETS + N_NEW * N_CHANGES

# --- Learner ---
STAT_RATE, VAR_FLOOR = 1e-3, 0.01    # standardization of the LTUs
INIT_STEP_SIZE = 1e-4                # Autostep's initial step-size
META_STEP_SIZE, TAU = 0.02, 1e4
N_LTUS = 32
FAST_STEP_SIZE = 3e-2                # initial step-size of a new LTU's input weights
GROW_Z = 6.0                         # an LTU connects an input whose gradient sum is this many RMS above the rest
FAN_IN = 20                          # a new LTU first reads this many inputs
REPLACED = 4                         # LTUs replaced at each search step, the least useful

# --- Experiment ---
NUM_STEPS = 500_000
WINDOW = 1_000                       # steps per recorded RMSE
N_SEEDS = 5
PERIOD = 10_000                      # steps between search steps
UNIFORM, LOSS, PROGRESS, EXPLAINED, MIXED, ORACLE, HARMONIC, ERR_PROGRESS, CREDITED, SUM, TRIALS = range(11)
PRIOR = 0      # TRIALS' pseudo-trial: 0 the mean outcome of all trials, 1 the explained fraction, 2 the larger
PARENT_ONLY = False  # a trial's outcome: its usefulness to its parent only (else to all outputs)
MEMORY = True  # per-output memory of trials (else the prior alone)
FORGET = True  # replaced trials' memory fades by the replaced share per search step
NOISE, LINEAR, NOISY_LINEAR = range(3)
DISJOINT = False
NO_LINEAR = False                    # new needles: a mirrored pair of thresholded target LTUs, no linear part
THRESHOLD = 0.674 * np.sqrt(N_SOURCES)
EXPLORED = 0.5                       # share of new LTUs whose parent is drawn uniformly, by PROGRESS
EXPLAINED_SHARE = 0.5                # share of the new LTUs drawn by error times explained fraction, if MIXED
# %% [markdown]
# ## Target
# %%
class Target(NamedTuple):
    w: jax.Array                # (N_ALL_TARGETS, N_INPUTS): the target LTUs' weights, +-1 on the inputs they read
    out: jax.Array              # (N_CHANGES + 1, N_OUTPUTS, N_ALL_TARGETS): the outputs' weights on them, by version
    noise_std: jax.Array        # (N_CHANGES + 1, N_OUTPUTS): each output's noise, by version
    needle: jax.Array           # (N_CHANGES + 1,): the newest output reading new target LTUs (-1 before any change)
    linear: jax.Array           # (N_OUTPUTS, N_INPUTS)
    bias: jax.Array             # (N_ALL_TARGETS,)


def make_target(key, kind):
    # Version 0: the first N_IMPROVABLE outputs each read OUTPUT_FAN_IN of the first N_TARGETS target LTUs, drawn at
    # random, with random +-1 weights scaled to unit variance, plus noise NOISE_STD; the others are noise of unit
    # variance. Each target LTU reads N_SOURCES inputs drawn at random, with random +-1 weights. Change c makes version
    # c + 1: if c is even, an output reading target LTUs, at random, reads the next N_NEW instead (an output changes);
    # if odd, one of them becomes noise of unit variance, and a noise output, at random, reads the next N_NEW (a noise
    # output becomes a target).
    read_key, w_key, out_key, out_w_key, change_key, linear_key, linear_w_key = jax.random.split(key, 7)
    pick = lambda k, shape, n: jnp.argsort(jnp.argsort(jax.random.uniform(k, shape))) < n   # n random entries per row
    w = (jax.random.rademacher(w_key, (N_ALL_TARGETS, N_INPUTS), dtype=float)
         * pick(read_key, (N_ALL_TARGETS, N_INPUTS), N_SOURCES))
    first = jnp.arange(N_OUTPUTS)[:, None] < N_IMPROVABLE
    own = jnp.arange(N_TARGETS) // OUTPUT_FAN_IN == jnp.arange(N_OUTPUTS)[:, None]
    reads = jnp.pad(own if DISJOINT else pick(out_key, (N_OUTPUTS, N_TARGETS), OUTPUT_FAN_IN),
                    ((0, 0), (0, N_ALL_TARGETS - N_TARGETS)))
    out = (jax.random.rademacher(out_w_key, (N_OUTPUTS, N_ALL_TARGETS), dtype=float) * reads * first
           / np.sqrt(OUTPUT_FAN_IN))
    noise_std = jnp.where(first[:, 0] | (kind == LINEAR), NOISE_STD, 1.0)
    linear = (jax.random.rademacher(linear_w_key, (N_OUTPUTS, N_INPUTS), dtype=float)
              * pick(linear_key, (N_OUTPUTS, N_INPUTS), N_SOURCES) / np.sqrt(N_SOURCES) * (~first & (kind != NOISE)))
    outs, noise_stds, needles = [out], [noise_std], [jnp.array(-1)]
    bias = jnp.zeros(N_ALL_TARGETS)
    for c in range(N_CHANGES):
        sign_key, old_key, new_key = jax.random.split(jax.random.fold_in(change_key, c), 3)
        improvable = (out != 0).any(1)
        targets = jnp.arange(N_ALL_TARGETS)
        new = (targets >= N_TARGETS + N_NEW * c) & (targets < N_TARGETS + N_NEW * (c + 1))
        row = jax.random.rademacher(sign_key, (N_ALL_TARGETS,), dtype=float) * new / np.sqrt(N_NEW)
        if NO_LINEAR:
            a = N_TARGETS + N_NEW * c
            w = w.at[a + 1].set(-w[a])
            bias = bias.at[a].set(-THRESHOLD).at[a + 1].set(-THRESHOLD)
            row = new * 1.0
        old = jax.random.categorical(old_key, jnp.log(improvable))
        if c % 2 == 0:
            needle = old
        else:
            needle = jax.random.categorical(new_key, jnp.log(~improvable))
            out, noise_std = out.at[old].set(0.0), noise_std.at[old].set(1.0)
        out, noise_std = out.at[needle].set(row), noise_std.at[needle].set(NOISE_STD)
        outs, noise_stds, needles = outs + [out], noise_stds + [noise_std], needles + [needle]
    return Target(w, jnp.stack(outs), jnp.stack(noise_stds), jnp.stack(needles), linear, bias)


def sample(target, version, key):
    # One input and its noisy targets, under the given version of the target.
    x_key, noise_key = jax.random.split(key)
    x = jax.random.normal(x_key, (N_INPUTS,))
    return x, (target.out[version] @ jnp.where(target.w @ x + target.bias > 0, 1.0, -1.0) + target.linear @ x
               + target.noise_std[version] * jax.random.normal(noise_key, (N_OUTPUTS,)))
# %% [markdown]
# ## Learner
#
# Notebook 4's learner. New here: each output's mean squared error over the last search period, for notebook 2's
# progress, and the oracle knows the needle.
# %%
OPTIMIZER = optax_idbd(meta_lr=META_STEP_SIZE, init_lr=INIT_STEP_SIZE, autostep=True, tau=TAU,
                       version='squared_inputs')


@jax.custom_vjp
def standardize(bits, mean, error_mean, error_h_mean):
    return (bits - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), VAR_FLOOR))


def standardize_fwd(bits, mean, error_mean, error_h_mean):
    variance = mean * (1 - mean)
    scale = jnp.sqrt(jnp.maximum(variance, VAR_FLOOR))
    h = (bits - mean) / scale
    return h, (h, scale, variance > VAR_FLOOR, error_mean, error_h_mean)


def standardize_bwd(saved, g):
    h, scale, unfloored, error_mean, error_h_mean = saved
    return (g - error_mean - jnp.where(unfloored, h * error_h_mean, 0.)) / scale, None, None, None


standardize.defvjp(standardize_fwd, standardize_bwd)


class Stats(NamedTuple):
    # Running statistics of the LTUs, each shaped (N_LTUS,).
    hidden_mean: jax.Array      # firing rate
    error_mean: jax.Array       # means of the error g arriving at each LTU, and of g * h
    error_h_mean: jax.Array


class Net(NamedTuple):
    # w['hidden'] is the LTUs' weights over [inputs, 1 (the bias)], of which each reads those in `reads`; w['output'] is
    # the outputs' weights over [inputs (the linear learner), LTUs, 1].
    w: dict                     # {'hidden': (N_LTUS, N_INPUTS + 1), 'output': (N_OUTPUTS, N_INPUTS + N_LTUS + 1)}
    reads: jax.Array            # shaped like w['hidden']
    added: jax.Array            # whether the LTUs feed the outputs yet
    opt: IDBDState
    stats: Stats
    shadow: jax.Array           # (N_OUTPUTS, N_LTUS): each LTU's shadow weight, +-1 on one output (its parent), 0 elsewhere
    candidates: jax.Array       # (N_LTUS, N_INPUTS): each LTU's sums of dL/dz * x_i since its last new connection
    loss_sum: jax.Array         # (N_OUTPUTS,): each output's summed squared error since the last search step
    removed_sum: jax.Array      # (N_OUTPUTS,): the squared error the LTUs removed from each output, summed likewise
    target_sum: jax.Array       # (N_OUTPUTS,): each output's targets, and their squares, summed likewise
    target_sq_sum: jax.Array
    last_removed: jax.Array     # (N_OUTPUTS,): mean squared error the LTUs removed over the last search period (NaN: none)
    last_loss: jax.Array        # (N_OUTPUTS,): mean squared error over the last search period (NaN: none)
    progress: jax.Array         # (N_OUTPUTS,): EMA of the period-to-period increase in removed error
    birth: jax.Array            # (N_LTUS,): its parent's mean squared error when it was added (NaN: not a trial)
    dead_sum: jax.Array         # (N_OUTPUTS,): replaced trials' outcomes (usefulness / birth), summed, by parent
    dead_count: jax.Array       # (N_OUTPUTS,)
    birth_fraction: jax.Array   # (N_LTUS,): its parent's explained fraction when it was added
    fit_sums: jax.Array         # (5,): replaced trials' n, sum e, sum f, sum e^2, sum e f (e: birth_fraction, f: outcome)


def init_net(key, target):
    # Each pair of LTUs wired like one target LTU in turn: reading its inputs and the bias, with LeCun uniform weights;
    # their shadow weights +1 and -1, on a random output that reads that target LTU. Every other weight 0.
    w_key, out_key = jax.random.split(key)
    assigned = jnp.arange(N_LTUS) // 2 % N_TARGETS
    reads = jnp.append(target.w[assigned] != 0, jnp.ones((N_LTUS, 1), bool), axis=1)
    fan_in = N_SOURCES
    readers = target.out[0][:, assigned] != 0
    parent = jnp.argmax(jnp.where(readers, jax.random.gumbel(out_key, readers.shape), -jnp.inf), 0)
    shadow = (jnp.arange(N_OUTPUTS)[:, None] == parent) * jnp.where(jnp.arange(N_LTUS) % 2 == 0, 1.0, -1.0)
    w = {'hidden': (jax.random.uniform(w_key, reads.shape, minval=-1.0, maxval=1.0)
                    * jnp.sqrt(3.0 / fan_in) * reads).at[:, -1].set(0.0),
         'output': jnp.zeros((N_OUTPUTS, N_INPUTS + N_LTUS + 1))}
    stats = Stats(jnp.full(N_LTUS, 0.5), jnp.zeros(N_LTUS), jnp.zeros(N_LTUS))
    return Net(w, reads, jnp.array(False), OPTIMIZER.init(w), stats, shadow, jnp.zeros((N_LTUS, N_INPUTS)),
               *(jnp.zeros(N_OUTPUTS),) * 4, jnp.full(N_OUTPUTS, jnp.nan), jnp.full(N_OUTPUTS, jnp.nan), jnp.zeros(N_OUTPUTS),
               jnp.full(N_LTUS, jnp.nan), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_LTUS), jnp.zeros(5))


def add(net):
    # The LTUs start feeding the outputs, the step-sizes of their input weights starting at FAST_STEP_SIZE.
    beta = jnp.where(net.reads.at[:, -1].set(False), jnp.log(FAST_STEP_SIZE), net.opt.beta['hidden'])
    return net._replace(added=jnp.array(True), opt=net.opt._replace(beta={**net.opt.beta, 'hidden': beta}))


def rewire(net, key, replaced, parent_logits, parent, signs):
    # Each `replaced` LTU becomes a new LTU. Its shadow weight, +1 or -1 by slot, goes on its parent, an output drawn
    # with log-probabilities parent_logits. It reads FAN_IN inputs and the bias, the inputs drawn without replacement
    # with probability proportional to the square of the parent's linear weight on each if parent, uniformly otherwise
    # (Gumbel top-k), with LeCun uniform weights, taking the signs of its shadow weight times the parent's linear
    # weights if signs, at step-size FAST_STEP_SIZE. Its output weights are 0, and its statistics, Autostep state and
    # gradient sums fresh.
    out_key, draw_key, w_key = jax.random.split(key, 3)
    sign = jnp.where(jnp.arange(N_LTUS) % 2 == 0, 1.0, -1.0)
    new_parent = jax.random.categorical(out_key, parent_logits, shape=(N_LTUS,))
    shadow = jnp.where(replaced, (jnp.arange(N_OUTPUTS)[:, None] == new_parent) * sign, net.shadow)
    parent_w = shadow.T @ net.w['output'][:, :N_INPUTS]      # each LTU's shadow weight times its parent's linear weights
    scores = jnp.where(parent, jnp.log(parent_w ** 2), 0.0) + jax.random.gumbel(draw_key, parent_w.shape)
    inputs = jnp.argsort(jnp.argsort(-scores)) < FAN_IN
    w = jax.random.uniform(w_key, inputs.shape, minval=-1.0, maxval=1.0) * jnp.sqrt(3.0 / FAN_IN) * inputs
    w = jnp.where(signs, jnp.abs(w) * jnp.sign(parent_w), w)

    where = {'hidden': replaced[:, None],
             'output': jnp.concatenate([jnp.zeros(N_INPUTS, bool), replaced, jnp.zeros(1, bool)])[None]}
    reset = lambda tree, fresh: jax.tree.map(lambda m, a, b: jnp.where(m, a, b), where, fresh, tree)
    zeros = jax.tree.map(jnp.zeros_like, net.w)
    fast = jnp.append(inputs, jnp.zeros((N_LTUS, 1), bool), axis=1)
    beta = {'hidden': jnp.where(fast, jnp.log(FAST_STEP_SIZE), net.opt.init_beta),
            'output': zeros['output'] + net.opt.init_beta}
    opt = net.opt._replace(beta=reset(net.opt.beta, beta), h=reset(net.opt.h, zeros), v=reset(net.opt.v, zeros))
    stats = Stats(*(jnp.where(replaced, fresh, old) for fresh, old in zip((0.5, 0.0, 0.0), net.stats)))
    return net._replace(w=reset(net.w, {'hidden': jnp.append(w, jnp.zeros((N_LTUS, 1)), axis=1),
                                        'output': zeros['output']}),
                        reads=jnp.where(replaced[:, None], jnp.append(inputs, jnp.ones((N_LTUS, 1), bool), axis=1),
                                        net.reads),
                        opt=opt, stats=stats, shadow=shadow, candidates=jnp.where(replaced[:, None], 0.0, net.candidates))


def learn(net, x, y):
    # One online network-Autostep update on half the squared error summed over the outputs, each LTU's shadow weight
    # standing in for its weight to its parent in the backward pass while the real one is smaller than 1 in magnitude.
    # Returns the new network and each output's squared error on this sample, before learning from it, which it also
    # adds to loss_sum; adds the squared error the LTUs remove from each output (with c their part of the prediction
    # and r the residual, (r - c)^2 - r^2) to removed_sum, the targets and their squares to target_sum and
    # target_sq_sum, and dL/dz * x_i to each LTU's gradient sums. `probe` is zero; its gradients are the error
    # arriving at each LTU and dL/dz.

    def loss_fn(w, probe):
        b = ltu((w['hidden'] * net.reads) @ jnp.append(x, 1.0) + probe[1])
        mean = net.stats.hidden_mean + STAT_RATE * (jax.lax.stop_gradient(b) - net.stats.hidden_mean)
        h = standardize(b, mean, net.stats.error_mean, net.stats.error_h_mean)
        a = (h + probe[0]) * net.added
        source = jnp.concatenate([x, a, jnp.ones(1)])
        unit_w = w['output'][:, N_INPUTS:-1]
        back = jax.lax.stop_gradient(jnp.where((net.shadow != 0) & (jnp.abs(unit_w) < 1.0), net.shadow, unit_w)) @ a
        prediction = w['output'] @ jax.lax.stop_gradient(source) + back - jax.lax.stop_gradient(back)
        residual = jax.lax.stop_gradient(prediction - y)
        removed = (residual - jax.lax.stop_gradient(unit_w @ a)) ** 2 - residual ** 2
        return 0.5 * jnp.sum((prediction - y) ** 2), (jax.lax.stop_gradient(source), jax.lax.stop_gradient(h), mean,
                                                      residual ** 2, removed)

    (_, (source, h, mean, squared, removed)), (grads, (error, dz)) = jax.value_and_grad(loss_fn, (0, 1), has_aux=True)(
        net.w, (jnp.zeros(N_LTUS),) * 2)
    grads = {**grads, 'hidden': grads['hidden'] * net.reads}
    sources = {'hidden': jnp.append(x, 1.0) * net.reads, 'output': jnp.broadcast_to(source, net.w['output'].shape)}
    updates, opt = OPTIMIZER.update((grads, None, sources), net.opt, net.w)
    stats = Stats(mean, net.stats.error_mean + STAT_RATE * (error - net.stats.error_mean),
                  net.stats.error_h_mean + STAT_RATE * (error * h - net.stats.error_h_mean))
    return net._replace(w=jax.tree.map(jnp.add, net.w, updates), opt=opt, stats=stats,
                        candidates=net.candidates + dz[:, None] * x, loss_sum=net.loss_sum + squared,
                        removed_sum=net.removed_sum + removed, target_sum=net.target_sum + y,
                        target_sq_sum=net.target_sq_sum + y ** 2), (squared, removed)


def grow(net):
    # Each LTU whose largest gradient sum, over the inputs it does not read, is more than GROW_Z times their root mean
    # square connects that input, at weight 0 with step-size FAST_STEP_SIZE, and restarts its sums.
    unread = ~net.reads[:, :N_INPUTS]
    sums = jnp.where(unread, jnp.abs(net.candidates), 0.0)
    rms = jnp.sqrt(jnp.sum(sums ** 2, 1) / unread.sum(1))
    grows = jnp.max(sums, 1) > GROW_Z * rms
    new = (jnp.arange(N_INPUTS + 1) == jnp.argmax(sums, 1)[:, None]) & grows[:, None]
    beta = jnp.where(new, jnp.log(FAST_STEP_SIZE), net.opt.beta['hidden'])
    return net._replace(w={**net.w, 'hidden': jnp.where(new, 0.0, net.w['hidden'])}, reads=net.reads | new,
                        opt=net.opt._replace(beta={**net.opt.beta, 'hidden': beta}),
                        candidates=jnp.where(grows[:, None], 0.0, net.candidates))


def trial_use(net):
    # Each LTU's usefulness for its trial outcome: to all outputs, or to its parent only.
    parent = jnp.argmax(jnp.abs(net.shadow), 0)
    return jnp.where(PARENT_ONLY, net.w['output'][parent, N_INPUTS + jnp.arange(N_LTUS)] ** 2, usefulness(net))


def usefulness(net):
    # Each LTU's sum of squared outgoing weights: about the increase in the squared error, summed over the outputs, if
    # it were removed (its activity has unit variance).
    return (net.w['output'][:, N_INPUTS:-1] ** 2).sum(0)


def parent_logits(net, rule, needle):
    # Log-probabilities of each output being a new LTU's parent, from its statistics since the last search step:
    # uniform; in proportion to its squared error (uniform before any is summed); by notebook 2's progress, EXPLORED
    # uniformly and the rest in proportion to the decrease in its mean squared error since the last search period, where
    # positive; in proportion to its squared error times its explained fraction; MIXED: EXPLAINED_SHARE so and the rest
    # in proportion to its credited progress, the increase in the squared error the LTUs removed from it since the last
    # search period, where positive; or the oracle: the needle (uniform among the first N_IMPROVABLE before any change).
    # Each part uniform while all its weights are 0.
    def share(weight):
        weight = jnp.maximum(weight, 0.0)
        return jnp.where(weight.sum() > 0, weight / jnp.maximum(weight.sum(), 1e-30), 1 / N_OUTPUTS)

    def since_last(now, last):
        return jnp.where(jnp.isnan(last), 0.0, now - last)

    loss = net.loss_sum / PERIOD
    variance = net.target_sq_sum / PERIOD - (net.target_sum / PERIOD) ** 2
    explained = share(loss * (variance - loss) / jnp.maximum(variance, 1e-30))
    credited = share(since_last(net.removed_sum / PERIOD, net.last_removed))
    removed = jnp.maximum(net.removed_sum / PERIOD, 0.0)
    positive = jnp.maximum(net.progress, 0.0)
    harmonic = share(loss * removed / jnp.maximum(loss + removed, 1e-30))
    err_progress = share(loss * positive)
    by_progress = share(positive)
    summed = share(loss * jnp.maximum(variance - loss, 0.0) / jnp.maximum(variance, 1e-30) + positive)
    parent = jnp.argmax(jnp.abs(net.shadow), 0)
    trial = ~jnp.isnan(net.birth)
    outcome = jnp.where(trial, trial_use(net) / jnp.maximum(jnp.nan_to_num(net.birth), 1e-30), 0.0)
    alive_sum = jnp.zeros(N_OUTPUTS).at[parent].add(outcome)
    alive_count = jnp.zeros(N_OUTPUTS).at[parent].add(trial * 1.0)
    total_count = net.dead_count.sum() + alive_count.sum()
    mean_outcome = jnp.where(total_count > 0, (net.dead_sum.sum() + alive_sum.sum()) / jnp.maximum(total_count, 1e-30), 1.0)
    fraction = jnp.maximum(variance - loss, 0.0) / jnp.maximum(variance, 1e-30)
    e, f = net.birth_fraction, outcome
    sums = net.fit_sums + jnp.array([2.0, 1.0, 1.0, 1.0, 1.0]) + jnp.stack(
        [trial.sum(), (trial * e).sum(), f.sum(), (trial * e * e).sum(), (e * f).sum()])
    slope = ((sums[4] - sums[1] * sums[2] / sums[0]) / jnp.maximum(sums[3] - sums[1] ** 2 / sums[0], 1e-30))
    intercept = (sums[2] - slope * sums[1]) / sums[0]
    fitted = jnp.maximum(intercept + slope * fraction, 0.0)
    prior = jnp.select([PRIOR == 1, PRIOR == 2, PRIOR == 3], [fraction, jnp.maximum(fraction, mean_outcome), fitted],
                       jnp.full(N_OUTPUTS, mean_outcome))
    trials = share(loss * prior) if not MEMORY else share(loss * (net.dead_sum + alive_sum + prior) / (net.dead_count + alive_count + 1))
    decrease = share(-since_last(loss, net.last_loss))
    outputs = jnp.arange(N_OUTPUTS)
    oracle = jnp.where(needle >= 0, outputs == needle, (outputs < N_IMPROVABLE) / N_IMPROVABLE)
    return jnp.log(jnp.select([rule == LOSS, rule == PROGRESS, rule == EXPLAINED, rule == MIXED, rule == ORACLE,
                               rule == HARMONIC, rule == ERR_PROGRESS, rule == CREDITED, rule == SUM, rule == TRIALS],
                              [share(loss), EXPLORED / N_OUTPUTS + (1 - EXPLORED) * decrease, explained,
                               EXPLAINED_SHARE * explained + (1 - EXPLAINED_SHARE) * credited, oracle,
                               harmonic, err_progress, by_progress, summed, trials],
                              jnp.full(N_OUTPUTS, 1 / N_OUTPUTS)))


# %% [markdown]
# ## Training
# %%
def pick(condition, a, b):
    # a where condition holds, b otherwise, for networks.
    return jax.tree.map(lambda u, v: jnp.where(condition, u, v), a, b)


def train(net, key, target, rule, changes, rate):
    # NUM_STEPS online network-Autostep steps on samples from `target`, its version changing at FIRST_CHANGE and every
    # CHANGE_EVERY steps after. The LTUs, added at step 0 as new LTUs with parents drawn by `rule`, each grow every
    # WINDOW steps, and every PERIOD steps the REPLACED least useful LTUs are replaced by new ones, and the outputs'
    # statistics restart. Returns, for each window: the mean squared error over the outputs then reading target LTUs,
    # the needle's (NaN before any change), and the share of LTUs whose parent is one of those outputs at its end.
    def window(net, inputs):
        index, key = inputs
        version = jnp.where(changes, jnp.clip((index * WINDOW - FIRST_CHANGE) // CHANGE_EVERY + 1, 0, N_CHANGES), 0)
        needle = target.needle[version]
        wire_key, key = jax.random.split(key)
        start = index == 0
        net = pick(start, add(net), net)
        # The first LTUs read random inputs with random signs: their parents have no weights yet.
        net = pick(start, rewire(net, wire_key, jnp.ones(N_LTUS, bool), parent_logits(net, rule, needle), False, False),
                   net)
        net, (squared, removed) = jax.lax.scan(lambda net, k: learn(net, *sample(target, version, k)), net,
                                               jax.random.split(key, WINDOW))
        net = grow(net)
        weakest = jnp.zeros(N_LTUS, bool).at[jnp.argsort(usefulness(net))[:REPLACED]].set(True)
        due = (index + 1) * WINDOW % PERIOD == 0
        increase = jnp.where(jnp.isnan(net.last_removed), 0.0, net.removed_sum / PERIOD - net.last_removed)
        net = pick(due, net._replace(progress=net.progress + rate * (increase - net.progress)), net)
        parent = jnp.argmax(jnp.abs(net.shadow), 0)
        trial = ~jnp.isnan(net.birth)
        outcome = jnp.where(trial, trial_use(net) / jnp.maximum(jnp.nan_to_num(net.birth), 1e-30), 0.0)
        keep = (1 - REPLACED / N_LTUS) if FORGET else 1.0
        ended = weakest & trial
        net = pick(due, net._replace(
            dead_sum=keep * net.dead_sum + jnp.zeros(N_OUTPUTS).at[parent].add(jnp.where(ended, outcome, 0.0)),
            dead_count=keep * net.dead_count + jnp.zeros(N_OUTPUTS).at[parent].add(ended * 1.0),
            fit_sums=keep * net.fit_sums + jnp.stack([ended.sum(), (ended * net.birth_fraction).sum(),
                                                      jnp.where(ended, outcome, 0.0).sum(),
                                                      (ended * net.birth_fraction ** 2).sum(),
                                                      jnp.where(ended, outcome * net.birth_fraction, 0.0).sum()]),
            birth=jnp.where(weakest, jnp.nan, net.birth)), net)
        new = rewire(net, jax.random.fold_in(wire_key, 1), weakest, parent_logits(net, rule, needle), True, True)
        new_parent = jnp.argmax(jnp.abs(new.shadow), 0)
        variance = net.target_sq_sum / PERIOD - (net.target_sum / PERIOD) ** 2
        fraction = jnp.maximum(variance - net.loss_sum / PERIOD, 0.0) / jnp.maximum(variance, 1e-30)
        new = new._replace(birth=jnp.where(weakest, (net.loss_sum / PERIOD)[new_parent], new.birth),
                           birth_fraction=jnp.where(weakest, fraction[new_parent], new.birth_fraction))
        zero = jnp.zeros(N_OUTPUTS)
        net = pick(due, new._replace(loss_sum=zero, removed_sum=zero, target_sum=zero, target_sq_sum=zero,
                                     last_removed=net.removed_sum / PERIOD, last_loss=net.loss_sum / PERIOD), net)
        improvable = (target.out[version] != 0).any(1)
        squared, removed = squared.mean(0), removed.mean(0)
        return net, ((squared * improvable).sum() / improvable.sum(),
                     jnp.where(needle >= 0, squared[needle], jnp.nan),
                     jnp.mean(improvable[jnp.argmax(jnp.abs(net.shadow), 0)]),
                     (removed * improvable).sum() / improvable.sum(),
                     (removed * ~improvable).sum() / (~improvable).sum())

    n = NUM_STEPS // WINDOW
    return jax.lax.scan(window, net, (jnp.arange(n), jax.random.split(key, n)))[1]


def set_linear_step(net, step):
    # The input-to-output (and bias) weights start at step-size `step`.
    linear = jnp.zeros(N_INPUTS + N_LTUS + 1, bool).at[:N_INPUTS].set(True).at[-1].set(True)
    beta = jnp.where(linear, jnp.log(step), net.opt.beta['output'])
    return net._replace(opt=net.opt._replace(beta={**net.opt.beta, 'output': beta}))


def run(configs):
    # configs: (kind, changes, rule, linear step, rate). Returns {config: (rmse, needle rmse, share, removed on
    # improvable, removed on the others), each (seed, window)}.
    kind, changes, rule, step, rate, seed = (jnp.array(v) for v in zip(*[(*c, s) for c in configs
                                                                       for s in range(N_SEEDS)]))
    keys = lambda k: jax.random.split(jax.random.PRNGKey(k), N_SEEDS)[seed]
    targets = jax.vmap(make_target)(keys(2), kind)
    nets = jax.vmap(set_linear_step)(jax.vmap(init_net)(keys(0), targets), step)
    outputs = jax.jit(jax.vmap(train))(nets, keys(1), targets, rule, changes, rate)
    mse, needle, share, rem_imp, rem_other = (np.asarray(v).reshape(len(configs), N_SEEDS, -1) for v in outputs)
    return dict(zip(configs, zip(np.sqrt(mse), np.sqrt(needle), share, rem_imp, rem_other)))
# %%
import pickle, sys
PILOT = eval(sys.argv[-1])   # list of (name, kinds, rule, linear step, rate); kinds: list of 0,1,2 and 'needle'
cfgs = []
for name, kinds, rule, step, rate in PILOT:
    for k in kinds:
        cfgs.append((0 if k == 'needle' else k, k == 'needle', rule, step, rate))
started = perf_counter()
res = run(sorted(set(cfgs), key=cfgs.index))
print(f'{len(res) * N_SEEDS} runs in {perf_counter() - started:.0f}s')
CH = FIRST_CHANGE + CHANGE_EVERY * np.arange(N_CHANGES)
for name, kinds, rule, step, rate in PILOT:
    out = []
    for k in kinds:
        c = (0 if k == 'needle' else k, k == 'needle', rule, step, rate)
        rmse, needle, share, rem_imp, rem_other = res[c]
        if k == 'needle':
            after = np.stack([needle[:, x // WINDOW:x // WINDOW + CHANGE_EVERY // WINDOW] for x in CH], 1)
            out.append(f"needle {after[:, 0::2].mean():.3f}/{after[:, 1::2].mean():.3f} {np.round(after.mean((1, 2)), 3)}")
        else:
            out.append(f"{['noise', 'linear', 'noisy lin'][k]} {rmse.mean():.3f} {np.round(rmse.mean(1), 3)} "
                       f"credit other {rem_other.mean():.4f} late {rem_other[:, -100:].mean():.4f} imp {rem_imp[:, -100:].mean():.3f}")
    print(f'{name}: ' + ' | '.join(out))
pickle.dump(res, open(f'pilot_{abs(hash(sys.argv[-1])) % 10**6}.pkl', 'wb'))
