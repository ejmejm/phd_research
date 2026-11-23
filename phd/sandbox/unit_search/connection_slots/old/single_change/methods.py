"""Two bounded connection methods and their controls, on one sparse LTU task."""

import os
from time import perf_counter
from typing import NamedTuple

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
jax.config.update('jax_platforms', 'cuda')
import jax.numpy as jnp
import numpy as np

from phd.jax_core.optimizers.idbd import optax_idbd


N_INPUTS, N_TARGETS, N_OUTPUTS = 108, 4, 4
TARGET_FAN_IN, OUTPUT_FAN_IN = 4, 2
INITIAL_INPUTS = 8
NOISE_STD = 0.1
N_SEEDS, STEPS, WINDOW, PERIOD = 5, 100_000, 1_000, 10_000
INIT_STEP, FAST_STEP, STAT_RATE, VAR_FLOOR = 1e-4, 0.03, 1e-3, 0.01
FAST_DECAY, SLOW_DECAY = np.exp(-1 / PERIOD), np.exp(-1 / (8 * PERIOD))
OPT = optax_idbd(meta_lr=0.02, init_lr=INIT_STEP, autostep=True, tau=1e4,
                version='squared_inputs')

# Count every slot and bias in both layers. The all-input control has exactly
# the same budget: 16 * 109 + 4 * 125 == (64 + 4) * 33 == 2244.
SHAPES = {'candidates': (64, 32, 32), 'single': (64, 32, 32),
          'slow_candidates': (64, 32, 32),
          'single_fast': (64, 32, 32),
          'fixed': (64, 32, 32), 'all_inputs': (16, 108, 124),
          'ideal': (4, 4, 2)}
LABELS = {'candidates': 'separate candidates (proposed)',
          'slow_candidates': 'candidates, small promotion step (control)',
          'single_fast': 'no candidates, large initial step (control)',
          'single': 'small initial steps (proposed)',
          'fixed': 'fixed input pools (baseline)',
          'all_inputs': 'all inputs, same budget (baseline)',
          'ideal': 'ideal connections (reference)'}


class Target(NamedTuple):
    inputs: jax.Array
    weights: jax.Array
    outputs: jax.Array
    output_weights: jax.Array


class Layer(NamedTuple):
    inputs: jax.Array
    weights: jax.Array                 # last column is the bias
    active: jax.Array
    opt: object
    g_sum: jax.Array
    g_sq: jax.Array
    log_imm: jax.Array                 # log(1 - maturity)
    error_sq: jax.Array
    count: jax.Array


class Net(NamedTuple):
    hidden: Layer
    output: Layer
    shadow: jax.Array                 # only actual, active output slots
    mean: jax.Array
    error_mean: jax.Array
    error_h_mean: jax.Array
    fast_abs: jax.Array
    slow_abs: jax.Array
    slow_sq: jax.Array
    sums: jax.Array                    # fast/slow weights, squares, product
    new_inputs: jax.Array
    signs: jax.Array                   # original sign-cut control only
    step: jax.Array


def draw_inputs(inputs, replace, n_sources, key):
    """Uniform draws from the complement of each row, using only its K slots.

    A random rank among unused identifiers maps to a source via sorted occupied
    identifiers. There are K passes over K slots, never a scan over n_sources
    or an unbounded rejection loop. Negative identifiers denote empty slots.
    """
    rows, slots = inputs.shape

    def fill(current, item):
        slot, rng = item
        occupied = (current >= 0) & (current < n_sources)
        # A full pool has only its old source available for this slot.
        occupied &= (jnp.arange(slots) != slot) | (n_sources > slots)
        ordered = jnp.sort(jnp.where(occupied, current, n_sources), axis=1)
        count = occupied.sum(1)
        rank = jax.random.randint(rng, (rows,), 0, jnp.maximum(n_sources - count, 1))
        source = rank + ((rank[:, None] >= ordered - jnp.arange(slots))
                         & (ordered < n_sources)).sum(1)
        change = replace[:, slot] & (count < n_sources)
        current = current.at[:, slot].set(jnp.where(change, source, current[:, slot]))
        return current, None

    return jax.lax.scan(fill, inputs, (jnp.arange(slots), jax.random.split(key, slots)))[0]


def alias_table(weights):
    """O(outputs) construction; subsequent parent draws cost O(1) each."""
    n = len(weights)
    weights = jnp.where(weights.sum() > 0, weights, jnp.ones_like(weights))
    q = weights * n / weights.sum()
    small = jnp.nonzero(q < 1, size=n, fill_value=0)[0]
    large = jnp.nonzero(q >= 1, size=n, fill_value=0)[0]
    state = (q, jnp.ones(n), jnp.arange(n), small, large, (q < 1).sum(), (q >= 1).sum())

    def step(state):
        q, prob, alias, small, large, ns, nl = state
        s, l = small[ns - 1], large[nl - 1]
        ns, nl = ns - 1, nl - 1
        prob, alias = prob.at[s].set(q[s]), alias.at[s].set(l)
        q = q.at[l].add(q[s] - 1)
        is_small = q[l] < 1
        small = small.at[ns].set(jnp.where(is_small, l, small[ns]))
        large = large.at[nl].set(jnp.where(is_small, large[nl], l))
        return q, prob, alias, small, large, ns + is_small, nl + ~is_small

    _, prob, alias, *_ = jax.lax.while_loop(lambda s: (s[-2] > 0) & (s[-1] > 0), step, state)
    return prob, alias


def make_target(key):
    a, b, c = jax.random.split(key, 3)
    inputs = draw_inputs(jnp.full((N_TARGETS, TARGET_FAN_IN), -1),
                         jnp.ones((N_TARGETS, TARGET_FAN_IN), bool), N_INPUTS, a)
    outputs = (jnp.arange(N_OUTPUTS)[:, None] + jnp.arange(OUTPUT_FAN_IN)) % N_TARGETS
    return Target(inputs, jax.random.rademacher(b, inputs.shape, dtype=jnp.float32), outputs,
                  jax.random.rademacher(c, outputs.shape, dtype=jnp.float32) / np.sqrt(OUTPUT_FAN_IN))


def sample(target, key, changed=False):
    a, b = jax.random.split(key)
    x = jax.random.normal(a, (N_INPUTS,))
    # Flip exactly half the output edges. Features and all magnitudes stay fixed.
    out = target.output_weights * jnp.where(changed, jnp.array([-1., 1.]), 1.)
    features = jnp.where((x[target.inputs] * target.weights).sum(1) > 0, 1., -1.)
    y = (features[target.outputs] * out).sum(1)
    return x, y + NOISE_STD * jax.random.normal(b, (N_OUTPUTS,))


def fresh_layer(inputs, active, weights=None):
    shape = active.shape
    weights = jnp.zeros(shape) if weights is None else weights
    opt = OPT.init(weights)
    return Layer(inputs, weights, active, opt, jnp.zeros(shape), jnp.zeros(shape),
                 jnp.zeros(shape), jnp.zeros(shape[0]), jnp.zeros(shape[0], jnp.int32))


def reset_slots(layer, reset, active, step_size=INIT_STEP):
    zero = lambda a: jnp.where(reset, 0., a)
    opt = layer.opt._replace(beta=jnp.where(reset, jnp.log(step_size), layer.opt.beta),
                            h=zero(layer.opt.h), v=zero(layer.opt.v))
    return layer._replace(weights=zero(layer.weights), active=active, opt=opt,
                          g_sum=zero(layer.g_sum), g_sq=zero(layer.g_sq), log_imm=zero(layer.log_imm))


def priority(net):
    """Gated progress from output_prioritization/3_progress.ipynb."""
    nf, ns = jnp.maximum(net.sums[:2], 1.)
    ff, ss, fs = net.sums[2:]
    fast, slow = net.fast_abs / nf, net.slow_abs / ns
    spread = jnp.maximum(ff / nf**2 + ss / ns**2 - 2 * fs / (nf * ns), 0.)
    variance = jnp.maximum(net.slow_sq / ns - slow**2, 0.)
    progress = (slow - fast) / jnp.maximum(slow, 1e-30) * fast / jnp.maximum(net.new_inputs, 1.)
    p = jnp.where(slow - fast > 3 * jnp.sqrt(variance * spread), progress, 0.)
    return jnp.where(p.sum() > 0, p + p.mean(), 1.)


def replace_units(net, chosen, key, method, initial=False):
    """Attach each new LTU to an actual output slot, then reset that LTU.

    Parents follow gated progress among outputs with room. A full sampled parent
    delays a replacement. A new unit never evicts an unrelated active connection.
    Existing edges to a replaced unit reset because their source has changed.
    """
    hidden, output = net.hidden, net.output
    n = len(hidden.inputs)
    a, b, c, d, e = jax.random.split(key, 5)
    selected = jnp.zeros(n, bool).at[chosen].set(True)
    source = jnp.clip(output.inputs - N_INPUTS, 0, n - 1)
    is_hidden = output.inputs >= N_INPUTS
    available = (~output.active[:, :-1] | (output.weights[:, :-1] == 0)) & (net.shadow[:, :-1] == 0)
    available |= is_hidden & selected[source]
    if method == 'all_inputs':
        available = jnp.ones_like(available)
    has_room = available.any(1)
    prob, alias = alias_table(priority(net) * has_room)
    parent_key, coin_key, sign_key = jax.random.split(a, 3)
    columns = jax.random.randint(parent_key, chosen.shape, 0, N_OUTPUTS)
    parents = jnp.where(jax.random.uniform(coin_key, chosen.shape) < prob[columns], columns, alias[columns])
    signs = jax.random.rademacher(sign_key, chosen.shape, dtype=jnp.float32)

    def attach(carry, item):
        ids, available, attached, shadow, success, parent_of = carry
        unit, parent, sign, rng = item
        count = available[parent].sum()
        rank = jax.random.randint(rng, (), 0, jnp.maximum(count, 1))
        slot = jnp.argmax((jnp.cumsum(available[parent]) > rank) & available[parent])
        if method == 'all_inputs':
            slot = N_INPUTS + unit
        ok = count > 0
        ids = ids.at[parent, slot].set(jnp.where(ok, N_INPUTS + unit, ids[parent, slot]))
        available = available.at[parent, slot].set(jnp.where(ok, False, available[parent, slot]))
        attached = attached.at[parent, slot].set(attached[parent, slot] | ok)
        shadow = shadow.at[parent, slot].set(jnp.where(ok, sign, shadow[parent, slot]))
        success = success.at[unit].set(ok)
        parent_of = parent_of.at[unit].set(parent)
        return (ids, available, attached, shadow, success, parent_of), None

    carry = (output.inputs, available, jnp.zeros_like(available), net.shadow[:, :-1],
             jnp.zeros(n, bool), jnp.zeros(n, jnp.int32))
    (ids, _, attached, shadow, success, parent_of), _ = jax.lax.scan(
        attach, carry, (chosen, parents, signs, jax.random.split(b, len(chosen))))
    old_source = jnp.clip(ids - N_INPUTS, 0, n - 1)
    stale = (ids >= N_INPUTS) & success[old_source] & ~attached
    reset = jnp.pad(stale | attached, ((0, 0), (0, 1)))
    active = output.active & ~jnp.pad(stale, ((0, 0), (0, 1)))
    active |= jnp.pad(attached, ((0, 0), (0, 1)))
    if method == 'all_inputs':
        active = jnp.ones_like(active)
    else:
        ids = jnp.where(stale, -1, ids)
    # Keep the common shadow-parent start unchanged. Only ordinary new
    # zero-weight connections use the diagnostic large initial step.
    output_step = (jnp.where(jnp.pad(attached, ((0, 0), (0, 1))), INIT_STEP, FAST_STEP)
                   if method == 'single_fast' else INIT_STEP)
    output = reset_slots(output._replace(inputs=ids), reset, active, output_step)
    shadow = jnp.pad(jnp.where(stale, 0., shadow), ((0, 0), (0, 1)))

    if method == 'all_inputs':
        ids_h = hidden.inputs
        parent_w = net.output.weights[parent_of, :N_INPUTS]
        use_parent = (jnp.abs(parent_w).sum(1) > 0) & (not initial)
        logits = jnp.where(use_parent[:, None], jnp.log(jnp.maximum(parent_w**2, 1e-30)), 0.)
        scores = logits + jax.random.gumbel(c, parent_w.shape)
        selected_inputs = jnp.argsort(jnp.argsort(-scores, axis=1), axis=1) < INITIAL_INPUTS
    else:
        ids_h = draw_inputs(jnp.full_like(hidden.inputs, -1), jnp.ones_like(hidden.inputs, bool), N_INPUTS, c)
        selected_inputs = jnp.broadcast_to(jnp.arange(ids_h.shape[1]) < INITIAL_INPUTS, ids_h.shape)
    weights = jax.random.uniform(d, ids_h.shape, minval=-1., maxval=1.) * np.sqrt(3 / INITIAL_INPUTS)
    if method == 'all_inputs':
        new_sign = jnp.zeros(n).at[chosen].set(signs)
        weights = jnp.where(use_parent[:, None], jnp.abs(weights) * jnp.sign(parent_w) * new_sign[:, None], weights)
    weights *= selected_inputs
    active_h = jnp.ones_like(selected_inputs) if method in ('single', 'single_fast') else selected_inputs
    active_h = jnp.pad(active_h, ((0, 0), (0, 1)), constant_values=True)
    fresh = fresh_layer(ids_h, active_h, jnp.pad(weights, ((0, 0), (0, 1))))
    fast = jnp.pad(jnp.ones_like(selected_inputs) if method == 'single_fast' else selected_inputs, ((0, 0), (0, 1)))
    fresh = fresh._replace(opt=fresh.opt._replace(beta=jnp.where(fast, jnp.log(FAST_STEP), fresh.opt.beta)))
    choose = lambda old, new: (jnp.where(success.reshape((n,) + (1,) * (old.ndim - 1)), new, old)
                               if jnp.ndim(old) else old)
    hidden = jax.tree.map(choose, hidden, fresh)
    vacant = output.inputs < 0
    if method != 'all_inputs':
        output = output._replace(inputs=draw_inputs(output.inputs, vacant, N_INPUTS + n, e))
    if method in ('single', 'single_fast'):
        output = output._replace(active=jnp.ones_like(output.active))
    return net._replace(hidden=hidden, output=output, shadow=shadow,
                        mean=jnp.where(success, 0.5, net.mean),
                        error_mean=jnp.where(success, 0., net.error_mean),
                        error_h_mean=jnp.where(success, 0., net.error_h_mean),
                        signs=jnp.where(success[:, None], 0., net.signs),
                        new_inputs=net.new_inputs + reset[:, :-1].sum(1))


def init_net(key, target, method):
    n, kh, ko = SHAPES[method]
    a, b, c, d = jax.random.split(key, 4)
    if method == 'ideal':
        ih, io = target.inputs, N_INPUTS + target.outputs
    elif method == 'all_inputs':
        ih = jnp.broadcast_to(jnp.arange(N_INPUTS), (n, kh))
        io = jnp.broadcast_to(jnp.arange(N_INPUTS + n), (N_OUTPUTS, ko))
    else:
        ih = draw_inputs(jnp.full((n, kh), -1), jnp.ones((n, kh), bool), N_INPUTS, a)
        io = draw_inputs(jnp.full((N_OUTPUTS, ko), -1), jnp.ones((N_OUTPUTS, ko), bool), N_INPUTS + n, b)
    hidden = fresh_layer(ih, jnp.ones((n, kh + 1), bool))
    out_active = jnp.ones((N_OUTPUTS, ko + 1), bool)
    if method in ('candidates', 'slow_candidates', 'fixed'):
        out_active = jnp.broadcast_to(jnp.arange(ko + 1) < INITIAL_INPUTS, out_active.shape).at[:, -1].set(True)
    output = fresh_layer(io, out_active)
    if method == 'single_fast':
        output = output._replace(opt=output.opt._replace(beta=output.opt.beta.at[:, :-1].set(jnp.log(FAST_STEP))))
    zeros = jnp.zeros(N_OUTPUTS)
    net = Net(hidden, output, jnp.zeros_like(output.weights), jnp.full(n, 0.5, jnp.float32), jnp.zeros(n), jnp.zeros(n),
              zeros, zeros, zeros, jnp.zeros(5), jnp.full(N_OUTPUTS, ko + 1., jnp.float32),
              jnp.zeros_like(hidden.weights), jnp.array(0, jnp.int32))
    if method != 'ideal':
        return replace_units(net, jnp.arange(n), c, method, initial=True)
    # Ideal topology is known, but all weights must learn from random starts.
    w = jax.random.uniform(c, (n, kh), minval=-1., maxval=1.) * np.sqrt(3 / kh)
    hidden = hidden._replace(weights=jnp.pad(w, ((0, 0), (0, 1))),
                             opt=hidden.opt._replace(beta=jnp.full_like(hidden.weights, jnp.log(FAST_STEP))))
    # Each target unit has an actual edge to the output with the same index.
    shadow = net.shadow.at[jnp.arange(n), 0].set(jax.random.rademacher(d, (n,), dtype=jnp.float32))
    return net._replace(hidden=hidden, shadow=shadow)


def update_layer(layer, inputs, delta):
    grads = delta[:, None] * inputs
    updates, opt = OPT.update((grads * layer.active, None, inputs * layer.active), layer.opt, layer.weights)
    alpha = jnp.exp(opt.beta)
    count = layer.count + 1
    c = -jnp.expm1(count * jnp.log1p(-STAT_RATE))
    log_imm = layer.log_imm + 2 * jnp.log1p(-jnp.minimum(alpha * c[:, None], 1.)) * layer.active
    return layer._replace(weights=layer.weights + updates, opt=opt, g_sum=layer.g_sum + grads,
                          g_sq=layer.g_sq + grads**2, log_imm=log_imm,
                          error_sq=layer.error_sq + STAT_RATE * (delta**2 - layer.error_sq), count=count)


def learn(net, x, y):
    hidden, output = net.hidden, net.output
    n = len(hidden.inputs)
    hx = jnp.pad(x[hidden.inputs], ((0, 0), (0, 1)), constant_values=1.)
    z = (hidden.weights * hidden.active * hx).sum(1)
    bits = (z > 0).astype(jnp.float32)
    mean = net.mean + STAT_RATE * (bits - net.mean)
    variance = mean * (1 - mean)
    scale = jnp.sqrt(jnp.maximum(variance, VAR_FLOOR))
    h = (bits - mean) / scale
    source = jnp.concatenate((x, h))
    ox = jnp.pad(source[output.inputs], ((0, 0), (0, 1)), constant_values=1.)
    prediction = (output.weights * output.active * ox).sum(1)
    residual = prediction - y
    back = jnp.where((net.shadow != 0) & (jnp.abs(output.weights) < 1.), net.shadow, output.weights) * output.active
    hidden_source = output.inputs >= N_INPUTS
    error = jnp.zeros(n).at[jnp.clip(output.inputs - N_INPUTS, 0, n - 1)].add(
        residual[:, None] * back[:, :-1] * hidden_source)
    centered = (error - net.error_mean - jnp.where(variance > VAR_FLOOR, h * net.error_h_mean, 0.)) / scale
    sigmoid = jax.nn.sigmoid(z)
    delta = centered * sigmoid * (1 - sigmoid)
    hidden = update_layer(hidden, hx, delta)
    output = update_layer(output, ox, residual)
    decay = jnp.array([FAST_DECAY, SLOW_DECAY, FAST_DECAY**2, SLOW_DECAY**2, FAST_DECAY * SLOW_DECAY])
    return net._replace(hidden=hidden, output=output, mean=mean,
                        error_mean=net.error_mean + STAT_RATE * (error - net.error_mean),
                        error_h_mean=net.error_h_mean + STAT_RATE * (error * h - net.error_h_mean),
                        fast_abs=FAST_DECAY * net.fast_abs + jnp.abs(residual),
                        slow_abs=SLOW_DECAY * net.slow_abs + jnp.abs(residual),
                        slow_sq=SLOW_DECAY * net.slow_sq + residual**2,
                        sums=decay * net.sums + 1, new_inputs=SLOW_DECAY * net.new_inputs,
                        step=net.step + 1), jnp.mean(residual**2)


def ci_cut(layer):
    """Existing |w|+|h| noise bound; bias is exempt."""
    unbias = -jnp.expm1(layer.count * jnp.log1p(-STAT_RATE))
    error_sq = layer.error_sq / jnp.maximum(unbias, 1e-30)
    bound = (jnp.sqrt(jnp.exp(layer.opt.beta) * error_sq[:, None] / 2 * -jnp.expm1(layer.log_imm))
             * (1.61 + 0.39 * jnp.exp(0.75 * layer.log_imm)))
    return (layer.active & (jnp.abs(layer.weights) + jnp.abs(layer.opt.h) < bound)).at[:, -1].set(False)


def revise_layer(layer, key, n_sources, method, cut):
    candidates = ~layer.active
    z = jnp.where(candidates, jnp.abs(layer.g_sum) / jnp.sqrt(jnp.maximum(layer.g_sq, 1e-30)), 0.)
    promote = (jnp.arange(z.shape[1]) == jnp.argmax(z, 1)[:, None]) & (z.max(1) > 6)[:, None]
    redraw = cut | (candidates & (z < 1))
    if method in ('fixed', 'all_inputs'):
        redraw = jnp.zeros_like(redraw)
    if method in ('single', 'single_fast'):
        promote = jnp.zeros_like(promote)
    active = (layer.active & ~cut) | promote
    if method in ('single', 'single_fast'):
        active = jnp.ones_like(active)
    fresh = cut | redraw
    layer = reset_slots(layer, fresh, active, FAST_STEP if method == 'single_fast' else INIT_STEP)
    promotion_step = INIT_STEP if method == 'slow_candidates' else FAST_STEP
    layer = reset_slots(layer, promote, active, promotion_step)
    inputs = (layer.inputs if method in ('fixed', 'all_inputs') else
              draw_inputs(layer.inputs, redraw[:, :-1], n_sources, key))
    changed = (fresh | promote).any(1)[:, None]
    return layer._replace(inputs=inputs, g_sum=jnp.where(changed, 0., layer.g_sum),
                          g_sq=jnp.where(changed, 0., layer.g_sq)), fresh, (redraw | promote)[:, :-1].sum(1)


def restructure(net, key, method, connections=True, units=True, remove_active=True):
    if method == 'ideal':
        return net
    a, b, c = jax.random.split(key, 3)
    if connections:
        hcut, ocut = ci_cut(net.hidden), ci_cut(net.output)
        if method in ('fixed', 'all_inputs'):
            hcut, ocut = jnp.zeros_like(hcut), jnp.zeros_like(ocut)
        if method == 'all_inputs':
            hcut = (net.hidden.active & (net.signs != 0) & (jnp.sign(net.hidden.weights) != net.signs)
                    & (net.step % PERIOD == 0)).at[:, -1].set(False)
        if not remove_active:
            hcut, ocut = jnp.zeros_like(hcut), jnp.zeros_like(ocut)
        hidden, _, _ = revise_layer(net.hidden, a, N_INPUTS, method, hcut)
        output, fresh, added = revise_layer(net.output, b, N_INPUTS + len(hidden.inputs), method, ocut)
        net = net._replace(hidden=hidden, output=output, shadow=jnp.where(fresh, 0., net.shadow),
                            new_inputs=net.new_inputs + added)
    if method == 'fixed' or not units:
        return net

    def search(net):
        n = len(net.hidden.inputs)
        valid = net.output.inputs >= N_INPUTS
        strength = jnp.zeros(n).at[jnp.clip(net.output.inputs - N_INPUTS, 0, n - 1)].add(
            (net.output.weights[:, :-1] * net.output.active[:, :-1])**2 * valid)
        # One weak unit per group of eight avoids a global sort of all units.
        groups = jnp.pad(strength, (0, (-n) % 8), constant_values=jnp.inf).reshape(-1, 8)
        chosen = jnp.argmin(groups, 1) + 8 * jnp.arange(len(groups))
        # Save old signs first. replace_units leaves new units at sign zero,
        # preserving the original rule's exemption at their first sign check.
        net = net._replace(signs=jnp.sign(net.hidden.weights) * net.hidden.active)
        return replace_units(net, chosen, c, method)

    return jax.lax.cond(net.step % PERIOD == 0, search, lambda net: net, net)


def train(net, target, stream_key, method, steps, changed=False, frozen=False,
          connections=True, units=True, remove_active=True):
    def window(net, _):
        # Generate the same online stream in a block. Only weight updates remain
        # sequential; targets and random draws do not depend on the learner.
        keys = jax.vmap(lambda i: jax.random.fold_in(stream_key, net.step + i))(jnp.arange(WINDOW))
        examples = jax.vmap(lambda k: sample(target, k, changed))(keys)
        net, mse = jax.lax.scan(lambda n, xy: learn(n, *xy), net, examples)
        if not frozen:
            net = restructure(net, jax.random.fold_in(stream_key, net.step + 1_000_000),
                              method, connections, units, remove_active)
        return net, mse.mean()
    return jax.lax.scan(window, net, None, length=steps // WINDOW)


def seeds():
    assert jax.default_backend() == 'gpu', 'Experiments require the GPU.'
    # Preserve the first five seeds of the original 30-seed experiments.
    keys = jax.random.split(jax.random.PRNGKey(42), 30)[:N_SEEDS]
    parts = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
    return parts[:, 0], jax.vmap(make_target)(parts[:, 1]), parts[:, 2]


def run_fixed(methods=('fixed', 'all_inputs', 'ideal', 'candidates', 'single'), steps=STEPS):
    init, targets, streams = seeds()
    result = {}
    started = perf_counter()
    for method in methods:
        nets = jax.jit(jax.vmap(lambda k, t: init_net(k, t, method)))(init, targets)
        _, mse = jax.jit(jax.vmap(lambda n, t, k: train(n, t, k, method, steps)))(nets, targets, streams)
        result[method] = np.sqrt(np.asarray(mse))
        print(f'{method}: final RMSE {result[method][:, -max(1, steps // WINDOW // 5):].mean():.3f}', flush=True)
    print(f'{perf_counter() - started:.1f}s for {N_SEEDS} seeds per method', flush=True)
    return result


def run_changed(steps=STEPS):
    init, targets, streams = seeds()
    result = {}
    started = perf_counter()
    for method in ('ideal', 'candidates', 'single'):
        nets = jax.jit(jax.vmap(lambda k, t: init_net(k, t, method)))(init, targets)
        nets, before = jax.jit(jax.vmap(lambda n, t, k: train(n, t, k, method, steps // 2)))(nets, targets, streams)
        for frozen in ((False,) if method == 'ideal' else (False, True)):
            _, after = jax.jit(jax.vmap(lambda n, t, k: train(n, t, k, method, steps // 2, True, frozen)))(
                nets, targets, streams)
            name = method + ('_frozen' if frozen else '')
            result[name] = np.sqrt(np.asarray(jnp.concatenate((before, after), axis=1)))
            print(f'{name}: post-change RMSE {result[name][:, steps // WINDOW // 2:].mean():.3f}', flush=True)
    print(f'{perf_counter() - started:.1f}s for {N_SEEDS} seeds per method', flush=True)
    return result


def run_recovery_parts(steps=STEPS):
    """Separate connection changes, unit replacement, and active removal."""
    init, targets, streams = seeds()
    started = perf_counter()
    nets = jax.jit(jax.vmap(lambda k, t: init_net(k, t, 'candidates')))(init, targets)
    nets, before = jax.jit(jax.vmap(lambda n, t, k: train(n, t, k, 'candidates', steps // 2)))(
        nets, targets, streams)
    result = {}
    for name, connections, units, remove_active in [
            ('fixed', False, False, True), ('connections_only', True, False, True),
            ('units_only', False, True, True), ('both', True, True, True),
            ('candidates_only', True, False, False)]:
        _, after = jax.jit(jax.vmap(lambda n, t, k: train(
            n, t, k, 'candidates', steps // 2, changed=True, connections=connections,
            units=units, remove_active=remove_active)))(
                nets, targets, streams)
        result[name] = np.sqrt(np.asarray(jnp.concatenate((before, after), axis=1)))
        print(f'{name}: post-change RMSE {result[name][:, steps // WINDOW // 2:].mean():.3f}', flush=True)
    print(f'{perf_counter() - started:.1f}s for {N_SEEDS} seeds per condition', flush=True)
    return result


def plot(result, changed=False, labels=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for name, values in result.items():
        label = (labels or LABELS).get(name, name.replace('_frozen', '') + ', connections fixed after change (control)')
        mean, se = values.mean(0), values.std(0, ddof=1) / np.sqrt(len(values))
        x = (np.arange(len(mean)) + 1) * WINDOW
        line, = ax.plot(x, mean, label=label)
        if len(values) == 30:
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=line.get_color(), alpha=0.12)
    ax.axhline(NOISE_STD, color='black', ls=':', label='noise standard deviation')
    if changed:
        ax.axvline(x[-1] / 2, color='black', ls='--', lw=1)
    ax.set(xlabel='Examples', ylabel=f'RMSE (mean over {len(values)} seeds)')
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
    plt.show()
