"""Shared definitions copied from 02d_retention_and_reuse.ipynb (setup, model, training, local backward).

Copied so that work in this folder does not depend on the notebook being edited concurrently.
"""
"""Two-layer retention: task performance against preserved learned representations."""
from functools import partial
from typing import NamedTuple
import time
import jax
jax.config.update('jax_platforms', 'cuda')
import jax.numpy as jnp
import numpy as np
from phd.jax_core.models import ltu
from phd.jax_core.optimizers.idbd import optax_idbd

assert jax.default_backend() == 'gpu'
INPUTS, FEATURES, OUTPUTS = 21, 10, 10
INIT_RATE, META_RATE, EMA_RATE, VAR_FLOOR = 1e-5, .02, 1e-3, .01
OPT = optax_idbd(meta_lr=META_RATE, init_lr=INIT_RATE, autostep=True,
                tau=1e4, version='squared_inputs')
CALIBRATION = jnp.asarray(np.random.default_rng(0).choice([-1., 1.], (10_000, INPUTS)), dtype=jnp.float32)

class Learner(NamedTuple):
    weights: tuple
    opt: object
    means: tuple
    slopes: tuple
    covs: tuple



import matplotlib.pyplot as plt

def make_task(key, first):
    ks, ko = jax.random.split(key)
    second = jax.random.rademacher(ks, (FEATURES, FEATURES), dtype=jnp.float32)
    out = jax.random.rademacher(ko, (OUTPUTS, FEATURES), dtype=jnp.float32)
    sums = second @ (first @ CALIBRATION.T > 2)
    candidates = jnp.arange(-FEATURES, FEATURES) + .5
    rates = (sums[:, None] > candidates[:, None]).mean(-1)
    return second, candidates[jnp.abs(rates - .3).argmin(1)], out


def target(t, x):
    first, second, thresholds, out = t
    h = (first @ x > 2).astype(jnp.float32)
    return out @ (second @ h > thresholds).astype(jnp.float32)


@partial(jax.jit, static_argnames=('shared', 'nseeds', 'ntasks'))
def initialize(shared, nseeds=5, ntasks=2):
    def one(seed):
        kt, kl, rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        kf, ks = jax.random.split(kt)
        groups = 1 if shared else ntasks
        first = jax.random.rademacher(kf, (groups, FEATURES, INPUTS), dtype=jnp.float32)
        first = jnp.repeat(first, ntasks, axis=0) if shared else first
        teachers = (first, *jax.vmap(make_task)(jax.random.split(ks, ntasks), first))
        widths = (15 * groups, 15 * ntasks)
        weights, fan_in = [], INPUTS
        for key, width in zip(jax.random.split(kl), widths):
            w = INIT_RATE * jax.random.uniform(key, (width, fan_in + 1), minval=-1., maxval=1.).at[:, -1].set(0.)
            weights.append(w)
            fan_in = width
        weights = (*weights, jnp.zeros((ntasks * OUTPUTS, fan_in + 1)))
        learner = Learner(weights, OPT.init(weights),
                          tuple(jnp.full(len(w), .5) for w in weights[:-1]),
                          tuple(jnp.zeros_like(w).at[:, -1].set(.25) for w in weights[:-1]),
                          tuple(jnp.zeros_like(w) for w in weights[:-1]))
        return learner, teachers, rng
    return jax.vmap(one)(jnp.arange(nseeds))



def forward(weights, x, learner, rate):
    sources, means, slopes, covs = [], [], [], []
    for w, mean, slope, cov in zip(weights[:-1], learner.means, learner.slopes, learner.covs):
        source = jnp.append(x, 1.)
        sources.append(source[None])
        z = w @ source
        bits = ltu(z)
        mean = mean + rate * (jax.lax.stop_gradient(bits) - mean)
        sigmoid = jax.nn.sigmoid(z)
        derivative = jax.lax.stop_gradient((sigmoid * (1 - sigmoid))[:, None] * source)
        slope = slope + rate * (derivative - slope)
        correction = jnp.sum(slope * w, axis=1)
        centered = jax.lax.stop_gradient(bits - mean)
        h = bits - mean + (jax.lax.stop_gradient(correction) - correction)
        cov = cov + rate * (centered[:, None] * (derivative - slope) - cov)
        var = mean * (1 - mean)
        change = jnp.sum(cov * w, axis=1)
        floored = jnp.maximum(var, VAR_FLOOR)
        h = h - centered * jnp.where(var > VAR_FLOOR, (change - jax.lax.stop_gradient(change)) / floored, 0.)
        x = h / jnp.sqrt(floored)
        means.append(mean); slopes.append(slope); covs.append(cov)
    sources.append(jnp.append(x, 1.)[None])
    return weights[-1] @ sources[-1][0], (tuple(sources), tuple(means), tuple(slopes), tuple(covs))


def predict(learner, x):
    """Evaluation changes neither weights nor normalization statistics."""
    for w, mean in zip(learner.weights[:-1], learner.means):
        h = (w[:, :-1] @ x + w[:, -1] > 0).astype(jnp.float32)
        x = (h - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), VAR_FLOOR))
    return learner.weights[-1][:, :-1] @ x + learner.weights[-1][:, -1]


def update(learner, anchor, x, y, task, mode):
    # 0: ordinary training. 2: preserved hidden representation. 5: preserve old predictions.
    def loss(weights):
        prediction, aux = forward(weights, x, learner, jnp.where(mode == 2, 0., EMA_RATE))
        prediction = prediction.reshape(-1, OUTPUTS)
        active = .5 * jnp.sum((prediction[task] - y) ** 2)
        objective = active
        if anchor is not None:
            old = jax.lax.stop_gradient(predict(anchor, x).reshape(-1, OUTPUTS))
            inactive = (jnp.arange(prediction.shape[0]) != task)[:, None]
            preservation = .5 * jnp.sum(jnp.where(inactive, (prediction - old) ** 2, 0.))
            objective = objective + jnp.where(mode == 5, preservation, 0.)
        return objective, (aux, 2 * active / OUTPUTS)
    (_, ((sources, *stats), mse)), grads = jax.value_and_grad(loss, has_aux=True)(learner.weights)
    # Inactive readouts stay fixed, including under the preservation objective.
    active_rows = (jnp.arange(grads[-1].shape[0]) // OUTPUTS == task)[:, None]
    grads = (*grads[:-1], jnp.where(active_rows, grads[-1], 0.))
    updates, opt = OPT.update((grads, None, sources), learner.opt, learner.weights)
    weights = jax.tree.map(jnp.add, learner.weights, updates)
    weights = tuple(jnp.where(mode == 2, a, b) for a, b in zip(learner.weights[:-1], weights[:-1])) + (weights[-1],)
    return Learner(weights, opt, *stats), mse



@partial(jax.jit, static_argnames=('steps', 'active', 'period'))
def train(learners, anchor, teachers, keys, steps, active, modes, period=10_000):
    def one(learner, saved, teacher, key, mode):
        def sample(carry, i):
            learner, key = carry
            key, kx, kn = jax.random.split(key, 3)
            x = jax.random.rademacher(kx, (INPUTS,), dtype=jnp.float32)
            task = (i // period) % teacher[0].shape[0] if active == 'all' else jnp.asarray(int(active))
            y = target(jax.tree.map(lambda a: a[task], teacher), x) + jax.random.normal(kn, (OUTPUTS,))
            learner, mse = update(learner, saved, x, y, task, mode)
            return (learner, key), mse
        def window(carry, i):
            carry, mse = jax.lax.scan(sample, carry, i * 1000 + jnp.arange(1000))
            return carry, mse.mean()
        (learner, key), mse = jax.lax.scan(window, (learner, key), jnp.arange(steps // 1000))
        return learner, key, mse
    seed_batch = jax.vmap(one, in_axes=(0, 0, 0, 0, None))
    return jax.vmap(seed_batch, in_axes=(0, None, None, 0, 0))(learners, anchor, teachers, keys, modes)


@jax.jit
def evaluate(learners, teachers):
    """Expected noisy RMSE: sqrt(mean squared signal error + known noise variance)."""
    x = jax.random.rademacher(jax.random.PRNGKey(926), (4096, INPUTS), dtype=jnp.float32)
    def one(learner, teacher):
        predictions = jax.vmap(lambda a: predict(learner, a))(x).reshape(4096, -1, OUTPUTS)
        truth = jax.vmap(lambda t: jax.vmap(lambda a: target(t, a))(x))(teacher).transpose(1, 0, 2)
        return jnp.sqrt(1 + jnp.mean((predictions - truth) ** 2, axis=(0, 2)))
    return jax.vmap(jax.vmap(one, in_axes=(0, 0)), in_axes=(0, None))(learners, teachers)


def repeat(learner, n):
    return jax.tree.map(lambda a: jnp.stack([a] * n), learner)


def ready(result):
    return jax.block_until_ready(result)



@jax.custom_vjp
def local_standardize(bits, mean, error_mean, error_product_mean):
    return (bits - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), VAR_FLOOR))


def _standardize_forward(bits, mean, error_mean, error_product_mean):
    var = mean * (1 - mean)
    scale = jnp.sqrt(jnp.maximum(var, VAR_FLOOR))
    h = (bits - mean) / scale
    return h, (h, scale, var > VAR_FLOOR, error_mean, error_product_mean)


def _standardize_backward(saved, g):
    h, scale, scale_active, error_mean, error_product_mean = saved
    corrected = (g - error_mean - jnp.where(scale_active, h * error_product_mean, 0.)) / scale
    return corrected, None, None, None


local_standardize.defvjp(_standardize_forward, _standardize_backward)


def local_update(learner, traces, x, y):
    # Forward values match the baseline; replace its parameter-specific correction.
    sources, means, bits_list, activations = [], [], [], []
    for w, mean in zip(learner.weights[:-1], learner.means):
        source = jnp.append(x, 1.)
        bits = ltu(w @ source)
        mean = mean + EMA_RATE * (bits - mean)
        x = (bits - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), VAR_FLOOR))
        sources.append(source); means.append(mean)
        bits_list.append(bits); activations.append(x)
    source = jnp.append(x, 1.)
    error = (learner.weights[-1] @ source)[:OUTPUTS] - y
    output_error = jnp.pad(error, (0, learner.weights[-1].shape[0] - OUTPUTS))
    grads = [output_error[:, None] * source]
    g = learner.weights[-1][:, :-1].T @ output_error
    updated_traces = []
    for i in reversed(range(len(means))):
        eg, egh = traces[i]
        eg = eg + EMA_RATE * (g - eg)
        egh = egh + EMA_RATE * (g * activations[i] - egh)
        _, backward = jax.vjp(lambda b: local_standardize(b, means[i], eg, egh), bits_list[i])
        db = backward(g)[0]
        sigmoid = jax.nn.sigmoid(learner.weights[i] @ sources[i])
        dz = db * sigmoid * (1 - sigmoid)  # Existing LTU surrogate.
        grads.insert(0, dz[:, None] * sources[i])
        g = learner.weights[i][:, :-1].T @ dz
        updated_traces.insert(0, (eg, egh))
    inputs = tuple(a[None] for a in (*sources, source))
    updates, opt = OPT.update((tuple(grads), None, inputs), learner.opt, learner.weights)
    weights = jax.tree.map(jnp.add, learner.weights, updates)
    # Old derivative traces are unused by this rule.
    return Learner(weights, opt, tuple(means), learner.slopes, learner.covs), tuple(updated_traces)





@partial(jax.jit, static_argnames=('shared', 'nseeds', 'ntasks'))
def initialize_wide(shared, nseeds=5, ntasks=2):
    def one(seed):
        kt, kl, rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        kf, ks = jax.random.split(kt)
        groups = 1 if shared else ntasks
        first = jax.random.rademacher(kf, (groups, FEATURES, INPUTS), dtype=jnp.float32)
        first = jnp.repeat(first, ntasks, axis=0) if shared else first
        teachers = (first, *jax.vmap(make_task)(jax.random.split(ks, ntasks), first))
        widths = (45 * groups, 45 * ntasks)
        weights, fan_in = [], INPUTS
        for key, width in zip(jax.random.split(kl), widths):
            w = INIT_RATE * jax.random.uniform(key, (width, fan_in + 1), minval=-1., maxval=1.).at[:, -1].set(0.)
            weights.append(w)
            fan_in = width
        weights = (*weights, jnp.zeros((ntasks * OUTPUTS, fan_in + 1)))
        learner = Learner(weights, OPT.init(weights),
                          tuple(jnp.full(len(w), .5) for w in weights[:-1]),
                          tuple(jnp.zeros_like(w).at[:, -1].set(.25) for w in weights[:-1]),
                          tuple(jnp.zeros_like(w) for w in weights[:-1]))
        return learner, teachers, rng
    return jax.vmap(one)(jnp.arange(nseeds))
