from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from phd.jax_core.optimizers.idbd import optax_idbd, IDBDState
from phd.jax_core.utils import tree_replace


DEFAULT_HYPERPARAMETERS = {
    'learning_rate': 2.0 ** -2,
    'meta_learning_rate': 2.0 ** -4,
    'replace_rate': 1e-3,
    'decay_rate': 0.999,
}


def linear_loss_and_grads(w, x, y):
    """Squared-error loss of the linear prediction and its exact gradient wrt `w`."""
    y_hat = jnp.sum(w * x[None, :])
    error = y_hat - y
    return error ** 2, 2.0 * error * x[None, :]


def reset_opt_state(opt_state, prune_mask):
    """Reset the Autostep (IDBD) step-size and traces for pruned (regenerated) features."""
    m = prune_mask[None, :]  # broadcast over the single output row -> (1, n_features)
    if isinstance(opt_state, IDBDState):
        return IDBDState(
            init_beta=opt_state.init_beta,
            beta=jnp.where(m, opt_state.init_beta, opt_state.beta),  # step-size -> initial
            h=jnp.where(m, 0.0, opt_state.h),
            v=jnp.where(m, 0.0, opt_state.v),
        )
    return opt_state


def recycle_features(w, opt_state, util, accum, replace_rate):
    """Continual-backprop recycling. Accumulate the replacement budget (`replace_rate` * n
    per call) and, once it reaches one whole feature, recycle the lowest-utility slot(s):
    zero the weight, reset the optimizer state, and set the utility to the median so the
    fresh slot is not immediately re-pruned. Returns the updated
    `(w, opt_state, util, accum, prune_mask)`; `prune_mask` marks the slots the task should
    regenerate on the next observation."""
    n = util.shape[0]
    accum = accum + replace_rate * n
    n_avail = jnp.floor(accum).astype(jnp.int32)
    rank = jnp.argsort(jnp.argsort(util))       # 0 = lowest utility
    prune_mask = rank < n_avail
    accum = accum - jnp.sum(prune_mask)
    w = jnp.where(prune_mask[None, :], 0.0, w)
    util = jnp.where(prune_mask, jnp.median(util), util)
    opt_state = reset_opt_state(opt_state, prune_mask)
    return w, opt_state, util, accum, prune_mask


class CBPAutostep(eqx.Module):
    """CBP + Autostep — continual-backprop feature recycling on an Autostep (IDBD) base."""
    DEFAULTS: ClassVar[dict] = DEFAULT_HYPERPARAMETERS

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    replace_rate: float = eqx.field(static=True)
    decay_rate: float = eqx.field(static=True)
    w: jax.Array
    opt_state: optax.OptState
    util: jax.Array
    accum: jax.Array
    prune_mask: jax.Array

    @classmethod
    def init(cls, input_dim, hparams, key):
        optimizer = optax_idbd(init_lr=hparams['learning_rate'],
                               meta_lr=hparams['meta_learning_rate'], autostep=True)
        w = jnp.zeros((1, input_dim))
        return cls(optimizer=optimizer, replace_rate=hparams['replace_rate'],
                   decay_rate=hparams['decay_rate'], w=w, opt_state=optimizer.init(w),
                   util=jnp.zeros(input_dim),
                   accum=jnp.array(hparams['replace_rate'] * input_dim),
                   prune_mask=jnp.zeros(input_dim, dtype=bool))

    def step(self, x, y):
        loss, loss_grads = linear_loss_and_grads(self.w, x, y)
        # Autostep/IDBD consumes both the loss gradient and the prediction gradient (= x).
        updates, opt_state = self.optimizer.update((loss_grads, x[None, :]), self.opt_state, self.w)
        w = optax.apply_updates(self.w, updates)
        util = (1.0 - self.decay_rate) * (jnp.abs(w[0]) * jnp.abs(x)) + self.decay_rate * self.util
        w, opt_state, util, accum, prune_mask = recycle_features(
            w, opt_state, util, self.accum, self.replace_rate)
        return tree_replace(self, w=w, opt_state=opt_state, util=util,
                            accum=accum, prune_mask=prune_mask), loss
