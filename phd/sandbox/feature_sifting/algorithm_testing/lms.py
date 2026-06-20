from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from phd.jax_core.utils import tree_replace


DEFAULT_HYPERPARAMETERS = {
    'learning_rate': 2.0 ** -12,
}


def linear_loss_and_grads(w, x, y):
    """Squared-error loss of the linear prediction and its exact gradient wrt `w`."""
    y_hat = jnp.sum(w * x[None, :])
    error = y_hat - y
    return error ** 2, 2.0 * error * x[None, :]


class LMS(eqx.Module):
    """LMS — SGD on a fixed candidate set; never recycles features."""
    DEFAULTS: ClassVar[dict] = DEFAULT_HYPERPARAMETERS
    
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    w: jax.Array
    opt_state: optax.OptState
    prune_mask: jax.Array

    @classmethod
    def init(cls, input_dim, hparams, key):
        optimizer = optax.sgd(hparams['learning_rate'])
        w = jnp.zeros((1, input_dim))
        return cls(optimizer=optimizer, w=w, opt_state=optimizer.init(w),
                   prune_mask=jnp.zeros(input_dim, dtype=bool))

    def step(self, x, y):
        loss, grads = linear_loss_and_grads(self.w, x, y)
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.w)
        w = optax.apply_updates(self.w, updates)
        return tree_replace(self, w=w, opt_state=opt_state), loss   # prune_mask stays all-False


# METHODS['lms'] = MethodSpec('lms', 'LMS', LMS, LMS.DEFAULTS)