"""Simple multi-GEOFF: multi-GEOFF with every input and teacher weight a random sign.

The target function is `n_tasks` independent GEOFF teachers with block-diagonal connectivity, as in
`MultiGEOFFTask`, but the inputs, the LTUs' input weights and the output weights are all independent
random +-1 signs, and every LTU threshold is 0. Only the hidden activations (0 or 1), the outputs and
the noisy targets take other values.
"""

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int

from ..utils import tree_replace


class SimpleMultiGEOFFTask(eqx.Module):
    """Multi-task, optionally non-stationary GEOFF regression task with +-1 inputs and weights.

    For task t, with every entry of x^(t), U^(t) and V^(t) uniform on {-1, +1}:

        h^(t)_j = 1[U^(t)_j . x^(t) > 0]
        y^(t)_i = sum_j V^(t)_ij h^(t)_j + eps,    eps ~ N(0, noise_std^2)

    With an odd `n_features_per_task` a pre-activation is never 0, so every LTU fires with probability
    exactly 1/2, and each output's noise-free signal has variance `n_hidden_per_task / 4` in expectation
    over teachers. Inputs and outputs are laid out task-major: the flat input reshapes to
    (n_tasks, n_features_per_task) and the flat output to (n_tasks, n_outputs_per_task).

    Non-stationarity is as in `MultiGEOFFTask`: every `perturb_period` steps, the columns of one
    uniformly chosen task's V are permuted.
    """

    n_tasks: int = eqx.field(static=True)
    n_features_per_task: int = eqx.field(static=True)
    n_hidden_per_task: int = eqx.field(static=True)
    n_outputs_per_task: int = eqx.field(static=True)
    noise_std: float = eqx.field(static=True)
    perturb_period: Optional[int] = eqx.field(static=True)

    input_weights: Float[Array, 'n_tasks n_hidden_per_task n_features_per_task']
    output_weights: Float[Array, 'n_tasks n_outputs_per_task n_hidden_per_task']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_tasks: int,
        perturb_period: Optional[int] = None,
        n_features_per_task: int = 21,
        n_hidden_per_task: int = 10,
        n_outputs_per_task: int = 10,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_tasks: Number of independent tasks
            perturb_period: Number of steps between perturbations (None if stationary)
            n_features_per_task: Number of inputs per task; odd, so no LTU's pre-activation is ever 0
            n_hidden_per_task: Number of hidden LTUs per task
            n_outputs_per_task: Number of outputs per task
            noise_std: Standard deviation of the per-output observation noise
            seed: Random seed for reproducibility
        """
        self.n_tasks = n_tasks
        self.n_features_per_task = n_features_per_task
        self.n_hidden_per_task = n_hidden_per_task
        self.n_outputs_per_task = n_outputs_per_task
        self.noise_std = noise_std
        self.perturb_period = perturb_period

        if seed is None:
            seed = np.random.randint(0, 2**31)
        input_key, output_key, key = random.split(random.PRNGKey(seed), 3)
        self.input_weights = random.rademacher(
            input_key, (n_tasks, n_hidden_per_task, n_features_per_task), jnp.float32)
        self.output_weights = random.rademacher(
            output_key, (n_tasks, n_outputs_per_task, n_hidden_per_task), jnp.float32)
        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

    def forward(self, x: Float[Array, 'batch_size n_features']) -> Float[Array, 'batch_size n_outputs']:
        """Noise-free targets of a batch of inputs."""
        x = x.reshape(x.shape[0], self.n_tasks, self.n_features_per_task)
        hidden = (jnp.einsum('tmn,btn->btm', self.input_weights, x) > 0).astype(jnp.float32)
        return jnp.einsum('tpm,btm->btp', self.output_weights, hidden).reshape(x.shape[0], -1)

    def _perturb(self, key: random.PRNGKey) -> Float[Array, 'n_tasks n_outputs_per_task n_hidden_per_task']:
        """The output weights with the columns of a uniformly chosen task's V permuted."""
        task_key, perm_key = random.split(key)
        task = random.randint(task_key, (), 0, self.n_tasks)
        permutation = random.permutation(perm_key, self.n_hidden_per_task)
        identity = jnp.arange(self.n_hidden_per_task)
        is_target_task = (jnp.arange(self.n_tasks) == task)[:, None]
        permutations = jnp.where(is_target_task, permutation[None, :], identity[None, :])
        return jax.vmap(lambda v, p: v[:, p])(self.output_weights, permutations)

    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a batch (x, y), with y including observation noise, and the new task state.

        `step` counts samples, so with online single-sample updates a task is perturbed exactly every
        `perturb_period` steps. At most one perturbation is applied per batch.
        """
        assert self.perturb_period is None or batch_size <= self.perturb_period, \
            f'batch_size ({batch_size}) exceeds perturb_period ({self.perturb_period})'

        new_rng, perturb_key, x_key, noise_key = random.split(self.rng, 4)
        new_step = self.step + batch_size

        output_weights = self.output_weights
        if self.perturb_period is not None:
            # True whenever a multiple of `perturb_period` falls inside this batch
            is_due = (self.step // self.perturb_period) != (new_step // self.perturb_period)
            output_weights = jnp.where(is_due, self._perturb(perturb_key), output_weights)

        new_task_state = tree_replace(self, output_weights=output_weights, step=new_step, rng=new_rng)
        x = random.rademacher(x_key, (batch_size, self.n_tasks * self.n_features_per_task), jnp.float32)
        y = new_task_state.forward(x)
        y = y + self.noise_std * random.normal(noise_key, y.shape)
        return new_task_state, (x, y)
