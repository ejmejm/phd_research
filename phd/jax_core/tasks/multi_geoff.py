"""Multi-GEOFF: a multi-task, non-stationary regression testbed.

A synthetic replacement for multi-MNIST that preserves the structure needed to study
connectivity and structural credit assignment (outputs that should be grouped, outputs
that should not), but uses per-output squared error so the loss never leaks the grouping.

The target function is `n_tasks` independent GEOFF sub-networks ("slots") with
block-diagonal connectivity: no teacher weight connects any unit of one slot to any unit
of another. Within a slot, every readout is a linear map of the exact same set of hidden
LTUs, so within-slot relatedness is real but is available only in the data.
"""

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int

from ..utils import tree_replace


def output_weight_scale(n_hidden_per_task: int) -> float:
    """Output weight magnitude giving unit signal variance per output in expectation.

    Each LTU has a symmetric, continuous pre-activation, so it fires with probability
    exactly q = 1/2 and Var(h_j) = q(1 - q) = 1/4. With independent, zero-mean output
    weights of magnitude c the cross-covariance terms vanish in expectation, leaving
    E[Var(f_i)] = c^2 * n_hidden_per_task / 4. Setting that to 1 gives
    c = 2 / sqrt(n_hidden_per_task).

    Args:
        n_hidden_per_task: Number of hidden LTUs per slot (m)

    Returns:
        Output weight magnitude c
    """
    return 2.0 / math.sqrt(n_hidden_per_task)


def perturbation_period(per_task_period: int, n_tasks: int) -> int:
    """Perturbation period keeping the per-slot rate of change fixed as `n_tasks` scales.

    Perturbations hit a uniformly chosen slot, so a period of `per_task_period / n_tasks`
    perturbs each individual slot on average once every `per_task_period` steps regardless
    of how many slots there are.

    Args:
        per_task_period: Steps a given slot should go between perturbations on average (T)
        n_tasks: Number of slots (k)

    Returns:
        Number of steps between perturbations (tau)
    """
    return max(1, int(per_task_period // n_tasks))


class MultiGEOFFTask(eqx.Module):
    """Multi-task, non-stationary GEOFF regression task with block-diagonal teacher.

    The teacher is `n_tasks` independent slots. For slot t, inputs x^(t) are uniform on
    `input_bounds`, hidden units are LTUs with binary input weights U^(t) and zero
    thresholds, and every output is a dense linear readout of that slot's hidden code via
    V^(t), whose entries are in {-c, +c}:

        h^(t)_j = 1[U^(t)_j . x^(t) > 0]
        y^(t)_i = sum_j V^(t)_ij h^(t)_j + eps,    eps ~ N(0, noise_std^2)

    Inputs and outputs are laid out slot-major, so the flat input of dimension
    `n_tasks * n_features_per_task` reshapes to (n_tasks, n_features_per_task) and the
    flat output of dimension `n_tasks * n_outputs_per_task` reshapes to
    (n_tasks, n_outputs_per_task).

    The scale c (see `output_weight_scale`) is chosen so the noise-free signal has unit
    variance per output in expectation over problem constructions, so the irreducible MSE
    is `noise_std ** 2` and Var(y_i) = 1 + noise_std ** 2. Use
    `fraction_variance_explained` to convert an MSE into the scale-free metric rho, which
    is 1 for an optimal predictor and 0 for a mean predictor.

    Non-stationarity permutes the columns of one slot's V every `perturb_period` steps.
    The same permutation applies to all of the slot's readouts, so the perturbation is a
    relabeling of the slot's shared feature space: it can be corrected either by one change
    in the trunk or by a consistent change across all of the slot's readouts. It also
    preserves the multiset of weights in every row, so the signal variance is unchanged.
    """

    # Static parameters (configuration)
    n_tasks: int = eqx.field(static=True)
    n_features_per_task: int = eqx.field(static=True)
    n_hidden_per_task: int = eqx.field(static=True)
    n_outputs_per_task: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    noise_std: float = eqx.field(static=True)
    input_bounds: Tuple[float, float] = eqx.field(static=True)
    perturb_period: Optional[int] = eqx.field(static=True)

    # Dynamic parameters (weights and state)
    input_weights: Float[Array, 'n_tasks n_hidden_per_task n_features_per_task']
    output_weights: Float[Array, 'n_tasks n_outputs_per_task n_hidden_per_task']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_tasks: int,
        perturb_period: Optional[int] = None,
        n_features_per_task: int = 20,
        n_hidden_per_task: int = 10,
        n_outputs_per_task: int = 10,
        noise_std: float = 1.0,
        input_bounds: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_tasks: Number of independent slots (k)
            perturb_period: Number of steps between perturbations (None if stationary).
                See `perturbation_period` to convert a per-slot period into this value.
            n_features_per_task: Number of inputs per slot (n)
            n_hidden_per_task: Number of hidden LTUs per slot (m)
            n_outputs_per_task: Number of outputs per slot (p)
            noise_std: Standard deviation of the per-output observation noise
            input_bounds: Bounds of the uniform input distribution
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Store static configuration
        self.n_tasks = n_tasks
        self.n_features_per_task = n_features_per_task
        self.n_hidden_per_task = n_hidden_per_task
        self.n_outputs_per_task = n_outputs_per_task
        self.weight_scale = output_weight_scale(n_hidden_per_task)
        self.noise_std = noise_std
        self.input_bounds = tuple(input_bounds)
        self.perturb_period = perturb_period

        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        # Sample the block-diagonal teacher
        input_key, output_key, key = random.split(key, 3)
        self.input_weights = random.bernoulli(
            input_key, 0.5, (n_tasks, n_hidden_per_task, n_features_per_task),
        ).astype(jnp.float32)
        signs = random.randint(
            output_key, (n_tasks, n_outputs_per_task, n_hidden_per_task), 0, 2,
        ) * 2 - 1
        self.output_weights = signs.astype(jnp.float32) * self.weight_scale

        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

    @property
    def n_features(self) -> int:
        """Total input dimension across all slots."""
        return self.n_tasks * self.n_features_per_task

    @property
    def n_hidden(self) -> int:
        """Total hidden dimension of the teacher across all slots."""
        return self.n_tasks * self.n_hidden_per_task

    @property
    def n_outputs(self) -> int:
        """Total output dimension across all slots."""
        return self.n_tasks * self.n_outputs_per_task

    @property
    def input_slot_ids(self) -> Int[Array, 'n_features']:
        """Slot index of each input feature."""
        return jnp.repeat(jnp.arange(self.n_tasks), self.n_features_per_task)

    @property
    def output_slot_ids(self) -> Int[Array, 'n_outputs']:
        """Slot index of each output."""
        return jnp.repeat(jnp.arange(self.n_tasks), self.n_outputs_per_task)

    @property
    def irreducible_mse(self) -> float:
        """Per-output MSE of an optimal predictor."""
        return self.noise_std ** 2

    def _forward_hidden(
        self, x: Float[Array, 'n_features'],
    ) -> Float[Array, 'n_tasks n_hidden_per_task']:
        """LTU activations of every slot for a single sample."""
        x = x.reshape(self.n_tasks, self.n_features_per_task)
        pre_activations = jnp.einsum('tmn,tn->tm', self.input_weights, x)
        return (pre_activations > 0).astype(jnp.float32)

    def _forward(self, x: Float[Array, 'n_features']) -> Float[Array, 'n_outputs']:
        """Noise-free targets of a single sample, flattened over slots."""
        hidden = self._forward_hidden(x)
        targets = jnp.einsum('tpm,tm->tp', self.output_weights, hidden)
        return targets.reshape(self.n_outputs)

    def forward(
        self, x: Float[Array, 'batch_size n_features'],
    ) -> Float[Array, 'batch_size n_outputs']:
        """Noise-free targets f of a batch of inputs."""
        return jax.vmap(self._forward)(x)

    def hidden_features(
        self, x: Float[Array, 'batch_size n_features'],
    ) -> Float[Array, 'batch_size n_hidden']:
        """Teacher LTU features of a batch of inputs, flattened over slots."""
        hidden = jax.vmap(self._forward_hidden)(x)
        return hidden.reshape(hidden.shape[0], self.n_hidden)

    def fraction_variance_explained(
        self, mse: Float[Array, '...'],
    ) -> Float[Array, '...']:
        """Fraction of explainable variance explained by a predictor with the given MSE.

        rho = 1 - (MSE - noise_var) / signal_var, which is 1 for an optimal predictor and
        0 for a mean predictor. Signal variance is 1 per output by construction, so with
        the default unit noise this is simply 2 - MSE.

        Args:
            mse: Per-output mean squared error, averaged over outputs and steps

        Returns:
            rho, bounded above by 1 and comparable across `n_tasks` and network size
        """
        return 1.0 - (mse - self.irreducible_mse)

    def _perturb(
        self, key: random.PRNGKey,
    ) -> Float[Array, 'n_tasks n_outputs_per_task n_hidden_per_task']:
        """Permutes the columns of a uniformly chosen slot's output weights.

        Args:
            key: PRNG key

        Returns:
            The perturbed output weights
        """
        slot_key, perm_key = random.split(key)
        slot = random.randint(slot_key, (), 0, self.n_tasks)

        # The chosen slot gets the permutation, every other slot gets the identity
        permutation = random.permutation(perm_key, self.n_hidden_per_task)
        identity = jnp.arange(self.n_hidden_per_task)
        is_target_slot = (jnp.arange(self.n_tasks) == slot)[:, None]
        permutations = jnp.where(is_target_slot, permutation[None, :], identity[None, :])

        return jax.vmap(lambda v, p: v[:, p])(self.output_weights, permutations)

    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.

        `step` counts samples, so with the intended online single-sample updates a slot is
        perturbed exactly every `perturb_period` steps. At most one perturbation is applied
        per batch, so `batch_size` may not exceed `perturb_period`.

        Args:
            batch_size: Size of batch to generate

        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y), with y including observation noise
        """
        assert self.perturb_period is None or batch_size <= self.perturb_period, \
            f"batch_size ({batch_size}) exceeds perturb_period ({self.perturb_period}), " \
            "which would silently drop perturbations"

        new_rng, perturb_key, x_key, noise_key = random.split(self.rng, 4)
        new_step = self.step + batch_size

        output_weights = self.output_weights
        if self.perturb_period is not None:
            # True whenever a multiple of `perturb_period` falls inside this batch
            is_due = (self.step // self.perturb_period) != (new_step // self.perturb_period)
            output_weights = jnp.where(is_due, self._perturb(perturb_key), output_weights)

        new_task_state: MultiGEOFFTask = tree_replace(
            self,
            output_weights = output_weights,
            step = new_step,
            rng = new_rng,
        )

        # Generate random input features
        x = random.uniform(
            x_key, (batch_size, self.n_features), jnp.float32, *self.input_bounds,
        )

        # Forward pass through the target network, then add per-output noise
        y = new_task_state.forward(x)
        y = y + self.noise_std * random.normal(noise_key, y.shape)

        return new_task_state, (x, y)
