"""Multi-linear: a multi-task, non-stationary linear regression testbed.

The linear counterpart of `multi_geoff.py` — the same block-diagonal, multi-task structure and
the same scale conventions, but the teacher has no hidden layer, so every output is a linear
function of its own slot's inputs. A linear learner wired to the right inputs therefore reaches
the irreducible MSE exactly: nothing about the function class stands between a method and the
noise floor, so what the MSE measures is the structure the method found.

The target function is `n_tasks` independent linear sub-problems ("slots") with block-diagonal
connectivity: no teacher weight connects any unit of one slot to any unit of another, so a
cross-slot input carries no information at all about an output.
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


def input_variance(input_bounds: Tuple[float, float] = (-1.0, 1.0)) -> float:
    """Variance of one uniform input feature.

    Args:
        input_bounds: Bounds of the uniform input distribution

    Returns:
        Var(x_j)
    """
    low, high = input_bounds
    return (high - low) ** 2 / 12.0


def output_weight_scale(
    n_features_per_task: int, input_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Output weight magnitude giving unit signal variance per output in expectation.

    Inputs are independent, so with zero-mean output weights of magnitude c the cross-covariance
    terms vanish in expectation, leaving E[Var(f_i)] = c^2 * n * Var(x). Setting that to 1 gives
    c = 1 / sqrt(n * Var(x)), which holds the signal variance — and with it the whole MSE scale —
    fixed as `input_bounds` changes.

    Args:
        n_features_per_task: Number of inputs per slot (n)
        input_bounds: Bounds of the uniform input distribution

    Returns:
        Output weight magnitude c
    """
    return 1.0 / math.sqrt(n_features_per_task * input_variance(input_bounds))


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


class MultiLinearTask(eqx.Module):
    """Multi-task, non-stationary linear regression task with a block-diagonal teacher.

    The teacher is `n_tasks` independent slots. For slot t, inputs x^(t) are uniform on
    `input_bounds` and every output is a dense linear readout of that slot's inputs via
    W^(t), whose entries are in {-c, +c}:

        y^(t)_i = sum_j W^(t)_ij x^(t)_j + eps,    eps ~ N(0, noise_std^2)

    Inputs and outputs are laid out slot-major, so the flat input of dimension
    `n_tasks * n_features_per_task` reshapes to (n_tasks, n_features_per_task) and the flat
    output of dimension `n_tasks * n_outputs_per_task` reshapes to
    (n_tasks, n_outputs_per_task).

    The scale c (see `output_weight_scale`) is chosen so the noise-free signal has unit variance
    per output in expectation over problem constructions, whatever `input_bounds` is, so the
    irreducible MSE is `noise_std ** 2` and Var(y_i) = 1 + noise_std ** 2. Use
    `fraction_variance_explained` to convert an MSE into the scale-free metric rho, which is 1
    for an optimal predictor and 0 for a mean predictor.

    Non-stationarity permutes the columns of one slot's weights every `perturb_period` steps,
    relabeling which of that slot's inputs each weight applies to. The same permutation applies
    to all of the slot's readouts, so it is a relabeling of the slot's shared input space rather
    than an independent change per output. It also preserves the multiset of weights in every
    row, so the signal variance is unchanged.
    """

    # Static parameters (configuration)
    n_tasks: int = eqx.field(static=True)
    n_features_per_task: int = eqx.field(static=True)
    n_outputs_per_task: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    noise_std: float = eqx.field(static=True)
    input_bounds: Tuple[float, float] = eqx.field(static=True)
    perturb_period: Optional[int] = eqx.field(static=True)

    # Dynamic parameters (weights and state)
    output_weights: Float[Array, 'n_tasks n_outputs_per_task n_features_per_task']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_tasks: int,
        perturb_period: Optional[int] = None,
        n_features_per_task: int = 20,
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
            n_outputs_per_task: Number of outputs per slot (p)
            noise_std: Standard deviation of the per-output observation noise
            input_bounds: Bounds of the uniform input distribution
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Store static configuration
        self.n_tasks = n_tasks
        self.n_features_per_task = n_features_per_task
        self.n_outputs_per_task = n_outputs_per_task
        self.weight_scale = output_weight_scale(n_features_per_task, input_bounds)
        self.noise_std = noise_std
        self.input_bounds = tuple(input_bounds)
        self.perturb_period = perturb_period

        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        # Sample the block-diagonal teacher
        weight_key, key = random.split(key)
        signs = random.randint(
            weight_key, (n_tasks, n_outputs_per_task, n_features_per_task), 0, 2,
        ) * 2 - 1
        self.output_weights = signs.astype(jnp.float32) * self.weight_scale

        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

    @property
    def n_features(self) -> int:
        """Total input dimension across all slots."""
        return self.n_tasks * self.n_features_per_task

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

    def _forward(self, x: Float[Array, 'n_features']) -> Float[Array, 'n_outputs']:
        """Noise-free targets of a single sample, flattened over slots."""
        x = x.reshape(self.n_tasks, self.n_features_per_task)
        targets = jnp.einsum('tpn,tn->tp', self.output_weights, x)
        return targets.reshape(self.n_outputs)

    def forward(
        self, x: Float[Array, 'batch_size n_features'],
    ) -> Float[Array, 'batch_size n_outputs']:
        """Noise-free targets f of a batch of inputs."""
        return jax.vmap(self._forward)(x)

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
            rho, bounded above by 1 and comparable across `n_tasks` and problem size
        """
        return 1.0 - (mse - self.irreducible_mse)

    def _perturb(
        self, key: random.PRNGKey,
    ) -> Float[Array, 'n_tasks n_outputs_per_task n_features_per_task']:
        """Permutes the columns of a uniformly chosen slot's weights.

        Args:
            key: PRNG key

        Returns:
            The perturbed output weights
        """
        slot_key, perm_key = random.split(key)
        slot = random.randint(slot_key, (), 0, self.n_tasks)

        # The chosen slot gets the permutation, every other slot gets the identity
        permutation = random.permutation(perm_key, self.n_features_per_task)
        identity = jnp.arange(self.n_features_per_task)
        is_target_slot = (jnp.arange(self.n_tasks) == slot)[:, None]
        permutations = jnp.where(is_target_slot, permutation[None, :], identity[None, :])

        return jax.vmap(lambda w, p: w[:, p])(self.output_weights, permutations)

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

        new_task_state: MultiLinearTask = tree_replace(
            self,
            output_weights = output_weights,
            step = new_step,
            rng = new_rng,
        )

        # Generate random input features
        x = random.uniform(
            x_key, (batch_size, self.n_features), jnp.float32, *self.input_bounds,
        )

        # Forward pass through the target function, then add per-output noise
        y = new_task_state.forward(x)
        y = y + self.noise_std * random.normal(noise_key, y.shape)

        return new_task_state, (x, y)
