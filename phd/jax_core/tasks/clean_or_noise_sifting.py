"""Clean-or-noise sifting: a feature-search task whose candidates are perfect or worthless.

A variant of :class:`~phd.jax_core.tasks.feature_sifting.FeatureSiftingTask` that changes one
thing — what a freshly drawn candidate feature is. The original mixes a true signal with pure
noise through a per-candidate noise coefficient drawn uniformly from [0, 1], so the candidate
pool is a *continuum* of partial views and a learner can make steady progress by trading a
noisier feature for a slightly cleaner one. Here a draw is a coin flip instead: with probability
``clean_prob`` the candidate is an **exact, uncorrupted copy** of one of the true signals, and
otherwise it is an **independent uniform draw** over the same range, carrying no information at
all. Nothing sits in between.

That makes the search discrete. There is no gradient of feature quality to follow and no way to
recognise a good candidate other than by using it: a clean copy and a distractor have the same
marginal distribution, so they are indistinguishable from a single observation. The learner's
only handle is the association with the target, which it has to discover by keeping a candidate
around long enough for its weight to move. With the default ``clean_prob`` of 0.05, 19 out of 20
draws are worthless, so the question the task asks is how cheaply a method can reject a
distractor and how reliably it holds on to a real signal once it lucks into one.

``with_replacement`` chooses which true signal a clean candidate copies, and the two settings
pose meaningfully different problems:

- **Without replacement** (the default): a clean candidate takes a signal that no other slot is
  currently holding, so the learner never ends up with two copies of the same signal. Every
  clean slot it accumulates is a signal newly covered, and the pool it is searching for shrinks
  as it succeeds. At most ``min(n_learner_features, n_target_features)`` slots can be clean at
  once (see ``max_clean_slots``); once every signal is spoken for, a candidate that draws
  "clean" finds nothing free and comes out as noise instead.
- **With replacement**: as in the original task, indices are drawn independently, so several
  slots may hold copies of the same signal. Duplicates are not harmful to the prediction — the
  weight simply splits between them — but they are wasted slots, and a pruning rule that only
  asks whether a weight is dead has no reason to cut one, since a split weight is not dead.

Either way a signal may go uncovered indefinitely. An optimal predictor needs a copy of all
``n_target_features`` signals somewhere among its slots; with ``k`` of them covered the best
reachable MSE is ``target_noise_std ** 2 + (n_target_features - k) * Var(x)``, since each
missing signal leaves its own variance in the residual.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Bool, Float, Int

from ..utils import tree_replace


class CleanOrNoiseSiftingTask(eqx.Module):
    """An online feature-selection task whose candidates are either exact signals or pure noise.

    A fixed target function generates each sample's label as a weighted sum of
    ``n_target_features`` true signals drawn uniformly from ``input_bounds``, with weights fixed
    at +1 or -1 and optional Gaussian noise on the output.

    The learner never sees the true signals directly. It observes ``n_learner_features``
    candidate slots, each of which was drawn as either an exact copy of one true signal (with
    probability ``clean_prob``) or an independent uniform draw over the same range (otherwise).
    Both look identical in isolation — same range, same marginal distribution — so the only way
    to tell them apart is by what they predict.

    On each ``step`` the learner supplies a ``prune_mask`` marking slots to discard; every marked
    slot is redrawn from scratch, taking a new clean/noise coin flip and a new signal. The task
    is therefore a continual generate-and-test problem in which the *test* is the whole
    difficulty: replacements are cheap and plentiful but almost always useless.

    ``with_replacement`` decides whether two slots may hold copies of the same signal; see the
    module docstring for what each setting is like to search.
    """

    # Static parameters (configuration)
    n_target_features: int = eqx.field(static=True)
    n_learner_features: int = eqx.field(static=True)
    clean_prob: float = eqx.field(static=True)
    with_replacement: bool = eqx.field(static=True)
    target_noise_std: float = eqx.field(static=True)
    input_bounds: Tuple[float, float] = eqx.field(static=True)
    flip_period: int = eqx.field(static=True)

    # Dynamic parameters (weights and state)
    target_idxs: Int[Array, 'n_learner_features']
    is_clean: Bool[Array, 'n_learner_features']
    weights: Float[Array, 'n_target_features']
    step_idx: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_target_features: int = 10,
        n_learner_features: int = 20,
        clean_prob: float = 0.05,
        with_replacement: bool = False,
        target_noise_std: float = 0.0,
        input_bounds: Tuple[float, float] = (-1.0, 1.0),
        flip_period: int = -1,
        key: Optional[random.PRNGKey] = None,
    ):
        """
        Args:
            n_target_features: Number of true signals the target function sums
            n_learner_features: Number of candidate slots the learner observes
            clean_prob: Probability that a drawn candidate is an exact copy of a true signal
                rather than an independent uniform draw
            with_replacement: Whether a clean candidate may copy a signal another slot already
                holds. False (the default) draws only from the signals no slot is currently
                holding, so the learner never accumulates duplicates and a draw that finds every
                signal taken comes out as noise instead. True reproduces the original task's
                independent draws.
            target_noise_std: Standard deviation of the observation noise on the target output
            input_bounds: Bounds of the uniform distribution, shared by the true signals and the
                distractors so the two are indistinguishable from their values alone
            flip_period: If > 0, the sign of one randomly chosen target weight is flipped every
                `flip_period` steps (including the first). Disabled when <= 0.
            key: PRNG key for the initial candidates, the teacher, and every redraw
        """
        assert 0.0 <= clean_prob <= 1.0, f"clean_prob ({clean_prob}) must be a probability!"

        self.n_target_features = n_target_features
        self.n_learner_features = n_learner_features
        self.clean_prob = clean_prob
        self.with_replacement = with_replacement
        self.target_noise_std = target_noise_std
        self.input_bounds = tuple(input_bounds)
        self.flip_period = flip_period
        self.step_idx = jnp.array(0, dtype=jnp.int32)

        # Set up RNG
        if key is None:
            key = random.PRNGKey(np.random.randint(0, 2**31))
        self.rng, feature_key, weight_key = random.split(key, 3)

        # Initial candidates, drawn exactly as replacements are: an empty pool with every slot
        # up for a draw, so without replacement they come out distinct.
        self.target_idxs, self.is_clean = self._redraw(
            feature_key,
            jnp.ones(self.n_learner_features, dtype=bool),
            jnp.zeros(self.n_learner_features, dtype=jnp.int32),
            jnp.zeros(self.n_learner_features, dtype=bool),
        )

        # Target weights, random -1 or +1
        self.weights = random.bernoulli(weight_key, 0.5, (self.n_target_features,)) * 2.0 - 1.0

    @property
    def input_variance(self) -> float:
        """Variance of one uniform input feature, true signal or distractor alike."""
        low, high = self.input_bounds
        return (high - low) ** 2 / 12.0

    @property
    def max_clean_slots(self) -> int:
        """Most slots that can hold a real signal at once.

        Without replacement the signals are handed out one to a slot, so the pool cannot hold
        more clean slots than there are signals however many slots the learner has.
        """
        if self.with_replacement:
            return self.n_learner_features
        return min(self.n_learner_features, self.n_target_features)

    @property
    def irreducible_mse(self) -> float:
        """MSE of a predictor holding a copy of every true signal."""
        return self.target_noise_std ** 2

    @property
    def baseline_mse(self) -> float:
        """MSE of the mean predictor, which is the floor with no signal covered at all."""
        return self.mse_floor(0)

    @property
    def covered(self) -> Bool[Array, 'n_target_features']:
        """True signals with at least one exact copy among the learner's slots.

        Redundant copies buy nothing, so this — not the number of clean slots — is what bounds
        how well the learner can do; see `mse_floor`. Without replacement the two coincide, no
        copy ever being redundant.
        """
        hits = jnp.zeros(self.n_target_features, jnp.int32)
        hits = hits.at[self.target_idxs].max(self.is_clean.astype(jnp.int32))
        return hits > 0

    @property
    def n_covered(self) -> Int[Array, '']:
        """Number of distinct true signals the learner currently has a copy of."""
        return jnp.sum(self.covered)

    def mse_floor(self, n_covered: Int[Array, '...']) -> Float[Array, '...']:
        """Best MSE reachable while covering `n_covered` of the true signals.

        The signals are independent and enter the target with unit-magnitude weights, so each
        one the learner cannot see leaves exactly its own variance in the residual — no
        combination of distractors can predict it, their being independent of everything.

        Args:
            n_covered: Number of distinct true signals covered

        Returns:
            The MSE floor, which is `irreducible_mse` when every signal is covered
        """
        missing = self.n_target_features - n_covered
        return self.irreducible_mse + missing * self.input_variance

    def _redraw(
        self,
        key: random.PRNGKey,
        prune_mask: Bool[Array, 'n_learner_features'],
        target_idxs: Int[Array, 'n_learner_features'],
        is_clean: Bool[Array, 'n_learner_features'],
    ) -> Tuple[Int[Array, 'n_learner_features'], Bool[Array, 'n_learner_features']]:
        """Draws a fresh candidate for every slot in `prune_mask`, leaving the rest untouched.

        Args:
            key: PRNG key
            prune_mask: Slots to redraw
            target_idxs: Current signal index of each slot
            is_clean: Which slots currently hold an exact copy

        Returns:
            Tuple containing:
            - The new signal index of each slot (meaningless for a noise slot)
            - Which slots now hold an exact copy
        """
        idx_key, clean_key, order_key, priority_key = random.split(key, 4)

        # Every slot draws a coin and an index; only the pruned ones use them. The index of a
        # noise candidate is drawn all the same and simply goes unused, which keeps the draw
        # shape-uniform and independent of the flips.
        new_is_clean = random.bernoulli(clean_key, self.clean_prob, (self.n_learner_features,))
        new_target_idxs = random.randint(
            idx_key, (self.n_learner_features,), 0, self.n_target_features,
        )

        if self.with_replacement:
            return (jnp.where(prune_mask, new_target_idxs, target_idxs),
                    jnp.where(prune_mask, new_is_clean, is_clean))

        # Without replacement, a clean candidate may only take a signal no *surviving* slot
        # holds; a pruned slot releases its own signal back, so it can be redrawn.
        kept = is_clean & ~prune_mask
        taken = jnp.zeros(self.n_target_features, jnp.int32).at[target_idxs].max(
            kept.astype(jnp.int32)) > 0

        # The free signals in a uniformly random order, the taken ones sorted past the end.
        scores = jnp.where(taken, jnp.inf, random.uniform(order_key, (self.n_target_features,)))
        free_order = jnp.argsort(scores)
        n_free = jnp.sum(~taken)

        # Slots that drew "clean" queue for one free signal each. The queue is ordered at random
        # rather than by slot index, so which slots go without when the signals run out does not
        # depend on where they sit in the pool.
        wants = prune_mask & new_is_clean
        priority = random.uniform(priority_key, (self.n_learner_features,))
        rank = jnp.sum(wants[None, :] & (priority[None, :] < priority[:, None]), axis=-1)
        served = wants & (rank < n_free)

        # A slot that wanted a signal and found none left comes out as noise.
        drawn_idxs = free_order[jnp.minimum(rank, self.n_target_features - 1)]
        return (jnp.where(served, drawn_idxs, jnp.where(prune_mask, new_target_idxs, target_idxs)),
                jnp.where(prune_mask, served, is_clean))

    def step(
        self,
        prune_mask: Bool[Array, 'n_learner_features'],
    ) -> Tuple['CleanOrNoiseSiftingTask', Tuple[Float[Array, 'n_learner_features'], Float[Array, '']]]:
        """Redraws every marked candidate, then generates one sample.

        Args:
            prune_mask: Slots to discard, each replaced by a fresh draw

        Returns:
            Tuple containing:
            - The new task state, whose `is_clean` describes the learner's new candidates
            - (learner_features, target_output):
                - learner_features: Array of shape (n_learner_features,), one value per slot
                - target_output: Scalar, the weighted sum of the true signals plus noise
        """
        new_rng, flip_key, draw_key, x_key, target_noise_key, distractor_key = random.split(self.rng, 6)

        # Periodically flip the sign of one randomly chosen target weight
        weights = self.weights
        if self.flip_period > 0:
            should_flip = (self.step_idx % self.flip_period) == 0
            flip_idx = random.randint(flip_key, (1,), 0, self.n_target_features)
            flipped = weights.at[flip_idx].set(-weights[flip_idx])
            weights = jnp.where(should_flip, flipped, weights)

        # Replace the pruned slots with fresh draws
        target_idxs, is_clean = self._redraw(
            draw_key, prune_mask, self.target_idxs, self.is_clean,
        )

        new_state: 'CleanOrNoiseSiftingTask' = tree_replace(
            self,
            weights = weights,
            target_idxs = target_idxs,
            is_clean = is_clean,
            step_idx = self.step_idx + 1,
            rng = new_rng,
        )

        # The true signals and the target they generate
        low, high = self.input_bounds
        target_features = random.uniform(x_key, (self.n_target_features,), minval=low, maxval=high)
        target_output = jnp.sum(target_features * weights)
        target_output += random.normal(target_noise_key, ()) * self.target_noise_std

        # A slot is either an exact copy of its signal or an independent draw over the same range
        distractors = random.uniform(
            distractor_key, (self.n_learner_features,), minval=low, maxval=high,
        )
        learner_features = jnp.where(is_clean, target_features[target_idxs], distractors)

        return new_state, (learner_features, target_output)
