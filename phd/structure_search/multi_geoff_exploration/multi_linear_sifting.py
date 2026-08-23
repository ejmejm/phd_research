"""Feature sifting on multi-linear: rewiring a linear learner into a fixed feature pool.

Wraps :class:`MultiLinearTask` into the feature-sifting setting of
``phd/jax_core/tasks/feature_sifting.py``: each step the learner marks connections for
removal, and every removed connection is immediately replaced by a fresh one. Two things
differ from the original. The candidate pool is *fixed* — it is the problem's own
``n_tasks * n_features_per_task`` inputs — rather than procedurally generated, so sifting
means discovering which of the real features each output should read. And the features are
handed over **unmodified**: the original mixes every candidate with pure noise through its
own noise coefficient, whereas here the only noise in the problem is the observation noise
on each target, so a feature is either exactly what the teacher used or exactly irrelevant.

A connection is one (output, feature) pair, so an output is never wired to the same
feature twice. Connections are therefore stored padded, as the full
``(n_outputs, n_features)`` grid with exactly ``n_connections`` entries active; the
learner keeps a weight per grid position and the inactive ones are held inert by zeroing
their input. A replacement picks a uniformly random output and then a uniformly random
feature that output is not already connected to, so the total stays fixed while the
number of connections *per output* drifts over a run.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Bool, Float

from phd.jax_core.tasks.multi_linear import MultiLinearTask
from phd.jax_core.utils import tree_replace


class MultiLinearSifting(eqx.Module):
    """A multi-linear task the learner may rewire, one connection at a time.

    ``step`` takes the learner's ``prune_mask`` over the connection grid, replaces every
    marked connection, and returns the next sample. The connection count is conserved, so
    the learner always has exactly ``n_connections`` weights doing work.
    """

    # Static parameters (configuration)
    n_connections: int = eqx.field(static=True)

    # Dynamic parameters (state)
    task: MultiLinearTask
    active: Bool[Array, 'n_outputs n_features']
    rng: random.PRNGKey

    def __init__(
        self,
        task: MultiLinearTask,
        n_connections_per_output: int,
        key: Optional[random.PRNGKey] = None,
    ):
        """
        Args:
            task: The multi-linear problem supplying the feature pool and the targets
            n_connections_per_output: Connections each output starts with, drawn without
                replacement from the pool. The total is conserved from then on, but the
                per-output counts are not.
            key: PRNG key for the initial wiring and for every replacement
        """
        self.task = task
        self.n_connections = task.n_outputs * n_connections_per_output

        # Every replacement needs a free feature on the output it picked. Keeping the total
        # no larger than the pool guarantees one always exists (see `step`).
        assert self.n_connections <= task.n_features, \
            f"{self.n_connections} connections exceeds the pool of {task.n_features} features!"

        if key is None:
            key = random.PRNGKey(np.random.randint(0, 2 ** 31))
        self.rng, init_key = random.split(key)

        # Initial wiring: `n_connections_per_output` distinct features per output, at random.
        scores = random.uniform(init_key, (task.n_outputs, task.n_features))
        ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
        self.active = ranks < n_connections_per_output

    @property
    def useful(self) -> Bool[Array, 'n_outputs n_features']:
        """Grid positions whose feature and output belong to the same task.

        The teacher is block-diagonal, so a cross-task feature carries no information about
        an output at all: the fraction of active connections that are useful measures how
        good a wiring the learner has found.
        """
        return self.task.output_slot_ids[:, None] == self.task.input_slot_ids[None, :]

    def step(
        self,
        prune_mask: Bool[Array, 'n_outputs n_features'],
    ) -> Tuple['MultiLinearSifting', Tuple[Float[Array, 'n_outputs n_features'], Float[Array, 'n_outputs']]]:
        """Replaces every marked connection, then generates one sample.

        Args:
            prune_mask: Active connections to discard, over the (output, feature) grid

        Returns:
            Tuple containing:
            - The new task state, whose `active` is the learner's new wiring
            - (inputs, y):
                - inputs: (n_outputs, n_features) feature values, zeroed where the output
                  is not connected, so an inactive position is inert for the learner
                - y: (n_outputs,) noisy targets
        """
        rng, output_key, feature_key = random.split(self.rng, 3)
        n_outputs = self.active.shape[0]

        surviving = self.active & ~prune_mask
        n_new = jnp.sum(prune_mask)

        # Every replacement draws a uniformly random output and then a uniformly random
        # feature that output is free of. The draws are padded out to `n_connections` (the
        # most that can ever be pruned at once) and only the first `n_new` are used.
        outputs = random.randint(output_key, (self.n_connections,), 0, n_outputs)
        used = jnp.arange(self.n_connections) < n_new

        # Each output's free features in a uniformly random order; occupied ones sort last.
        scores = jnp.where(surviving, jnp.inf, random.uniform(feature_key, surviving.shape))
        free_order = jnp.argsort(scores, axis=-1)

        # Draws landing on the same output take successive free features, so two replacements
        # never collide. A draw's rank is how many earlier used draws share its output, which
        # stays below that output's free count because survivors plus replacements can never
        # exceed `n_connections <= n_features`.
        idxs = jnp.arange(self.n_connections)
        shares_output = (outputs[:, None] == outputs[None, :]) & (idxs[:, None] > idxs[None, :])
        rank = jnp.sum(shares_output & used[None, :], axis=-1)
        features = free_order[outputs, rank]

        added = jnp.zeros(surviving.shape, jnp.int32).at[outputs, features].max(used.astype(jnp.int32))
        active = surviving | (added > 0)

        task, (x, y) = self.task.generate_batch(1)
        state: 'MultiLinearSifting' = tree_replace(self, task=task, active=active, rng=rng)
        return state, (jnp.where(active, x[0][None, :], 0.0), y[0])
