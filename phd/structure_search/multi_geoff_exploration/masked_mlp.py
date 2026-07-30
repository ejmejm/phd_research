"""Masked two-layer MLP and connectivity builders for the multi-GEOFF experiments.

Shared by Experiment 1 (does Autostep separate useful from useless connections?) and
Experiment 2 (learn a good structure with online connectivity changes). Everything here is
about *which connections exist* and *which of them are known to be useless by construction*.

Design notes
------------
* Connectivity is a float 0/1 mask multiplied into the weights at forward time, the same
  representation `PaddedMLP` uses in ``experiments/train_weight_pruning.py``. Arbitrary
  per-connection topologies are expressible, and adding / removing a single connection is a
  ``.at[i, j].set(...)`` on the mask.
* The trainable weights live in a plain ``dict`` (``{'w1': ..., 'w2': ...}``) that is handed to
  the optimizer directly, so the masks never enter the optimizer's parameter tree and no
  ``eqx.filter`` / ``filter_spec`` plumbing is needed. The same dict structure is what
  ``param_inputs`` uses for Autostep's ``version='squared_inputs'``.
* Masked-out connections are inert under Autostep: their loss gradient and curvature term are
  both zero, so ``v`` stays 0, ``jnp.where(v != 0, ...)`` leaves the step-size at its initial
  value, ``h`` stays 0, and the parameter update is 0. Activating a connection is therefore
  just: flip the mask, zero the weight, and reset that position's optimizer state
  (`reset_idbd_state_at`).
* No biases anywhere — Autostep asserts against them.
"""

from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int

from phd.jax_core.models import ACTIVATION_MAP, lecun_uniform
from phd.jax_core.optimizers.idbd import IDBDState


# Group labels used for the useful / useless split. `INACTIVE` marks positions where no
# connection exists, so they can be excluded from every statistic.
INACTIVE = -1
USEFUL = 0
USELESS = 1

# Hidden-unit group labels for the Experiment 1a contamination.
GROUP_EXTRA_OUTPUTS = 0
GROUP_EXTRA_INPUTS = 1
GROUP_CLEAN = 2


Masks = Dict[str, Float[Array, '...']]
Weights = Dict[str, Float[Array, '...']]


class MaskedTwoLayerMLP(eqx.Module):
    """Two-layer MLP with an arbitrary per-connection connectivity mask and no biases.

    The forward pass returns ``(output, param_inputs)`` following the convention every model in
    this package uses. ``param_inputs`` maps each weight matrix to the value of its source units
    — the network input for ``w1`` and the hidden activations for ``w2`` — which is exactly the
    curvature basis Autostep's ``version='squared_inputs'`` consumes. Leaves are shaped
    ``(1, in_features)`` and broadcast over the output-row axis, matching the convention of the
    original torch implementation.
    """

    weights: Weights
    masks: Masks
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        masks: Masks,
        activation: str = 'leaky_relu',
        *,
        key: random.PRNGKey,
    ):
        """
        Args:
            masks: Connectivity masks, ``{'w1': (n_hidden, n_features), 'w2': (n_outputs, n_hidden)}``
            activation: Hidden activation, a key of `ACTIVATION_MAP`
            key: PRNG key

        Both layers use `lecun_uniform`. Do not zero-initialize the second layer: that makes the
        *prediction* gradients of the first layer identically zero, pinning the first layer's
        step-sizes at their initial value under Autostep's ``version='prediction_grads'``.
        """
        assert set(masks) == {'w1', 'w2'}, f"Expected masks for w1 and w2, got {sorted(masks)}!"
        assert activation in ACTIVATION_MAP, f"Unknown activation: {activation}!"

        w1_key, w2_key = random.split(key)
        # Masked-out weights are held at exactly zero so a mask flip cannot resurrect a stale
        # value, and so weight statistics over inactive positions are unambiguous.
        self.weights = {
            'w1': lecun_uniform(w1_key, masks['w1'].shape) * masks['w1'],
            'w2': lecun_uniform(w2_key, masks['w2'].shape) * masks['w2'],
        }
        self.masks = {k: m.astype(jnp.float32) for k, m in masks.items()}
        self.activation = activation

    @property
    def activation_fn(self):
        return ACTIVATION_MAP[self.activation]

    def __call__(
        self, x: Float[Array, 'n_features'],
    ) -> Tuple[Float[Array, 'n_outputs'], Weights]:
        """Forward pass of a single sample, returning the output and the per-weight source values."""
        hidden = self.activation_fn((self.weights['w1'] * self.masks['w1']) @ x)
        output = (self.weights['w2'] * self.masks['w2']) @ hidden
        return output, {'w1': x[None, :], 'w2': hidden[None, :]}


# --------------------------------------------------------------------------- connectivity
def slot_ids(n_tasks: int, per_task: int) -> Int[Array, 'n_tasks*per_task']:
    """Task index of each unit in a slot-major layout, matching `MultiGEOFFTask`."""
    return jnp.repeat(jnp.arange(n_tasks), per_task)


def block_sparse_masks(
    n_tasks: int,
    n_features_per_task: int,
    n_outputs_per_task: int,
    n_hidden: int,
) -> Tuple[Masks, Int[Array, 'n_hidden']]:
    """Block-sparse connectivity: each hidden unit sees only its own task's inputs and outputs.

    Hidden units are split as evenly as possible across tasks and laid out slot-major, so the
    task of hidden unit ``j`` is ``hidden_task_ids[j]``. This is the reference good structure —
    every hidden unit's connections belong to a single subtask, so any cross-task connection
    added on top of it is known to be useless.

    Args:
        n_tasks: Number of subtasks (k)
        n_features_per_task: Inputs per subtask
        n_outputs_per_task: Outputs per subtask
        n_hidden: Total hidden units, split across tasks

    Returns:
        Tuple of the masks and the per-hidden-unit task ids
    """
    assert n_hidden % n_tasks == 0, \
        f"n_hidden ({n_hidden}) must divide evenly across n_tasks ({n_tasks})!"
    hidden_per_task = n_hidden // n_tasks

    hidden_task_ids = slot_ids(n_tasks, hidden_per_task)
    input_task_ids = slot_ids(n_tasks, n_features_per_task)
    output_task_ids = slot_ids(n_tasks, n_outputs_per_task)

    w1 = (hidden_task_ids[:, None] == input_task_ids[None, :]).astype(jnp.float32)
    w2 = (output_task_ids[:, None] == hidden_task_ids[None, :]).astype(jnp.float32)
    return {'w1': w1, 'w2': w2}, hidden_task_ids


def thirds_group_ids(
    hidden_task_ids: Int[Array, 'n_hidden'], n_tasks: int,
) -> Int[Array, 'n_hidden']:
    """Split each task's hidden units into three contamination groups as evenly as possible.

    The split is done *within* each task so every task is contaminated the same way. With 64
    hidden units over 2 tasks that is 32 per task split 10 / 11 / 11.

    Args:
        hidden_task_ids: Task index of each hidden unit
        n_tasks: Number of subtasks

    Returns:
        A group id in {`GROUP_EXTRA_OUTPUTS`, `GROUP_EXTRA_INPUTS`, `GROUP_CLEAN`} per hidden unit
    """
    group_ids = jnp.zeros_like(hidden_task_ids)
    for task in range(n_tasks):
        idxs = jnp.where(hidden_task_ids == task)[0]
        n = len(idxs)
        # Sizes differ by at most one; the earlier groups take the remainder.
        bounds = [(n * i) // 3 for i in range(4)]
        for group, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            group_ids = group_ids.at[idxs[lo:hi]].set(group)
    return group_ids


def contaminate_masks(
    masks: Masks,
    hidden_task_ids: Int[Array, 'n_hidden'],
    group_ids: Int[Array, 'n_hidden'],
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Masks:
    """The Experiment 1a modification of a block-sparse base structure.

    Hidden units in `GROUP_EXTRA_OUTPUTS` are additionally fully connected to every *other*
    task's outputs; those in `GROUP_EXTRA_INPUTS` to every other task's inputs;
    `GROUP_CLEAN` units are left alone. Because the base structure gives each hidden unit a
    single task, every connection added here is useless by construction.

    Args:
        masks: Block-sparse base masks
        hidden_task_ids: Task index of each hidden unit
        group_ids: Contamination group of each hidden unit (see `thirds_group_ids`)
        input_task_ids: Task index of each input feature
        output_task_ids: Task index of each output

    Returns:
        The contaminated masks
    """
    cross_in = hidden_task_ids[:, None] != input_task_ids[None, :]
    cross_out = output_task_ids[:, None] != hidden_task_ids[None, :]

    extra_in = cross_in & (group_ids == GROUP_EXTRA_INPUTS)[:, None]
    extra_out = cross_out & (group_ids == GROUP_EXTRA_OUTPUTS)[None, :]

    return {
        'w1': jnp.maximum(masks['w1'], extra_in.astype(jnp.float32)),
        'w2': jnp.maximum(masks['w2'], extra_out.astype(jnp.float32)),
    }


def add_random_cross_task_connections(
    masks: Masks,
    hidden_task_ids: Int[Array, 'n_hidden'],
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
    key: random.PRNGKey,
    n_out_range: Tuple[int, int] = (1, 3),
    n_in_range: Tuple[int, int] = (1, 10),
) -> Tuple[Masks, Masks]:
    """The Experiment 1b injection: give every hidden unit some cross-task connections.

    Per hidden unit, a uniformly sampled ``n_out`` in ``n_out_range`` outgoing connections to
    other tasks' outputs and ``n_in`` in ``n_in_range`` incoming connections from other tasks'
    inputs, with the specific endpoints sampled without replacement. Ranges are inclusive.

    Each affected hidden unit then reads some of another task's inputs *and* feeds some of its
    outputs, so the added connections are bad but not strictly useless — there is some incentive
    for their weights to move off zero. The best solution still holds them all at 0.

    Args:
        masks: Current masks (expected to be the block-sparse structure)
        hidden_task_ids: Task index of each hidden unit
        input_task_ids: Task index of each input feature
        output_task_ids: Task index of each output
        key: PRNG key
        n_out_range: Inclusive range for the number of added outgoing connections per hidden unit
        n_in_range: Inclusive range for the number of added incoming connections per hidden unit

    Returns:
        Tuple of the new masks and the mask of *newly added* connections per layer
    """
    n_hidden = hidden_task_ids.shape[0]
    in_key, out_key = random.split(key)

    def sample_row(row_key, eligible, count_range):
        """A 0/1 row selecting `n` of the eligible positions, `n` uniform in `count_range`."""
        count_key, order_key = random.split(row_key)
        n = random.randint(count_key, (), count_range[0], count_range[1] + 1)
        # Rank the eligible positions in a random order and keep the first `n` of them. Random
        # ranking is how sampling without replacement is done branchlessly under `jit`.
        noise = jnp.where(eligible, random.uniform(order_key, eligible.shape), jnp.inf)
        rank = jnp.argsort(jnp.argsort(noise))
        return (eligible & (rank < n)).astype(jnp.float32)

    # Incoming: rows of w1 are hidden units, columns are input features.
    cross_in = (hidden_task_ids[:, None] != input_task_ids[None, :]) & (masks['w1'] == 0)
    new_w1 = jax.vmap(sample_row, in_axes=(0, 0, None))(
        random.split(in_key, n_hidden), cross_in, n_in_range)

    # Outgoing: rows of w2 are outputs, columns are hidden units, so sample per *column* and
    # transpose back.
    cross_out = (output_task_ids[:, None] != hidden_task_ids[None, :]) & (masks['w2'] == 0)
    new_w2 = jax.vmap(sample_row, in_axes=(0, 0, None))(
        random.split(out_key, n_hidden), cross_out.T, n_out_range).T

    new_masks = {'w1': new_w1, 'w2': new_w2}
    return (
        {k: jnp.maximum(masks[k], new_masks[k]) for k in masks},
        new_masks,
    )


def connection_labels(
    masks: Masks,
    hidden_task_ids: Int[Array, 'n_hidden'],
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Dict[str, Int[Array, '...']]:
    """Label every position `INACTIVE`, `USEFUL`, or `USELESS`.

    A connection is useless exactly when it crosses tasks. For a hidden unit that only reads its
    own task's inputs, an outgoing cross-task connection carries no information about the task it
    feeds; for one that only feeds its own task's outputs, an incoming cross-task connection
    injects nothing but noise.

    Args:
        masks: Current connectivity masks
        hidden_task_ids: Task index of each hidden unit
        input_task_ids: Task index of each input feature
        output_task_ids: Task index of each output

    Returns:
        Per-layer integer label arrays with the same shapes as the masks
    """
    cross = {
        'w1': hidden_task_ids[:, None] != input_task_ids[None, :],
        'w2': output_task_ids[:, None] != hidden_task_ids[None, :],
    }
    return {
        k: jnp.where(masks[k] > 0, jnp.where(cross[k], USELESS, USEFUL), INACTIVE)
        for k in masks
    }


def added_connection_labels(
    masks: Masks, new_masks: Masks,
) -> Dict[str, Int[Array, '...']]:
    """Label positions by whether they are pre-existing (`USEFUL`) or newly added (`USELESS`).

    Used by Experiment 1b, where the split of interest is original-vs-added rather than
    same-task-vs-cross-task (in that setup the two coincide, because the pre-existing structure
    is pure block-sparse and every added connection crosses tasks).
    """
    return {
        k: jnp.where(masks[k] > 0, jnp.where(new_masks[k] > 0, USELESS, USEFUL), INACTIVE)
        for k in masks
    }


# ------------------------------------------------------------------------ optimizer state
def reset_idbd_state_at(
    opt_state: IDBDState, reset_masks: Masks,
) -> IDBDState:
    """Put the Autostep state of the selected connections back to its initial value.

    Step-size back to ``init_lr``, gradient trace ``h`` and normalizer ``v`` to zero — the same
    treatment continual backprop gives a recycled feature, applied here to newly activated
    connections.

    Args:
        opt_state: Current `IDBDState`, with trees matching the weight dict
        reset_masks: Per-layer 0/1 masks selecting the positions to reset

    Returns:
        The updated `IDBDState`
    """
    select = {k: m > 0 for k, m in reset_masks.items()}
    return IDBDState(
        init_beta=opt_state.init_beta,
        beta=jax.tree.map(
            lambda s, b: jnp.where(s, opt_state.init_beta, b), select, opt_state.beta),
        h=jax.tree.map(lambda s, h: jnp.where(s, 0.0, h), select, opt_state.h),
        v=None if opt_state.v is None else
          jax.tree.map(lambda s, v: jnp.where(s, 0.0, v), select, opt_state.v),
    )


# ------------------------------------------------------------------------------ statistics
def masked_quantiles(
    values: Float[Array, '...'],
    labels: Int[Array, '...'],
    group: int,
    quantiles: Tuple[float, ...],
) -> Float[Array, 'n_quantiles']:
    """Quantiles of `values` over the positions labelled `group`.

    Positions outside the group are pushed to ``+inf`` and the quantile is taken over the first
    ``count`` entries of the sorted array. Doing it this way keeps the shape static, so it works
    inside ``jax.lax.scan`` — a boolean-indexed selection would not. Returns NaN for an empty
    group.

    Args:
        values: Per-connection values (step-sizes or absolute weights)
        labels: Per-connection group labels from `connection_labels`
        group: The label to select
        quantiles: Quantiles in [0, 1]

    Returns:
        One value per requested quantile
    """
    in_group = labels == group
    count = jnp.sum(in_group)
    sorted_vals = jnp.sort(jnp.where(in_group, values, jnp.inf).reshape(-1))

    def at_q(q):
        # Nearest-rank quantile over the `count` real entries at the front of `sorted_vals`.
        idx = jnp.clip(jnp.floor(q * (count - 1) + 0.5).astype(jnp.int32), 0, sorted_vals.size - 1)
        return jnp.where(count > 0, sorted_vals[idx], jnp.nan)

    return jnp.stack([at_q(q) for q in quantiles])


def subsample_indices(
    labels: Int[Array, '...'], group: int, n: int, seed: int = 0,
) -> Optional[Int[Array, 'n 2']]:
    """Pick `n` random positions labelled `group`, as (row, col) index pairs.

    Called once outside of `jit` to fix which individual connections are traced, so the traces
    follow the same connections for the whole run.

    Args:
        labels: Per-connection group labels
        group: The label to sample from
        n: Number of positions to sample (capped at the group size)
        seed: Random seed

    Returns:
        An ``(n, 2)`` index array, or None if the group is empty
    """
    rows, cols = jnp.where(labels == group)
    if rows.size == 0:
        return None
    n = min(n, int(rows.size))
    pick = random.choice(random.PRNGKey(seed), rows.size, (n,), replace=False)
    return jnp.stack([rows[pick], cols[pick]], axis=-1)


def gather_at(values: Float[Array, '...'], idxs: Int[Array, 'n 2']) -> Float[Array, 'n']:
    """Values at the given (row, col) index pairs."""
    return values[idxs[:, 0], idxs[:, 1]]
