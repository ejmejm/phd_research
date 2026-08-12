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
* Biases (``bias=True``) are ordinary weights on a constant-1 input appended to each layer's
  input vector, never a separate parameter array. Autostep normalizes each output unit's
  step-sizes by ``sum_i alpha_i x_i^2`` taken over the last axis of a weight matrix, so a bias
  has to sit *in* that matrix for its own ``alpha_b * 1`` to enter the sum and for both to be
  divided by the same normalizer. A separate 1-D bias leaf gets its own normalizer reduced over
  the wrong axis, which is why `optax_idbd` asserts against bias parameters outright.
"""

from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int

from phd.jax_core.models import ACTIVATION_MAP, lecun_uniform
from phd.jax_core.optimizers.idbd import IDBDState


# Group labels used for the useful / useless split. `INACTIVE` marks positions where no
# connection exists, so they can be excluded from every statistic. `BIAS` marks the weights on
# the constant-1 input: they are trained like any other weight, but they belong to no task and so
# fall outside the useful / useless comparison.
INACTIVE = -1
USEFUL = 0
USELESS = 1
BIAS = 2

# Hidden-unit group labels for the Experiment 1a contamination.
GROUP_EXTRA_OUTPUTS = 0
GROUP_EXTRA_INPUTS = 1

# Task id of the constant-1 input column, which belongs to no task.
BIAS_TASK = -1


Masks = Dict[str, Float[Array, '...']]
Weights = Dict[str, Float[Array, '...']]


class MaskedTwoLayerMLP(eqx.Module):
    """Two-layer MLP with an arbitrary per-connection connectivity mask.

    The forward pass returns ``(output, param_inputs)`` following the convention every model in
    this package uses. ``param_inputs`` maps each weight matrix to the value of its source units
    — the network input for ``w1`` and the hidden activations for ``w2`` — which is exactly the
    curvature basis Autostep's ``version='squared_inputs'`` consumes. Leaves are shaped
    ``(1, in_features)`` and broadcast over the output-row axis, matching the convention of the
    original torch implementation.

    With ``bias=True`` a constant 1 is appended to each layer's input, so every weight matrix
    carries one extra column and the biases are ordinary weights inside it. There is no separate
    bias array: Autostep normalizes over a weight matrix's last axis, so a bias only gets folded
    into its output unit's normalizer if it lives in that matrix.
    """

    weights: Weights
    masks: Masks
    activation: str = eqx.field(static=True)
    bias: bool = eqx.field(static=True)

    def __init__(
        self,
        masks: Masks,
        activation: str = 'leaky_relu',
        bias: bool = False,
        *,
        key: random.PRNGKey,
    ):
        """
        Args:
            masks: Connectivity masks, ``{'w1': (n_hidden, n_features), 'w2': (n_outputs, n_hidden)}``,
                each with one extra trailing column for the constant-1 input when `bias`
            activation: Hidden activation, a key of `ACTIVATION_MAP`
            bias: Whether the last column of each mask is the constant-1 input
            key: PRNG key

        Both layers use `lecun_uniform`. Do not zero-initialize the second layer: that makes the
        *prediction* gradients of the first layer identically zero, pinning the first layer's
        step-sizes at their initial value under Autostep's ``version='prediction_grads'``.
        """
        assert set(masks) == {'w1', 'w2'}, f"Expected masks for w1 and w2, got {sorted(masks)}!"
        assert activation in ACTIVATION_MAP, f"Unknown activation: {activation}!"

        w1_key, w2_key = random.split(key)
        weights = {
            'w1': lecun_uniform(w1_key, masks['w1'].shape),
            'w2': lecun_uniform(w2_key, masks['w2'].shape),
        }
        if bias:
            # Biases start at 0 as usual. Only `w2`'s *hidden* columns enter the first layer's
            # prediction gradients, so this is not the zero-initialization hazard warned about
            # above — it leaves the network's initial function exactly what it is without biases.
            weights = {k: w.at[:, -1].set(0.0) for k, w in weights.items()}

        # Masked-out weights are held at exactly zero so a mask flip cannot resurrect a stale
        # value, and so weight statistics over inactive positions are unambiguous.
        self.weights = {k: w * masks[k] for k, w in weights.items()}
        self.masks = {k: m.astype(jnp.float32) for k, m in masks.items()}
        self.activation = activation
        self.bias = bias

    @property
    def activation_fn(self):
        return ACTIVATION_MAP[self.activation]

    def _with_bias(self, values: Float[Array, 'n']) -> Float[Array, 'n+1']:
        """Append the constant-1 input a layer's bias column reads from."""
        return jnp.append(values, 1.0) if self.bias else values

    def __call__(
        self, x: Float[Array, 'n_features'],
    ) -> Tuple[Float[Array, 'n_outputs'], Weights]:
        """Forward pass of a single sample, returning the output and the per-weight source values."""
        x = self._with_bias(x)
        hidden = self._with_bias(
            self.activation_fn((self.weights['w1'] * self.masks['w1']) @ x))
        output = (self.weights['w2'] * self.masks['w2']) @ hidden
        return output, {'w1': x[None, :], 'w2': hidden[None, :]}


# --------------------------------------------------------------------------- connectivity
def slot_ids(n_tasks: int, per_task: int) -> Int[Array, 'n_tasks*per_task']:
    """Task index of each unit in a slot-major layout, matching `MultiGEOFFTask`."""
    return jnp.repeat(jnp.arange(n_tasks), per_task)


def col_task_ids(unit_task_ids: Int[Array, 'n_units'], n_cols: int) -> Int[Array, 'n_cols']:
    """Task id of each *column* of a weight matrix, given the task of each source unit.

    One column wider than there are source units means the layer has a bias, so the trailing
    column reads the constant-1 input and gets `BIAS_TASK`. Deriving this from the mask width is
    what keeps the bias out of every builder's signature below.

    Args:
        unit_task_ids: Task index of each source unit (inputs for ``w1``, hidden units for ``w2``)
        n_cols: Number of columns of the corresponding weight matrix

    Returns:
        A task id per column
    """
    n_units = unit_task_ids.shape[0]
    assert n_cols in (n_units, n_units + 1), \
        f"Expected {n_units} columns, or {n_units + 1} with a bias, got {n_cols}!"
    return unit_task_ids if n_cols == n_units else jnp.append(unit_task_ids, BIAS_TASK)


def cross_task(
    row_task_ids: Int[Array, 'n_rows'], col_task_ids: Int[Array, 'n_cols'],
) -> Array:
    """Mark the (row, col) positions whose two endpoints belong to different tasks.

    A bias column belongs to no task and is never a cross-task connection, so ``~cross_task(...)``
    is exactly the within-task structure *including* the biases.
    """
    return (row_task_ids[:, None] != col_task_ids[None, :]) & (col_task_ids[None, :] != BIAS_TASK)


def block_sparse_masks(
    n_tasks: int,
    n_features_per_task: int,
    n_outputs_per_task: int,
    n_hidden: int,
    bias: bool = False,
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
        bias: Append a constant-1 input column to each layer, connected to every unit of that
            layer. Biases belong to no task, so they are part of the good structure.

    Returns:
        Tuple of the masks and the per-hidden-unit task ids (the bias column is not a hidden unit,
        so `hidden_task_ids` stays ``n_hidden`` long either way)
    """
    assert n_hidden % n_tasks == 0, \
        f"n_hidden ({n_hidden}) must divide evenly across n_tasks ({n_tasks})!"
    hidden_per_task = n_hidden // n_tasks

    hidden_task_ids = slot_ids(n_tasks, hidden_per_task)
    input_task_ids = slot_ids(n_tasks, n_features_per_task)
    output_task_ids = slot_ids(n_tasks, n_outputs_per_task)

    n_w1_cols = n_tasks * n_features_per_task + int(bias)
    n_w2_cols = n_hidden + int(bias)
    w1 = ~cross_task(hidden_task_ids, col_task_ids(input_task_ids, n_w1_cols))
    w2 = ~cross_task(output_task_ids, col_task_ids(hidden_task_ids, n_w2_cols))
    return {'w1': w1.astype(jnp.float32), 'w2': w2.astype(jnp.float32)}, hidden_task_ids


def halves_group_ids(
    hidden_task_ids: Int[Array, 'n_hidden'],
) -> Int[Array, 'n_hidden']:
    """Assign each hidden unit to a contamination group using a repeating cycle: 0, 1, 0, 1, ...

    Half of the hidden units get extra outgoing cross-task connections, the other half extra
    incoming ones; no unit is left clean.

    Args:
        hidden_task_ids: Task index of each hidden unit

    Returns:
        A group id in {`GROUP_EXTRA_OUTPUTS`, `GROUP_EXTRA_INPUTS`} per hidden unit
    """
    n_hidden = hidden_task_ids.shape[0]
    return jnp.arange(n_hidden) % 2


def contaminate_masks(
    masks: Masks,
    hidden_task_ids: Int[Array, 'n_hidden'],
    group_ids: Int[Array, 'n_hidden'],
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Masks:
    """The Experiment 1a modification of a block-sparse base structure.

    Hidden units in `GROUP_EXTRA_OUTPUTS` are additionally fully connected to every *other*
    task's outputs, and those in `GROUP_EXTRA_INPUTS` to every other task's inputs. Because the
    base structure gives each hidden unit a single task, every connection added here is useless
    by construction.

    Args:
        masks: Block-sparse base masks
        hidden_task_ids: Task index of each hidden unit
        group_ids: Contamination group of each hidden unit (see `halves_group_ids`)
        input_task_ids: Task index of each input feature
        output_task_ids: Task index of each output

    Returns:
        The contaminated masks
    """
    cross_in = cross_task(hidden_task_ids, col_task_ids(input_task_ids, masks['w1'].shape[1]))
    cross_out = cross_task(output_task_ids, col_task_ids(hidden_task_ids, masks['w2'].shape[1]))

    # `cross_out`'s columns are hidden units plus a possible bias column, which no group owns.
    out_units = jnp.pad(group_ids == GROUP_EXTRA_OUTPUTS,
                        (0, cross_out.shape[1] - group_ids.shape[0]))

    extra_in = cross_in & (group_ids == GROUP_EXTRA_INPUTS)[:, None]
    extra_out = cross_out & out_units[None, :]

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

    # Incoming: rows of w1 are hidden units, columns are input features (plus a possible bias).
    cross_in = cross_task(
        hidden_task_ids, col_task_ids(input_task_ids, masks['w1'].shape[1])) & (masks['w1'] == 0)
    new_w1 = jax.vmap(sample_row, in_axes=(0, 0, None))(
        random.split(in_key, n_hidden), cross_in, n_in_range)

    # Outgoing: rows of w2 are outputs, columns are hidden units, so sample per *column* and
    # transpose back. A bias column is eligible nowhere, so its row samples nothing.
    n_w2_cols = masks['w2'].shape[1]
    cross_out = cross_task(
        output_task_ids, col_task_ids(hidden_task_ids, n_w2_cols)) & (masks['w2'] == 0)
    new_w2 = jax.vmap(sample_row, in_axes=(0, 0, None))(
        random.split(out_key, n_w2_cols), cross_out.T, n_out_range).T

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
    """Label every position `INACTIVE`, `USEFUL`, `USELESS`, or `BIAS`.

    A connection is useless exactly when it crosses tasks. For a hidden unit that only reads its
    own task's inputs, an outgoing cross-task connection carries no information about the task it
    feeds; for one that only feeds its own task's outputs, an incoming cross-task connection
    injects nothing but noise. Bias weights belong to no task and get their own label, so they
    stay out of both sides of that comparison.

    Args:
        masks: Current connectivity masks
        hidden_task_ids: Task index of each hidden unit
        input_task_ids: Task index of each input feature
        output_task_ids: Task index of each output

    Returns:
        Per-layer integer label arrays with the same shapes as the masks
    """
    cols = {'w1': col_task_ids(input_task_ids, masks['w1'].shape[1]),
            'w2': col_task_ids(hidden_task_ids, masks['w2'].shape[1])}
    rows = {'w1': hidden_task_ids, 'w2': output_task_ids}
    return {
        k: jnp.where(
            masks[k] > 0,
            jnp.where((cols[k] == BIAS_TASK)[None, :], BIAS,
                      jnp.where(cross_task(rows[k], cols[k]), USELESS, USEFUL)),
            INACTIVE)
        for k in masks
    }


def added_connection_labels(
    masks: Masks, new_masks: Masks, bias: bool = False,
) -> Dict[str, Int[Array, '...']]:
    """Label positions by whether they are pre-existing (`USEFUL`) or newly added (`USELESS`).

    Used by Experiment 1b, where the split of interest is original-vs-added rather than
    same-task-vs-cross-task (in that setup the two coincide, because the pre-existing structure
    is pure block-sparse and every added connection crosses tasks). Bias weights are pre-existing
    but get the `BIAS` label, matching `connection_labels`.

    Args:
        masks: Current connectivity masks
        new_masks: Mask of the newly added connections
        bias: Whether the last column of each mask is the constant-1 input
    """
    labels = {
        k: jnp.where(masks[k] > 0, jnp.where(new_masks[k] > 0, USELESS, USEFUL), INACTIVE)
        for k in masks
    }
    if bias:
        labels = {k: v.at[:, -1].set(jnp.where(masks[k][:, -1] > 0, BIAS, INACTIVE))
                  for k, v in labels.items()}
    return labels


def penalty_masks(masks: Masks, bias: bool = False) -> Masks:
    """The masks a weight penalty should use: every active connection except the biases.

    An L1 penalty on a bias fights the target function's nonzero mean rather than pruning any
    structure, so biases are left out of it the same way weight decay conventionally is.

    Args:
        masks: Current connectivity masks
        bias: Whether the last column of each mask is the constant-1 input

    Returns:
        The masks, with the bias column zeroed when there is one
    """
    return {k: m.at[:, -1].set(0.0) for k, m in masks.items()} if bias else dict(masks)


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
) -> Int[Array, 'n 2']:
    """Pick up to `n` random positions labelled `group`, as (row, col) index pairs.

    Called once outside of `jit` to fix which individual connections are traced, so the traces
    follow the same connections for the whole run.

    Groups are usually smaller than `n`, and different groups have different sizes, so the result
    is always padded out to exactly `n` rows with the sentinel ``-1``. A fixed length is what lets
    the per-group traces be stacked into one array; `gather_at` turns the padding into NaN.

    Args:
        labels: Per-connection group labels
        group: The label to sample from
        n: Number of positions to return, padding included
        seed: Random seed

    Returns:
        An ``(n, 2)`` index array whose trailing rows are ``-1`` when the group holds fewer than
        `n` positions (all of them, if the group is empty)
    """
    rows, cols = jnp.where(labels == group)
    n_sampled = min(n, int(rows.size))
    pick = random.choice(random.PRNGKey(seed), max(int(rows.size), 1), (n_sampled,), replace=False)
    return jnp.concatenate([
        jnp.stack([rows[pick], cols[pick]], axis=-1).astype(jnp.int32),
        jnp.full((n - n_sampled, 2), -1, dtype=jnp.int32),
    ])


def gather_at(values: Float[Array, '...'], idxs: Int[Array, 'n 2']) -> Float[Array, 'n']:
    """Values at the given (row, col) index pairs, NaN wherever `idxs` is the ``-1`` padding."""
    rows, cols = idxs[:, 0], idxs[:, 1]
    return jnp.where(rows >= 0, values[jnp.maximum(rows, 0), jnp.maximum(cols, 0)], jnp.nan)
