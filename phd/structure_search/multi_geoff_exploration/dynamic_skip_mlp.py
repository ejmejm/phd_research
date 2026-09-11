"""Skip-connected masked network with biases, fixed +-1 feedback, and online restructuring of the
input connections — Experiment 3, revised.

Experiment 3 tries to *learn* a good structure by changing connectivity while training. This module
is the successor of ``old/dynamic_masked_mlp.py`` with the rules from the Experiment 1a follow-ups
folded in; the old module's stage machinery, gradient masking, and h-based prune test are gone.

Layout
------
An output unit's incoming weights live in *one* row over ``[inputs | hidden | 1]``, a hidden unit's
over ``[inputs | 1]``::

    weights['w1']: (n_hidden,  n_features + 1)               # hidden  <- [inputs | 1]
    weights['w2']: (n_outputs, n_features + n_hidden + 1)    # outputs <- [inputs | hidden | 1]

Autostep's effective-step-size normalizer is a row sum inside `optax_idbd`, so everything feeding one
unit has to sit in one row — the skip connections and the bias included. The trailing column reads a
constant 1, so a bias is an ordinary weight, as in `masked_mlp`. Connectivity is a float 0/1 mask
multiplied into the weights at forward time; bias columns are always active.

The rules
---------
* **Fixed +-1 feedback.** On the way back, an output unit's error reaches the hidden units through a
  fixed random +-1 per hidden->output connection instead of the actual weight. The input layer then
  has a learning signal from step 0 even though ``w2`` starts at 0, and that signal is not tied to
  whatever ``w2`` happens to be. The forward pass and ``w2``'s own gradient are the ordinary ones.
  ``feedback=None`` gives plain backpropagation, which the reference structures use. With
  ``feedback_steps`` set, the feedback is used for that many steps only and the error then comes
  through the actual hidden->output weights, like any other connection.
* **Initialization** (`small_uniform_weights`): input connections start uniform in
  ``(-input_weight_scale, input_weight_scale)``; hidden->output weights and biases start at 0.
  Step-sizes start at ``init_lr`` everywhere except the bias columns, which start at
  ``bias_init_lr``. A generated connection is initialized exactly like an initial one of its kind.
* **L1** is the subgradient form, added to the gradient Autostep consumes, on every connection except
  the biases.
* **Pruning** (when ``restructure``): a prunable connection is pruned on the step its weight changes
  sign. Input connections are always prunable; the hidden->output connections present at
  initialization (``protected``) never are, so every hidden unit keeps the output it was assigned;
  hidden->output connections created by generation are prunable. Biases are never pruned.
* **Generation.** Every pruned connection is replaced one-for-one, with the source drawn uniformly
  over the units in the layers before the destination that it is not yet connected to — the inputs,
  and, for an output destination once ``hidden_sources`` is in effect (after ``feedback_steps``), the
  hidden units as well. Two rules pick the destination (``regeneration``): ``'uniform'``, the plan's,
  draws it uniformly over all hidden and output units; ``'same_destination'`` puts the replacement on
  the unit that just lost a connection, so every unit's number of incoming connections is fixed for
  the whole run. See `DynamicAutostepMLP._restructure` for the fixed-shape accounting.

Masked-out connections are inert, exactly as in `masked_mlp`: zero gradient, so Autostep's h and v
stay at 0 and the step-size stays at its initial value. Activating a connection is a mask flip plus
a state reset (`reset_idbd_state_at`) and a fresh weight draw.
"""

from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jaxtyping import Array, Bool, Float, Int

from phd.jax_core.models import ACTIVATION_MAP, lecun_uniform
from phd.jax_core.optimizers.idbd import IDBDState, optax_idbd
from phd.jax_core.utils import tree_replace

from masked_mlp import (
    INACTIVE, USEFUL, USELESS, block_sparse_masks, masked_quantiles, reset_idbd_state_at,
)


LAYERS = ('w1', 'w2')
NO_TASK = -1      # task id of a hidden unit that belongs to no task (the dense reference)

Masks = Dict[str, Float[Array, '...']]
Weights = Dict[str, Float[Array, '...']]
Labels = Dict[str, Int[Array, '...']]


# =============================================================================== the network
class SkipMaskedMLP(eqx.Module):
    """Masked two-layer network with input->output skip connections, biases, and optional fixed
    feedback weights for the backward pass.

    Output units read ``[inputs | hidden | 1]``, so ``weights['w2']`` holds the skip weights in its
    first `n_features` columns, the hidden->output weights in the next `n_hidden`, and the biases in
    the last. Hidden units read ``[inputs | 1]``.
    """

    weights: Weights
    masks: Masks
    feedback: Optional[Float[Array, 'n_outputs n_hidden']]
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        masks: Masks,
        activation: str = 'ltu',
        feedback: Optional[Float[Array, 'n_outputs n_hidden']] = None,
        *,
        key: random.PRNGKey,
    ):
        """
        Args:
            masks: ``{'w1': (n_hidden, n_features + 1), 'w2': (n_outputs, n_features + n_hidden + 1)}``,
                the last column of each being the constant-1 input
            activation: Hidden activation, a key of `ACTIVATION_MAP`
            feedback: Fixed weights that stand in for the hidden->output block on the backward pass,
                or None for ordinary backpropagation. Masked to the hidden->output connectivity.
            key: PRNG key

        Weights get `lecun_uniform` on the connections and 0 on the biases — the reference
        structures' initialization. The experiment's own initialization is `small_uniform_weights`.
        Masked-out weights are held at exactly 0 so a mask flip cannot resurrect a stale value.
        """
        assert set(masks) == set(LAYERS), f"Expected masks for w1 and w2, got {sorted(masks)}!"
        assert activation in ACTIVATION_MAP, f"Unknown activation: {activation}!"
        n_hidden, n_features = masks['w1'].shape[0], masks['w1'].shape[1] - 1
        assert masks['w2'].shape[1] == n_features + n_hidden + 1, (
            f"w2 must have n_features + n_hidden + 1 = {n_features + n_hidden + 1} columns, "
            f"got {masks['w2'].shape[1]}!")
        if feedback is not None:
            assert feedback.shape == (masks['w2'].shape[0], n_hidden), (
                f"feedback must be (n_outputs, n_hidden) = {(masks['w2'].shape[0], n_hidden)}, "
                f"got {feedback.shape}!")

        w1_key, w2_key = random.split(key)
        weights = {
            'w1': lecun_uniform(w1_key, masks['w1'].shape, in_dim=n_features),
            'w2': lecun_uniform(w2_key, masks['w2'].shape, in_dim=n_features + n_hidden),
        }
        self.masks = {k: m.astype(jnp.float32) for k, m in masks.items()}
        self.weights = {k: w.at[:, -1].set(0.0) * self.masks[k] for k, w in weights.items()}
        self.feedback = None if feedback is None else (
            feedback.astype(jnp.float32) * self.masks['w2'][:, n_features:n_features + n_hidden])
        self.activation = activation

    @property
    def n_hidden(self) -> int:
        return self.masks['w1'].shape[0]

    @property
    def n_features(self) -> int:
        return self.masks['w1'].shape[1] - 1

    @property
    def n_outputs(self) -> int:
        return self.masks['w2'].shape[0]

    @property
    def skip(self) -> slice:
        """Columns of ``w2`` holding the input->output skip connections."""
        return slice(0, self.n_features)

    @property
    def hidden_block(self) -> slice:
        """Columns of ``w2`` holding the hidden->output connections."""
        return slice(self.n_features, self.n_features + self.n_hidden)

    @property
    def activation_fn(self):
        return ACTIVATION_MAP[self.activation]

    def activation_grad(self, pre: Float[Array, 'n_hidden']) -> Float[Array, 'n_hidden']:
        """Elementwise derivative of the hidden activation (the surrogate one, for the LTU)."""
        return jax.vmap(jax.grad(self.activation_fn))(pre)

    def hidden_error_weights(self) -> Float[Array, 'n_outputs n_hidden']:
        """What carries an output unit's error back to the hidden units: the fixed feedback when
        there is one, otherwise the actual hidden->output weights. Masked either way."""
        block = self.masks['w2'][:, self.hidden_block]
        if self.feedback is None:
            return self.weights['w2'][:, self.hidden_block] * block
        return self.feedback * block

    def __call__(
        self, x: Float[Array, 'n_features'],
    ) -> Tuple[Float[Array, 'n_outputs'], Float[Array, 'n_hidden'], Weights]:
        """Forward pass of a single sample.

        Returns:
            Tuple of the output, the hidden pre-activations, and the per-weight source values
            (Autostep's `param_inputs`, shaped ``(1, in_features)`` to broadcast over output rows).
        """
        x1 = jnp.append(x, 1.0)
        hidden_pre = (self.weights['w1'] * self.masks['w1']) @ x1
        source = jnp.concatenate([x, self.activation_fn(hidden_pre), jnp.ones((1,), x.dtype)])
        out = (self.weights['w2'] * self.masks['w2']) @ source
        return out, hidden_pre, {'w1': x1[None, :], 'w2': source[None, :]}


# ================================================================================ initializers
def input_columns(masks: Masks) -> Dict[str, Bool[Array, '...']]:
    """Boolean arrays, shaped like the masks, marking the input-sourced columns of each layer."""
    n_features = masks['w1'].shape[1] - 1
    return {k: jnp.broadcast_to(jnp.arange(m.shape[1]) < n_features, m.shape) for k, m in masks.items()}


def small_uniform_weights(masks: Masks, scale: float, key: random.PRNGKey) -> Weights:
    """The experiment's initialization: input connections uniform in ``(-scale, scale)``, and the
    hidden->output weights and biases at 0. Masked-out positions are 0."""
    cols = input_columns(masks)
    return {
        k: jnp.where(cols[k], random.uniform(kk, masks[k].shape, minval=-scale, maxval=scale), 0.0)
        * masks[k]
        for k, kk in zip(LAYERS, random.split(key, len(LAYERS)))
    }


# ============================================================================== structures
def _with_bias_column(mask: Array) -> Float[Array, 'rows cols+1']:
    """Append the always-active constant-1 column to a connectivity mask."""
    return jnp.concatenate(
        [mask.astype(jnp.float32), jnp.ones((mask.shape[0], 1), jnp.float32)], axis=1)


def initial_structure(
    n_features: int,
    n_hidden: int,
    n_outputs: int,
    budget: int,
    key: random.PRNGKey,
) -> Tuple[Masks, Int[Array, 'n_hidden']]:
    """Experiment 3's initialization, spending exactly `budget` connections (biases not counted).

    1. Each hidden unit gets exactly one outgoing connection, to a uniformly chosen output unit.
       These are the only hidden->output connections, and they never change during a run.
    2. The remaining budget is divided evenly over all output *and* hidden units: each hidden unit
       gets that many random input connections, and each output unit that many random
       input->output skip connections.

    The budget rarely divides evenly; the remainder goes one connection each to a random subset of
    units, so the total is exactly `budget`. Generation replaces pruned connections one-for-one, so
    the initial count *is* the budget for the whole run.

    Args:
        n_features: Number of input units
        n_hidden: Number of hidden units
        n_outputs: Number of output units
        budget: Total number of connections
        key: PRNG key

    Returns:
        Tuple of the masks (bias columns included) and, per hidden unit, the index of the output
        unit it feeds
    """
    n_units = n_hidden + n_outputs
    remaining = budget - n_hidden
    assert remaining >= n_units, (
        f"budget ({budget}) leaves {remaining} connections after one outgoing connection per "
        f"hidden unit, fewer than one per unit ({n_units})!")
    per_unit, leftover = divmod(remaining, n_units)
    assert per_unit + 1 <= n_features, (
        f"budget ({budget}) asks for {per_unit + 1} input connections per unit, more than the "
        f"{n_features} inputs available!")

    out_key, in_key, extra_key = random.split(key, 3)

    # (1) one outgoing connection per hidden unit
    hidden_output = random.randint(out_key, (n_hidden,), 0, n_outputs)
    hidden_block = jnp.zeros((n_outputs, n_hidden), jnp.float32).at[
        hidden_output, jnp.arange(n_hidden)].set(1.0)

    # (2) `per_unit` random inputs per unit, the remainder spread over random units. Ranking each
    # unit's inputs in a random order and keeping the first `counts[u]` samples without replacement
    # branchlessly.
    counts = jnp.full((n_units,), per_unit, jnp.int32)
    counts = counts.at[random.permutation(extra_key, n_units)[:leftover]].add(1)
    rank = jnp.argsort(jnp.argsort(random.uniform(in_key, (n_units, n_features)), axis=1), axis=1)
    chosen = (rank < counts[:, None]).astype(jnp.float32)

    masks = {
        'w1': _with_bias_column(chosen[:n_hidden]),
        'w2': _with_bias_column(jnp.concatenate([chosen[n_hidden:], hidden_block], axis=1)),
    }
    return masks, hidden_output


def random_feedback(masks: Masks, key: random.PRNGKey) -> Float[Array, 'n_outputs n_hidden']:
    """A fixed random +-1 per hidden->output connection, 0 where there is none."""
    n_hidden, n_features = masks['w1'].shape[0], masks['w1'].shape[1] - 1
    block = masks['w2'][:, n_features:n_features + n_hidden]
    signs = random.bernoulli(key, 0.5, block.shape).astype(jnp.float32) * 2.0 - 1.0
    return signs * block


def to_skip_layout(masks: Masks, n_features: int) -> Masks:
    """Lift `masked_mlp`-style two-layer masks (no biases) into this file's layout: an empty skip
    block and a bias column on each layer."""
    n_outputs = masks['w2'].shape[0]
    return {
        'w1': _with_bias_column(masks['w1']),
        'w2': _with_bias_column(jnp.concatenate(
            [jnp.zeros((n_outputs, n_features), jnp.float32), masks['w2'].astype(jnp.float32)],
            axis=1)),
    }


def dense_skip_masks(n_features: int, n_hidden: int, n_outputs: int) -> Masks:
    """Fully connected two-layer masks (no skip connections) — the dense reference."""
    return to_skip_layout(
        {'w1': jnp.ones((n_hidden, n_features)), 'w2': jnp.ones((n_outputs, n_hidden))}, n_features)


def block_sparse_skip_masks(
    n_tasks: int, n_features_per_task: int, n_outputs_per_task: int, n_hidden: int,
) -> Tuple[Masks, Int[Array, 'n_hidden']]:
    """Block-sparse masks (no skip connections) and the hidden units' task ids — the reference good
    structure."""
    masks, hidden_task_ids = block_sparse_masks(
        n_tasks, n_features_per_task, n_outputs_per_task, n_hidden)
    return to_skip_layout(masks, n_tasks * n_features_per_task), hidden_task_ids


def n_connections(masks: Masks) -> int:
    """Total number of active connections, biases not counted."""
    return int(sum(int(masks[k][:, :-1].sum()) for k in LAYERS))


# ==================================================================================== labels
def task_labels(
    masks: Masks,
    hidden_task_ids: Int[Array, 'n_hidden'],
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Labels:
    """Label every active connection `USEFUL` (same task at both ends) or `USELESS`.

    Everything else is `INACTIVE`: masked-out positions, biases, and connections touching a hidden
    unit whose task id is `NO_TASK`. A hidden unit's task is the task of the output it was initially
    assigned, so the initial hidden->output connections are all useful and a generated one is useful
    exactly when it feeds an output of that same task. Pass all-ones masks to label *positions*
    regardless of activity.

    Args:
        masks: Connectivity masks
        hidden_task_ids: Task of each hidden unit (`NO_TASK` for none)
        input_task_ids: Task of each input
        output_task_ids: Task of each output

    Returns:
        Per-layer integer labels with the masks' shapes
    """
    n_hidden, n_features = masks['w1'].shape[0], masks['w1'].shape[1] - 1
    hidden = slice(n_features, n_features + n_hidden)

    def label(row_tasks, col_tasks, active):
        same = row_tasks[:, None] == col_tasks[None, :]
        known = (row_tasks >= 0)[:, None] & (col_tasks >= 0)[None, :]
        return jnp.where(active & known, jnp.where(same, USEFUL, USELESS), INACTIVE)

    w1 = jnp.full(masks['w1'].shape, INACTIVE, jnp.int32).at[:, :n_features].set(
        label(hidden_task_ids, input_task_ids, masks['w1'][:, :n_features] > 0))
    w2 = jnp.full(masks['w2'].shape, INACTIVE, jnp.int32).at[:, :n_features].set(
        label(output_task_ids, input_task_ids, masks['w2'][:, :n_features] > 0))
    w2 = w2.at[:, hidden].set(label(output_task_ids, hidden_task_ids, masks['w2'][:, hidden] > 0))
    return {'w1': w1, 'w2': w2}


VIEW_NAMES = ('hidden <- inputs', 'outputs <- inputs (skips)', 'outputs <- hidden')


def connection_views(masks: Masks) -> Tuple[Tuple[str, slice], ...]:
    """The three kinds of connection, as ``(layer, columns)`` in `VIEW_NAMES` order."""
    n_hidden, n_features = masks['w1'].shape[0], masks['w1'].shape[1] - 1
    return (('w1', slice(0, n_features)), ('w2', slice(0, n_features)),
            ('w2', slice(n_features, n_features + n_hidden)))


def group_quantiles(
    values: Weights,
    labels: Labels,
    masks: Masks,
    quantiles: Tuple[float, ...],
    groups: Tuple[int, ...] = (USEFUL, USELESS),
) -> Float[Array, 'n_views n_groups n_quantiles']:
    """Quantiles of `values` over the *currently active* connections of each group, per view.

    `labels` are position labels (from `task_labels` with all-ones masks); the current masks decide
    which of them count. Views are `connection_views`. NaN for an empty group.
    """
    return jnp.stack([
        jnp.stack([
            masked_quantiles(values[k][:, cols],
                             jnp.where(masks[k][:, cols] > 0, labels[k][:, cols], INACTIVE),
                             g, quantiles)
            for g in groups])
        for k, cols in connection_views(masks)])


# ================================================================================ the learner
class DynamicAutostepMLP(eqx.Module):
    """`SkipMaskedMLP` trained online with Autostep (Variant B), optionally restructuring its input
    connections every step.

    One `step` call is one sample: forward, explicit backward (through the fixed feedback while it is
    in force), the L1 subgradient, the Autostep update, and, when `restructure`, the zero-crossing
    prune test and the matching generation. Everything is fixed-shape and branchless, so it scans
    under `jit`.
    """

    # --- static configuration
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    init_lr: float = eqx.field(static=True)
    input_weight_scale: float = eqx.field(static=True)
    l1: float = eqx.field(static=True)
    restructure: bool = eqx.field(static=True)
    gen_cap: int = eqx.field(static=True)
    feedback_steps: Optional[int] = eqx.field(static=True)
    hidden_sources: bool = eqx.field(static=True)
    regeneration: str = eqx.field(static=True)

    # --- dynamic state
    model: SkipMaskedMLP
    opt_state: IDBDState
    labels: Labels                # position labels of every connection, for the bookkeeping
    protected: Bool[Array, 'n_outputs n_hidden']    # hidden->output connections never pruned
    deficit: Int[Array, '']                          # replacements still owed, in total ...
    row_deficit: Int[Array, 'n_units']               # ... and per destination unit (same_destination)
    step_count: Int[Array, '']
    rng: random.PRNGKey

    @classmethod
    def init(
        cls,
        masks: Masks,
        meta_lr: float,
        key: random.PRNGKey,
        *,
        init_lr: float,
        bias_init_lr: Optional[float] = None,
        input_weight_scale: Optional[float] = None,
        l1: float = 0.0,
        activation: str = 'ltu',
        feedback: Optional[Float[Array, 'n_outputs n_hidden']] = None,
        feedback_steps: Optional[int] = None,
        hidden_sources: bool = False,
        weight_init: str = 'small_uniform',
        restructure: bool = False,
        regeneration: str = 'uniform',
        gen_cap: int = 512,
        labels: Optional[Labels] = None,
    ) -> 'DynamicAutostepMLP':
        """
        Args:
            masks: Initial connectivity, bias columns included. Its hidden->output connections are
                the protected ones.
            meta_lr: Autostep meta step-size
            key: PRNG key (split for the weights and for generation)
            init_lr: Initial step-size of every connection, including generated ones
            bias_init_lr: Initial step-size of the bias columns; None leaves them at `init_lr`
            input_weight_scale: Half-width of the input weights' uniform initialization under
                ``'small_uniform'``, for initial and generated connections alike; None uses `init_lr`
            l1: L1 coefficient, added to the gradient as a subgradient (biases exempt)
            activation: Hidden activation
            feedback: Fixed +-1 feedback for the backward pass (`random_feedback`), or None for
                ordinary backpropagation
            feedback_steps: Use the feedback for this many steps only, then the actual hidden->output
                weights; None keeps the feedback for the whole run
            hidden_sources: Let generation create hidden->output connections (prunable, unlike the
                protected initial ones) — from the start, or from `feedback_steps` on when that is set
            weight_init: ``'small_uniform'`` (`small_uniform_weights`, the experiment's) or
                ``'lecun'`` (the model's own, for the reference structures)
            restructure: Prune prunable connections when their weight changes sign, and regenerate
            regeneration: Where a replacement goes: ``'uniform'`` (a destination uniformly over all
                hidden and output units) or ``'same_destination'`` (the unit that lost the
                connection, keeping every unit's in-degree fixed)
            gen_cap: Maximum connections generated per step under ``'uniform'``; see `_restructure`
            labels: Position labels from `task_labels`, used to split the crossing and prune counts
                into same-task and cross-task; None counts nothing
        """
        weight_key, rng = random.split(key)
        model = SkipMaskedMLP(masks, activation=activation, feedback=feedback, key=weight_key)
        input_weight_scale = init_lr if input_weight_scale is None else input_weight_scale
        if weight_init == 'small_uniform':
            model = tree_replace(model, weights=small_uniform_weights(
                model.masks, input_weight_scale, weight_key))
        elif weight_init != 'lecun':
            raise ValueError(f"Unknown weight_init: {weight_init}!")
        if regeneration not in ('uniform', 'same_destination'):
            raise ValueError(f"Unknown regeneration rule: {regeneration}!")

        optimizer = optax_idbd(
            init_lr=init_lr, meta_lr=meta_lr, autostep=True, version='squared_inputs')
        opt_state = optimizer.init(model.weights)
        if bias_init_lr is not None:
            # `init_beta` stays at `init_lr`: it is what a reset connection goes back to, and the
            # biases are never reset.
            opt_state = opt_state._replace(beta={
                k: b.at[:, -1].set(jnp.log(bias_init_lr)) for k, b in opt_state.beta.items()})
        if labels is None:
            labels = {k: jnp.full(m.shape, INACTIVE, jnp.int32) for k, m in model.masks.items()}

        return cls(
            optimizer=optimizer, init_lr=init_lr, input_weight_scale=input_weight_scale, l1=l1,
            restructure=restructure, gen_cap=gen_cap, feedback_steps=feedback_steps,
            hidden_sources=hidden_sources, regeneration=regeneration,
            model=model, opt_state=opt_state, labels=labels,
            protected=model.masks['w2'][:, model.hidden_block] > 0,
            deficit=jnp.array(0, jnp.int32),
            row_deficit=jnp.zeros((model.n_hidden + model.n_outputs,), jnp.int32),
            step_count=jnp.array(0, jnp.int32), rng=rng,
        )

    # ------------------------------------------------------------------------- what is where
    @property
    def feedback_active(self) -> Bool[Array, '']:
        """Whether the hidden units' error comes through the fixed feedback on this step."""
        if self.model.feedback is None:
            return jnp.array(False)
        if self.feedback_steps is None:
            return jnp.array(True)
        return self.step_count < self.feedback_steps

    @property
    def hidden_sources_active(self) -> Bool[Array, '']:
        """Whether generation may create hidden->output connections on this step."""
        if not self.hidden_sources:
            return jnp.array(False)
        if self.feedback_steps is None:
            return jnp.array(True)
        return self.step_count >= self.feedback_steps

    def prunable_columns(self) -> Dict[str, Bool[Array, '...']]:
        """Input connections always, hidden->output connections unless protected, biases never."""
        cols = input_columns(self.model.masks)
        return {'w1': cols['w1'],
                'w2': cols['w2'].at[:, self.model.hidden_block].set(~self.protected)}

    # ---------------------------------------------------------------- forward and backward
    def loss_and_grads(
        self, x: Float[Array, 'n_features'], y: Float[Array, 'n_outputs'],
    ) -> Tuple[Float[Array, ''], Weights, Weights]:
        """Forward pass and the explicit backward pass, without the L1 term.

        Written out rather than taken from `jax.value_and_grad` so that the hidden units' error can
        come through the fixed feedback: ``d_hidden = feedback.T @ d_out`` instead of
        ``w2_hidden.T @ d_out``. Without feedback, or once `feedback_steps` have passed, this is
        exactly the autodiff gradient.

        Returns:
            Tuple of the squared error summed over outputs, the per-weight gradients, and Autostep's
            `param_inputs`
        """
        model, masks = self.model, self.model.masks
        out, hidden_pre, param_inputs = model(x)
        x1, source = param_inputs['w1'][0], param_inputs['w2'][0]

        error = out - y
        loss = jnp.sum(error ** 2)                       # summed over outputs, as in Experiment 1
        d_out = 2.0 * error
        hidden_block = masks['w2'][:, model.hidden_block]
        actual = model.weights['w2'][:, model.hidden_block] * hidden_block
        error_weights = actual if model.feedback is None else jnp.where(
            self.feedback_active, model.feedback * hidden_block, actual)
        d_hidden_pre = (error_weights.T @ d_out) * model.activation_grad(hidden_pre)
        grads = {
            'w1': (d_hidden_pre[:, None] * x1[None, :]) * masks['w1'],
            'w2': (d_out[:, None] * source[None, :]) * masks['w2'],
        }
        return loss, grads, param_inputs

    # ------------------------------------------------------------------------------ one step
    def step(
        self, x: Float[Array, 'n_features'], y: Float[Array, 'n_outputs'],
    ) -> Tuple['DynamicAutostepMLP', Dict[str, Array]]:
        """One online update.

        Returns:
            Tuple of the new learner and per-step scalars: the squared error summed over outputs
            (`loss`); how many prunable connections changed sign (`n_crossed`, split into `_useful`
            and `_useless` by the position labels and `_hidden_out` by kind); how many were pruned
            and generated; and the running generation `deficit`. Without `restructure` the crossings
            are counted but nothing is pruned.
        """
        model = self.model
        masks, w_prev = model.masks, model.weights

        loss, grads, param_inputs = self.loss_and_grads(x, y)

        if self.l1 > 0.0:
            # Subgradient L1 on every connection but the biases, inside the gradient Autostep
            # consumes, so it shapes the step-size adaptation too (the Experiment 1 convention).
            penalty = {k: (self.l1 * jnp.sign(w_prev[k]) * masks[k]).at[:, -1].set(0.0)
                       for k in LAYERS}
            grads = {k: g + penalty[k] for k, g in grads.items()}

        # Autostep, Variant B: the curvature term is the squared value of each weight's source unit.
        updates, opt_state = self.optimizer.update(
            (grads, None, param_inputs), self.opt_state, w_prev)
        weights = {k: (w_prev[k] + updates[k]) * masks[k] for k in LAYERS}
        model = tree_replace(model, weights=weights)

        # A prunable connection whose weight changed sign this step. The protected hidden->output
        # connections and the biases stay for the whole run.
        prunable = self.prunable_columns()
        crossed = {k: (masks[k] > 0) & prunable[k] & (w_prev[k] * weights[k] < 0) for k in LAYERS}
        counts = self._count(crossed, 'n_crossed')

        zero = jnp.array(0, jnp.int32)
        rng, deficit, row_deficit = self.rng, self.deficit, self.row_deficit
        counts.update(n_pruned=zero, n_generated=zero)
        if self.restructure:
            model, opt_state, rng, deficit, row_deficit, restructured = self._restructure(
                model, opt_state, crossed)
            counts.update(restructured)
        counts['deficit'] = deficit

        return (tree_replace(self, model=model, opt_state=opt_state, deficit=deficit,
                             row_deficit=row_deficit, rng=rng, step_count=self.step_count + 1),
                {'loss': loss, **counts})

    def _count(self, selected: Dict[str, Bool[Array, '...']], name: str) -> Dict[str, Array]:
        """Total, same-task, cross-task, and hidden->output counts of the selected connections."""
        def total(condition):
            return sum(jnp.sum(selected[k] & condition(k)) for k in LAYERS).astype(jnp.int32)
        hidden_out = {k: jnp.zeros_like(selected[k]) for k in LAYERS}
        hidden_out['w2'] = hidden_out['w2'].at[:, self.model.hidden_block].set(True)
        return {name: total(lambda k: True),
                f'{name}_useful': total(lambda k: self.labels[k] == USEFUL),
                f'{name}_useless': total(lambda k: self.labels[k] == USELESS),
                f'{name}_hidden_out': total(lambda k: hidden_out[k])}

    # -------------------------------------------------------------------- prune and generate
    def _restructure(self, model, opt_state, crossed):
        """Prune the crossed connections, generate one replacement each, initialize the new ones.

        The source of a replacement is always drawn uniformly over the units in earlier layers that
        the destination is not already connected to — inputs for a hidden destination; inputs, and
        hidden units while `hidden_sources_active`, for an output destination. Where the replacement
        goes is the `regeneration` rule (`_propose_uniform`, `_propose_same_destination`). Either
        way the connection count is exactly the budget minus the current `deficit`, which is logged
        rather than hidden.
        """
        masks = {k: jnp.where(crossed[k], 0.0, model.masks[k]) for k in LAYERS}
        n_pruned = sum(jnp.sum(crossed[k]) for k in LAYERS).astype(jnp.int32)
        rng, propose_key, weight_key = random.split(self.rng, 3)

        if self.regeneration == 'uniform':
            generated, deficit, row_deficit = self._propose_uniform(
                model, masks, n_pruned, propose_key)
        else:
            generated, deficit, row_deficit = self._propose_same_destination(
                model, masks, crossed, propose_key)
        masks = {k: jnp.maximum(masks[k], generated[k]) for k in LAYERS}
        n_generated = sum(jnp.sum(generated[k]) for k in LAYERS).astype(jnp.int32)

        # A generated connection is initialized exactly like an initial one of its kind: an input
        # connection gets a fresh uniform draw in (-input_weight_scale, input_weight_scale), a
        # hidden->output connection starts at 0; either way the step-size goes back to init_lr and h
        # and Autostep's normalizer to 0. Pruned positions that were not refilled get the same reset
        # and a weight of 0 via the mask.
        reset = {k: crossed[k] | (generated[k] > 0) for k in LAYERS}
        fresh = small_uniform_weights(masks, self.input_weight_scale, weight_key)
        weights = {k: jnp.where(reset[k], fresh[k], model.weights[k]) * masks[k] for k in LAYERS}
        opt_state = reset_idbd_state_at(opt_state, {k: r.astype(jnp.float32) for k, r in reset.items()})
        model = tree_replace(model, weights=weights, masks=masks)
        return (model, opt_state, rng, deficit, row_deficit,
                {'n_pruned': n_pruned, 'n_generated': n_generated})

    def _allowed_sources(self, model) -> Bool[Array, 'n_features+n_hidden']:
        """Which source units an output destination may draw from, in the ``[inputs | hidden]`` layout."""
        return jnp.concatenate([jnp.ones((model.n_features,), bool),
                                jnp.broadcast_to(self.hidden_sources_active, (model.n_hidden,))])

    def _propose_uniform(self, model, masks, n_pruned, key):
        """The plan's rule: a destination uniformly over all hidden and output units per replacement.

        Sampling replacements one at a time is not expressible at a fixed shape, so `gen_cap`
        proposals are drawn at once and the first ``demand = min(pruned + deficit, gen_cap)`` of them
        are applied. Two proposals can collide on the same pathway, or a proposal can land on a
        destination with no free source, in which case fewer connections are created than were
        pruned; the shortfall is carried in `deficit` and filled on later steps.
        """
        n_features, n_hidden, n_outputs = model.n_features, model.n_hidden, model.n_outputs
        n_units, cap = n_hidden + n_outputs, self.gen_cap
        dest_key, source_key = random.split(key)
        demand = jnp.minimum(n_pruned + self.deficit, cap)

        # A destination per proposal. Eligible sources live in the padded `[inputs | hidden]` layout
        # so both destination kinds share one array: a hidden destination may only draw from the
        # inputs; an output destination from the inputs and, while hidden sources are allowed, from
        # the hidden units.
        width = n_features + n_hidden
        destination = random.randint(dest_key, (cap,), 0, n_units)
        into_hidden = destination < n_hidden
        hidden_row = jnp.minimum(destination, n_hidden - 1)
        output_row = jnp.maximum(destination - n_hidden, 0)
        free_into_hidden = jnp.concatenate([
            masks['w1'][hidden_row, :n_features] == 0, jnp.zeros((cap, n_hidden), bool)], axis=1)
        free_into_output = (masks['w2'][output_row, :width] == 0) & self._allowed_sources(model)[None, :]
        free = jnp.where(into_hidden[:, None], free_into_hidden, free_into_output)
        source = jnp.argmax(
            jnp.where(free, random.uniform(source_key, (cap, width)), -jnp.inf), axis=1)
        accepted = (jnp.arange(cap) < demand) & jnp.any(free, axis=1)

        # Scatter the accepted proposals. `max` makes duplicate indices harmless and lets rejected
        # proposals write a no-op 0; a hidden destination's source is always an input, so clipping
        # it to the w1 width only keeps rejected proposals in range.
        generated = {
            'w1': jnp.zeros_like(masks['w1']).at[hidden_row, jnp.minimum(source, n_features - 1)].max(
                jnp.where(accepted & into_hidden, 1.0, 0.0)),
            'w2': jnp.zeros_like(masks['w2']).at[output_row, source].max(
                jnp.where(accepted & ~into_hidden, 1.0, 0.0)),
        }
        n_generated = sum(jnp.sum(generated[k]) for k in LAYERS).astype(jnp.int32)
        deficit = n_pruned + self.deficit - n_generated
        return generated, deficit, self.row_deficit

    def _propose_same_destination(self, model, masks, crossed, key):
        """Fixed in-degree: each unit that lost connections this step gets as many new ones, each to a
        source it is not connected to, drawn uniformly and never the one just pruned.

        Every unit is handled at once: rank its free sources in a random order and keep the first
        ``demand`` of them, where ``demand`` is the unit's prunes this step plus anything still owed
        to it. A unit can only be owed anything if it had fewer free sources than prunes, which does
        not happen at these in-degrees, but the accounting is kept exact regardless.
        """
        n_features, n_hidden, n_outputs = model.n_features, model.n_hidden, model.n_outputs
        n_units, width = n_hidden + n_outputs, n_features + n_hidden

        demand = self.row_deficit + jnp.concatenate([
            crossed['w1'][:, :n_features].sum(axis=1),
            crossed['w2'][:, :width].sum(axis=1)]).astype(jnp.int32)

        # Free sources per unit in the padded `[inputs | hidden]` layout: hidden units read inputs
        # only; output units read inputs and, while allowed, hidden units. The position just pruned
        # is excluded so the replacement really is a new source.
        free_hidden_rows = jnp.concatenate([
            (masks['w1'][:, :n_features] == 0) & ~crossed['w1'][:, :n_features],
            jnp.zeros((n_hidden, n_hidden), bool)], axis=1)
        free_output_rows = ((masks['w2'][:, :width] == 0) & ~crossed['w2'][:, :width]
                            & self._allowed_sources(model)[None, :])
        free = jnp.concatenate([free_hidden_rows, free_output_rows], axis=0)

        noise = jnp.where(free, random.uniform(key, (n_units, width)), jnp.inf)
        rank = jnp.argsort(jnp.argsort(noise, axis=1), axis=1)
        chosen = free & (rank < demand[:, None])

        generated = {
            'w1': jnp.zeros_like(masks['w1']).at[:, :n_features].set(
                chosen[:n_hidden, :n_features].astype(jnp.float32)),
            'w2': jnp.zeros_like(masks['w2']).at[:, :width].set(chosen[n_hidden:].astype(jnp.float32)),
        }
        row_deficit = demand - chosen.sum(axis=1).astype(jnp.int32)
        return generated, row_deficit.sum(), row_deficit

    # ------------------------------------------------------------------------------ readouts
    @property
    def alpha(self) -> Weights:
        """Per-connection step-sizes."""
        return {k: jnp.exp(self.opt_state.beta[k]) for k in LAYERS}


# ==================================================================================== metrics
def separation_metrics(
    model: SkipMaskedMLP,
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Dict[str, Float[Array, '']]:
    """Connectivity and signal separation, both in [0, 1] and averaged over output units.

    **Connectivity separation.** From an output unit, trace every backward path along connections
    toward the input layer — a direct skip connection is a length-1 path, an
    output <- hidden <- input chain a length-2 path. The metric is the fraction of those paths that
    end at an input of the output's own subtask. Paths are counted, not inputs, so an input reached
    along several paths counts several times. Biases are not paths.

    **Signal separation.** The same quantity with each path weighted by the absolute product of the
    weights along it, normalized per output unit.

    Both are sums over paths without enumerating them::

        paths(o) = sum_i M_skip[o, i] + sum_h M_hidden[o, h] sum_i M_1[h, i]
        same(o)  = the same two sums restricted to inputs of o's own task

    An output unit with no paths contributes NaN and is dropped from the average.
    """
    n_features = model.n_features
    m1 = model.masks['w1'][:, :n_features]
    skip, hidden_out = model.masks['w2'][:, model.skip], model.masks['w2'][:, model.hidden_block]
    same = (output_task_ids[:, None] == input_task_ids[None, :]).astype(jnp.float32)

    def fraction_same(a1, a_skip, a_hidden):
        total = a_skip.sum(axis=1) + a_hidden @ a1.sum(axis=1)
        matched = (a_skip * same).sum(axis=1) + jnp.einsum('oh,hi,oi->o', a_hidden, a1, same)
        return jnp.nanmean(jnp.where(total > 0, matched / jnp.where(total > 0, total, 1.0), jnp.nan))

    return {
        'connectivity_separation': fraction_same(m1, skip, hidden_out),
        'signal_separation': fraction_same(
            jnp.abs(model.weights['w1'][:, :n_features] * m1),
            jnp.abs(model.weights['w2'][:, model.skip] * skip),
            jnp.abs(model.weights['w2'][:, model.hidden_block] * hidden_out),
        ),
    }


def structure_metrics(
    model: SkipMaskedMLP,
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
    n_tasks: Optional[int] = None,
) -> Dict[str, Float[Array, '']]:
    """`separation_metrics` plus connection-count summaries, all scalars and jit-safe.

    With `n_tasks` given, adds `hidden_input_purity`: per hidden unit, the fraction of its incoming
    connections that come from its most-represented subtask, averaged over hidden units that have
    any. 1 means every hidden unit reads a single subtask; 1 / n_tasks means its inputs are spread
    evenly over all of them. Unlike the separation metrics this looks at the hidden layer alone.
    """
    n_features, n_hidden = model.n_features, model.n_hidden
    m1 = model.masks['w1'][:, :n_features]
    skip, hidden_out = model.masks['w2'][:, model.skip], model.masks['w2'][:, model.hidden_block]

    metrics = {
        **separation_metrics(model, input_task_ids, output_task_ids),
        'hidden_incoming': m1.sum() / n_hidden,
        'hidden_outgoing': hidden_out.sum() / n_hidden,
        'output_incoming': (skip.sum() + hidden_out.sum()) / model.n_outputs,
        'output_skip_fraction': skip.sum() / jnp.maximum(skip.sum() + hidden_out.sum(), 1.0),
        'dead_hidden_fraction': jnp.mean((hidden_out.sum(axis=0) == 0) | (m1.sum(axis=1) == 0)),
        'n_connections': m1.sum() + skip.sum() + hidden_out.sum(),
    }
    if n_tasks is not None:
        per_task = m1 @ (input_task_ids[None, :] == jnp.arange(n_tasks)[:, None]).astype(
            jnp.float32).T
        total = per_task.sum(axis=1)
        metrics['hidden_input_purity'] = jnp.nanmean(jnp.where(
            total > 0, per_task.max(axis=1) / jnp.where(total > 0, total, 1.0), jnp.nan))
    return metrics
