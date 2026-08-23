"""Dynamic masked network with skip connections, online pruning and generation — Experiment 3.

Experiment 1 asked whether Autostep can *tell* useful connections from useless ones. Experiment 3
uses that signal to actually change the connectivity while training: connections whose weight
crossed zero with a small gradient trace are pruned and replaced one-for-one elsewhere, so the
total connection count never leaves its budget.

What is different from `masked_mlp.py` (Experiments 1 and 2), and why
--------------------------------------------------------------------
* **Skip connections.** Experiment 3's initialization connects output units directly to inputs, so
  the network is not a plain two-layer MLP. Rather than carry a third weight matrix, an output
  unit's incoming weights are stored as a *single* row over ``[inputs | hidden]``::

      weights['w1']: (n_hidden,  n_features)                # hidden <- inputs
      weights['w2']: (n_outputs, n_features + n_hidden)     # outputs <- [inputs | hidden]

  That layout is not cosmetic: everything in this file is defined *per unit over its incoming
  weights* — Autostep's effective-step-size normalizer (a row sum inside `optax_idbd`), the
  squared-error trace v, the occupancy c, and the jitter thresholds derived from them. With the
  skips in their own matrix, each output unit's incoming weights would straddle two leaves and
  every one of those row reductions would silently be taken over half a unit.
* **Manual forward/backward.** Experiment 1 got its gradients from `jax.value_and_grad`. Here the
  backward pass is written out, for two reasons. The per-unit error signal delta_u is needed in its
  own right (it drives the v trace and hence every threshold), and the base version of 3b masks the
  gradient flowing from output units back into hidden units except through the hidden->output
  connections created at initialization — which is one array multiply in an explicit backward pass.
* **Restructuring inside the scan.** Masks are dynamic state, and pruning / generation happen every
  step under `jit`, so both are branchless and fixed-shape: pruning is a `where` on the mask, and
  generation samples a fixed `gen_cap` proposals per step of which only the first `demand` are
  applied. See `DynamicAutostepMLP.step` for the accounting that keeps the budget exact.

Conventions worth knowing before reading the code
-------------------------------------------------
* **Autostep Variant B everywhere** (`version='squared_inputs'`): wherever the linear algorithm
  uses the squared input x^2, this uses the squared value of the weight's *source unit* — actual
  inputs for `w1`, and ``[inputs | hidden]`` for `w2`. Experiment 1 settled on it.
* **The stage rule is Experiment 2's**, imported from `connection_stages` rather than reimplemented,
  so that if Experiment 2's validation revises the thresholds, 3b Variant 2 follows. Its per-unit
  traces are indexed by destination unit, which is exactly this file's row convention.
* **delta_u is signed so that the loss gradient is ``-delta_u * source``**, i.e. delta_out =
  -dL/d(out) = 2 (y - out) for the summed-squared-error loss this file uses. That is what makes the
  plan's algebra hold as written: h is built from ``alpha * g`` (so |h| ~ alpha |delta| |x|) and v
  from delta_u^2, and the jitter thresholds relate the two. Using an unscaled ``y - out`` here while
  the optimizer sees a gradient of ``2 (out - y) x`` would put every threshold out by 16x.
* **The letter v is overloaded.** `IDBDState.v` is Autostep's *per-weight* normalizer. The plan's v
  is a *per-unit* trace of squared error, called `err_trace` here, with occupancy `occupancy` (the
  plan's c). They never interact.
* **Masked-out connections are inert**, exactly as in `masked_mlp.py`: zero gradient, so h and
  Autostep's v stay at 0 and the step-size stays at its initial value. Activating a connection is
  therefore a mask flip plus a state reset (`reset_idbd_state_at`).
* **No biases** — Autostep asserts against them.
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

from masked_mlp import block_sparse_masks
# Experiment 2's stage rule, used verbatim: a connection is nascent when created, enters growth when
# its weight crosses the jitter threshold, and matures when its h crosses zero while in growth.
# Re-exported so callers of this module do not need to know where the definitions live.
from connection_stages import (  # noqa: F401
    GROWTH, MATURE, NASCENT, STAGE_NAMES, TRACE_K, StageState, hidden_unit_stages,
    init_stage_state, reset_connections_at, update_stage_state,
)


LAYERS = ('w1', 'w2')

Masks = Dict[str, Float[Array, '...']]
Weights = Dict[str, Float[Array, '...']]


# =============================================================================== the network
class SkipMaskedMLP(eqx.Module):
    """Masked two-layer network with input->output skip connections and no biases.

    Connectivity is a float 0/1 mask multiplied into the weights at forward time, so an arbitrary
    per-connection topology is expressible and a single connection is toggled with an `.at[]` write.
    Output units read a concatenation of the inputs and the hidden units, so ``weights['w2']`` holds
    the skip weights in its first `n_features` columns and the hidden->output weights in the rest.
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
            masks: ``{'w1': (n_hidden, n_features), 'w2': (n_outputs, n_features + n_hidden)}``
            activation: Hidden activation, a key of `ACTIVATION_MAP`
            key: PRNG key

        Both layers use `lecun_uniform`, not zeros: with zero incoming weights every hidden unit
        outputs 0 and nothing flows into the layer below it. Masked-out weights are held at exactly
        zero so a mask flip cannot resurrect a stale value.
        """
        assert set(masks) == set(LAYERS), f"Expected masks for w1 and w2, got {sorted(masks)}!"
        assert activation in ACTIVATION_MAP, f"Unknown activation: {activation}!"
        n_hidden, n_features = masks['w1'].shape
        assert masks['w2'].shape[1] == n_features + n_hidden, (
            f"w2 must have n_features + n_hidden = {n_features + n_hidden} columns, "
            f"got {masks['w2'].shape[1]}!")

        w1_key, w2_key = random.split(key)
        self.weights = {
            'w1': lecun_uniform(w1_key, masks['w1'].shape) * masks['w1'],
            'w2': lecun_uniform(w2_key, masks['w2'].shape) * masks['w2'],
        }
        self.masks = {k: m.astype(jnp.float32) for k, m in masks.items()}
        self.activation = activation

    @property
    def n_hidden(self) -> int:
        return self.masks['w1'].shape[0]

    @property
    def n_features(self) -> int:
        return self.masks['w1'].shape[1]

    @property
    def n_outputs(self) -> int:
        return self.masks['w2'].shape[0]

    @property
    def activation_fn(self):
        return ACTIVATION_MAP[self.activation]

    def activation_grad(self, pre: Float[Array, 'n_hidden']) -> Float[Array, 'n_hidden']:
        """Elementwise derivative of the hidden activation, for the manual backward pass."""
        return jax.vmap(jax.grad(self.activation_fn))(pre)

    @property
    def skip_weights(self) -> Float[Array, 'n_outputs n_features']:
        """The input->output skip block of the output layer."""
        return self.weights['w2'][:, :self.n_features]

    @property
    def hidden_out_weights(self) -> Float[Array, 'n_outputs n_hidden']:
        """The hidden->output block of the output layer."""
        return self.weights['w2'][:, self.n_features:]

    def __call__(
        self, x: Float[Array, 'n_features'],
    ) -> Tuple[Float[Array, 'n_outputs'], Float[Array, 'n_hidden'], Weights]:
        """Forward pass of a single sample.

        Returns:
            Tuple of the output, the hidden pre-activations, and the per-weight source values
            (Autostep's `param_inputs`, shaped ``(1, in_features)`` to broadcast over output rows).
        """
        hidden_pre = (self.weights['w1'] * self.masks['w1']) @ x
        source = jnp.concatenate([x, self.activation_fn(hidden_pre)])
        out = (self.weights['w2'] * self.masks['w2']) @ source
        return out, hidden_pre, {'w1': x[None, :], 'w2': source[None, :]}


# ========================================================================== initial structures
def initial_structure(
    n_features: int,
    n_hidden: int,
    n_outputs: int,
    budget: int,
    key: random.PRNGKey,
) -> Tuple[Masks, Float[Array, 'n_outputs n_features+n_hidden']]:
    """Experiment 3's initialization, spending exactly `budget` connections.

    1. Each hidden unit gets exactly one outgoing connection, to a uniformly chosen output unit.
       These are the only hidden->output connections at initialization.
    2. The remaining budget is divided evenly over all output *and* hidden units: each hidden unit
       gets that many random input connections, and each output unit that many random
       input->output skip connections. Rule (1) is preserved — nothing here touches the
       hidden->output block.

    The budget rarely divides evenly over the units; the remainder is handed out one connection
    each to a random subset of units, so the total is exactly `budget` rather than a floor of it.
    That matters because generation replaces pruned connections one-for-one, so the initial count
    *is* the budget for the whole run.

    Args:
        n_features: Number of input units
        n_hidden: Number of hidden units
        n_outputs: Number of output units
        budget: Total number of connections
        key: PRNG key

    Returns:
        Tuple of the masks and the mask of the initial hidden->output connections (in the full
        ``(n_outputs, n_features + n_hidden)`` layout), which the base version of 3b treats
        specially: they alone carry gradient into the hidden layer, and they are exempt from pruning.
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
    destinations = random.randint(out_key, (n_hidden,), 0, n_outputs)
    initial_out = jnp.zeros((n_outputs, n_features + n_hidden), jnp.float32).at[
        destinations, n_features + jnp.arange(n_hidden)].set(1.0)

    # (2) `per_unit` random inputs per hidden and output unit, remainder spread over random units
    counts = jnp.full((n_units,), per_unit, jnp.int32)
    counts = counts.at[random.permutation(extra_key, n_units)[:leftover]].add(1)
    # Rank each unit's inputs in a random order and keep the first `counts[u]` — sampling without
    # replacement, branchlessly.
    order = jnp.argsort(random.uniform(in_key, (n_units, n_features)), axis=1)
    rank = jnp.argsort(order, axis=1)
    chosen = (rank < counts[:, None]).astype(jnp.float32)

    masks = {
        'w1': chosen[:n_hidden],
        'w2': initial_out.at[:, :n_features].set(chosen[n_hidden:]),
    }
    return masks, initial_out


def to_skip_layout(masks: Masks, n_features: int) -> Masks:
    """Lift `masked_mlp`-style two-layer masks into this file's ``[inputs | hidden]`` layout.

    The baselines (dense and block-sparse) have no skip connections, so their skip block is zero.

    Args:
        masks: ``{'w1': (n_hidden, n_features), 'w2': (n_outputs, n_hidden)}``
        n_features: Number of input units

    Returns:
        Masks with ``w2`` widened to ``(n_outputs, n_features + n_hidden)``
    """
    n_outputs = masks['w2'].shape[0]
    return {
        'w1': masks['w1'].astype(jnp.float32),
        'w2': jnp.concatenate([
            jnp.zeros((n_outputs, n_features), jnp.float32),
            masks['w2'].astype(jnp.float32),
        ], axis=1),
    }


def dense_skip_masks(n_features: int, n_hidden: int, n_outputs: int) -> Masks:
    """Fully connected two-layer masks (no skip connections) — the dense baseline."""
    return to_skip_layout(
        {'w1': jnp.ones((n_hidden, n_features)), 'w2': jnp.ones((n_outputs, n_hidden))},
        n_features,
    )


def block_sparse_skip_masks(
    n_tasks: int, n_features_per_task: int, n_outputs_per_task: int, n_hidden: int,
) -> Tuple[Masks, Int[Array, 'n_hidden']]:
    """Block-sparse masks (no skip connections) — the reference good structure."""
    masks, hidden_task_ids = block_sparse_masks(
        n_tasks, n_features_per_task, n_outputs_per_task, n_hidden)
    return to_skip_layout(masks, n_tasks * n_features_per_task), hidden_task_ids


def n_connections(masks: Masks) -> int:
    """Total number of active connections."""
    return int(sum(int(masks[k].sum()) for k in LAYERS))


def _hidden_mature(
    stages: Dict[str, Int[Array, '...']], masks: Masks, n_features: int,
) -> Bool[Array, 'n_hidden']:
    """Which hidden units count as mature — Variant 2's pool of legal generation sources.

    `hidden_unit_stages` takes the most advanced stage down each *column* of the layer whose columns
    are the units in question. Here that layer is the hidden->output block of `w2`, so the skip
    columns are sliced off first — they belong to input units, which are mature from the first step.

    **A unit with no outgoing connections counts as mature**, which is where this departs from
    `hidden_unit_stages` (it reads the empty maximum as nascent). Maturity is defined over a unit's
    outgoing connections, so a unit that has none has no stage to read, and calling that "not yet
    mature" makes the state absorbing: it can never be chosen as a source, so it can never get an
    outgoing connection, so it can never mature. Worse, it is a sink — generation can still hand it
    *incoming* connections, and those are stranded, because with no outgoing connection its
    backpropagated error is exactly zero, its squared-error trace v decays to zero, and the prune
    test ``h² < α v / 4`` can then never fire. Treating it as mature lets generation revive it,
    after which error flows again and its incoming connections become prunable like any others.
    """
    hidden_block = slice(n_features, None)
    outgoing = masks['w2'][:, hidden_block]
    stage = hidden_unit_stages({'w2': stages['w2'][:, hidden_block]}, {'w2': outgoing})
    return (stage == MATURE) | (outgoing.sum(axis=0) == 0)


# ================================================================================ the learner
class DynamicAutostepMLP(eqx.Module):
    """`SkipMaskedMLP` trained online with Autostep, optionally restructuring every step.

    One `step` call is one sample: forward, manual backward, Autostep update, the per-unit traces
    and stage transitions, then (when `restructure`) the prune test and the matching generation.
    Everything is fixed-shape and branchless, so the whole thing scans under `jit`.
    """

    # --- static configuration
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    init_lr: float = eqx.field(static=True)
    l1: float = eqx.field(static=True)
    restructure: bool = eqx.field(static=True)
    gen_cap: int = eqx.field(static=True)
    require_mature_source: bool = eqx.field(static=True)

    # --- dynamic state
    model: SkipMaskedMLP
    opt_state: IDBDState
    # Per-connection stages plus the per-destination-unit squared-error trace v and its occupancy c
    # that the thresholds come from. v also drives the prune test, which is why the stage state is
    # carried even when nothing gates on maturity.
    stage_state: StageState
    grad_mask: Float[Array, 'n_outputs n_hidden']
    prune_exempt: Dict[str, Bool[Array, '...']]
    deficit: Int[Array, '']
    rng: random.PRNGKey

    @classmethod
    def init(
        cls,
        masks: Masks,
        meta_lr: float,
        key: random.PRNGKey,
        *,
        init_lr: float = 1e-6,
        l1: float = 0.0,
        activation: str = 'leaky_relu',
        restructure: bool = False,
        gen_cap: int = 256,
        require_mature_source: bool = False,
        protected: Optional[Float[Array, 'n_outputs n_features+n_hidden']] = None,
    ) -> 'DynamicAutostepMLP':
        """
        Args:
            masks: Initial connectivity
            meta_lr: Autostep meta step-size
            key: PRNG key (split for the weights and for generation)
            init_lr: Initial step-size of every connection, including generated ones
            l1: L1 coefficient, added to the gradient (the subgradient form, as in Experiment 1)
            activation: Hidden activation
            restructure: Whether to prune and generate connections online
            gen_cap: Maximum connections generated per step; see `step` for the accounting
            require_mature_source: 3b Variant 2 — a generated connection's source unit must be mature
            protected: Mask of hidden->output connections that are exempt from pruning and are the
                only ones carrying gradient into the hidden layer (3b's base version). None gives
                the standard backward pass with nothing protected.
        """
        weight_key, rng = random.split(key)
        model = SkipMaskedMLP(masks, activation=activation, key=weight_key)
        optimizer = optax_idbd(
            init_lr=init_lr, meta_lr=meta_lr, autostep=True, version='squared_inputs')

        n_features, n_hidden = model.n_features, model.n_hidden
        if protected is None:
            grad_mask = jnp.ones((model.n_outputs, n_hidden), jnp.float32)
            exempt = jnp.zeros_like(model.masks['w2'], bool)
        else:
            grad_mask = protected[:, n_features:].astype(jnp.float32)
            exempt = protected > 0

        return cls(
            optimizer=optimizer, init_lr=init_lr, l1=l1,
            restructure=restructure, gen_cap=gen_cap,
            require_mature_source=require_mature_source,
            model=model,
            opt_state=optimizer.init(model.weights),
            stage_state=init_stage_state(model.masks),
            grad_mask=grad_mask,
            prune_exempt={'w1': jnp.zeros_like(model.masks['w1'], bool), 'w2': exempt},
            deficit=jnp.array(0, jnp.int32),
            rng=rng,
        )

    # ---------------------------------------------------------------- forward and backward
    def loss_and_grads(
        self, x: Float[Array, 'n_features'], y: Float[Array, 'n_outputs'],
    ) -> Tuple[Float[Array, ''], Weights, Dict[str, Float[Array, 'n_units']], Weights]:
        """Forward pass and the explicit backward pass, without the L1 term.

        Written out rather than taken from `jax.value_and_grad` because the per-unit error signals
        are needed in their own right, and because the gradient mask (3b's base version: no error
        flows into the hidden layer except through the initial hidden->output connections) is one
        array multiply here. Up to that mask the gradients are exactly the autodiff ones — see
        ``tests/test_multi_geoff_dynamic_mlp.py``.

        Returns:
            Tuple of the summed squared error, the per-weight gradients, the per-unit error signals
            delta_u (signed so that the gradient is ``-delta_u * source``), and Autostep's
            `param_inputs`
        """
        model, masks = self.model, self.model.masks
        n_features = model.n_features

        out, hidden_pre, param_inputs = model(x)
        source = param_inputs['w2'][0]
        error = out - y
        loss = jnp.sum(error ** 2)   # summed over outputs, as in Experiment 1

        d_out = 2.0 * error
        d_hidden = (model.hidden_out_weights * masks['w2'][:, n_features:] * self.grad_mask).T @ d_out
        d_hidden_pre = d_hidden * model.activation_grad(hidden_pre)
        grads = {
            'w1': (d_hidden_pre[:, None] * x[None, :]) * masks['w1'],
            'w2': (d_out[:, None] * source[None, :]) * masks['w2'],
        }
        return loss, grads, {'w1': -d_hidden_pre, 'w2': -d_out}, param_inputs

    # ------------------------------------------------------------------------------ one step
    def step(
        self, x: Float[Array, 'n_features'], y: Float[Array, 'n_outputs'],
    ) -> Tuple['DynamicAutostepMLP', Dict[str, Array]]:
        """One online update.

        Returns:
            Tuple of the new learner and a dict of per-step scalars: the squared error summed over
            outputs (`loss`), how many connections were pruned and generated, and the current
            generation `deficit` (see `_restructure`).
        """
        model = self.model
        masks, weights = model.masks, model.weights

        loss, grads, delta, param_inputs = self.loss_and_grads(x, y)

        if self.l1 > 0.0:
            # Subgradient L1, matching Experiment 1: the penalty joins the gradient Autostep
            # consumes, so it shapes the h and v traces and the meta-update as well as the weight
            # update. (The plan's Variant 2 pseudocode writes the penalty into the weight update
            # only; the plan resolves the ambiguity by pointing at the Experiment 1 implementation.)
            grads = {k: g + self.l1 * jnp.sign(weights[k]) * masks[k] for k, g in grads.items()}

        # --- Autostep (Variant B: the curvature term is the squared source value)
        h_prev = self.opt_state.h
        updates, opt_state = self.optimizer.update(
            (grads, None, param_inputs), self.opt_state, weights)
        w_prev = weights
        weights = {k: (w_prev[k] + updates[k]) * masks[k] for k in LAYERS}
        alpha = {k: jnp.exp(opt_state.beta[k]) for k in LAYERS}

        # --- per-unit traces and stage transitions, exactly as Experiment 2 defines them
        stage_state, _, _ = update_stage_state(
            self.stage_state, masks, alpha, param_inputs, delta, weights, h_prev, opt_state.h)

        # --- the prune test: the weight crossed zero this step and h is jitter-sized. The trace v
        # starts at 0, so nothing is prunable until its destination unit has seen some error.
        prune = {
            k: ((masks[k] > 0) & (w_prev[k] * weights[k] < 0)
                & (opt_state.h[k] ** 2 < alpha[k] * stage_state.v[k][:, None] / 4.0)
                & ~self.prune_exempt[k])
            for k in LAYERS
        }

        zero = jnp.array(0, jnp.int32)
        rng, n_pruned, n_generated, deficit = self.rng, zero, zero, self.deficit
        model = tree_replace(model, weights=weights, masks=masks)
        if self.restructure:
            model, opt_state, stage_state, rng, n_pruned, n_generated, deficit = self._restructure(
                model, opt_state, stage_state, prune)

        return (
            tree_replace(self, model=model, opt_state=opt_state, stage_state=stage_state,
                         deficit=deficit, rng=rng),
            {'loss': loss, 'n_pruned': n_pruned, 'n_generated': n_generated, 'deficit': deficit},
        )

    # -------------------------------------------------------------------- prune and generate
    def _restructure(self, model, opt_state, stage_state, prune):
        """Apply the prune test, generate one replacement per pruned connection, reset both.

        Generation follows the plan: a destination uniformly over all hidden and output units, then
        a source uniformly over the units in prior layers that the destination is not already
        connected to (and, under `require_mature_source`, that are mature).

        The one place this is not literally the plan is *how many* are generated per step. Sampling
        replacements one at a time is not expressible at a fixed shape, so `gen_cap` proposals are
        drawn at once and the first `demand = min(pruned + deficit, gen_cap)` of them are applied.
        Two proposals can collide on the same pathway (or a proposal can land on a destination with
        no eligible source), in which case fewer connections are created than were pruned; the
        shortfall is carried in `deficit` and filled on later steps. So the connection count is
        exactly the budget minus the current deficit, and `deficit` is logged rather than hidden —
        if it does not sit at ~0, the budget is not being honoured and `gen_cap` is too small.
        """
        n_features, n_hidden, n_outputs = model.n_features, model.n_hidden, model.n_outputs
        n_units, width, cap = n_hidden + n_outputs, n_features + n_hidden, self.gen_cap

        masks = {k: jnp.where(prune[k], 0.0, model.masks[k]) for k in LAYERS}
        n_pruned = sum(jnp.sum(prune[k]) for k in LAYERS).astype(jnp.int32)

        rng, dest_key, source_key = random.split(self.rng, 3)
        demand = jnp.minimum(n_pruned + self.deficit, cap)
        destination = random.randint(dest_key, (cap,), 0, n_units)
        into_hidden = destination < n_hidden

        # Eligible sources per proposal, in the padded `width` layout so both destination kinds
        # share one array: a hidden destination may only draw from the inputs.
        eligible_hidden = jnp.concatenate([
            masks['w1'][jnp.minimum(destination, n_hidden - 1)] == 0,
            jnp.zeros((cap, n_hidden), bool),
        ], axis=1)
        eligible_output = masks['w2'][jnp.maximum(destination - n_hidden, 0)] == 0
        if self.require_mature_source:
            # A hidden unit's stage is the most advanced stage among its outgoing connections, so it
            # is mature once any of them is. Input units are mature from the first step.
            hidden_mature = _hidden_mature(stage_state.stage, masks, n_features)
            eligible_output = eligible_output & jnp.concatenate(
                [jnp.ones(n_features, bool), hidden_mature])[None, :]
        eligible = jnp.where(into_hidden[:, None], eligible_hidden, eligible_output)

        source = jnp.argmax(
            jnp.where(eligible, random.uniform(source_key, (cap, width)), -jnp.inf), axis=1)
        accepted = (jnp.arange(cap) < demand) & jnp.any(eligible, axis=1)

        # Scatter the accepted proposals. `max` makes duplicate indices harmless and lets rejected
        # proposals write a no-op 0, so the index clipping below only keeps them in range.
        generated = {
            'w1': jnp.zeros_like(masks['w1']).at[
                jnp.minimum(destination, n_hidden - 1),
                jnp.minimum(source, n_features - 1),
            ].max(jnp.where(accepted & into_hidden, 1.0, 0.0)),
            'w2': jnp.zeros_like(masks['w2']).at[
                jnp.maximum(destination - n_hidden, 0), source,
            ].max(jnp.where(accepted & ~into_hidden, 1.0, 0.0)),
        }
        masks = {k: jnp.maximum(masks[k], generated[k]) for k in LAYERS}
        n_generated = sum(jnp.sum(generated[k]) for k in LAYERS).astype(jnp.int32)
        deficit = n_pruned + self.deficit - n_generated

        # A pruned and a generated connection look identical afterwards: weight 0, h 0, Autostep's
        # normalizer 0, step-size back at `init_lr`, stage nascent. The per-*unit* traces are
        # deliberately untouched — see `connection_stages.reset_connections_at`.
        reset = {k: (prune[k] | (generated[k] > 0)).astype(jnp.float32) for k in LAYERS}
        model, opt_state, stage_state = reset_connections_at(
            tree_replace(model, masks=masks), opt_state, stage_state, reset)
        model = tree_replace(
            model, weights={k: model.weights[k] * masks[k] for k in LAYERS})
        return model, opt_state, stage_state, rng, n_pruned, n_generated, deficit

    # ------------------------------------------------------------------------------ readouts
    @property
    def alpha(self) -> Weights:
        """Per-connection step-sizes."""
        return {k: jnp.exp(self.opt_state.beta[k]) for k in LAYERS}

    @property
    def stages(self) -> Dict[str, Int[Array, '...']]:
        """Per-connection stages."""
        return self.stage_state.stage

    @property
    def hidden_mature(self) -> Bool[Array, 'n_hidden']:
        """Which hidden units are mature — those with at least one mature outgoing connection."""
        return _hidden_mature(self.stages, self.model.masks, self.model.n_features)


# ==================================================================================== metrics
def separation_metrics(
    model: SkipMaskedMLP,
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
) -> Dict[str, Float[Array, '']]:
    """Connectivity and signal separation, both in [0, 1] and averaged over output units.

    **Connectivity separation.** From an output unit, trace every backward path along connections
    toward the input layer — a direct skip connection is a length-1 path, and an
    output <- hidden <- input chain is a length-2 path. The metric is the fraction of those paths
    that terminate at an input belonging to the output's own subtask. Paths are counted, not inputs,
    so the same input reached along several paths counts several times.

    **Signal separation.** The same quantity with each path weighted by the absolute product of the
    weights along it, normalized per output unit.

    Both are computed as sums over paths without ever enumerating them::

        paths(o)      = sum_i M_skip[o, i]  +  sum_h M_hidden[o, h] sum_i M_1[h, i]
        same(o)       = the same two sums restricted to inputs of o's own task

    An output unit with no paths at all contributes NaN and is dropped from the average, rather than
    silently counting as perfectly separated.

    Args:
        model: The network
        input_task_ids: Subtask index of each input feature
        output_task_ids: Subtask index of each output unit

    Returns:
        ``{'connectivity_separation': ..., 'signal_separation': ...}``
    """
    n_features = model.n_features
    m1 = model.masks['w1']
    skip, hidden_out = model.masks['w2'][:, :n_features], model.masks['w2'][:, n_features:]
    same = (output_task_ids[:, None] == input_task_ids[None, :]).astype(jnp.float32)

    def fraction_same(a1, a_skip, a_hidden):
        total = a_skip.sum(axis=1) + a_hidden @ a1.sum(axis=1)
        matched = (a_skip * same).sum(axis=1) + jnp.einsum('oh,hi,oi->o', a_hidden, a1, same)
        return jnp.nanmean(jnp.where(total > 0, matched / jnp.where(total > 0, total, 1.0), jnp.nan))

    absw1 = jnp.abs(model.weights['w1'] * m1)
    return {
        'connectivity_separation': fraction_same(m1, skip, hidden_out),
        'signal_separation': fraction_same(
            absw1,
            jnp.abs(model.weights['w2'][:, :n_features] * skip),
            jnp.abs(model.weights['w2'][:, n_features:] * hidden_out),
        ),
    }


def structure_metrics(
    model: SkipMaskedMLP,
    input_task_ids: Int[Array, 'n_features'],
    output_task_ids: Int[Array, 'n_outputs'],
    stages: Optional[Dict[str, Int[Array, '...']]] = None,
    n_tasks: Optional[int] = None,
) -> Dict[str, Float[Array, '']]:
    """`separation_metrics` plus the connection-count and (optionally) stage summaries.

    Everything here is a scalar and jit-safe, so it can be emitted from inside the training scan.

    Args:
        model: The network
        input_task_ids: Subtask index of each input feature
        output_task_ids: Subtask index of each output unit
        stages: Per-connection stages; omit to skip the stage fractions
        n_tasks: Number of subtasks; given, adds `hidden_input_purity` — per hidden unit, the
            fraction of its incoming connections that come from its most-represented subtask,
            averaged over the hidden units that have any. 1 means every hidden unit reads a single
            subtask (the block-sparse ideal, one subnetwork per task); 1 / n_tasks means its inputs
            are spread evenly over all of them. Unlike the separation metrics this looks at the
            hidden layer alone, so it distinguishes "the output units happen to be reading clean
            paths" from "the hidden units have actually specialized".

    Returns:
        A flat dict of scalar metrics
    """
    n_features, n_hidden = model.n_features, model.n_hidden
    m1 = model.masks['w1']
    skip, hidden_out = model.masks['w2'][:, :n_features], model.masks['w2'][:, n_features:]

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

    if stages is not None:
        active = jnp.concatenate([m1.reshape(-1) > 0, model.masks['w2'].reshape(-1) > 0])
        stage = jnp.concatenate([stages['w1'].reshape(-1), stages['w2'].reshape(-1)])
        n_active = jnp.maximum(active.sum(), 1)
        for value, name in zip((NASCENT, GROWTH, MATURE), STAGE_NAMES):
            metrics[f'fraction_{name}'] = jnp.sum(active & (stage == value)) / n_active
        metrics['mature_hidden_fraction'] = jnp.mean(
            _hidden_mature(stages, model.masks, n_features))
    return metrics
