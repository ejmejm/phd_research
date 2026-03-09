"""Connectivity manager for DynamicNetwork structure search.

Tracks per-unit utility and periodically prunes low-utility units,
replacing them with new randomly-connected units. Modeled after
CBPTracker from feature_search, adapted for sparse padded connectivity.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray

from phd.jax_core.optimizers import EqxOptimizer
from phd.structure_search.dynamic_network import DynamicNetwork, build_outgoing_indices


class UnitStats(eqx.Module):
    """Per-unit tracking statistics across all layers.

    Stored as padded 2D arrays matching DynamicNetwork's
    (max_layers, max_units_per_layer) layout. Inactive unit slots
    have age=0 and utility=0.

    Attributes:
        age: Steps since each unit was created or last reset.
            Incremented each step for active units only.
        utility: Exponential moving average of per-unit utility.
            Computed as input_magnitude * outgoing_weight_sum.
        replacement_accumulator: Fractional budget for replacements.
            Accumulates replace_rate * n_active_units each step;
            when >= 1, that many units can be pruned and replaced.
    """
    age: Int[Array, 'max_layers max_units_per_layer']
    utility: Float[Array, 'max_layers max_units_per_layer']
    replacement_accumulator: Float[Array, '']


class ConnectivityManager(eqx.Module):
    """Manages unit-level utility tracking and structural pruning/generation
    for DynamicNetwork.

    This is the structure search analog of CBPTracker from feature_search.
    It sits in TrainState, gets called each training step to update per-unit
    utility estimates, and periodically (every prune_frequency steps) prunes
    the lowest-utility units and generates new random replacements.

    Lifecycle in the training loop:
        1. Every step: call update_stats(model, buffer) to update the EMA
           utility of each active unit using the CBP utility formula, adapted
           for sparse padded connectivity.
        2. Every nth step: call modify_structure(model, optimizer) to prune
           the lowest-utility mature units and generate random replacements.

    Key differences from CBPTracker:
        - Operates on units in a sparse DynamicNetwork, not features in a
          dense MLP. All arrays are 2D (max_layers, max_units_per_layer)
          with masking for inactive/padded slots.
        - Utility computation accounts for padded connectivity: inactive
          connections (input_indices == -1) and inactive units (unit_mask == 0)
          are excluded from both the activation magnitude and the outgoing
          weight sum.
        - The outgoing weight sum for a unit includes ALL downstream
          connections: hidden-to-hidden contributions (from the outgoing_weights
          cache, which covers all consuming hidden layers) plus hidden-to-output
          contributions (from output_weights * output_mask).
        - Pruning deactivates units (unit_mask=0, clears incoming connections
          and output connections) without cascading to downstream units that
          had connections from the pruned unit. Those stale references are
          already masked out by unit_mask in the forward pass.
        - Instead of regenerating features in-place (as CBP does), new units
          are generated at a random layer with random input connections and a
          single output connection (weight=0). The new unit has no outgoing
          connections to other hidden units initially — it connects directly
          to the output layer only.
        - After structural changes, build_outgoing_indices() must be called
          to rebuild the reverse mapping used by the custom VJP backward pass.

    Utility formula (per unit, per step):
        step_utility = mean(|activation|) * sum(|outgoing_weights|)

        where activation is the unit's buffer value across the batch, and
        outgoing_weights includes all downstream connections (hidden + output).

        The EMA update is:
        utility = (1 - decay_rate) * step_utility + decay_rate * old_utility

    Replacement budget:
        Each step, replacement_accumulator += replace_rate * n_active_units.
        When modify_structure is called, floor(accumulator) units are pruned
        (subject to maturity_threshold eligibility). The accumulator is
        decremented by the number actually pruned.

    Attributes:
        decay_rate: EMA decay for utility tracking. Higher values give more
            weight to historical utility. Typical value: 0.99.
        maturity_threshold: Minimum age (in steps) before a unit is eligible
            for pruning. Protects newly created units from immediate removal.
            Set to -1 to disable (all units eligible).
        replace_rate: Fraction of active units to budget for replacement per
            step. Accumulated fractionally until >= 1 unit can be replaced.
        unit_stats: Current per-unit statistics (age, utility, accumulator).
        rng: PRNG key for random operations (unit generation, tie-breaking).
    """

    # Static config (not updated during training)
    decay_rate: float = eqx.field(static=True)
    maturity_threshold: int = eqx.field(static=True)

    # Dynamic state (carried in TrainState, updated each step)
    replace_rate: float
    unit_stats: UnitStats
    rng: PRNGKeyArray

    def __init__(
        self,
        model: DynamicNetwork,
        replace_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = -1,
        *,
        rng: PRNGKeyArray,
    ):
        """Initialize ConnectivityManager from a DynamicNetwork.

        Creates zero-initialized UnitStats matching the network's padded
        layout. Active units (per model.unit_mask) start with age=0 and
        utility=0; inactive slots remain zeroed throughout.

        Args:
            model: The DynamicNetwork whose structure will be managed.
                Used to determine array shapes (max_layers, max_units_per_layer).
            replace_rate: Fraction of total active units to replace per step.
                Accumulated fractionally; pruning happens when >= 1.
            decay_rate: EMA decay for utility tracking. Values closer to 1
                give more weight to historical utility.
            maturity_threshold: Minimum age (steps) before a unit is eligible
                for pruning. Set to -1 to disable the threshold.
            rng: PRNG key for random operations.
        """
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.replace_rate = replace_rate
        self.rng = rng

        shape = (model.max_layers, model.max_units_per_layer)
        self.unit_stats = UnitStats(
            age=jnp.zeros(shape, dtype=jnp.int32),
            utility=jnp.zeros(shape, dtype=jnp.float32),
            replacement_accumulator=jnp.array(0.0),
        )

    def update_stats(
        self,
        model: DynamicNetwork,
        buffer: Float[Array, 'batch_size buffer_size'],
    ) -> 'ConnectivityManager':
        """Update per-unit utility estimates using CBP-style formula.

        Called every training step. For each active unit at position (l, u):

        1. Reads the unit's activation from the buffer at position
           (input_dim + l * max_units_per_layer + u), computes the mean
           absolute activation across the batch.

        2. Computes the total absolute outgoing weight:
           - Hidden-to-hidden: sum of |outgoing_weights[l', buf_pos, :]|
             for all consuming layers l'. The outgoing_weights array is
             already zero-masked by sync_outgoing_weights for inactive slots.
           - Hidden-to-output: sum of |output_weights[:, buf_pos]| masked
             by output_mask[:, buf_pos].

        3. step_utility = mean_abs_activation * total_abs_outgoing_weight

        4. Updates the EMA:
           utility = (1 - decay_rate) * step_utility + decay_rate * old_utility

        5. Increments age for all active units (unit_mask == 1).

        6. Accumulates replacement budget:
           accumulator += replace_rate * n_active_units

        Inactive units (unit_mask == 0) are not updated; their stats
        remain at whatever they were (typically zero from initialization
        or from being pruned).

        Args:
            model: Current DynamicNetwork with up-to-date weights, masks,
                and outgoing_weights cache.
            buffer: Activation buffer from the forward pass, shape
                (batch_size, buffer_size). This is the param_inputs
                returned by DynamicNetwork.__call__ via vmap.

        Returns:
            Updated ConnectivityManager with new unit_stats.
        """
        return self  # Stub

    def modify_structure(
        self,
        model: DynamicNetwork,
        optimizer: EqxOptimizer,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple['ConnectivityManager', DynamicNetwork, EqxOptimizer]:
        """Prune lowest-utility units and generate new random replacements.

        Called every prune_frequency steps. The number of units to prune
        is floor(replacement_accumulator), subject to maturity eligibility.

        Steps:
        1. Compute prune mask via _make_prune_mask: select the N units with
           lowest utility among those exceeding maturity_threshold.

        2. Prune selected units:
           - Set unit_mask[l, u] = 0
           - Set input_indices[l, u, :] = -1 (clear incoming connections)
           - Set weights[l, u, :] = 0
           - Clear output connections: output_mask[:, buf_pos] = 0,
             output_weights[:, buf_pos] = 0
           - Reset unit_stats (age=0, utility=0) for pruned slots

        3. Generate replacement units (one per pruned unit):
           - Choose a random layer (uniform over 0..max_layers-1)
           - Find an inactive slot in that layer (unit_mask == 0)
           - If no inactive slot available in chosen layer, try others
           - Choose random input connections from available buffer positions
             (all positions from prior layers: inputs + earlier hidden units)
           - Initialize incoming weights with lecun_uniform
           - Connect to output layer with weight 0:
             output_mask[:, buf_pos] = 1, output_weights[:, buf_pos] = 0
           - No outgoing connections to other hidden units
           - Set unit_mask[l, u] = 1, age = 0

        4. Call build_outgoing_indices(model) to rebuild the reverse mapping.

        5. Reset optimizer state for all affected weight positions via
           _reset_optimizer_state.

        6. Decrement replacement_accumulator by the number of units pruned.

        Args:
            model: Current DynamicNetwork.
            optimizer: Current EqxOptimizer (state will be reset for
                weights associated with pruned/generated units).
            rng: PRNG key for random layer/connection selection and
                weight initialization.

        Returns:
            (connectivity_manager, model, optimizer) — all potentially modified.
        """
        return self, model, optimizer  # Stub

    def _compute_unit_utility(
        self,
        model: DynamicNetwork,
        buffer: Float[Array, 'batch_size buffer_size'],
    ) -> Float[Array, 'max_layers max_units_per_layer']:
        """Compute per-unit step utility from buffer activations and weights.

        For each unit at layer l, position u:
            buf_pos = input_dim + l * max_units_per_layer + u
            activation = buffer[:, buf_pos]  (batch_size,)
            mean_abs_act = mean(|activation|)

            # Hidden-to-hidden outgoing weights (all consuming layers)
            h2h = sum over l' of sum(|outgoing_weights[l', buf_pos, :]|)

            # Hidden-to-output weights
            h2o = sum(|output_weights[:, buf_pos]| * output_mask[:, buf_pos])

            step_utility = mean_abs_act * (h2h + h2o)

        Inactive units (unit_mask == 0) get step_utility = 0.

        Args:
            model: DynamicNetwork with current weights and structure.
            buffer: (batch_size, buffer_size) activation buffer.

        Returns:
            (max_layers, max_units_per_layer) step utility per unit slot.
        """
        raise NotImplementedError

    def _make_prune_mask(
        self,
        unit_mask: Int[Array, 'max_layers max_units_per_layer'],
        *,
        rng: PRNGKeyArray,
    ) -> Tuple[Bool[Array, 'max_layers max_units_per_layer'], Int[Array, '']]:
        """Determine which units to prune based on utility ranking.

        Uses the same approach as CBPTracker._make_prune_mask:

        1. Compute eligibility: unit must be active (unit_mask == 1) and
           have age > maturity_threshold (or threshold disabled).

        2. Determine n_replacements = min(floor(accumulator), n_eligible).

        3. Among eligible units, find the n_replacements with lowest utility.
           Utilities are perturbed slightly with random noise to break ties.

        4. Return boolean mask and count.

        Args:
            unit_mask: Active unit mask from DynamicNetwork.
            rng: PRNG key for tie-breaking perturbation.

        Returns:
            (prune_mask, n_pruned) where prune_mask is boolean over
            (max_layers, max_units_per_layer) and n_pruned is a scalar.
        """
        raise NotImplementedError

    def _prune_unit(
        self,
        model: DynamicNetwork,
        layer: int,
        unit_idx: int,
    ) -> DynamicNetwork:
        """Deactivate a single unit at the given layer and slot.

        Sets unit_mask[layer, unit_idx] = 0, clears its incoming
        connections (input_indices = -1, weights = 0), and removes
        its output connections (output_mask = 0, output_weights = 0).

        Does NOT cascade to downstream units — stale references in
        other units' input_indices are harmlessly masked by unit_mask
        during the forward pass.

        Args:
            model: DynamicNetwork to modify.
            layer: Layer index of the unit to prune.
            unit_idx: Unit slot index within the layer.

        Returns:
            Updated DynamicNetwork with the unit deactivated.
        """
        raise NotImplementedError

    def _generate_unit(
        self,
        model: DynamicNetwork,
        layer: int,
        unit_idx: int,
        *,
        rng: PRNGKeyArray,
    ) -> DynamicNetwork:
        """Generate a single new unit at the given layer and slot.

        The new unit:
        - Gets random input connections sampled from available buffer
          positions (input features + hidden units from all prior layers).
          The number of connections is min(n_available, max_connections_per_unit).
        - Incoming weights initialized with lecun_uniform.
        - Connects to ALL output dimensions with weight 0:
          output_mask[:, buf_pos] = 1, output_weights[:, buf_pos] = 0.
        - Has NO outgoing connections to other hidden units initially.
          Other units must independently form connections to this unit
          through future structural modifications.
        - unit_mask[layer, unit_idx] = 1.
        - Activation function assigned randomly from available activations,
          or defaults to the first activation (e.g., relu).

        Args:
            model: DynamicNetwork to modify.
            layer: Layer index for the new unit.
            unit_idx: Unit slot index within the layer.
            rng: PRNG key for connection sampling and weight initialization.

        Returns:
            Updated DynamicNetwork with the new unit added.
        """
        raise NotImplementedError

    def _reset_optimizer_state(
        self,
        optimizer: EqxOptimizer,
        model: DynamicNetwork,
        prune_mask: Bool[Array, 'max_layers max_units_per_layer'],
    ) -> EqxOptimizer:
        """Reset optimizer state for weights associated with pruned/generated units.

        For each unit flagged in prune_mask at (layer, unit_idx):
        - Zero the optimizer state for weights[layer, unit_idx, :] (incoming
          connections to this unit).
        - Compute buf_pos = input_dim + layer * max_units_per_layer + unit_idx
          and zero the optimizer state for output_weights[:, buf_pos]
          (outgoing connections to the output layer).

        This ensures the optimizer doesn't carry stale momentum/variance
        from the old unit into the new replacement.

        Note: DynamicNetwork's optimizer filter_spec only selects `weights`
        and `output_weights` as trainable, so only those state entries
        need resetting.

        Args:
            optimizer: Current EqxOptimizer.
            model: DynamicNetwork (for computing buffer positions from
                input_dim and max_units_per_layer).
            prune_mask: Boolean mask of units being pruned/regenerated,
                shape (max_layers, max_units_per_layer).

        Returns:
            Updated EqxOptimizer with reset state for affected weights.
        """
        raise NotImplementedError
