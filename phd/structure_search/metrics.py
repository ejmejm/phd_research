import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class StepMetrics(eqx.Module):
    loss: jax.Array
    correct: jax.Array
    pruned_per_layer: jax.Array   # (max_layers,) — 0 on non-restructure steps
    generated_per_layer: jax.Array  # (max_layers,) — 0 on non-restructure steps


def compute_structure_metrics(train_state) -> dict:
    """Compute per-layer structure metrics from a (possibly vmapped) TrainState.

    Works with both single and multi-seed train_state (extra leading seed dim).
    Returns a flat dict of scalar metrics suitable for logging.
    """
    model = train_state.model
    if not hasattr(model, 'unit_mask'):
        return {}

    unit_mask = np.array(model.unit_mask)        # (n_layers, max_units)  or  (n_seeds, n_layers, max_units)
    input_indices = np.array(model.input_indices) # (n_layers, max_units, max_conns)  or  (n_seeds, ...)
    output_mask = np.array(model.output_mask)     # (output_dim, buffer_size)  or  (n_seeds, ...)

    # Handle vmapped (multi-seed) by averaging across seeds
    multi_seed = unit_mask.ndim == 3
    if multi_seed:
        n_seeds = unit_mask.shape[0]
    else:
        unit_mask = unit_mask[None]
        input_indices = input_indices[None]
        output_mask = output_mask[None]
        n_seeds = 1

    n_layers = model.max_layers if multi_seed else model.max_layers
    max_units = model.max_units_per_layer if multi_seed else model.max_units_per_layer
    input_dim = model.input_dim if multi_seed else model.input_dim
    output_dim = model.output_dim if multi_seed else model.output_dim

    metrics = {}

    # Per-layer metrics (averaged across seeds)
    for l in range(n_layers):
        # Active units
        active = unit_mask[:, l]  # (n_seeds, max_units)
        n_active = active.sum(axis=1).mean()  # scalar
        metrics[f'layer_{l}/active_units'] = float(n_active)

        # Average incoming connections per active unit
        layer_indices = input_indices[:, l]  # (n_seeds, max_units, max_conns)
        incoming_per_unit = (layer_indices >= 0).sum(axis=2)  # (n_seeds, max_units)
        # Mean over active units only (per seed, then average)
        avg_incoming = []
        for s in range(n_seeds):
            active_mask = active[s].astype(bool)
            if active_mask.any():
                avg_incoming.append(incoming_per_unit[s][active_mask].mean())
            else:
                avg_incoming.append(0.0)
        metrics[f'layer_{l}/avg_incoming_conns'] = float(np.mean(avg_incoming))

        # Average outgoing connections per active unit (to output layer)
        # Buffer positions for this layer's units
        offsets = input_dim + l * max_units
        buf_positions = np.arange(offsets, offsets + max_units)
        outgoing_to_output = output_mask[:, :, buf_positions].sum(axis=1)  # (n_seeds, max_units)

        # Hidden-to-hidden outgoing: count how many times each unit in this layer
        # is referenced in input_indices of later layers
        outgoing_to_hidden = np.zeros((n_seeds, max_units))
        for l2 in range(l + 1, n_layers):
            later_indices = input_indices[:, l2]  # (n_seeds, max_units_later, max_conns)
            # For each connection, check if it points to a buffer position in this layer
            # buf_positions for layer l: [offsets, offsets+1, ..., offsets+max_units-1]
            shifted = later_indices - offsets  # (n_seeds, max_units_later, max_conns)
            in_range = (shifted >= 0) & (shifted < max_units)
            # Bin count: for each seed, count references to each unit in this layer
            for s in range(n_seeds):
                valid = in_range[s]
                if valid.any():
                    unit_refs = shifted[s][valid]
                    outgoing_to_hidden[s] += np.bincount(unit_refs, minlength=max_units)[:max_units]

        total_outgoing = outgoing_to_output + outgoing_to_hidden  # (n_seeds, max_units)
        avg_outgoing = []
        for s in range(n_seeds):
            active_mask = active[s].astype(bool)
            if active_mask.any():
                avg_outgoing.append(total_outgoing[s][active_mask].mean())
            else:
                avg_outgoing.append(0.0)
        metrics[f'layer_{l}/avg_outgoing_conns'] = float(np.mean(avg_outgoing))

    # Average incoming connections per output unit (averaged across seeds)
    output_incoming = output_mask.sum(axis=2)  # (n_seeds, output_dim)
    metrics['avg_output_incoming_conns'] = float(output_incoming.mean())

    # Total active connections (averaged across seeds)
    total_incoming = (input_indices >= 0).sum(axis=(1, 2, 3))  # (n_seeds,)
    total_output = output_mask.sum(axis=(1, 2))  # (n_seeds,)
    metrics['total_active_connections'] = float((total_incoming + total_output).mean())

    # Utility metrics (from structure tracker)
    tracker = train_state.structure_tracker
    if hasattr(tracker, 'unit_stats'):
        utility = np.array(tracker.unit_stats.utility)  # (n_layers, max_units) or (n_seeds, ...)
        if utility.ndim == 2:
            utility = utility[None]

        # Median utility across all active units
        all_median_utils = []
        for s in range(n_seeds):
            active_mask = unit_mask[s].astype(bool)
            if active_mask.any():
                all_median_utils.append(float(np.median(utility[s][active_mask])))
            else:
                all_median_utils.append(0.0)
        metrics['median_utility'] = float(np.mean(all_median_utils))

        # Per-layer average utility
        for l in range(n_layers):
            avg_utils = []
            for s in range(n_seeds):
                active_mask = unit_mask[s, l].astype(bool)
                if active_mask.any():
                    avg_utils.append(float(utility[s, l][active_mask].mean()))
                else:
                    avg_utils.append(0.0)
            metrics[f'layer_{l}/avg_utility'] = float(np.mean(avg_utils))

    elif hasattr(tracker, 'connection_stats'):
        hidden_util = np.array(tracker.connection_stats.hidden_utility)
        output_util = np.array(tracker.connection_stats.output_utility)
        if hidden_util.ndim == 3:
            hidden_util = hidden_util[None]
            output_util = output_util[None]

        all_median_conn_utils = []
        for s in range(n_seeds):
            h_active = (input_indices[s] >= 0)
            o_active = (output_mask[s] == 1)
            all_active_utils = np.concatenate([
                hidden_util[s][h_active],
                output_util[s][o_active],
            ])
            if len(all_active_utils) > 0:
                all_median_conn_utils.append(float(np.median(all_active_utils)))
            else:
                all_median_conn_utils.append(0.0)
        metrics['median_connection_utility'] = float(np.mean(all_median_conn_utils))

        # Per-layer average connection utility (hidden connections only)
        for l in range(n_layers):
            avg_conn_utils = []
            for s in range(n_seeds):
                h_active_l = (input_indices[s, l] >= 0)
                active_utils = hidden_util[s, l][h_active_l]
                if len(active_utils) > 0:
                    avg_conn_utils.append(float(active_utils.mean()))
                else:
                    avg_conn_utils.append(0.0)
            metrics[f'layer_{l}/avg_connection_utility'] = float(np.mean(avg_conn_utils))

    return metrics
