"""Analyze and plot feature utility trajectories from mixed_generation experiments.

Pulls unit_snapshots.npz from MLflow, reconstructs per-feature lifetimes, and
produces interactive plotly plots showing utility evolution of column-constrained
vs free units, with mean/median/pruning-threshold overlays.

Usage:
    python experiments/plot_mixed_generation.py --run-id <RUN_ID> [options]

Options:
    --run-id        MLflow run ID to pull data from
    --max-steps     Only plot up to this many training steps (default: all)
    --fractions     Slider fraction steps (default: 0.01 0.05 0.1)
    --seed          Which seed index to plot (default: 0)
    --output-dir    Directory to save HTML plots (default: ./plots)
    --prune-freq    Restructure frequency in training steps (default: 200)
"""

import argparse
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_snapshots(run_id: str) -> dict:
    """Download and load unit_snapshots.npz from an MLflow run."""
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = client.download_artifacts(
            run_id, 'snapshots', tmpdir)
        npz_files = list(Path(local_path).glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(
                f'No .npz files found in snapshots artifact for run {run_id}')
        data = dict(np.load(npz_files[0]))
    return data


def extract_lifetimes(data: dict, seed: int):
    """Extract per-feature lifetimes from snapshot data.

    A new feature lifetime starts when:
      - unit_mask transitions 0 → 1 (slot was empty, now occupied), OR
      - unit_mask stays 1 but age resets (unit was pruned and a new one was
        generated in the same restructure step — the most common case)

    Returns a list of dicts, each representing one feature lifetime:
        {layer, position, birth_idx, death_idx, column_tag,
         birth_step, death_step, utility_trajectory, step_indices,
         alive_at_end, pruned_same_step}
    """
    unit_mask = data['unit_mask'][seed]   # (T, L, U)
    utility = data['utility'][seed]       # (T, L, U)
    age = data['age'][seed]               # (T, L, U)
    column_tag = data['column_tag'][seed] # (T, L, U)
    step_indices = data['step_indices']   # (T,)

    T, L, U = unit_mask.shape
    lifetimes = []

    for l in range(L):
        for u in range(U):
            mask_seq = unit_mask[:, l, u]
            util_seq = utility[:, l, u]
            age_seq = age[:, l, u]
            tag_seq = column_tag[:, l, u]

            # Detect all birth events:
            # 1. mask 0→1 (normal birth)
            # 2. mask 1→1 but age resets to 0 while prev age > 0
            #    (prune+regen in same step)
            is_active = mask_seq.astype(bool)
            birth_mask = np.zeros(T, dtype=bool)

            # Type 1: mask transition 0→1
            if T > 1:
                birth_mask[1:] |= (~is_active[:-1]) & is_active[1:]
            # Active at very start
            if is_active[0]:
                birth_mask[0] = True

            # Type 2: age reset while staying active
            if T > 1:
                age_reset = is_active[:-1] & is_active[1:] & (age_seq[1:] == 0) & (age_seq[:-1] > 0)
                birth_mask[1:] |= age_reset

            birth_indices = np.where(birth_mask)[0]

            for i, birth_idx in enumerate(birth_indices):
                # Death is the next birth at this slot, or mask→0, or end
                # Find next birth
                next_birth = birth_indices[i + 1] if i + 1 < len(birth_indices) else T

                # Find next mask→0
                remaining_mask = mask_seq[birth_idx:]
                zeros = np.where(remaining_mask == 0)[0]
                next_zero = birth_idx + zeros[0] if zeros.size > 0 else T

                death_idx = min(next_birth, next_zero)
                alive_at_end = (death_idx >= T)
                # Was it pruned via same-step regen (death_idx == next_birth)?
                pruned_same_step = (death_idx == next_birth) and (death_idx < T)

                lifetimes.append({
                    'layer': l,
                    'position': u,
                    'birth_idx': int(birth_idx),
                    'death_idx': int(death_idx),
                    'birth_step': int(step_indices[birth_idx]),
                    'death_step': (int(step_indices[death_idx - 1])
                                   if death_idx < T
                                   else int(step_indices[-1])),
                    'column_tag': int(tag_seq[birth_idx]),
                    'utility_trajectory': util_seq[birth_idx:death_idx].copy(),
                    'step_indices': step_indices[birth_idx:death_idx].copy(),
                    'alive_at_end': alive_at_end,
                    'pruned_same_step': pruned_same_step,
                })

    return lifetimes


def _masked_stats(utility, mask):
    """Vectorized mean/median/5th-percentile over last axis with a boolean mask.

    Args:
        utility: (T, U) array of values
        mask:    (T, U) boolean array — True for active units

    Returns:
        mean, median, threshold arrays each of shape (T,)
    """
    T, U = utility.shape
    # Replace inactive with nan, then use nanmean/nanmedian/nanpercentile
    masked = np.where(mask, utility, np.nan)

    with np.errstate(all='ignore'):
        mean = np.nanmean(masked, axis=1)
        median = np.nanmedian(masked, axis=1)
        # Sort-based 5th percentile: sort each row, pick the index
        # corresponding to 5% of active count
        sorted_vals = np.sort(masked, axis=1)  # nans go to end
        n_active = mask.sum(axis=1)  # (T,)
        pct_idx = np.clip((n_active * 0.05).astype(int), 0, U - 1)
        threshold = sorted_vals[np.arange(T), pct_idx]

    # Fill nans (no active units) with 0
    mean = np.nan_to_num(mean, nan=0.0)
    median = np.nan_to_num(median, nan=0.0)
    threshold = np.nan_to_num(threshold, nan=0.0)
    return mean, median, threshold


def compute_aggregate_curves(data: dict, seed: int):
    """Compute mean/median utility and pruning threshold per timestep."""
    utility = data['utility'][seed]       # (T, L, U)
    unit_mask = data['unit_mask'][seed]   # (T, L, U)
    step_indices = data['step_indices']

    T, L, U = utility.shape

    # Global: flatten layers into unit dimension → (T, L*U)
    global_util = utility.reshape(T, -1)
    global_mask = unit_mask.reshape(T, -1).astype(bool)
    mean_util, median_util, threshold = _masked_stats(global_util, global_mask)

    # Per-layer
    per_layer = {}
    for l in range(L):
        m, med, thr = _masked_stats(utility[:, l], unit_mask[:, l].astype(bool))
        per_layer[l] = {'mean': m, 'median': med, 'threshold': thr}

    return {
        'step_indices': step_indices,
        'mean': mean_util,
        'median': median_util,
        'threshold': threshold,
        'per_layer': per_layer,
    }


def _build_batched_traces(lts, max_steps, tag, colors):
    """Build batched line/birth/death data for all lifetimes of one tag type.

    Lines use None separators so a single Scatter trace draws all segments.
    Birth and death markers are batched into single Scatter traces.
    """
    line_x, line_y = [], []
    birth_x, birth_y = [], []
    death_x, death_y = [], []

    for lt in lts:
        steps = lt['step_indices']
        utils = lt['utility_trajectory']
        if max_steps is not None:
            keep = steps <= max_steps
            steps = steps[keep]
            utils = utils[keep]
        if len(steps) == 0:
            continue

        line_x.extend(steps.tolist() + [None])
        line_y.extend(utils.tolist() + [None])
        birth_x.append(steps[0])
        birth_y.append(utils[0])
        if not lt['alive_at_end'] and len(steps) > 0:
            death_x.append(steps[-1])
            death_y.append(utils[-1])

    traces = []
    if line_x:
        traces.append(go.Scattergl(
            x=line_x, y=line_y, mode='lines',
            line=dict(color=colors['line'], width=0.8),
            showlegend=False, hoverinfo='skip',
        ))
    if birth_x:
        traces.append(go.Scattergl(
            x=birth_x, y=birth_y, mode='markers',
            marker=dict(color=colors['birth'], size=4, symbol='circle'),
            showlegend=False, hoverinfo='skip',
        ))
    if death_x:
        traces.append(go.Scattergl(
            x=death_x, y=death_y, mode='markers',
            marker=dict(color=colors['death'], size=5, symbol='x',
                        line=dict(width=1.5, color=colors['death'])),
            showlegend=False, hoverinfo='skip',
        ))
    return traces


def plot_layer(lifetimes, agg, layer, step_indices, max_steps, fractions, title):
    """Create a plotly figure for one layer (or global) with a fraction slider."""
    fig = go.Figure()

    # Filter lifetimes for this layer (or all for global)
    if layer is not None:
        lts = [lt for lt in lifetimes if lt['layer'] == layer]
    else:
        lts = lifetimes

    # Filter by max_steps
    if max_steps is not None:
        lts = [lt for lt in lts if lt['birth_step'] <= max_steps]

    # Assign each lifetime a stable random rank for slider-based filtering
    rng = np.random.default_rng(42)
    ranks = rng.random(len(lts))

    TAG_COLORS = {
        0: {'line': 'rgba(160, 160, 160, 0.25)', 'birth': 'rgba(120, 120, 120, 0.4)',
            'death': 'rgba(80, 80, 80, 0.6)'},
        1: {'line': 'rgba(30, 100, 220, 0.35)', 'birth': 'rgba(30, 100, 220, 0.6)',
            'death': 'rgba(15, 50, 110, 0.85)'},
        2: {'line': 'rgba(220, 50, 30, 0.35)', 'birth': 'rgba(220, 50, 30, 0.6)',
            'death': 'rgba(140, 25, 15, 0.85)'},
    }
    TAG_IDS = sorted(TAG_COLORS.keys())

    # Sort lifetimes by rank within each tag so we can slice for fractions
    tag_sorted = {}  # tag -> list of lifetimes sorted by rank (ascending)
    for tag in TAG_IDS:
        indices = [i for i, lt in enumerate(lts) if lt['column_tag'] == tag]
        indices.sort(key=lambda i: ranks[i])
        tag_sorted[tag] = [lts[i] for i in indices]

    # Pre-build batched traces for each slider fraction by slicing the sorted lists
    slider_fracs = sorted(fractions)
    traces_per_step = []
    for frac in slider_fracs:
        step_traces = []
        for tag in TAG_IDS:
            sorted_lts = tag_sorted[tag]
            n_keep = max(1, int(len(sorted_lts) * frac)) if sorted_lts else 0
            step_traces.extend(
                _build_batched_traces(sorted_lts[:n_keep], max_steps, tag, TAG_COLORS[tag]))
        traces_per_step.append(step_traces)

    # Add all traces for all steps, initially only the active step is visible
    active_idx = len(slider_fracs) - 1  # default to max

    trace_step_ranges = []  # (start, end) index in fig.data for each step
    for si, step_traces in enumerate(traces_per_step):
        start = len(fig.data)
        for tr in step_traces:
            tr.visible = (si == active_idx)
            fig.add_trace(tr)
        trace_step_ranges.append((start, len(fig.data)))

    n_lifetime_traces = len(fig.data)

    # Aggregate overlay lines (always visible)
    if layer is not None:
        agg_data = agg['per_layer'][layer]
    else:
        agg_data = agg

    s = agg['step_indices']
    if max_steps is not None:
        mask = s <= max_steps
        s = s[mask]
        mean_v = agg_data['mean'][mask]
        median_v = agg_data['median'][mask]
        thresh_v = agg_data['threshold'][mask]
    else:
        mean_v = agg_data['mean']
        median_v = agg_data['median']
        thresh_v = agg_data['threshold']

    fig.add_trace(go.Scatter(
        x=s, y=mean_v, mode='lines',
        line=dict(color='black', width=2), name='Mean utility',
    ))
    fig.add_trace(go.Scatter(
        x=s, y=median_v, mode='lines',
        line=dict(color='black', width=2, dash='dash'), name='Median utility',
    ))
    fig.add_trace(go.Scatter(
        x=s, y=thresh_v, mode='lines',
        line=dict(color='gray', width=1.5, dash='dot'), name='Prune threshold (5th pct)',
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='gray', width=2), name='Initial'))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='blue', width=2), name='Column-constrained'))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='red', width=2), name='Free'))

    n_total = len(fig.data)
    n_agg = n_total - n_lifetime_traces

    # Build slider steps — toggle visibility of pre-built trace groups
    steps = []
    for si, frac in enumerate(slider_fracs):
        vis = [False] * n_lifetime_traces + [True] * n_agg
        start, end = trace_step_ranges[si]
        for ti in range(start, end):
            vis[ti] = True
        steps.append(dict(
            method='restyle',
            args=[{'visible': vis}],
            label=f'{frac:.0%}',
        ))

    fig.update_layout(
        sliders=[dict(
            active=active_idx,
            currentvalue=dict(prefix='Fraction: '),
            pad=dict(t=50),
            steps=steps,
        )],
        title=title,
        xaxis_title='Training Step',
        yaxis_title='Utility (EMA)',
        height=600,
        width=1400,
    )
    return fig


def print_summary_stats(lifetimes, data, seed):
    """Print summary statistics answering Q2 and Q3."""
    utility = data['utility'][seed]
    unit_mask = data['unit_mask'][seed]
    column_tag = data['column_tag'][seed]
    step_indices = data['step_indices']
    T = utility.shape[0]

    TAG_NAMES = {0: 'Initial', 1: 'Column-constrained', 2: 'Free'}
    groups = {tag: [lt for lt in lifetimes if lt['column_tag'] == tag]
              for tag in sorted(TAG_NAMES)}

    print('\n' + '=' * 70)
    print('SUMMARY STATISTICS')
    print('=' * 70)

    # Q2: Average utility comparison
    print('\n--- Q2: Utility Comparison ---')
    for tag, name in TAG_NAMES.items():
        lts = groups[tag]
        if not lts:
            print(f'  {name}: no features')
            continue
        alive = [lt for lt in lts if lt['alive_at_end']]
        dead = [lt for lt in lts if not lt['alive_at_end']]
        final_utils = [lt['utility_trajectory'][-1] for lt in alive if len(lt['utility_trajectory']) > 0]
        mean_util = np.mean(final_utils) if final_utils else 0
        median_util = np.median(final_utils) if final_utils else 0
        print(f'  {name}:')
        print(f'    Total lifetimes:     {len(lts)}')
        print(f'    Alive at end:        {len(alive)} ({100*len(alive)/max(len(lts),1):.1f}%)')
        print(f'    Final mean utility:  {mean_util:.6f}')
        print(f'    Final median utility:{median_util:.6f}')

    # Q3: Survival and pruning dynamics
    print('\n--- Q3: Survival and Pruning Dynamics ---')
    for tag, name in TAG_NAMES.items():
        lts = groups[tag]
        if not lts:
            continue
        lifespans = [lt['death_step'] - lt['birth_step'] for lt in lts]
        alive = [lt for lt in lts if lt['alive_at_end']]
        dead = [lt for lt in lts if not lt['alive_at_end']]

        print(f'  {name}:')
        print(f'    Mean lifetime (steps):   {np.mean(lifespans):.0f}')
        print(f'    Median lifetime (steps): {np.median(lifespans):.0f}')
        print(f'    Survival rate:           {100*len(alive)/max(len(lts),1):.1f}%')
        if dead:
            dead_lifespans = [lt['death_step'] - lt['birth_step'] for lt in dead]
            print(f'    Mean lifetime (pruned):  {np.mean(dead_lifespans):.0f} steps')

    # Per-layer breakdown
    n_layers = utility.shape[1]
    print(f'\n--- Per-Layer Breakdown ---')
    for l in range(n_layers):
        print(f'  Layer {l}:')
        for tag, name in TAG_NAMES.items():
            layer_lts = [lt for lt in groups[tag] if lt['layer'] == l]
            n_alive = sum(1 for lt in layer_lts if lt['alive_at_end'])
            alive_utils = [lt['utility_trajectory'][-1] for lt in layer_lts
                           if lt['alive_at_end'] and len(lt['utility_trajectory']) > 0]
            util_str = f', mean final util={np.mean(alive_utils):.6f}' if alive_utils else ''
            print(f'    {name:22s}: {len(layer_lts):4d} total, {n_alive:4d} alive{util_str}')

    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Plot feature utility trajectories from mixed_generation')
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Plot up to this many training steps')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.01, 0.05, 0.1],
                        help='Slider fraction steps (e.g., 0.01 0.05 0.1 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed index to plot')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save HTML plots')
    args = parser.parse_args()

    print(f'Loading snapshots from run {args.run_id}...')
    data = load_snapshots(args.run_id)

    n_seeds = data['utility'].shape[0]
    n_layers = data['utility'].shape[2]
    print(f'Data: {data["utility"].shape[1]} timesteps, {n_seeds} seeds, '
          f'{n_layers} layers, {data["utility"].shape[3]} max units')

    seed = args.seed
    if seed >= n_seeds:
        raise ValueError(f'Seed index {seed} >= n_seeds {n_seeds}')

    print(f'Extracting feature lifetimes for seed {seed}...')
    lifetimes = extract_lifetimes(data, seed)
    print(f'Found {len(lifetimes)} feature lifetimes')

    initial = sum(1 for lt in lifetimes if lt['column_tag'] == 0)
    tagged = sum(1 for lt in lifetimes if lt['column_tag'] == 1)
    free = sum(1 for lt in lifetimes if lt['column_tag'] == 2)
    print(f'  Initial: {initial}, Column-constrained: {tagged}, Free: {free}')

    print('Computing aggregate curves...')
    agg = compute_aggregate_curves(data, seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-layer plots
    for l in range(n_layers):
        fig = plot_layer(
            lifetimes, agg, layer=l,
            step_indices=data['step_indices'],
            max_steps=args.max_steps, fractions=args.fractions,
            title=f'Feature Utility Trajectories — Layer {l}',
        )
        path = os.path.join(args.output_dir, f'layer_{l}_trajectories.html')
        fig.write_html(path)
        print(f'Saved {path}')

    # Global plot
    fig = plot_layer(
        lifetimes, agg, layer=None,
        step_indices=data['step_indices'],
        max_steps=args.max_steps, fractions=args.fractions,
        title='Feature Utility Trajectories — All Layers',
    )
    path = os.path.join(args.output_dir, 'global_trajectories.html')
    fig.write_html(path)
    print(f'Saved {path}')

    # Summary stats
    print_summary_stats(lifetimes, data, seed)


if __name__ == '__main__':
    main()
