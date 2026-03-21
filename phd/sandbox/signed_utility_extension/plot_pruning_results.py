"""Plot results from the pruning experiment using Plotly (interactive HTML).

Produces 4 figures:
1. Learning curves of top-N methods + baselines
2. Box plots of avg_loss per config option (with IQM markers)
3. Distribution of avg_loss across all methods + baseline markers
4. Box plots of avg_accuracy per config option
"""

import argparse
import os
import pickle

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASELINE_NAMES = {'Contribution', 'UPGD', 'SI'}
CONFIG_FIELDS = ['child_score_method', 'output_normalization', 'pseudo_error',
                 'hidden_normalization', 'cap']

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def load_results(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data.get('config', {})


def filter_valid(results, metric='avg_loss'):
    """Filter out methods with NaN/Inf in the target metric."""
    return {k: v for k, v in results.items() if np.isfinite(v.get(metric, np.nan))}


def iqm(values):
    """Interquartile mean: mean of values between 25th and 75th percentile."""
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return np.mean(values[mask]) if mask.any() else np.nan


def smooth(values, window=10):
    """Simple moving average, NaN-aware."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


BASELINE_COLORS = {'Contribution': '#d62728', 'UPGD': '#9467bd', 'SI': '#8c564b'}
BASELINE_DASHES = {'Contribution': 'dash', 'UPGD': 'dashdot', 'SI': 'dot'}


# ==============================================================================
# Figure 1: Learning curves (top N + baselines)
# ==============================================================================

def plot_learning_curves(results, n_top=5, save_path=None):
    """Plot smoothed loss curves for the top-N methods and all baselines."""
    valid = filter_valid(results)
    if not valid:
        print("No valid results to plot.")
        return

    sorted_methods = sorted(valid.items(), key=lambda x: x[1]['avg_loss'])

    top_methods = []
    for name, r in sorted_methods:
        if name not in BASELINE_NAMES and len(top_methods) < n_top:
            top_methods.append((name, r))

    fig = go.Figure()

    # Baselines
    for name in ['Contribution', 'UPGD', 'SI']:
        if name in valid:
            r = valid[name]
            y = smooth(r['loss_history'], window=5)
            fig.add_trace(go.Scatter(
                y=y, mode='lines',
                name=f'{name} (avg={r["avg_loss"]:.4f})',
                line=dict(color=BASELINE_COLORS[name], dash=BASELINE_DASHES[name], width=2),
            ))

    # Top methods
    for name, r in top_methods:
        y = smooth(r['loss_history'], window=5)
        short_name = name[:50] + '...' if len(name) > 50 else name
        fig.add_trace(go.Scatter(
            y=y, mode='lines',
            name=f'{short_name} (avg={r["avg_loss"]:.4f})',
            line=dict(width=1.5),
        ))

    fig.update_layout(
        title=f'Learning Curves: Top {n_top} Methods + Baselines',
        xaxis_title='Log step', yaxis_title='Loss',
        legend=dict(font=dict(size=9)),
        hovermode='x unified',
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")


# ==============================================================================
# Figure 2 & 4: Box plots per config option
# ==============================================================================

def plot_config_boxplots(results, metric='avg_loss', save_path=None):
    """Box plots of a metric grouped by each config option value."""
    configurable = {k: v for k, v in results.items()
                    if v.get('config') is not None and np.isfinite(v.get(metric, np.nan))}

    if not configurable:
        print(f"No valid configurable results for {metric}.")
        return

    fig = make_subplots(
        rows=1, cols=len(CONFIG_FIELDS),
        subplot_titles=[f.replace('_', ' ').title() for f in CONFIG_FIELDS],
        shared_yaxes=True,
    )

    for col, field in enumerate(CONFIG_FIELDS, 1):
        groups = {}
        for name, r in configurable.items():
            val = str(getattr(r['config'], field))
            groups.setdefault(val, []).append((name, r[metric]))

        labels = sorted(groups.keys())

        for label in labels:
            names_and_vals = groups[label]
            vals = [v for _, v in names_and_vals]
            method_names = [n for n, _ in names_and_vals]
            fig.add_trace(go.Box(
                y=vals, name=label,
                text=method_names,
                boxpoints='all', jitter=0.3, pointpos=-1.5,
                marker=dict(size=3, opacity=0.5),
                legendgroup=field, showlegend=False,
            ), row=1, col=col)

        # IQM markers
        for label in labels:
            vals = np.array([v for _, v in groups[label]])
            if len(vals) > 0:
                iqm_val = iqm(vals)
                fig.add_trace(go.Scatter(
                    x=[label], y=[iqm_val], mode='markers',
                    marker=dict(symbol='diamond', size=10, color='red'),
                    name='IQM', legendgroup='iqm',
                    showlegend=(col == 1 and label == labels[0]),
                    hovertext=f'IQM: {iqm_val:.6f}',
                ), row=1, col=col)

    metric_label = metric.replace('_', ' ').title()
    fig.update_layout(
        title=f'{metric_label} by Config Option',
        height=500, width=300 * len(CONFIG_FIELDS),
        hovermode='closest',
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")


# ==============================================================================
# Figure 3: Distribution of all methods + baseline markers
# ==============================================================================

def plot_distribution(results, save_path=None):
    """Histogram of avg_loss for all configurable methods + baseline vertical lines."""
    valid = filter_valid(results)

    config_vals = [(k, v['avg_loss']) for k, v in valid.items()
                   if v.get('config') is not None]

    if not config_vals:
        print("No valid configurable results for distribution plot.")
        return

    names, vals = zip(*config_vals)
    vals = np.array(vals)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=vals, nbinsx=30, name=f'Configurable methods (n={len(vals)})',
        marker_color='steelblue', opacity=0.7,
        hovertext=list(names),
    ))

    for bname in ['Contribution', 'UPGD', 'SI']:
        if bname in valid:
            val = valid[bname]['avg_loss']
            fig.add_vline(
                x=val, line_dash=BASELINE_DASHES[bname],
                line_color=BASELINE_COLORS[bname], line_width=2,
                annotation_text=f'{bname} ({val:.4f})',
                annotation_position='top',
            )

    fig.update_layout(
        title='Distribution of Average Loss Across All Methods',
        xaxis_title='Average Loss', yaxis_title='Count',
        hovermode='closest',
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot pruning experiment results')
    parser.add_argument('results_file', type=str, default='results.pkl', nargs='?',
                        help='Path to results.pkl')
    parser.add_argument('--n_top', type=int, default=10,
                        help='Number of top methods to show in learning curves')
    parser.add_argument('--output_dir', type=str, default=FIGURE_DIR,
                        help='Directory to save figures')
    args = parser.parse_args()

    results, cfg = load_results(args.results_file)
    os.makedirs(args.output_dir, exist_ok=True)

    n_valid = len(filter_valid(results))
    n_total = len(results)
    print(f"Loaded {n_total} methods, {n_valid} valid (non-NaN)")

    plot_learning_curves(
        results, n_top=args.n_top,
        save_path=os.path.join(args.output_dir, 'fig1_learning_curves.html'))

    plot_config_boxplots(
        results, metric='avg_loss',
        save_path=os.path.join(args.output_dir, 'fig2_loss_boxplots.html'))

    plot_distribution(
        results,
        save_path=os.path.join(args.output_dir, 'fig3_loss_distribution.html'))

    plot_config_boxplots(
        results, metric='avg_accuracy',
        save_path=os.path.join(args.output_dir, 'fig4_accuracy_boxplots.html'))

    print("All figures generated.")


if __name__ == '__main__':
    main()
