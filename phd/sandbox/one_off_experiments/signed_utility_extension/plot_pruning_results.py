"""Plot results from the pruning experiment.

Produces 4 figures:
1. Learning curves of top-N methods + baselines
2. Box plots of avg_loss per config option (with IQM and median)
3. Distribution of avg_loss across all methods + baseline markers
4. Box plots of avg_accuracy per config option
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

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
    # Pad to avoid edge effects
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# ==============================================================================
# Figure 1: Learning curves (top N + baselines)
# ==============================================================================

def plot_learning_curves(results, n_top=5, save_path=None):
    """Plot smoothed loss curves for the top-N methods and all baselines."""
    valid = filter_valid(results)
    if not valid:
        print("No valid results to plot.")
        return

    # Sort by avg_loss
    sorted_methods = sorted(valid.items(), key=lambda x: x[1]['avg_loss'])

    # Collect top-N non-baseline methods
    top_methods = []
    for name, r in sorted_methods:
        if name not in BASELINE_NAMES and len(top_methods) < n_top:
            top_methods.append((name, r))

    # Baselines
    baselines = [(name, valid[name]) for name in BASELINE_NAMES if name in valid]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot baselines with distinct styles
    baseline_styles = {'Contribution': ('--', 'C3'), 'UPGD': ('-.', 'C4'), 'SI': (':', 'C5')}
    for name, r in baselines:
        ls, color = baseline_styles.get(name, ('-', 'gray'))
        y = smooth(r['loss_history'], window=5)
        ax.plot(y, linestyle=ls, color=color, linewidth=2, label=f'{name} (avg={r["avg_loss"]:.4f})')

    # Plot top methods
    for i, (name, r) in enumerate(top_methods):
        y = smooth(r['loss_history'], window=5)
        short_name = name[:50] + '...' if len(name) > 50 else name
        ax.plot(y, linewidth=1.5, label=f'{short_name} (avg={r["avg_loss"]:.4f})')

    ax.set_xlabel('Log step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Curves: Top {n_top} Methods + Baselines')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ==============================================================================
# Figure 2 & 4: Box plots per config option
# ==============================================================================

def plot_config_boxplots(results, metric='avg_loss', save_path=None):
    """Box plots of a metric grouped by each config option value."""
    # Only include configurable methods (not baselines)
    configurable = {k: v for k, v in results.items()
                    if v.get('config') is not None and np.isfinite(v.get(metric, np.nan))}

    if not configurable:
        print(f"No valid configurable results for {metric}.")
        return

    fig, axes = plt.subplots(1, len(CONFIG_FIELDS), figsize=(4 * len(CONFIG_FIELDS), 5))
    if len(CONFIG_FIELDS) == 1:
        axes = [axes]

    for ax, field in zip(axes, CONFIG_FIELDS):
        # Group by field value
        groups = {}
        for name, r in configurable.items():
            val = str(getattr(r['config'], field))
            groups.setdefault(val, []).append(r[metric])

        labels = sorted(groups.keys())
        data = [np.array(groups[l]) for l in labels]

        # Robust y-limits: clip at 95th percentile of all data
        all_vals = np.concatenate(data)
        if len(all_vals) > 0:
            ylim_upper = np.percentile(all_vals, 95) * 1.1

        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        # Add IQM and median markers
        for i, d in enumerate(data):
            if len(d) > 0:
                iqm_val = iqm(d)
                ax.plot(i + 1, iqm_val, 'D', color='red', markersize=6, zorder=5,
                        label='IQM' if i == 0 else '')

        ax.set_title(field.replace('_', ' ').title(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        if len(all_vals) > 0:
            ax.set_ylim(top=ylim_upper)
        ax.grid(True, alpha=0.3, axis='y')

    # Add legend for IQM marker
    axes[0].legend(fontsize=8)

    metric_label = metric.replace('_', ' ').title()
    fig.suptitle(f'{metric_label} by Config Option', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ==============================================================================
# Figure 3: Distribution of all methods + baseline markers
# ==============================================================================

def plot_distribution(results, save_path=None):
    """Histogram/KDE of avg_loss for all configurable methods + baseline vertical lines."""
    valid = filter_valid(results)

    # Configurable methods only
    config_vals = [v['avg_loss'] for k, v in valid.items()
                   if v.get('config') is not None]

    if not config_vals:
        print("No valid configurable results for distribution plot.")
        return

    config_vals = np.array(config_vals)

    # Robust limits: clip extreme outliers
    p5, p95 = np.percentile(config_vals, [5, 95])
    margin = (p95 - p5) * 0.3
    xlim = (p5 - margin, p95 + margin)
    clipped = config_vals[(config_vals >= xlim[0]) & (config_vals <= xlim[1])]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(clipped, bins=30, alpha=0.6, color='steelblue', edgecolor='white',
            density=True, label=f'Configurable methods (n={len(config_vals)})')

    # Baseline vertical lines
    baseline_styles = {'Contribution': ('--', 'C3'), 'UPGD': ('-.', 'C4'), 'SI': (':', 'C5')}
    for name in BASELINE_NAMES:
        if name in valid:
            val = valid[name]['avg_loss']
            ls, color = baseline_styles.get(name, ('-', 'gray'))
            ax.axvline(val, linestyle=ls, color=color, linewidth=2,
                       label=f'{name} ({val:.4f})')

    ax.set_xlabel('Average Loss')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Average Loss Across All Methods')
    ax.legend(fontsize=9)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot pruning experiment results')
    parser.add_argument('results_file', type=str, default='results.pkl', nargs='?',
                        help='Path to results.pkl')
    parser.add_argument('--n_top', type=int, default=5,
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
        save_path=os.path.join(args.output_dir, 'fig1_learning_curves.png'))

    plot_config_boxplots(
        results, metric='avg_loss',
        save_path=os.path.join(args.output_dir, 'fig2_loss_boxplots.png'))

    plot_distribution(
        results,
        save_path=os.path.join(args.output_dir, 'fig3_loss_distribution.png'))

    plot_config_boxplots(
        results, metric='avg_accuracy',
        save_path=os.path.join(args.output_dir, 'fig4_accuracy_boxplots.png'))

    print("All figures generated.")


if __name__ == '__main__':
    main()
