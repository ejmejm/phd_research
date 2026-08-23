"""Logging and plotting helpers for the multi-GEOFF experiments.

Kept out of the notebooks so the experiment cells stay focused on the algorithms rather than the
bookkeeping. Two groups of helpers, matching the two kinds of data a run produces:

* **Per-step scalars** (loss, fraction of variance explained) — `log_curves` bins them and logs
  them to MLflow as metric series; `summarize_runs` and `plot_learning_curve` read them back.
* **Per-connection snapshots** (step-size and |weight| quantiles per group, plus traces of
  individual connections) — far too many series for a metric store, so `save_traces` /
  `load_traces` cache them as `.npz` next to the metrics, and `plot_group_bands` / `plot_traces`
  draw them. Nothing is ever replayed: one run writes both.

Uncertainty bands are confidence intervals on the mean over seeds, derived at plot time from the
logged across-seed std by `ci_halfwidth` — nothing CI-specific is logged, so `CI_LEVEL` can change
without re-running anything.
"""

import math
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.entities import Metric
from scipy import stats


CI_LEVEL = 0.95   # confidence level of the band drawn by `plot_learning_curve`

# Colours of the two connection groups. Blue = connections that should carry signal, red =
# connections that are useless (or newly added) by construction.
GROUP_COLORS = {0: '#3b6fb0', 1: '#c0392b'}


def ci_halfwidth(std, n_seeds, level=CI_LEVEL):
    """Half-width of a `level` CI on the mean, from the *population* std across seeds.

    ``std`` is the across-seed std with ``ddof=0`` (what numpy's ``.std()`` returns and what
    `log_curves` computes), so the sample std is ``std * sqrt(n / (n - 1))`` and the standard error
    of the mean is that divided by ``sqrt(n)`` — i.e. ``std / sqrt(n - 1)``. Uses the t
    distribution, which matters at these seed counts (``t = 2.045`` vs ``z = 1.96`` at n=30).
    """
    if n_seeds < 2:
        return np.zeros_like(np.asarray(std, dtype=float))
    t = float(stats.t.ppf(0.5 + level / 2.0, n_seeds - 1))
    return t * np.asarray(std, dtype=float) / math.sqrt(n_seeds - 1)


# --------------------------------------------------------------------------- logging
def log_curves(series, n_log_points=250, final_fraction=0.05, objective_key='loss'):
    """Bin and log named per-step series; return `objective_key`'s final value for Optuna.

    Each series is binned into `n_log_points` windows (per-seed mean within a window, then mean and
    std across seeds) and logged as ``name`` / ``name_std``, plus a scalar ``final_{name}`` over the
    last `final_fraction` of steps. Non-finite values (divergence) become NaN so they propagate
    through the means; a fully-diverged run reports a large finite sentinel so Optuna always
    receives a number.

    Args:
        series: ``{name: (n_seeds, n_steps)}`` arrays recorded every step.
        n_log_points: Number of binned points logged per series.
        final_fraction: Fraction of the final steps summarised by the scalar metrics.
        objective_key: Which series' final value is returned.

    Returns:
        The scalar final value of `objective_key`.
    """
    metrics, objective = [], None

    for name, values in series.items():
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan   # treat divergence (inf) as NaN
        n_seeds, n_steps = values.shape

        window = n_steps // n_log_points
        binned = values[:, :window * n_log_points].reshape(n_seeds, n_log_points, window).mean(axis=2)
        for step, mean, std in zip(np.arange(1, n_log_points + 1) * window,
                                   binned.mean(axis=0), binned.std(axis=0)):
            metrics.append(Metric(name, float(mean), 0, int(step)))
            metrics.append(Metric(f'{name}_std', float(std), 0, int(step)))

        k = max(1, int(final_fraction * n_steps))
        per_seed_final = values[:, -k:].mean(axis=1)
        final, final_std = float(per_seed_final.mean()), float(per_seed_final.std())
        final = final if np.isfinite(final) else 1e6
        metrics.append(Metric(f'final_{name}', final, 0, 0))
        metrics.append(Metric(f'final_{name}_std', final_std if np.isfinite(final_std) else 0.0, 0, 0))
        if name == objective_key:
            objective = final

    # Logging each point with its own `mlflow.log_metric` is dramatically slower than one batched
    # transaction, and `log_batch` caps at 1000 metrics per call.
    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])

    assert objective is not None, \
        f"objective_key '{objective_key}' not among the series {sorted(series)}!"
    return objective


# ------------------------------------------------------------------------- trace caching
def traces_path(data_dir, sweep_name):
    """Path of the ``.npz`` holding a sweep's per-connection snapshots."""
    return os.path.join(data_dir, f'traces_{sweep_name}.npz')


def save_traces(data_dir, sweep_name, **arrays):
    """Cache the per-connection snapshots of a run next to the logged metrics."""
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(traces_path(data_dir, sweep_name), **arrays)


def load_traces(data_dir, sweep_name):
    """Load a sweep's cached snapshots, or None if it has not been run yet."""
    path = traces_path(data_dir, sweep_name)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# -------------------------------------------------------------------------- plotting
def summarize_runs(cfgs, runs, sweep_name, sweep_vars, final_fraction, baseline_value,
                   metric='loss'):
    """Reduce each run to a single point: its final `metric` value plus a `converged` flag.

    ``value`` = mean `metric` over the last `final_fraction` of steps (NaN-propagating, so a
    diverged seed propagates), ``value_std`` = mean logged ``{metric}_std``, and ``converged`` =
    ``value`` finite and below `baseline_value`.
    """
    cdf, rdf = cfgs[sweep_name], runs[sweep_name]
    max_step, min_step = rdf['step'].max(), rdf['step'].min()
    final = rdf[rdf['step'] >= max_step - final_fraction * (max_step - min_step)]
    agg = final.groupby('run_id').agg(
        value=(metric, lambda x: x.mean(skipna=False)),
        value_std=(f'{metric}_std', 'mean'),
    ).reset_index()
    out = agg.merge(cdf[['run_id'] + sweep_vars], on='run_id', how='left')
    for var in sweep_vars:
        out[var] = out[var].astype(float)
    out['converged'] = np.isfinite(out['value']) & (out['value'] < baseline_value)
    return out


def plot_learning_curve(runs, sweep_name, label=None, color=None, ax=None, n_seeds=None,
                        show_ci=True, metric='loss'):
    """Binned mean learning curve with an uncertainty band.

    With `n_seeds` given the band is a `CI_LEVEL` confidence interval on the mean over seeds,
    computed from the logged across-seed std by `ci_halfwidth`; without it the band falls back to
    ±1 std (the across-seed spread). Pass ``show_ci=False`` for the mean line alone — useful when
    several curves share an axis and overlapping bands hide them.
    """
    ax = ax or plt.gca()
    rdf = runs[sweep_name].dropna(subset=[metric])
    rdf = rdf.groupby('step').agg(
        value=(metric, 'mean'), value_std=(f'{metric}_std', 'mean')).reset_index()
    rdf = rdf.sort_values('step')
    steps, mean, std = rdf['step'].values, rdf['value'].values, rdf['value_std'].values
    band = ci_halfwidth(std, n_seeds) if n_seeds is not None else std
    ax.plot(steps, mean, color=color, label=label or sweep_name)
    if show_ci:
        ax.fill_between(steps, mean - band, mean + band, color=color, alpha=0.2)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('step')
    ax.set_ylabel(metric)


def _panels(n, axes):
    """One axis per layer, created if not supplied."""
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2))
    return np.atleast_1d(axes)


def _finish(axes, layer_names, ylabel, title, log_scale, hline):
    for ax, name in zip(axes, layer_names):
        if hline is not None:
            ax.axhline(hline, ls='--', c='gray', lw=1, label='initial step-size')
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(name)
        ax.set_xlabel('step')
        ax.grid(True, alpha=0.35, which='both')
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc='best')
    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return axes


def plot_group_bands(quantiles, log_steps, layer_names, group_names, quantile_levels,
                     title, ylabel, log_scale=True, axes=None, x_offset=0, hline=None):
    """Median line + inter-quantile bands per connection group, one panel per layer.

    The distributional view, and the quantity Experiment 1 actually cares about. Bands span the
    outermost pair of `quantile_levels` outward-in, so a group whose bulk collapses toward the
    initial step-size is visible even when a few connections escape. Quantiles are averaged over
    seeds, giving the typical run's distribution rather than the pooled one.

    Args:
        quantiles: ``(n_seeds, n_snapshots, n_layers, n_groups, n_quantiles)``.
        log_steps: ``(n_snapshots,)`` step index of each snapshot.
        layer_names: One label per layer.
        group_names: One label per group.
        quantile_levels: The levels the last axis holds, ascending.
        title: Figure title.
        ylabel: Shared y-axis label.
        log_scale: Log-scale the y axis (right for step-sizes, wrong where zero is meaningful).
        axes: Existing axes to draw into; created if omitted.
        x_offset: Added to `log_steps`, for panels that start mid-training.
        hline: Optional reference value drawn as a dashed grey line.
    """
    mean_q = np.nanmean(np.asarray(quantiles, dtype=np.float64), axis=0)
    n_layers, n_groups, n_q = mean_q.shape[1], mean_q.shape[2], mean_q.shape[3]
    mid = n_q // 2
    steps = np.asarray(log_steps) + x_offset
    axes = _panels(n_layers, axes)

    for layer in range(n_layers):
        for group in range(n_groups):
            curves = mean_q[:, layer, group, :]
            if not np.isfinite(curves).any():
                continue
            color = GROUP_COLORS.get(group, f'C{group}')
            axes[layer].plot(steps, curves[:, mid], color=color, lw=1.8,
                             label=f'{group_names[group]}  (median)')
            for depth, (lo, hi) in enumerate(zip(range(mid), range(n_q - 1, mid, -1))):
                axes[layer].fill_between(
                    steps, curves[:, lo], curves[:, hi], color=color, lw=0,
                    alpha=0.14 + 0.12 * depth,
                    label=(f'{group_names[group]}  '
                           f'({quantile_levels[lo]:.0%}–{quantile_levels[hi]:.0%})'))
    return _finish(axes, layer_names, ylabel, title, log_scale, hline)


def plot_traces(traces, log_steps, layer_names, group_names, title, ylabel,
                log_scale=True, seed=0, max_traces=30, axes=None, x_offset=0, hline=None):
    """Individual connection traces, one panel per layer.

    The starting point for reading Experiment 1: real connections rather than a summary. Drawn
    from a single seed, because averaging individual connections across seeds would mix unrelated
    connections.

    Args:
        traces: ``(n_seeds, n_snapshots, n_layers, n_groups, n_traces)``.
        log_steps: ``(n_snapshots,)`` step index of each snapshot.
        layer_names: One label per layer.
        group_names: One label per group.
        title: Figure title.
        ylabel: Shared y-axis label.
        log_scale: Log-scale the y axis.
        seed: Which seed's traces to draw.
        max_traces: Cap on traces drawn per group.
        axes: Existing axes to draw into; created if omitted.
        x_offset: Added to `log_steps`, for panels that start mid-training.
        hline: Optional reference value drawn as a dashed grey line.
    """
    t = np.asarray(traces, dtype=np.float64)[seed]
    n_layers, n_groups = t.shape[1], t.shape[2]
    steps = np.asarray(log_steps) + x_offset
    axes = _panels(n_layers, axes)

    for layer in range(n_layers):
        for group in range(n_groups):
            curves = t[:, layer, group, :max_traces]
            if not np.isfinite(curves).any():
                continue
            color = GROUP_COLORS.get(group, f'C{group}')
            axes[layer].plot(steps, curves, color=color, lw=0.7, alpha=0.5)
            axes[layer].plot([], [], color=color, lw=1.5, label=group_names[group])  # legend proxy
    return _finish(axes, layer_names, ylabel, title, log_scale, hline)
