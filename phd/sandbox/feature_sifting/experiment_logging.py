"""Logging and plotting helpers for the feature-sifting sweeps.

Kept out of the notebook so the experiment cells stay focused on the algorithms rather
than the bookkeeping. Two groups of helpers:

* ``log_run`` — reduce a batch of per-seed loss curves to a binned mean/std learning
  curve plus a scalar final loss, logged to the active MLflow run in one batched call.
* ``summarize_runs`` / ``plot_sensitivity`` / ``plot_learning_curve`` — turn the
  downloaded ``config_dfs`` / ``run_dfs`` into the notebook's figures.
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities import Metric

from phd.research_utils.analysis.analysis_utils import get_color_palette


# --------------------------------------------------------------------------- logging
def log_run(losses, n_log_points=250, final_fraction=0.05):
    """Bin + log the learning curve and a scalar final loss; return a finite metric for Optuna.

    ``losses`` has shape ``(n_seeds, n_steps)``. The curve is binned into ``n_log_points``
    windows (per-seed mean within a window, then mean / std across seeds) and logged as
    ``loss`` / ``loss_std``. ``final_loss`` / ``final_loss_std`` summarise the last
    ``final_fraction`` of steps. Non-finite values (divergence) are treated as NaN so they
    propagate through the means; a fully-diverged run reports a large finite sentinel so
    Optuna always receives a number.
    """
    losses = np.asarray(losses, dtype=np.float64)
    losses[~np.isfinite(losses)] = np.nan   # treat divergence (inf) as NaN
    n_seeds, n_steps = losses.shape

    # Bin into n_log_points windows; mean within each window per seed (NaN-propagating),
    # then mean / std across seeds.
    window = n_steps // n_log_points
    usable = window * n_log_points
    binned = losses[:, :usable].reshape(n_seeds, n_log_points, window).mean(axis=2)
    bin_steps = np.arange(1, n_log_points + 1) * window
    curve_mean = binned.mean(axis=0)
    curve_std = binned.std(axis=0)

    # Final-window summary: per-seed mean over the last final_fraction of steps.
    k = max(1, int(final_fraction * n_steps))
    per_seed_final = losses[:, -k:].mean(axis=1)
    final_loss = float(per_seed_final.mean())
    final_loss_std = float(per_seed_final.std())
    final_loss = final_loss if np.isfinite(final_loss) else 1e6
    final_loss_std = final_loss_std if np.isfinite(final_loss_std) else 0.0

    # Log everything in one batched transaction. Logging each of the ~n_log_points*2
    # points with a separate mlflow.log_metric call is dramatically slower on a
    # file-backed (sqlite) store, so we use log_batch instead.
    metrics = []
    for step, m, s in zip(bin_steps, curve_mean, curve_std):
        metrics.append(Metric('loss', float(m), 0, int(step)))
        metrics.append(Metric('loss_std', float(s), 0, int(step)))
    metrics.append(Metric('final_loss', final_loss, 0, 0))
    metrics.append(Metric('final_loss_std', final_loss_std, 0, 0))

    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    for i in range(0, len(metrics), 1000):   # log_batch caps at 1000 metrics/call
        client.log_batch(run_id, metrics=metrics[i:i + 1000])

    # Optuna needs a finite objective; diverged configs return the sentinel above.
    return final_loss


# -------------------------------------------------------------------------- plotting
def _is_pow2(v):
    return v > 0 and abs(math.log2(v) - round(math.log2(v))) < 1e-9


def _fmt_val(v):
    v = float(v)
    if _is_pow2(v):
        return f"$2^{{{int(round(math.log2(v)))}}}$"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e3):
        return f"{v:.0e}"
    return f"{v:g}"


def summarize_runs(cfgs, runs, sweep_name, sweep_vars, final_fraction, baseline_loss):
    """Reduce each run to a single point for the sensitivity plots.

    ``value`` = mean ``loss`` over the last ``final_fraction`` of steps (NaN-propagating,
    so a diverged seed propagates), ``value_std`` = mean logged ``loss_std``, and a
    ``converged`` flag (``value`` finite and below ``baseline_loss``).
    """
    cdf, rdf = cfgs[sweep_name], runs[sweep_name]
    max_step, min_step = rdf['step'].max(), rdf['step'].min()
    thresh = max_step - final_fraction * (max_step - min_step)
    final = rdf[rdf['step'] >= thresh]
    agg = final.groupby('run_id').agg(
        value=('loss', lambda x: x.mean(skipna=False)),
        value_std=('loss_std', 'mean'),
    ).reset_index()
    out = agg.merge(cdf[['run_id'] + sweep_vars], on='run_id', how='left')
    for v in sweep_vars:
        out[v] = out[v].astype(float)
    out['converged'] = np.isfinite(out['value']) & (out['value'] < baseline_loss)
    return out


def _profile(summary, var):
    '''Profile-best over `var`: for each value, the best (min) finite config holding `var` fixed.'''
    rows = []
    for val, g in summary.groupby(var):
        finite = g[np.isfinite(g['value'])]
        if len(finite):
            best = finite.loc[finite['value'].idxmin()]
            rows.append({var: val, 'value': best['value'], 'value_std': best['value_std'],
                         'converged': bool(best['converged'])})
        else:
            rows.append({var: val, 'value': np.nan, 'value_std': 0.0, 'converged': False})
    return pd.DataFrame(rows).sort_values(var)


def _robust_ylim(values):
    '''Tight y-limits around the bulk of the data, ignoring high outliers so the plot
    is filled rather than dominated by a few large (or diverged) values.'''
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if len(v) >= 4:
        q1, q3 = np.percentile(v, [25, 75])
        fence = q3 + 1.5 * (q3 - q1)             # upper outlier fence (Tukey)
        hi = min(hi, max(fence, float(np.median(v))))
    span = (hi - lo) or (abs(hi) or 1.0)
    pad = 0.08 * span
    return max(0.0, lo - pad), hi + pad


def _draw_sensitivity(ax, df, xvar, xpos, marker_y, color, label=None):
    df = df.sort_values(xvar)
    x = df[xvar].map(xpos).values.astype(float)
    y = df['value'].values.astype(float)
    # yerr = np.nan_to_num(df['value_std'].values.astype(float), nan=0.0)
    conv = df['converged'].values.astype(bool)
    yc = np.where(conv, y, np.nan)
    ax.plot(x, yc, '-o', color=color, label=label, ms=6, zorder=3)
    # ax.fill_between(x, np.where(conv, y - yerr, np.nan), np.where(conv, y + yerr, np.nan),
    #                 color=color, alpha=0.15, zorder=1)
    if (~conv).any():   # diverged / NaN configs: faded x near the top of the panel
        ax.scatter(x[~conv], np.full((~conv).sum(), marker_y), color=color, marker='x',
                   s=55, alpha=0.55, zorder=4)


def _axis(ax, xvar, xvals):
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([_fmt_val(v) for v in xvals])
    ax.set_xlabel(xvar.replace('_', ' '))
    ax.grid(True, alpha=0.4)


def plot_sensitivity(summary, sweep_vars, title):
    """Sensitivity plot: 1 swept var -> single line; 2 -> x-axis + hue; 3+ -> one
    profile-best panel per var (each point = best config holding that var fixed).
    Non-converged configs are drawn as faded x markers pinned near the top."""
    n = len(sweep_vars)
    note = '   (× = did not converge)'

    # y-limits come from the points we actually draw (the profile-best points for 3+
    # vars, not every config), with high outliers ignored so the plot is well filled.
    if n >= 3:
        panels = [(var, _profile(summary, var)) for var in sweep_vars]
        drawn = pd.concat([p for _, p in panels])
    else:
        panels = None
        drawn = summary
    ylo, yhi = _robust_ylim(drawn.loc[drawn['converged'], 'value'].tolist())
    marker_y = ylo + 0.96 * (yhi - ylo)   # where non-converged markers sit

    if n <= 2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if n == 1:
            xvar = sweep_vars[0]
            xvals = sorted(summary[xvar].unique())
            xpos = {v: i for i, v in enumerate(xvals)}
            _draw_sensitivity(ax, summary, xvar, xpos, marker_y, get_color_palette(n=1)[0])
            _axis(ax, xvar, xvals)
        else:
            xvar, hvar = sweep_vars
            xvals = sorted(summary[xvar].unique())
            xpos = {v: i for i, v in enumerate(xvals)}
            hvals = sorted(summary[hvar].unique())
            colors = get_color_palette(n=len(hvals))
            for c, hv in zip(colors, hvals):
                _draw_sensitivity(ax, summary[summary[hvar] == hv], xvar, xpos, marker_y, c,
                                  label=f'{hvar.replace("_", " ")}={_fmt_val(hv)}')
            _axis(ax, xvar, xvals)
            ax.legend()
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel('final loss')
        ax.set_title(title + note)
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
        color = get_color_palette(n=1)[0]
        for ax, (var, prof) in zip(axes, panels):
            xvals = sorted(prof[var].unique())
            xpos = {v: i for i, v in enumerate(xvals)}
            _draw_sensitivity(ax, prof, var, xpos, marker_y, color)
            _axis(ax, var, xvals)
            ax.set_ylim(ylo, yhi)
        axes[0].set_ylabel('final loss')
        fig.suptitle(f'{title} — profile-best sensitivity{note}', y=1.02)
        plt.tight_layout()


def plot_learning_curve(runs, sweep_name, label=None, color=None, ax=None):
    """Binned mean learning curve with a ±1 std band (std across seeds, as logged)."""
    ax = ax or plt.gca()
    rdf = runs[sweep_name].dropna(subset=['loss'])
    rdf = rdf.groupby('step').agg(loss=('loss', 'mean'), loss_std=('loss_std', 'mean')).reset_index()
    rdf = rdf.sort_values('step')
    steps, mean, std = rdf['step'].values, rdf['loss'].values, rdf['loss_std'].values
    ax.plot(steps, mean, color=color, label=label or sweep_name)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('step'); ax.set_ylabel('loss')
