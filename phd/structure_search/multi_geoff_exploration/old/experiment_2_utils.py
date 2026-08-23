"""Bookkeeping, statistics and figures for Experiment 2 (`experiment_2.ipynb`).

Kept out of the notebook so that it reads as the experiment itself: configuration, connectivity,
one training loop, then one cell per check. Nothing here touches the learner — every function is
either plumbing (drive a sweep, cache its arrays, reload them) or a reduction of the arrays a
completed run left behind.

Three groups:

* `SweepRunner` — runs a single-point mlflow-sweeper study, or reloads it if it has already been
  run. Per-step curves go to MLflow; the per-connection arrays go to an ``.npz`` beside them,
  because they are thousands of series nothing would ever query from a metric store.
* Statistics — every check is over the connections the run's own schedule *reset*, read from the
  exact per-connection transition steps recorded inside the training loop rather than from the
  snapshot grid. Fractions are computed per seed and averaged, with a 95% CI on that mean.
* Figures — the stage raster, the threshold overlays, and the summary bars / histograms.
"""

import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from omegaconf import OmegaConf
from scipy import stats

from mlflow_sweeper import SweepConfig, delete_sweep, run_sweep

from phd.research_utils.analysis.analysis_utils import get_color_palette, load_experiment_data
from phd.research_utils.scripts.mlflow_download import download_experiment

from connection_stages import GROWTH, MATURE, NASCENT, NEVER, STAGE_NAMES
from experiment_logging import (
    ci_halfwidth, load_traces, log_curves, plot_learning_curve, save_traces,
)
from masked_mlp import USEFUL, USELESS


LAYERS = ('w1', 'w2')
LAYER_NAMES = ('input layer (hidden $\\leftarrow$ inputs)',
               'output layer (outputs $\\leftarrow$ hidden)')
GROUPS = (USEFUL, USELESS)
GROUP_NAMES = ('useful (same task)', 'useless (cross task)')

# Grey for "nothing has happened yet", amber for growing, green for mature. Red is not a stage: it
# marks a connection Experiment 3's prune rule would already have removed.
STAGE_COLORS = ('#d9d9d9', '#e8a33d', '#2e7d5b')
PRUNED_COLOR = '#c0392b'
PRUNED = len(STAGE_COLORS)


# ------------------------------------------------------------------------------ sweep driving
class SweepRunner:
    """Runs one-point mlflow-sweeper studies, caching each run's arrays beside its metrics.

    Args:
        experiment: MLflow experiment name
        notebook_dir: Directory holding the notebook; ``data/``, ``sweep_output/`` and ``output/``
            are created inside it
        metrics: ``losses -> {name: (n_seeds, n_steps)}``, the per-step curves to log
        local_stores: Use local sqlite stores instead of whatever the environment points at
        n_log_points: Points logged per curve
        final_fraction: Fraction of the final steps the scalar ``final_*`` metrics summarize
        n_trace_seeds: Seeds kept for the per-connection ``*_tr`` arrays (all 30 would be ~170 MB)
        n_jobs: Trial concurrency; 1, since a trial already saturates the device
    """

    def __init__(self, experiment, notebook_dir, metrics, local_stores=False,
                 n_log_points=250, final_fraction=0.05, n_trace_seeds=None, n_jobs=1):
        self.experiment = experiment
        self.metrics = metrics
        self.n_log_points = n_log_points
        self.final_fraction = final_fraction
        self.n_trace_seeds = n_trace_seeds
        self.n_jobs = n_jobs

        self.data_dir = os.path.join(notebook_dir, 'data')
        self.sweep_dir = os.path.join(notebook_dir, 'sweep_output')
        store_dir = os.path.join(notebook_dir, 'output')
        for path in (self.data_dir, self.sweep_dir, store_dir):
            os.makedirs(path, exist_ok=True)

        # The shared server by default, local sqlite files if the environment does not name one.
        # The env vars are only ever read, never assigned, so running the notebook cannot repoint
        # the session's tracking server.
        def uri(env_var, filename):
            local = f"sqlite:///{os.path.join(store_dir, filename)}"
            return local if local_stores else os.environ.get(env_var, local)

        self.mlflow_uri = uri('MLFLOW_TRACKING_URI', 'mlruns.db')
        self.optuna_uri = uri('OPTUNA_TRACKING_URI', 'optuna.db')
        mlflow.set_tracking_uri(self.mlflow_uri)

    def run(self, sweep_name, params, fn, required_keys=(), n_traces=None):
        """Run `fn(**params)` under `sweep_name`, or reload it, and return its cached arrays.

        Step counts and seed counts are passed as sweep *parameters* rather than closed over, so
        changing one invalidates the study instead of silently serving results from the old length.

        Args:
            sweep_name: Sweep name; doubles as the trace-cache filename
            params: Parameters `fn` is called with, and the grid the study is defined by
            fn: ``(**params) -> {name: array}``, the run itself
            required_keys: Arrays the analysis needs; a cache missing any of them is discarded
            n_traces: Expected size of the traced-connection axis; a cache with a different one is
                discarded, since the analysis would silently mispair it against the current indices

        Returns:
            The run's cached arrays
        """
        self._run_sweep(sweep_name, params, fn)
        traces = self._validate(load_traces(self.data_dir, sweep_name), sweep_name,
                                required_keys, n_traces)
        if traces is None:
            # The study is complete but its arrays are not on disk — deleted, or produced on
            # another machine. Every figure reads them and nothing in MLflow can reconstruct them,
            # so the only repair is to run it again.
            print(f'Sweep {sweep_name} is complete but its cached arrays are missing or stale; '
                  f're-running it.')
            self._run_sweep(sweep_name, params, fn, overwrite=True)
            traces = load_traces(self.data_dir, sweep_name)
            assert traces is not None, f"Sweep {sweep_name} ran but wrote no traces!"
        return traces

    def sweep_data(self):
        """The experiment's logged curves, downloaded from MLflow only when new runs appeared."""
        params_csv, metrics_csv = csv_paths(self.data_dir, self.experiment)
        marker_path = os.path.join(self.data_dir, f'.{_stem(self.experiment)}_run_count')

        n_runs = str(self._n_finished_runs())
        marker = open(marker_path).read().strip() if os.path.exists(marker_path) else None
        if marker != n_runs or not os.path.exists(metrics_csv):
            download_experiment(self.experiment, self.data_dir, tracking_uri=self.mlflow_uri)
            with open(marker_path, 'w') as f:
                f.write(n_runs)
        return load_experiment_data(params_csv, metrics_csv)

    # -- internals
    def _run_sweep(self, sweep_name, params, fn, overwrite=False):
        config = SweepConfig.from_dict_config(OmegaConf.create({
            'experiment': self.experiment,
            'sweep_name': sweep_name,
            'optuna_storage': self.optuna_uri,
            'mlflow_storage': self.mlflow_uri,
            'output_dir': self.sweep_dir,
            'algorithm': 'grid',
            'spec': {'direction': 'minimize', 'metric': 'final_loss'},
            'parameters': dict(params),
        }))
        if mlflow.active_run() is not None:
            mlflow.end_run()     # a lingering run makes a completed sweep raise "already active"
        if overwrite:
            delete_sweep(config)

        def objective(**swept):
            # Sweep parameters come back as floats; counts must stay ints (scan lengths).
            kwargs = {k: int(v) if k.startswith('n_') else float(v) for k, v in swept.items()}
            return self._log(fn(**kwargs), kwargs['n_steps'], sweep_name)

        # executor='thread' is required: `objective` is a closure, and the process pool only
        # accepts top-level picklable functions.
        run_sweep(config, objective, n_jobs=self.n_jobs, log_params=True, no_plots=True,
                  allow_param_change=True, executor='thread')

    def _log(self, result, n_steps, sweep_name):
        """Log the per-step curves, cache the arrays, return the objective."""
        arrays = {}
        for key, value in result.items():
            if key == 'losses' or key in ('reset_step', 'n_snapshots'):
                continue        # logged in binned form / not arrays
            value = np.asarray(value)
            arrays[key] = value[:self.n_trace_seeds] if key.endswith('_tr') else value
        save_traces(self.data_dir, sweep_name,
                    log_steps=snapshot_steps(n_steps, result['n_snapshots']),
                    **{f'reset_step_{k}': np.asarray(v) for k, v in result['reset_step'].items()},
                    **arrays)
        return log_curves(self.metrics(result['losses']), n_log_points=self.n_log_points,
                          final_fraction=self.final_fraction, objective_key='loss')

    def _validate(self, traces, sweep_name, required_keys, n_traces):
        """The cache, or None if it predates an array the analysis needs or has the wrong shape."""
        if traces is None:
            return None
        missing = [k for k in required_keys if k not in traces]
        if missing:
            print(f'Cached traces for {sweep_name} predate {missing[0]}; discarding them.')
            return None
        if n_traces is not None and traces['stage_tr'].shape[-1] != n_traces:
            print(f'Cached traces for {sweep_name} hold {traces["stage_tr"].shape[-1]} traced '
                  f'connections, not {n_traces}; discarding them.')
            return None
        return traces

    def _n_finished_runs(self):
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(self.experiment)
        if exp is None:
            return 0
        return len(client.search_runs([exp.experiment_id], max_results=50000,
                                      filter_string="attributes.status = 'FINISHED'"))


def _stem(experiment):
    # `download_experiment` sanitizes spaces and dashes to underscores when naming its output
    # (mlflow_download.py:269), so derive the CSV paths the same way.
    return experiment.replace(' ', '_').replace('-', '_')


def csv_paths(data_dir, experiment):
    """Paths of the params / metrics CSVs `download_experiment` writes for `experiment`."""
    stem = _stem(experiment)
    return (os.path.join(data_dir, f'{stem}_params.csv'),
            os.path.join(data_dir, f'{stem}_metrics.csv'))


def snapshot_steps(n_steps, n_snapshots):
    """Step index each snapshot was taken at (the end of its block)."""
    return np.arange(1, n_snapshots + 1) * (n_steps // n_snapshots)


def final_value(runs, sweep_name, metric='loss', final_fraction=0.05):
    """Mean `metric` over the last `final_fraction` of a run's logged curve."""
    rdf = runs[sweep_name].dropna(subset=[metric])
    if rdf.empty:
        return np.nan
    curve = rdf.groupby('step')[metric].mean().sort_index()
    return float(curve.iloc[-max(1, int(final_fraction * len(curve))):].mean())


# -------------------------------------------------------------------------------- statistics
def reset_selection(traces, labels, layer, group):
    """Boolean mask of the connections in `group` that this run's own schedule resets.

    Read from the run's cached schedule rather than the notebook's, so a run made under a different
    one — the ablation — is analysed against its own.
    """
    reset_step = np.asarray(traces[f'reset_step_{layer}'])
    return (np.asarray(labels[layer]) == group) & (reset_step != NEVER)


def per_seed_fraction(traces, labels, key, layer, group):
    """Per-seed fraction of the group's reset connections with a recorded `key` transition."""
    reached = np.asarray(traces[f'final_{key}_{layer}']) != NEVER      # (seeds, out, in)
    return reached[:, reset_selection(traces, labels, layer, group)].mean(axis=1)


def mean_ci(values):
    """Mean over seeds and the half-width of a 95% CI on it, ignoring non-finite entries."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return values.mean(), float(ci_halfwidth(values.std(), len(values)))


def time_to(traces, labels, layer, group=USEFUL, key='mature_step'):
    """Per-connection delay from reset to `key` in steps, and the fraction that never reached it."""
    reached_at = np.asarray(traces[f'final_{key}_{layer}'])            # (seeds, out, in)
    sel = reset_selection(traces, labels, layer, group)
    reset = np.broadcast_to(
        np.asarray(traces[f'reset_step_{layer}'])[None], reached_at.shape)[:, sel]
    reached_at = reached_at[:, sel]
    reached = reached_at != NEVER
    return reached_at[reached] - reset[reached], 1.0 - reached.mean()


def maturity_table(traces_by_key, configs, labels):
    """Fractions of reset connections that reached each stage, per run / layer / group."""
    rows = []
    for key, cfg in configs.items():
        traces = traces_by_key[key]
        for layer in LAYERS:
            for group, name in zip(GROUPS, ('useful', 'useless')):
                sel = reset_selection(traces, labels, layer, group)
                row = {'run': cfg['label'], 'layer': layer, 'group': name,
                       'n reset (per seed)': int(sel.sum())}
                for stage_key, stage in (('growth_step', 'growth'), ('mature_step', 'mature')):
                    mean, ci = mean_ci(per_seed_fraction(traces, labels, stage_key, layer, group))
                    row[f'reached {stage}'], row[f'reached {stage} CI'] = mean, ci

                # Separates the two gates: near 1 means the h-crossing test lets through everything
                # the growth threshold admitted, so maturity carries no extra information.
                grew = np.asarray(traces[f'final_growth_step_{layer}'])[:, sel] != NEVER
                matured = np.asarray(traces[f'final_mature_step_{layer}'])[:, sel] != NEVER
                n_grew = grew.sum(axis=1)
                row['mature | growth'], _ = mean_ci(np.where(
                    n_grew > 0, matured.sum(axis=1) / np.maximum(n_grew, 1), np.nan))
                row['would be pruned'], _ = mean_ci(
                    per_seed_fraction(traces, labels, 'prune_step', layer, group))
                rows.append(row)
    return pd.DataFrame(rows)


def growth_vs_chance(traces_by_key, configs, labels, log_steps, n_steps, tail_fraction=0.25):
    """Measured growth rate of reset useless connections against the jitter model.

    A useless connection's weight is driven by an error uncorrelated with its input, so under LMS
    it is an Ornstein-Uhlenbeck process with stationary sd ``sigma = sqrt(alpha v / 2)`` = sqrt(2)
    J_h and correlation time ``tau = 1 / (alpha E[x^2])``. The growth threshold sits at
    ``J_w = 2 J_h``, so a purely jittering weight crosses it roughly once per tau steps. The chance
    prediction is reported twice: from the analytic sigma, and from the sigma measured on this
    run's still-nascent useless weights over the last `tail_fraction` of the run.
    """
    rows = []
    for key, cfg in configs.items():
        traces = traces_by_key[key]
        for li, layer in enumerate(LAYERS):
            measured, ci = mean_ci(
                per_seed_fraction(traces, labels, 'growth_step', layer, USELESS))
            row = {'run': cfg['label'], 'layer': layer,
                   'measured growth rate': measured, 'CI': ci}

            # The sub-threshold weight scale, measured on traced useless connections still nascent
            # over the tail of the run — exactly the connections the jitter model describes. Only
            # the cached trace seeds contribute; the measured rate above uses all of them.
            tail = slice(int((1 - tail_fraction) * len(log_steps)), None)
            def tailed(name):
                return np.asarray(traces[name])[:, tail, li, USELESS, :]
            nascent = tailed('stage_tr') == NASCENT
            if nascent.sum() < 2:
                rows.append({**row, 'note': 'no nascent useless traces left to fit the model'})
                continue

            sigma_obs = float(np.std(tailed('w_tr')[nascent]))
            j_h = float(np.mean(tailed('jh_tr')[nascent]))
            j_w = float(np.mean(tailed('jw_tr')[nascent]))

            # tau = 1 / (alpha E[x^2]), from the final step-sizes and the run-mean squared source
            # values of the reset useless connections.
            sel = reset_selection(traces, labels, layer, USELESS)
            alpha = np.asarray(traces[f'final_alpha_{layer}'])[:, sel].mean()
            mean_x2 = np.asarray(traces[f'mean_x2_{layer}']).mean(axis=0)
            x2 = np.broadcast_to(mean_x2[None, :], np.asarray(labels[layer]).shape)[sel].mean()
            tau = 1.0 / (alpha * x2)
            horizon = float(np.mean(n_steps - np.asarray(traces[f'reset_step_{layer}'])[sel]))

            def p_ever(sigma):
                p_step = 2 * stats.norm.cdf(-j_w / max(sigma, 1e-30))
                return 1.0 - (1.0 - p_step) ** (horizon / tau), p_step

            chance_obs, p_obs = p_ever(sigma_obs)
            chance_analytic, _ = p_ever(np.sqrt(2) * j_h)
            rows.append({**row,
                         'sigma_obs / J_h': sigma_obs / j_h,
                         'P(|w| > J_w) per step': p_obs,
                         'tau (steps)': tau,
                         'crossings expected': horizon / tau,
                         'chance, observed sigma': chance_obs,
                         'chance, jitter model': chance_analytic})
    return pd.DataFrame(rows)


def delay_table(traces_by_key, configs, labels, key='mature_step'):
    """Median and IQR of the delay from reset to `key`, per run / layer / group."""
    rows = []
    for cfg_key, cfg in configs.items():
        for layer in LAYERS:
            for group, name in zip(GROUPS, ('useful', 'useless')):
                delays, never = time_to(traces_by_key[cfg_key], labels, layer, group, key=key)
                q1, med, q3 = (np.percentile(delays, [25, 50, 75]) if len(delays)
                               else (np.nan,) * 3)
                rows.append({'run': cfg['label'], 'layer': layer, 'group': name,
                             'fraction reached': 1.0 - never,
                             'median steps': med, 'IQR lo': q1, 'IQR hi': q3})
    return pd.DataFrame(rows)


def churn_table(traces_by_key, configs, labels, masks, n_steps, linear_ref=None):
    """How often the prune rule fires per connection per step, against the linear reference."""
    rows = []
    if linear_ref is not None:
        rows.append({'run': 'linear feature sifting', 'layer': '-',
                     'fires / connection / step': linear_ref['fires / feature / step'],
                     'median newborn lifetime': linear_ref['median newborn lifetime'],
                     'ever fires after a reset': np.nan})
    for key, cfg in configs.items():
        traces = traces_by_key[key]
        for layer in LAYERS:
            counts = np.asarray(traces[f'final_prune_count_{layer}'])      # (seeds, out, in)
            active = np.asarray(masks[layer]) > 0
            useful, never_useful = time_to(traces, labels, layer, USEFUL, key='prune_step')
            useless, never_useless = time_to(traces, labels, layer, USELESS, key='prune_step')
            delays = np.concatenate([useful, useless])
            rows.append({
                'run': cfg['label'], 'layer': layer,
                'fires / connection / step': counts[:, active].mean() / n_steps,
                'median newborn lifetime': np.median(delays) if len(delays) else np.nan,
                'ever fires after a reset': 1.0 - (never_useful + never_useless) / 2,
            })
    return pd.DataFrame(rows)


def show(df):
    """Print a table without the index and without scientific-notation noise."""
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))


# ----------------------------------------------------------------------------------- figures
def plot_learning_curves(runs, configs, n_seeds, reset_window, hlines=(), exp1_runs=None):
    """Experiment 2's curves against their Experiment 1 counterparts.

    Returns the pre-reset agreement with Experiment 1: the same problem, learner and L1, with only
    the read-only stage machinery added, so the two should coincide up to the first reset.
    """
    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    palette = get_color_palette(n=2 * len(configs))
    rows = []

    def pre_reset(rdf):
        rdf = rdf.dropna(subset=['loss'])
        return rdf[rdf['step'] <= reset_window[0]].groupby('step')['loss'].mean()

    for i, cfg in enumerate(configs.values()):
        plot_learning_curve(runs, cfg['sweep'], label=f"Experiment 2 — {cfg['label']}",
                            color=palette[2 * i], ax=ax, n_seeds=n_seeds, show_ci=False)
        if exp1_runs is None or cfg['exp1_sweep'] not in exp1_runs:
            continue
        plot_learning_curve(exp1_runs, cfg['exp1_sweep'], ax=ax, color=palette[2 * i + 1],
                            label=f"Experiment 1a — {cfg['label']} (no resets)", show_ci=False)

        a, b = pre_reset(runs[cfg['sweep']]), pre_reset(exp1_runs[cfg['exp1_sweep']])
        shared = a.index.intersection(b.index)
        rel = np.abs(a[shared] - b[shared]) / b[shared]
        rows.append({'run': cfg['label'], 'vs': cfg['exp1_sweep'], 'shared log points': len(shared),
                     'max rel. diff': rel.max() if len(shared) else np.nan,
                     'mean rel. diff': rel.mean() if len(shared) else np.nan})

    for step in reset_window:
        ax.axvline(step, c='k', lw=1, ls='--', alpha=0.5)
    for value, style, label in hlines:
        ax.axhline(value, ls=style, c='gray', lw=1, label=label)
    ax.set_ylabel('per-output MSE')
    ax.set_title('Learning curves — the dashed verticals bracket the reset window')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(rows)


def plot_stage_raster(traces, log_steps, trace_reset_steps, title, seed=0, show_pruned=True):
    """One panel per layer; useful connections above useless, each block sorted by reset step.

    With `show_pruned`, red overrides the stage colour from the first step Experiment 3's prune
    rule would have fired, because past that point the connection would not exist.
    """
    stages = np.asarray(traces['stage_tr'], dtype=int)[seed]     # (snapshots, layer, group, n)
    colors = STAGE_COLORS
    if show_pruned:
        stages = np.where(np.asarray(traces['pruned_tr'], dtype=bool)[seed], PRUNED, stages)
        colors = colors + (PRUNED_COLOR,)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(cmap.N + 1) - 0.5, cmap.N)

    _, axes = plt.subplots(len(LAYERS), 1, figsize=(11, 7), sharex=True)
    for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
        blocks, resets, boundaries = [], [], []
        for group in GROUPS:
            order = np.argsort(trace_reset_steps[layer][group])
            blocks.append(stages[:, li, group, :].T[order])      # (n_traces, snapshots)
            resets.append(np.asarray(trace_reset_steps[layer][group])[order])
            boundaries.append(sum(b.shape[0] for b in blocks))

        image = np.concatenate(blocks, axis=0)
        ax.imshow(image, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm,
                  extent=[log_steps[0], log_steps[-1], image.shape[0], 0])
        # One staircase per block: the two are sorted independently, so a single line would jump
        # backwards in x at the boundary.
        for block_resets, lo in zip(resets, [0] + boundaries[:-1]):
            ax.step(block_resets, np.arange(len(block_resets)) + lo + 0.5, where='mid',
                    c='k', lw=1.2)
        ax.axhline(boundaries[0], c='k', lw=1.5)
        for name, lo, hi in zip(GROUP_NAMES, [0] + boundaries[:-1], boundaries):
            ax.text(log_steps[-1] * 1.01, (lo + hi) / 2, name.replace(' (', '\n('),
                    va='center', fontsize=8)
        ax.set_ylabel('connection')
        ax.set_title(LAYER_NAMES[li], fontsize=10)

    names = STAGE_NAMES + (('would be pruned',) if show_pruned else ())
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    axes[0].legend(handles, names, loc='upper left', fontsize=8, ncol=len(names), framealpha=0.9)
    axes[-1].set_xlabel('step')
    plt.suptitle(f'{title} — stages over time (seed {seed}; black line = reset step)'
                 + (', prune rule overlaid' if show_pruned else ''), y=1.0)
    plt.tight_layout()
    plt.show()


def plot_threshold_overlays(traces, log_steps, trace_reset_steps, layer, group, title,
                            n=4, seed=0):
    """|w| vs J_w and h vs +/- J_h for `n` reset connections, one column each."""
    li = LAYERS.index(layer)
    get = lambda key: np.asarray(traces[key])[seed, :, li, group, :]
    reset_steps = trace_reset_steps[layer][group]
    stages = get('stage_tr').astype(int)

    # Rank by how far each connection got *after* its reset — before it they have all been growing
    # since step 0 and are all mature. Show the most and least successful half each, so the figure
    # is neither four flat lines nor a curated set of successes.
    post_reset = log_steps[None, :] > reset_steps[:, None]
    order = np.argsort(-np.where(post_reset, stages.T, NASCENT).max(axis=1))
    ranked = np.concatenate([order[:n - n // 2], order[::-1][:n // 2]])

    _, axes = plt.subplots(2, len(ranked), figsize=(3.6 * len(ranked), 5.4), sharex=True,
                           squeeze=False)
    for col, idx in enumerate(ranked):
        reset = reset_steps[idx]
        transitions = [('reset', reset, '#555555')]
        for stage, name in ((GROWTH, 'growth'), (MATURE, 'mature')):
            after = np.where((stages[:, idx] >= stage) & (log_steps > reset))[0]
            if len(after):
                transitions.append((name, log_steps[after[0]], STAGE_COLORS[stage]))

        top, bottom = axes[0][col], axes[1][col]
        top.plot(log_steps, get('absw_tr')[:, idx], c='#3b6fb0', lw=1.2, label='$|w|$')
        top.plot(log_steps, get('jw_tr')[:, idx], c='#c0392b', lw=1.0, ls='--', label='$J_w$')
        top.set_yscale('log')

        bottom.plot(log_steps, get('h_tr')[:, idx], c='#3b6fb0', lw=1.2, label='$h$')
        bottom.plot(log_steps, get('jh_tr')[:, idx], c='#c0392b', lw=1.0, ls='--',
                    label='$\\pm J_h$')
        bottom.plot(log_steps, -get('jh_tr')[:, idx], c='#c0392b', lw=1.0, ls='--')
        bottom.axhline(0, c='gray', lw=0.8)

        for ax in (top, bottom):
            for name, step, color in transitions:
                ax.axvline(step, c=color, lw=1.1, alpha=0.85,
                           ls=':' if name == 'reset' else '-')
            ax.grid(True, alpha=0.3)
        top.set_title(' | '.join(f'{name}@{int(s):,}' for name, s, _ in transitions), fontsize=8)
        bottom.set_xlabel('step')

    axes[0][0].set_ylabel('$|w|$ vs $J_w$')
    axes[1][0].set_ylabel('$h$ vs $\\pm J_h$')
    axes[0][-1].legend(fontsize=8)
    axes[1][-1].legend(fontsize=8)
    plt.suptitle(f'{title} — {LAYER_NAMES[li]} — {GROUP_NAMES[group]}', y=1.0)
    plt.tight_layout()
    plt.show()


def plot_maturity_bars(maturity):
    """The stages-reached table as grouped bars, one panel per run."""
    runs = list(dict.fromkeys(maturity['run']))
    positions = {f'{layer} {stage}': i for i, (layer, stage) in enumerate(
        [(l, s) for l in LAYERS for s in ('growth', 'mature')])}
    colors = {'useful': '#3b6fb0', 'useless': '#c0392b'}

    _, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        for group, offset in (('useful', -0.18), ('useless', 0.18)):
            sub = maturity[(maturity['run'] == run) & (maturity['group'] == group)]
            xs, ys, es = [], [], []
            for _, r in sub.iterrows():
                for stage in ('growth', 'mature'):
                    xs.append(positions[f"{r['layer']} {stage}"] + offset)
                    ys.append(r[f'reached {stage}'])
                    es.append(r[f'reached {stage} CI'])
            ax.bar(xs, ys, 0.35, yerr=es, capsize=3, color=colors[group], label=group, alpha=0.85)
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(list(positions), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.35)
        ax.set_title(run, fontsize=10)
    axes[0][0].set_ylabel('fraction of reset connections')
    axes[0][-1].legend()
    plt.suptitle('Stages reached after a reset (mean $\\pm$ 95% CI over seeds)', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_maturity_delays(traces_by_key, configs, labels):
    """Histograms of the time from reset to maturity for useful connections, pooled over seeds."""
    _, axes = plt.subplots(len(configs), len(LAYERS), figsize=(11, 3.6 * len(configs)),
                           sharex=True, squeeze=False)
    for row, (key, cfg) in zip(axes, configs.items()):
        for ax, layer in zip(row, LAYERS):
            delays, never = time_to(traces_by_key[key], labels, layer, USEFUL)
            if len(delays):
                ax.hist(delays / 1000, bins=40, color='#3b6fb0', alpha=0.85)
                ax.axvline(np.median(delays) / 1000, c='k', lw=1.2)
            ax.set_title(f"{cfg['label']} — {layer} — never matured: {never:.1%}", fontsize=9)
            ax.grid(True, alpha=0.35)
        row[0].set_ylabel('reset useful connections')
    for ax in axes[-1]:
        ax.set_xlabel('steps from reset to maturity (thousands)')
    plt.suptitle('Time from reset to maturity, useful connections', y=1.0)
    plt.tight_layout()
    plt.show()
