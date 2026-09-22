"""Loading sweep results and reducing them to per-cell numbers.

A Comet export is two CSVs -- one row per trial of parameters, one row per
logged step of metrics -- plus, for runs with ``log_individual_seeds``, a
``per_seed_losses.npy`` / ``per_seed_accs.npy`` asset per trial. The metric
stream carries only the across-seed mean, so confidence intervals come from
the assets.

Three things this handles that a naive groupby would get wrong:

**Retries.** Comet re-assigns a trial after a timeout or crash, so one
experimental cell can have several trials. Everything here dedupes on the
parameters that define a cell, keeping the most complete attempt.

**Divergence.** A diverged run is a real measurement of a bad step-size, and
the trainer stops it early. Its curve is therefore short and its metrics are
meaningless. Reductions here propagate NaN rather than skipping, so a cell
containing a diverged run reads as a gap in the plot instead of silently
averaging over the seeds that happened to survive.

**Uneven lengths.** Stacking curves of different lengths pads with NaN, which
combined with NaN propagation means a combined curve ends where its earliest
constituent ended.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def download_project(project: str, output_dir='data', discovery_samples: int = 10_000):
    """Download a Comet project's parameter and metric CSVs.

    ``discovery_samples`` controls how many experiments are scanned to
    discover parameter names. The default is deliberately high: a parameter
    present on only a small fraction of trials (a sweep axis that only one
    config sets) is missed by a small sample.
    """
    from multi_mnist.scripts.comet_download import run as _run
    return _run(project=project, output_dir=output_dir,
                discovery_samples=discovery_samples)


def load_export(project: str, data_dir='data'):
    """Load the two CSVs of a downloaded Comet project.

    Returns ``(params, metrics)``: one row per trial, and one row per logged
    step.
    """
    data_dir = Path(data_dir)
    cfg = pd.read_csv(data_dir / f'{project}_params.csv', index_col=0)
    run = pd.read_csv(data_dir / f'{project}_metrics.csv', index_col=0)
    return _resolve_pipe_params(cfg), run


def _resolve_pipe_params(cfg: pd.DataFrame) -> pd.DataFrame:
    """Prefer the resolved value of a swept parameter over the raw override.

    Sweep axes are logged twice: ``foo.bar`` is the raw command-line override
    string, which may still contain an unevaluated ``${eval:...}``, while
    ``foo|bar`` is the value after Hydra resolved it. Where both exist, keep
    the resolved one under the dotted name.
    """
    out = cfg.copy()
    for col in list(out.columns):
        if '|' not in col:
            continue
        dotted = col.replace('|', '.')
        if dotted in out.columns:
            out[dotted] = out[dotted].where(out[col].isna(), out[col])
        else:
            out[dotted] = out[col]
    return out


# ---------------------------------------------------------------------------
# Trial bookkeeping
# ---------------------------------------------------------------------------

def diverged_run_ids(run: pd.DataFrame) -> set:
    """Trials whose ``diverged`` summary metric is set."""
    if 'diverged' not in run.columns:
        return set()
    last = run.groupby('run_id')['diverged'].apply(
        lambda s: s.dropna().iloc[-1] if s.notna().any() else 0)
    return set(last[last > 0.5].index)


def annotate_trials(cfg: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """Add per-trial completeness diagnostics.

    A trial counts as complete if it logged every expected period *or* it was
    flagged diverged -- an early stop on divergence is a finished measurement,
    not a truncated one.
    """
    out = cfg.copy()
    n_rows = run.groupby('run_id').size().rename('n_rows')
    if 'diverged' in run.columns:
        div = run.groupby('run_id')['diverged'].apply(
            lambda s: s.dropna().iloc[-1] if s.notna().any() else 0).rename('div_flag')
    else:
        div = pd.Series(name='div_flag', dtype=float)

    out = out.merge(n_rows, left_on='run_id', right_index=True, how='left')
    out = out.merge(div, left_on='run_id', right_index=True, how='left')
    out['n_rows'] = out['n_rows'].fillna(0).astype(int)
    out['div_flag'] = out['div_flag'].fillna(0)
    out['expected'] = (
        out['train.total_steps'] / out['train.log_freq']).fillna(0).astype(int)
    out['completed'] = (out['n_rows'] >= out['expected']) | (out['div_flag'] > 0.5)
    return out


def dedupe_trials(cfg: pd.DataFrame, run: pd.DataFrame, cell_cols: Sequence[str]):
    """Keep one trial per experimental cell, preferring the most complete."""
    annotated = annotate_trials(cfg, run)
    annotated = annotated.sort_values(['completed', 'n_rows'], ascending=[False, False])
    dedup = annotated.drop_duplicates(subset=list(cell_cols), keep='first')
    dedup_cfg = dedup.drop(columns=['n_rows', 'expected', 'div_flag', 'completed'])
    return dedup_cfg, run[run['run_id'].isin(dedup_cfg['run_id'])]


def filter_sweep(cfg_all, run_all, sweep_names, cell_cols=None, dedup=True):
    """Slice to one logical sweep, which may span several Comet sweep names."""
    if isinstance(sweep_names, str):
        sweep_names = [sweep_names]
    cfg = cfg_all[cfg_all['sweep_name'].isin(sweep_names)].copy()
    run = run_all[run_all['run_id'].isin(cfg['run_id'])].copy()
    if dedup and cell_cols:
        # Dedupe after merging, so retries spanning addendum sweeps collapse too.
        cfg, run = dedupe_trials(cfg, run, cell_cols)
    return cfg, run


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def asymptotic_metric(cfg, run, metric_col, group_cols, tail_frac=0.05):
    """Mean of ``metric_col`` over the final ``tail_frac`` of each trial.

    This is the number every summary plot in the paper is built from. In the
    stationary problem it measures converged performance; in the
    non-stationary problem, where nothing converges, it measures how well the
    network keeps up -- which is the distinction Figure 2 turns on.
    """
    diverged = diverged_run_ids(run)
    per_run = (
        run.groupby('run_id')[metric_col]
        .apply(lambda s: (
            s.dropna().iloc[max(1, int((1 - tail_frac) * len(s.dropna()))):].mean()
            if len(s.dropna()) > 1 else np.nan))
        .rename('_metric')
    )
    if diverged:
        per_run.loc[per_run.index.intersection(diverged)] = np.nan

    merged = cfg.merge(per_run, left_on='run_id', right_index=True, how='left')
    return (
        merged.groupby(list(group_cols) + ['optimizer.learning_rate'])['_metric']
        .agg(lambda x: x.mean(skipna=False))   # one bad run nulls the cell
        .reset_index()
    )


def filter_to_best(cfg, run, group_cols, metric_col, direction='max', tail_frac=0.05):
    """Keep only the trials at the best step-size within each group.

    Returns ``(cfg, run, best)``, where ``best`` is the per-group winning row.
    """
    asym = asymptotic_metric(cfg, run, metric_col, group_cols, tail_frac)
    grouped = asym.groupby(list(group_cols))['_metric']
    idxs = (grouped.idxmax() if direction == 'max' else grouped.idxmin()).dropna()
    best = asym.loc[idxs.astype(int)]

    key_cols = list(group_cols) + ['optimizer.learning_rate']
    keep = set(map(tuple, best[key_cols].values))
    mask = pd.Series(
        [k in keep for k in map(tuple, cfg[key_cols].values)], index=cfg.index)
    return cfg[mask].copy(), run[run['run_id'].isin(cfg.loc[mask, 'run_id'])].copy(), best


# ---------------------------------------------------------------------------
# Per-seed assets
# ---------------------------------------------------------------------------

_api_singleton = None


def _api():
    global _api_singleton
    if _api_singleton is None:
        from comet_ml.api import API
        _api_singleton = API()
    return _api_singleton


def fetch_per_seed_arrays(run_ids, cache_dir='data/per_seed_cache', n_threads=8):
    """Download per-seed curve arrays for a set of trials, cached on disk.

    Returns ``{run_id: {'losses': arr, 'accs': arr}}``, each of shape
    ``(n_log_periods, n_seeds)``.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch_one(run_id):
        cache_l = cache / f'{run_id}_losses.npy'
        cache_a = cache / f'{run_id}_accs.npy'
        if cache_l.exists() and cache_a.exists():
            return run_id, {'losses': np.load(cache_l), 'accs': np.load(cache_a)}
        try:
            exp = _api().get_experiment_by_key(run_id)
            by_name = {a['fileName']: a for a in exp.get_asset_list()}
            for asset_name, cache_path in [('per_seed_losses.npy', cache_l),
                                           ('per_seed_accs.npy', cache_a)]:
                if asset_name not in by_name:
                    return run_id, None
                data = exp.get_asset(by_name[asset_name]['assetId'], return_type='binary')
                np.save(cache_path, np.load(io.BytesIO(data)))
            return run_id, {'losses': np.load(cache_l), 'accs': np.load(cache_a)}
        except Exception as exc:
            print(f'failed for {run_id}: {exc}')
            return run_id, None

    out = {}
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(fetch_one, rid): rid for rid in run_ids}
        for f in tqdm(as_completed(futures), total=len(futures), desc='fetch per-seed'):
            rid, data = f.result()
            if data is not None:
                out[rid] = data
    return out


def stack_per_seed(asset_dict, run_ids):
    """Concatenate per-seed arrays across trials along the seed axis.

    This is how the two 15-seed halves of a 30-seed run are rejoined.
    Diverged trials produce shorter arrays, so shorter ones are NaN-padded to
    the longest; with NaN-propagating means the combined curve then ends at
    the earliest divergence rather than quietly continuing on the survivors.

    Returns ``(losses, accs)`` of shape ``(max_periods, total_seeds)``.
    """
    present = [rid for rid in run_ids if rid in asset_dict]
    if not present:
        return None, None
    max_len = max(asset_dict[rid]['losses'].shape[0] for rid in present)

    def pad(arr):
        if arr.shape[0] == max_len:
            return arr
        block = np.full((max_len - arr.shape[0], arr.shape[1]), np.nan, dtype=arr.dtype)
        return np.concatenate([arr, block], axis=0)

    losses = np.concatenate([pad(asset_dict[r]['losses']) for r in present], axis=1)
    accs = np.concatenate([pad(asset_dict[r]['accs']) for r in present], axis=1)
    return losses, accs


#: Parameter names changed when the code was extracted for release. The
#: published Comet runs carry the left-hand names; configs in this repository
#: produce the right-hand ones. ``normalize_columns`` maps old to new so one
#: notebook reads both.
LEGACY_COLUMN_MAP = {
    'model.dense_fill_step': 'algorithm.transition_step',
    'structure_search.connection_budget': 'model.sparse.connection_budget',
    'set.zeta': 'algorithm.zeta',
    'set.evolve_frequency': 'algorithm.evolve_frequency',
    'set.prune_metric': 'algorithm.prune_metric',
}


def normalize_columns(cfg: pd.DataFrame) -> pd.DataFrame:
    """Rename published-run parameter columns to their current names.

    Where both exist (a mixed export of old and new runs), the two are merged
    into the current name, preferring whichever is present per row.
    """
    out = cfg.copy()
    for old, new in LEGACY_COLUMN_MAP.items():
        if old not in out.columns:
            continue
        if new in out.columns:
            out[new] = out[new].where(out[new].notna(), out[old])
        else:
            out[new] = out[old]
    return out
