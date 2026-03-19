"""Download run data from MLflow and convert it into CSV files.

Outputs two CSVs compatible with load_experiment_data():
  - {name}_params.csv: one row per run (run_id, sweep_name, curr_step, params...)
  - {name}_metrics.csv: multiple rows per run (run_id, step, metric1, metric2, ...)
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download experiment data from MLflow and save as CSV"
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or sqlite:///mlruns.db)",
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help="MLflow experiment name",
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for CSV files (default: current directory)",
    )
    parser.add_argument(
        '--include-failed',
        action='store_true',
        default=False,
        help="Include failed/crashed runs",
    )
    parser.add_argument(
        '--include-running',
        action='store_true',
        default=False,
        help="Include currently running runs",
    )
    parser.add_argument(
        '--include-children',
        action='store_true',
        default=False,
        help="Include child (per-seed) runs in addition to parent runs",
    )
    parser.add_argument(
        '--n-threads',
        type=int,
        default=8,
        help="Number of threads for parallel metric history download (default: 8)",
    )
    parser.add_argument(
        '--max-runs',
        type=int,
        default=None,
        help="Maximum number of runs to process (for testing)",
    )
    parser.add_argument(
        '--metrics',
        nargs='*',
        type=str,
        default=None,
        help="Specific metrics to download (default: all available)",
    )
    return parser.parse_args()


def get_tracking_uri(args) -> str:
    import os
    if args.tracking_uri:
        return args.tracking_uri
    env_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if env_uri:
        return env_uri
    return 'sqlite:///mlruns.db'


def discover_metrics(client, run_ids: List[str], sample_size: int = 10) -> Set[str]:
    """Discover available metric names by sampling runs."""
    all_metrics = set()
    sample = run_ids[:sample_size]
    for run_id in sample:
        run = client.get_run(run_id)
        all_metrics.update(run.data.metrics.keys())
    logger.info(f"Discovered {len(all_metrics)} metric names from {len(sample)} sample runs")
    return all_metrics


def process_run(
    client,
    run,
    metric_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract params and metric history from a single MLflow run.

    Returns:
        (param_dict, metric_rows) where param_dict has one entry per run,
        and metric_rows has one entry per (step, metrics) tuple.
    """
    run_id = run.info.run_id

    # Build param dict
    param_dict = dict(run.data.params)
    param_dict['run_id'] = run_id

    # Get sweep_name from tags
    tags = run.data.tags
    sweep_name = tags.get('sweep_name', 'default')
    param_dict['sweep_name'] = sweep_name

    # Download metric histories and merge by step
    rows_by_step = defaultdict(dict)
    for metric_name in metric_names:
        try:
            history = client.get_metric_history(run_id, metric_name)
            for entry in history:
                rows_by_step[entry.step][metric_name] = entry.value
        except Exception:
            pass

    metric_rows = []
    max_step = 0
    for step in sorted(rows_by_step.keys()):
        row = {'run_id': run_id, 'step': step}
        row.update(rows_by_step[step])
        metric_rows.append(row)
        max_step = max(max_step, step)

    param_dict['curr_step'] = max_step

    return param_dict, metric_rows


def main():
    args = parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = get_tracking_uri(args)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    logger.info(f"Tracking URI: {tracking_uri}")

    # Find experiment
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        logger.error(f"Experiment '{args.experiment}' not found")
        return
    experiment_id = experiment.experiment_id
    logger.info(f"Found experiment '{args.experiment}' (id={experiment_id})")

    # Query runs
    filter_parts = []
    status_filters = ["attributes.status = 'FINISHED'"]
    if args.include_failed:
        status_filters.append("attributes.status = 'FAILED'")
    if args.include_running:
        status_filters.append("attributes.status = 'RUNNING'")

    # MLflow search_runs doesn't support OR on status, so we query multiple times
    all_runs = []
    for status_filter in status_filters:
        filter_string = status_filter
        if not args.include_children:
            # Exclude child (per-seed) runs — they have mlflow.parentRunId tag
            filter_string += " and tags.`mlflow.parentRunId` = ''"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=50000,
        )
        all_runs.extend(runs)

    # Also try without the parentRunId filter in case some runs don't have it
    if not args.include_children:
        # The tag filter above may miss runs that don't have the tag at all.
        # Re-query without tag filter and deduplicate.
        run_ids_found = {r.info.run_id for r in all_runs}
        for status_filter in status_filters:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=status_filter,
                max_results=50000,
            )
            for r in runs:
                if r.info.run_id not in run_ids_found:
                    # Include only if it's NOT a child run
                    if 'mlflow.parentRunId' not in r.data.tags:
                        all_runs.append(r)
                        run_ids_found.add(r.info.run_id)

    logger.info(f"Found {len(all_runs)} runs")

    if args.max_runs is not None:
        all_runs = all_runs[:args.max_runs]
        logger.info(f"Limited to {len(all_runs)} runs")

    if len(all_runs) == 0:
        logger.warning("No runs found, exiting")
        return

    # Discover metrics
    run_ids = [r.info.run_id for r in all_runs]
    if args.metrics:
        metric_names = args.metrics
    else:
        metric_names = sorted(discover_metrics(client, run_ids))
    logger.info(f"Metrics to download: {metric_names}")

    # Process runs in parallel
    all_param_rows = []
    all_metric_rows = []

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        future_to_run = {
            executor.submit(process_run, client, run, metric_names): run
            for run in all_runs
        }

        for future in tqdm(as_completed(future_to_run), total=len(all_runs), desc="Downloading"):
            run = future_to_run[future]
            try:
                param_dict, metric_rows = future.result()
                if param_dict:
                    all_param_rows.append(param_dict)
                if metric_rows:
                    all_metric_rows.extend(metric_rows)
            except Exception as e:
                logger.warning(f"Failed to process run {run.info.run_id}: {e}")

    # Save CSVs
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment.replace(' ', '_').replace('-', '_')
    params_file = output_dir / f"{experiment_name}_params.csv"
    metrics_file = output_dir / f"{experiment_name}_metrics.csv"

    if all_param_rows:
        params_df = pd.DataFrame(all_param_rows)
        # Ensure run_id is first column
        cols = ['run_id', 'sweep_name', 'curr_step'] + sorted(
            [c for c in params_df.columns if c not in ('run_id', 'sweep_name', 'curr_step')]
        )
        params_df = params_df.reindex(columns=cols)
        params_df.to_csv(params_file, index=True)
        logger.info(f"Saved {len(params_df)} param rows to {params_file}")
    else:
        logger.warning("No param data to save")

    if all_metric_rows:
        metrics_df = pd.DataFrame(all_metric_rows)
        # Ensure run_id and step are first columns
        cols = ['run_id', 'step'] + sorted(
            [c for c in metrics_df.columns if c not in ('run_id', 'step')]
        )
        metrics_df = metrics_df.reindex(columns=cols)
        metrics_df.to_csv(metrics_file, index=True)
        logger.info(f"Saved {len(metrics_df)} metric rows to {metrics_file}")
    else:
        logger.warning("No metric data to save")

    logger.info(f"{len(all_param_rows)}/{len(all_runs)} runs saved successfully")


if __name__ == '__main__':
    main()
