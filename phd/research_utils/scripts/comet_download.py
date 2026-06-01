"""Download run data from CometML and convert it into a CSV file."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import comet_ml
from comet_ml.api import API
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
        """Create and return argument parser for Comet data download.
        
        Returns:
            argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Download experiment data from CometML and save as CSV"
        )
        parser.add_argument(
            '--project', 
            type = str, 
            required = True,
            help = "CometML project name",
        )
        parser.add_argument(
            '--workspace', 
            type = str, 
            default = None,
            help = "CometML workspace name",
        )
        parser.add_argument(
            '--history_vars', 
            nargs = '*', 
            type = str, 
            default = None,
            help = "Specific metrics to download (default: all available)",
        )
        parser.add_argument(
            '--params', 
            nargs = '*', 
            type = str, 
            default = None,
            help = "Specific parameters to download (default: all available)",
        )
        parser.add_argument(
            '--max_experiments',
            type = int,
            default = None,
            help = "Maximum number of experiments to process (for testing)",
        )
        parser.add_argument(
            '--output_dir',
            type = str,
            default = None,
            help = "Output directory for the CSV file. If not provided, the "
                   "CSV will be saved in the current working directory.",
        )
        parser.add_argument(
            '--index_metric',
            type = str,
            default = None,
            help = "Metric to use as the index of the CSV file. "
                   "CometML's builtin step index (separate from metrics) will be used if not provided.",
        )
        parser.add_argument(
            '--include_crashed',
            action = 'store_true',
            default = False,
            help = "Include experiments that crashed in the output.",
        )
        parser.add_argument(
            '--include_running',
            action = 'store_true',
            default = False,
            help = "Include experiments that crashed in the output.",
        )
        parser.add_argument(
            '--n_threads',
            type = int,
            default = 8,
            help = "Number of threads to use for parallel processing (default: 4).",
        )
        parser.add_argument(
            '--save_batch_size',
            type = int,
            default = 300,
            help = "Number of experiments to process before saving data to disk (default: 100).",
        )
        
        return parser.parse_args()


METRIC_PARAM_DISCOVERY_SAMPLES = 10


def discover_all_metrics_and_params(
        experiments: List[Any],
        sample_size: Optional[int] = None,
    ) -> Tuple[Set[str], Set[str]]:
    """Discover all available metrics and parameters from experiments.

    Args:
        experiments: List of CometML experiment objects.
        sample_size: How many experiments to sample for the discovery scan.
            If None (default), uses the module constant
            ``METRIC_PARAM_DISCOVERY_SAMPLES``. Pass a larger value to reduce
            the chance of missing rare params that only appear on a small
            subset of experiments (e.g. one sweep out of several).

    Returns:
        tuple: (set of all metric names, set of all parameter names)
    """
    logger.info("Discovering available metrics and parameters...")
    all_metrics = set()
    all_params = set()

    # Sample a random subset of experiments to discover available metrics/params
    if sample_size is None:
        sample_size = METRIC_PARAM_DISCOVERY_SAMPLES
    sample_size = min(sample_size, len(experiments))
    indices = list(range(len(experiments)))
    random.shuffle(indices)
    sample_experiments = [experiments[i] for i in indices[:sample_size]]
    
    for experiment in tqdm(sample_experiments, desc='Sampling experiments'):
        try:
            param_names = [p['name'] for p in experiment.get_parameters_summary()]
            all_params.update(param_names)
            
            system_metric_names = experiment.get_system_metric_names()
            metric_names = [m['name'] for m in experiment.get_metrics_summary()]
            metric_names = list(set(metric_names) - set(system_metric_names))
            all_metrics.update(metric_names)
            
        except Exception as e:
            logger.warning(f"Failed to get metrics/params for experiment {experiment.id}: {e}")
            continue
    
    logger.info(f"Discovered {len(all_metrics)} unique metrics and {len(all_params)} unique parameters")
    return all_metrics, all_params


def get_experiment_data(
        experiment: comet_ml.api.APIExperiment,
        metric_names: List[str],
        param_names: List[str],
        index_metric: Optional[str] = None,
        include_crashed: bool = False,
        include_running: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert a CometML experiment into a dictionary of paramters and a list of metric rows.

    Args:
        experiment: CometML experiment object.
        metric_names: List of metric names to include.
        param_names: List of parameter names to include.
        index_metric: Optional metric to use as the index of the CSV file.
        include_crashed: Include experiments that crashed.
        include_running: Include experiments that are still running.
    """
    if filter_experiment(experiment, include_crashed=include_crashed, include_running=include_running):
        return {}, []
    
    all_metrics = experiment.get_metrics()
    
    metric_names = set(metric_names)
    param_names = set(param_names)
    
    # Default is to use Comet's builtin step index, separate from metrics
    if index_metric is None:
        rows = defaultdict(dict)
        for entry in all_metrics:
            if entry['metricName'] in metric_names and entry['step'] is not None:
                rows[entry['step']][entry['metricName']] = entry['metricValue']

        metric_rows = [v for k, v in sorted(rows.items())]
    
    # If an index metric is provided, use it to index the rows
    else:
        valid_timesteps = [x['timestamp'] for x in all_metrics if x['metricName'] == index_metric]
        if len(valid_timesteps) == 0:
            raise ValueError(f"The index metric {index_metric} is not present in the experiments!")
        
        rows = {t: {} for t in valid_timesteps}
        for entry in all_metrics:
            timestamp = entry['timestamp']
            if entry['metricName'] in metric_names and timestamp in rows:
                rows[timestamp][entry['metricName']] = entry['metricValue']
        
        metric_rows = list(sorted(rows.values(), key=lambda x: float(x[index_metric])))
    
    param_data = experiment.get_parameters_summary()
    param_dict = {x['name']: x['valueCurrent'] for x in param_data if x['name'] in param_names}
    
    # Add experiment key to the param dict and metric rows so they are joinable
    param_dict.update({'run_id': experiment.id})
    for row in metric_rows:
        row.update({'run_id': experiment.id})
    
    return param_dict, metric_rows


def filter_experiment(
        experiment: comet_ml.api.APIExperiment,
        include_crashed: bool = False,
        include_running: bool = False,
    ) -> bool:
    """Determine if an experiment should be filtered based on its state.

    Args:
        experiment: CometML experiment object.
        include_crashed: Include experiments that crashed.
        include_running: Include experiments that are still running.

    Returns:
        True if experiment should be filtered, False if it should be included.
    """
    state = experiment.get_state()

    if state == 'finished':
        return False
    elif state == 'crashed' and include_crashed:
        return False
    elif state == 'running' and include_running:
        return False
    else:
        logger.warning(f"Experiment {experiment.id} has unknown state {state}, skipping...")
        return True


def process_single_experiment(
        experiment: comet_ml.api.APIExperiment,
        metrics: List[str],
        params: List[str],
        index_metric: Optional[str] = None,
        include_crashed: bool = False,
        include_running: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process a single experiment and return its data.

    Args:
        experiment: CometML experiment object.
        metrics: List of metric names to include.
        params: List of parameter names to include.
        index_metric: Optional metric to use as the index.
        include_crashed: Include experiments that crashed.
        include_running: Include experiments that are still running.

    Returns:
        Tuple of (param_dict, metric_rows) or ({}, []) if experiment should be filtered.
    """
    try:
        return get_experiment_data(
            experiment, metrics, params,
            index_metric=index_metric,
            include_crashed=include_crashed,
            include_running=include_running,
        )
    except Exception as e:
        logger.warning(f"Failed to process experiment {experiment.id}: {e}")
        return {}, []


def save_batch_data(
        all_param_rows: List[Dict[str, Any]], 
        all_metric_rows: List[Dict[str, Any]], 
        params_file: Path, 
        metrics_file: Path,
        param_columns: Optional[List[str]] = None,
        metric_columns: Optional[List[str]] = None,
        param_start_index: int = 0,
        metric_start_index: int = 0,
    ) -> Tuple[int, int]:
    """Save current batch data to CSV files and clear memory.
    
    Args:
        all_param_rows: List of parameter dictionaries to save.
        all_metric_rows: List of metric dictionaries to save.
        params_file: Path to the parameters CSV file.
        metrics_file: Path to the metrics CSV file.
        param_columns: Fixed column order for parameters.
        metric_columns: Fixed column order for metrics.
        param_start_index: Starting index for parameters DataFrame.
        metric_start_index: Starting index for metrics DataFrame.
        
    Returns:
        Tuple of (next_param_index, next_metric_index) for the next batch.
    """
    next_param_index = param_start_index
    next_metric_index = metric_start_index
    
    if len(all_param_rows) > 0:
        params_df = pd.DataFrame(all_param_rows)
        
        # Ensure consistent column ordering
        if param_columns is not None:
            # Reorder columns to match the fixed order, fill missing columns with NaN
            params_df = params_df.reindex(columns=param_columns)
        
        # Set continuous index starting from param_start_index
        params_df.index = range(param_start_index, param_start_index + len(params_df))
        next_param_index = param_start_index + len(params_df)
        
        # Append to existing file or create new one
        if params_file.exists():
            params_df.to_csv(params_file, mode='a', header=False, index=True)
        else:
            params_df.to_csv(params_file, index=True)
    
    if len(all_metric_rows) > 0:
        metrics_df = pd.DataFrame(all_metric_rows)
        
        # Ensure consistent column ordering
        if metric_columns is not None:
            # Reorder columns to match the fixed order, fill missing columns with NaN
            metrics_df = metrics_df.reindex(columns=metric_columns)
        
        # Set continuous index starting from metric_start_index
        metrics_df.index = range(metric_start_index, metric_start_index + len(metrics_df))
        next_metric_index = metric_start_index + len(metrics_df)
        
        # Append to existing file or create new one
        if metrics_file.exists():
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=True)
        else:
            metrics_df.to_csv(metrics_file, index=True)
    
    return next_param_index, next_metric_index


def run(
    project: str,
    workspace: Optional[str] = None,
    history_vars: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    max_experiments: Optional[int] = None,
    output_dir: Optional[str] = None,
    index_metric: Optional[str] = None,
    include_crashed: bool = False,
    include_running: bool = False,
    n_threads: int = 8,
    save_batch_size: int = 300,
    discovery_samples: Optional[int] = None,
):
    """Download experiment data from CometML and save as CSV.

    Args:
        project: CometML project name.
        workspace: CometML workspace name (default: your default workspace).
        history_vars: Specific metrics to download (default: all available).
        params: Specific parameters to download (default: all available).
        max_experiments: Maximum number of experiments to process.
        output_dir: Output directory for CSV files (default: current directory).
        index_metric: Metric to use as the CSV index (default: Comet's step index).
        include_crashed: Include experiments that crashed.
        include_running: Include experiments that are still running.
        n_threads: Number of threads for parallel processing.
        save_batch_size: Number of experiments per batch before saving to disk.
        discovery_samples: How many experiments to scan for param/metric name
            discovery (default ``METRIC_PARAM_DISCOVERY_SAMPLES``=10). Set to
            ``len(experiments)`` (or higher) to scan every experiment when
            params only appear on a small subset (e.g. one sweep out of
            several in the project) and the default sample misses them.
    """
    api = API()
    api.use_cache(False)

    if workspace is None:
        workspace = api.get_default_workspace()

    logger.info("Looking for experiments...")
    experiments = api.get_experiments(workspace, project)
    logger.info(f"Found {len(experiments)} experiments")

    if max_experiments is not None:
        logger.info(f"Limiting to {max_experiments} experiments")
        experiments = experiments[:max_experiments]

    if not history_vars or not params:
        logger.info("Discovering available metrics and parameters...")
        all_metrics, all_params = discover_all_metrics_and_params(
            experiments, sample_size=discovery_samples,
        )

    metrics = history_vars if history_vars is not None else all_metrics
    params_to_use = params if params is not None else all_params

    # Create consistent column ordering with run_id as first column
    param_columns = ['run_id'] + sorted(list(params_to_use))
    metric_columns = ['run_id'] + sorted(list(metrics))

    logger.info(f"\nParameters to collect: {params_to_use}")
    logger.info(f"\nMetrics to collect: {metrics}\n")

    logger.info(f"Querying experiment data using {n_threads} threads...")

    all_param_rows = []
    all_metric_rows = []
    n_valid_runs = 0
    processed_count = 0

    # Initialize continuous index tracking
    param_index = 0
    metric_index = 0

    # Setup output directory and file paths
    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    params_file = out_dir / f"{project}_params.csv"
    metrics_file = out_dir / f"{project}_metrics.csv"

    # Remove existing files to start fresh
    if params_file.exists():
        params_file.unlink()
    if metrics_file.exists():
        metrics_file.unlink()

    # Process experiments in batches to control memory usage
    for batch_start in range(0, len(experiments), save_batch_size):
        batch_end = min(batch_start + save_batch_size, len(experiments))
        batch = experiments[batch_start:batch_end]
        current_batch_num = batch_start // save_batch_size + 1
        total_batches = (len(experiments) + save_batch_size - 1) // save_batch_size

        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} experiments)")

        # Process this batch in parallel
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit all experiments in this batch
            future_to_experiment = {
                executor.submit(
                    process_single_experiment, experiment, metrics, params_to_use,
                    index_metric=index_metric,
                    include_crashed=include_crashed,
                    include_running=include_running,
                ): experiment
                for experiment in batch
            }

            # Process completed tasks in this batch
            for future in tqdm(as_completed(future_to_experiment), total=len(batch), desc=f"Batch {current_batch_num}"):
                experiment = future_to_experiment[future]
                try:
                    param_dict, metric_rows = future.result()

                    if len(metric_rows) > 0 and len(param_dict) > 0:
                        n_valid_runs += 1
                        all_metric_rows.extend(metric_rows)
                        all_param_rows.append(param_dict)

                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to get result for experiment {experiment.id}: {e}")
                    processed_count += 1

        # Save batch data and clear memory after each batch
        if len(all_param_rows) > 0 or len(all_metric_rows) > 0:
            param_index, metric_index = save_batch_data(
                all_param_rows,
                all_metric_rows,
                params_file,
                metrics_file,
                param_columns=param_columns,
                metric_columns=metric_columns,
                param_start_index=param_index,
                metric_start_index=metric_index
            )
            all_param_rows.clear()
            all_metric_rows.clear()
            logger.info(f"Saved and cleared batch {current_batch_num} data after processing {processed_count} experiments")

    logger.info(f"{n_valid_runs}/{len(experiments)} runs saved.")

    # Count total metric rows in the final file
    if metrics_file.exists():
        final_metrics_df = pd.read_csv(metrics_file)
        logger.info(f"{len(final_metrics_df)} metric rows saved.")
    else:
        logger.info("0 metric rows saved.")


def main():
    args = parse_args()
    run(
        project=args.project,
        workspace=args.workspace,
        history_vars=args.history_vars,
        params=args.params,
        max_experiments=args.max_experiments,
        output_dir=args.output_dir,
        index_metric=args.index_metric,
        include_crashed=args.include_crashed,
        include_running=args.include_running,
        n_threads=args.n_threads,
        save_batch_size=args.save_batch_size,
    )


if __name__ == '__main__':
    main()