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


def discover_all_metrics_and_params(experiments: List[Any]) -> Tuple[Set[str], Set[str]]:
    """Discover all available metrics and parameters from experiments.
    
    Args:
        experiments: List of CometML experiment objects.
        
    Returns:
        tuple: (set of all metric names, set of all parameter names)
    """
    logger.info("Discovering available metrics and parameters...")
    all_metrics = set()
    all_params = set()
    
    # Sample a random subset of experiments to discover available metrics/params
    sample_size = min(METRIC_PARAM_DISCOVERY_SAMPLES, len(experiments))
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
        args: argparse.Namespace,
        index_metric: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert a CometML experiment into a dictionary of paramters and a list of metric rows.
    
    Args:
        experiment: CometML experiment object.
        metric_names: List of metric names to include.
        param_names: List of parameter names to include.
        index_metric: Optional metric to use as the index of the CSV file.
    """
    if filter_experiment(experiment, args):
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
        args: argparse.Namespace,
    ) -> bool:
    """Determine if an experiment should be filtered based on its state.
    
    Args:
        experiment: CometML experiment object.
        args: Command line arguments.
        
    Returns:
        True if experiment should be filtered, False if it should be included.
    """
    state = experiment.get_state()
    
    if state == 'finished':
        return False
    elif state == 'crashed' and args.include_crashed:
        return False
    elif state == 'running' and args.include_running:
        return False
    else:
        logger.warning(f"Experiment {experiment.id} has unknown state {state}, skipping...")
        return True


def process_single_experiment(
        experiment: comet_ml.api.APIExperiment,
        metrics: List[str],
        params: List[str],
        args: argparse.Namespace,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process a single experiment and return its data.
    
    Args:
        experiment: CometML experiment object.
        metrics: List of metric names to include.
        params: List of parameter names to include.
        args: Command line arguments.
        
    Returns:
        Tuple of (param_dict, metric_rows) or ({}, []) if experiment should be filtered.
    """
    try:
        return get_experiment_data(experiment, metrics, params, args, index_metric=args.index_metric)
    except Exception as e:
        logger.warning(f"Failed to process experiment {experiment.id}: {e}")
        return {}, []


def save_batch_data(
        all_param_rows: List[Dict[str, Any]], 
        all_metric_rows: List[Dict[str, Any]], 
        params_file: Path, 
        metrics_file: Path, 
    ) -> None:
    """Save current batch data to CSV files and clear memory.
    
    Args:
        all_param_rows: List of parameter dictionaries to save.
        all_metric_rows: List of metric dictionaries to save.
        params_file: Path to the parameters CSV file.
        metrics_file: Path to the metrics CSV file.
    """
    if len(all_param_rows) > 0:
        params_df = pd.DataFrame(all_param_rows)
        params_df.reset_index(drop=True, inplace=True)
        
        # Append to existing file or create new one
        if params_file.exists():
            params_df.to_csv(params_file, mode='a', header=False, index=False)
        else:
            params_df.to_csv(params_file, index=False)
    
    if len(all_metric_rows) > 0:
        metrics_df = pd.DataFrame(all_metric_rows)
        metrics_df.reset_index(drop=True, inplace=True)
        
        # Append to existing file or create new one
        if metrics_file.exists():
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_file, index=False)


def main():
    args = parse_args()

    api = API()
    api.use_cache(False)
    
    if args.workspace is None:
        args.workspace = api.get_default_workspace()

    logger.info("Looking for experiments...")
    experiments = api.get_experiments(args.workspace, args.project)
    logger.info(f"Found {len(experiments)} experiments")
    
    if args.max_experiments is not None:
        logger.info(f"Limiting to {args.max_experiments} experiments")
        experiments = experiments[:args.max_experiments]

    if not args.history_vars or not args.params:
        logger.info("Discovering available metrics and parameters...")
        all_metrics, all_params = discover_all_metrics_and_params(experiments)
        
    metrics = args.history_vars if args.history_vars is not None else all_metrics
    params = args.params if args.params is not None else all_params
    
    logger.info(f"\nParameters to collect: {params}")
    logger.info(f"\nMetrics to collect: {metrics}\n")

    logger.info(f"Querying experiment data using {args.n_threads} threads...")

    all_param_rows = []
    all_metric_rows = []
    n_valid_runs = 0
    batch_size = args.save_batch_size
    processed_count = 0
    
    # Setup output directory and file paths
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path.cwd()
    params_file = output_dir / f"{args.project}_params.csv"
    metrics_file = output_dir / f"{args.project}_metrics.csv"
    
    # Remove existing files to start fresh
    if params_file.exists():
        params_file.unlink()
    if metrics_file.exists():
        metrics_file.unlink()
    
    # Process experiments in batches to control memory usage
    for batch_start in range(0, len(experiments), batch_size):
        batch_end = min(batch_start + batch_size, len(experiments))
        batch = experiments[batch_start:batch_end]
        current_batch_num = batch_start // batch_size + 1
        total_batches = (len(experiments) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} experiments)")
        
        # Process this batch in parallel
        with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            # Submit all experiments in this batch
            future_to_experiment = {
                executor.submit(process_single_experiment, experiment, metrics, params, args): experiment
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
            save_batch_data(all_param_rows, all_metric_rows, params_file, metrics_file)
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


if __name__ == '__main__':
        main()