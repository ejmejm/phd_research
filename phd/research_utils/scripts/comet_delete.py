"""Delete runs from a CometML experiment filtered by date range."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
from typing import List

from comet_ml.api import API, APIExperiment
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Create and return argument parser for Comet experiment deletion.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Delete experiments from CometML filtered by date range"
    )
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help="CometML project name",
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default=None,
        help="CometML workspace name",
    )

    # Date filtering - at least one required
    date_group = parser.add_argument_group('date filtering (at least one required)')
    date_group.add_argument(
        '--before',
        type=str,
        default=None,
        help="Delete experiments created before this date (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    date_group.add_argument(
        '--after',
        type=str,
        default=None,
        help="Delete experiments created after this date (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )

    parser.add_argument(
        '--include_running',
        action='store_true',
        default=False,
        help="Also delete experiments that are still running.",
    )
    parser.add_argument(
        '--n_threads',
        type=int,
        default=8,
        help="Number of threads to use for parallel deletion (default: 8).",
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        default=False,
        help="List experiments that would be deleted without actually deleting them.",
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        default=False,
        help="Skip confirmation prompt.",
    )

    args = parser.parse_args()

    if args.before is None and args.after is None:
        parser.error("At least one of --before or --after is required.")

    return args


def parse_date(date_str: str) -> datetime:
    """Parse a date string into a timezone-aware UTC datetime.

    Args:
        date_str: Date string in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.

    Returns:
        datetime: Parsed datetime in UTC.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")


def _check_experiment_date(
    experiment: APIExperiment,
    before: datetime | None,
    after: datetime | None,
    include_running: bool,
) -> APIExperiment | None:
    """Check if a single experiment matches the date and state filters.

    Returns:
        The experiment if it matches, None otherwise.
    """
    if not include_running:
        try:
            state = experiment.get_state()
            if state == 'running':
                return None
        except Exception:
            pass

    try:
        start_time_ms = experiment.start_server_timestamp
        if start_time_ms is None:
            metadata = experiment.get_metadata()
            start_time_ms = metadata.get('startTimeMillis')
        if start_time_ms is None:
            return None
        exp_time = datetime.fromtimestamp(int(start_time_ms) / 1000, tz=timezone.utc)
    except Exception:
        return None

    if before is not None and exp_time >= before:
        return None
    if after is not None and exp_time <= after:
        return None

    return experiment


def filter_experiments_by_date(
    experiments: List[APIExperiment],
    before: datetime | None,
    after: datetime | None,
    include_running: bool,
    n_threads: int = 8,
) -> List[APIExperiment]:
    """Filter experiments by creation date and state.

    Args:
        experiments: List of CometML experiment objects.
        before: Only include experiments created before this date.
        after: Only include experiments created after this date.
        include_running: Whether to include running experiments.
        n_threads: Number of threads for parallel filtering.

    Returns:
        List of experiments matching the filters.
    """
    matched = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(_check_experiment_date, exp, before, after, include_running): exp
            for exp in experiments
        }
        for future in tqdm(as_completed(futures), total=len(experiments), desc="Filtering experiments by date"):
            result = future.result()
            if result is not None:
                matched.append(result)

    return matched


def delete_experiment(api: API, experiment: APIExperiment) -> tuple[str, bool, str]:
    """Delete a single experiment.

    Returns:
        Tuple of (experiment_id, success, message).
    """
    try:
        api.delete_experiment(experiment.id)
        return (experiment.id, True, "deleted")
    except Exception as e:
        return (experiment.id, False, str(e))


def run(
    project: str,
    workspace: str | None = None,
    before: str | datetime | None = None,
    after: str | datetime | None = None,
    include_running: bool = False,
    n_threads: int = 8,
    dry_run: bool = False,
    yes: bool = False,
):
    """Delete experiments from CometML filtered by date range.

    Args:
        project: CometML project name.
        workspace: CometML workspace name (default: your default workspace).
        before: Delete experiments created before this date. String (YYYY-MM-DD
            or YYYY-MM-DD HH:MM:SS) or datetime object.
        after: Delete experiments created after this date. String or datetime.
        include_running: Also delete experiments that are still running.
        n_threads: Number of threads for parallel operations.
        dry_run: List experiments that would be deleted without deleting.
        yes: Skip confirmation prompt.
    """
    if before is None and after is None:
        raise ValueError("At least one of 'before' or 'after' is required.")

    api = API()
    api.use_cache(False)

    if workspace is None:
        workspace = api.get_default_workspace()

    # Parse dates if given as strings
    before_dt = parse_date(before) if isinstance(before, str) else before
    after_dt = parse_date(after) if isinstance(after, str) else after

    logger.info(f"Looking for experiments in {workspace}/{project}...")
    experiments = api.get_experiments(workspace, project)
    logger.info(f"Found {len(experiments)} total experiments")

    # Filter by date
    to_delete = filter_experiments_by_date(experiments, before_dt, after_dt, include_running, n_threads)

    date_desc = []
    if after_dt:
        date_desc.append(f"after {after_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if before_dt:
        date_desc.append(f"before {before_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Found {len(to_delete)} experiments {' and '.join(date_desc)}")

    if len(to_delete) == 0:
        logger.info("No experiments to delete.")
        return

    if dry_run:
        logger.info("Dry run — the following experiments would be deleted:")
        for exp in to_delete:
            try:
                name = exp.get_name() or exp.id
            except Exception:
                name = exp.id
            try:
                start_ms = exp.start_server_timestamp or exp.get_metadata().get('startTimeMillis')
                ts = datetime.fromtimestamp(int(start_ms) / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts = "unknown"
            logger.info(f"  {name} (id={exp.id}, created={ts} UTC)")
        return

    # Confirmation prompt
    if not yes:
        response = input(f"\nDelete {len(to_delete)} experiments from {workspace}/{project}? [y/N] ")
        if response.lower() not in ('y', 'yes'):
            logger.info("Aborted.")
            return

    # Delete in parallel
    logger.info(f"Deleting {len(to_delete)} experiments using {n_threads} threads...")
    n_success = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_exp = {
            executor.submit(delete_experiment, api, exp): exp
            for exp in to_delete
        }
        for future in tqdm(as_completed(future_to_exp), total=len(to_delete), desc="Deleting"):
            exp_id, success, msg = future.result()
            if success:
                n_success += 1
            else:
                n_failed += 1
                logger.warning(f"Failed to delete {exp_id}: {msg}")

    logger.info(f"Done. {n_success} deleted, {n_failed} failed.")


def main():
    args = parse_args()
    run(
        project=args.project,
        workspace=args.workspace,
        before=args.before,
        after=args.after,
        include_running=args.include_running,
        n_threads=args.n_threads,
        dry_run=args.dry_run,
        yes=args.yes,
    )


if __name__ == '__main__':
    main()
