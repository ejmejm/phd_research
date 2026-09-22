"""Turning sweep results into the paper's figures.

``loading``  Comet exports -> per-cell numbers, handling retries and divergence.
``plotting`` figure styling and the two plot types the paper uses.
"""

from .loading import (
    asymptotic_metric, dedupe_trials, diverged_run_ids, download_project,
    fetch_per_seed_arrays, filter_sweep, filter_to_best, load_export,
    normalize_columns, stack_per_seed,
)
from .plotting import (
    COLOR_PALETTE, get_color_palette, plot_ci_curve, plot_sensitivity,
    save_fig, set_style,
)

__all__ = [
    'COLOR_PALETTE', 'asymptotic_metric', 'dedupe_trials', 'diverged_run_ids',
    'download_project', 'fetch_per_seed_arrays', 'filter_sweep',
    'filter_to_best', 'get_color_palette', 'load_export', 'normalize_columns',
    'plot_ci_curve',
    'plot_sensitivity', 'save_fig', 'set_style', 'stack_per_seed',
]
