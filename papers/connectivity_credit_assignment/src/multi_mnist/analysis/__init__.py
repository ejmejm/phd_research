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
    COLOR_PALETTE, INLINE_LABEL_SIZE, get_color_palette, label_curves_inline,
    measure_inline_labels, plot_ci_curve, plot_sensitivity, save_fig, set_style,
)

__all__ = [
    'COLOR_PALETTE', 'INLINE_LABEL_SIZE', 'asymptotic_metric', 'dedupe_trials',
    'diverged_run_ids', 'download_project', 'fetch_per_seed_arrays', 'filter_sweep',
    'filter_to_best', 'get_color_palette', 'label_curves_inline',
    'load_export', 'measure_inline_labels', 'normalize_columns', 'plot_ci_curve',
    'plot_sensitivity', 'save_fig', 'set_style', 'stack_per_seed',
]
