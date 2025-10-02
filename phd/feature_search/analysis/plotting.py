from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from phd.feature_search.analysis.plotting_utils import *


def plot_param_sensitivity(
        run_df: pd.DataFrame,
        config_df: Optional[pd.DataFrame],
        x_col: str,
        title: str,
        x_label: str,
        y_label: str,
        metric_col: str = 'loss',
        metric_type: str = 'cumulative', # {'cumulative', 'final_avg', 'last'}
        step_col: str = 'step',
        id_col: str = 'run_id',
        hue_col: Optional[str] = None,
        legend_title: Optional[str] = None,
        show_ci: bool = False,
        baseline_col: Optional[str] = None,
        pow_2_x_axis: bool = False,
    ):
    if metric_type.lower() == 'cumulative':
        final_step_df = run_df.groupby(id_col).agg({
            metric_col: 'sum',
            **({baseline_col: 'sum'} if baseline_col is not None else {}),
            **{col: 'last' for col in run_df.columns if col not in [metric_col, baseline_col]}
        }).reset_index()
        if config_df is not None:
            final_step_df = final_step_df.merge(config_df, on=id_col, how='left')
        y_label = f'Cumulative {y_label}'
        
    elif metric_type.lower() == 'last':
        final_step_df = run_df.groupby(id_col).last().reset_index()
        if config_df is not None:
            final_step_df = final_step_df.merge(config_df, on=id_col, how='left')

    elif metric_type.lower() == 'final_avg':
        # Get the last 5% of steps for each run
        max_steps_per_run = run_df.groupby(id_col)[step_col].max()
        threshold_steps = max_steps_per_run * 0.95

        # Filter for last 5% of steps and calculate mean loss
        final_step_df = run_df.merge(
            threshold_steps.rename('threshold_step'),
            left_on = id_col,
            right_index = True
        )
        final_step_df = final_step_df[final_step_df[step_col] >= final_step_df['threshold_step']]
        
        # Calculate mean of metric columns while preserving other columns
        agg_dict = {col: 'mean' if col in [metric_col, baseline_col] else 'last' 
                   for col in final_step_df.columns 
                   if col not in [id_col, 'threshold_step']}
            
        final_step_df = final_step_df.groupby(id_col).agg(agg_dict).reset_index()
        
        # Merge with config data
        if config_df is not None:
            final_step_df = final_step_df.merge(config_df, on=id_col, how='left')

        y_label = f'{y_label} (Final 5%)'
        
    else:
        raise ValueError(f"Invalid metric type: {metric_type}!")

    # Create a mapping of actual values to evenly spaced positions
    x_values = sorted(final_step_df[x_col].unique())
    x_positions = np.arange(len(x_values))
    x_map = dict(zip(x_values, x_positions))

    # Create temporary column with evenly spaced x positions
    final_step_df['x_position'] = final_step_df[x_col].map(x_map)

    # Create plot with evenly spaced x positions
    sns.lineplot(
        data = final_step_df,
        x = 'x_position',
        y = metric_col, 
        hue = hue_col,
        marker = 'o',
        palette = 'deep',
        errorbar = ('ci', 95) if show_ci else None,
    )

    if baseline_col is not None:
        sns.lineplot(
            data = final_step_df,
            x = 'x_position',
            y = baseline_col,
            color = 'black',
            label = 'Baseline',
        )

    # Customize plot
    plt.grid(True, alpha=0.4)  # Add light grid
    plt.xlim(-0.1, len(x_values) - 0.9)  # Add small padding on both sides
    plt.xticks(x_positions, x_values)  # Use original values as labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if legend_title is not None:
        plt.legend(title=legend_title)
        
    if pow_2_x_axis:
        for ax in plt.gcf().axes:
            x_vals = [float(x.get_text()) for x in ax.get_xticklabels()]
            ax.set_xticklabels([f'{"0" if x == 0 else f"$2^{{{int(np.log2(x))}}}$"}' for x in x_vals])


def plot_learning_curves(
        run_df: pd.DataFrame,
        config_df: Optional[pd.DataFrame] = None,
        subplot_col = None,
        subplot_values = None,
        n_bins = 200,
        figsize = (12, 10),
        max_cols = 2,
        subplot_col_label = None,
        same_y_axis = True,
        x_col = 'step',
        y_col = 'loss',
        y_label = None,
        hue_col = None,
        legend_title = None,
    ):
    """Creates subplots of learning curves for different values of a variable.
    
    Args:
        run_df: DataFrame containing run data
        config_df: DataFrame containing config data
        subplot_col: Column name to split subplots by. If None, creates single plot.
        subplot_values: List of values to plot. If None, uses all unique values
        n_bins: Number of bins for the learning curves
        figsize: Figure size as (width, height) tuple
        max_cols: Maximum number of columns in subplot grid
        subplot_col_label: Label for the subplot column
        same_y_axis: Whether all subplots should share the same y-axis scale
        x_col: Column to plot on x-axis (default: 'step')
        y_col: Column to plot on y-axis (default: 'loss')
        y_label: Label for y-axis (default: same as y_col)
        hue_col: Column to use for line colors
        legend_title: Title for the legend
    """
    # Get full dataset
    plot_df = run_df
    if config_df:
        plot_df = plot_df.merge(config_df, on='run_id', how='left')
        
    # Remove runs that contain any NaN or inf values
    nan_inf_runs = plot_df.groupby('run_id')[y_col].apply(lambda x: x.isna().any() | np.isinf(x).any())
    valid_runs = nan_inf_runs[~nan_inf_runs].index
    plot_df = plot_df[plot_df['run_id'].isin(valid_runs)]
    
    # Get subplot values if not provided
    if subplot_col is not None and subplot_values is None:
        subplot_values = sorted(plot_df[subplot_col].unique())
    elif subplot_col is None:
        subplot_values = [None]
    
    # Calculate number of rows/cols for subplots
    n_plots = len(subplot_values)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Calculate mid 98% percentile for consistent y-axis if requested
    if same_y_axis:
        filtered_df = plot_df[
            (plot_df[y_col] >= np.percentile(plot_df[y_col], 1)) &
            (plot_df[y_col] <= np.percentile(plot_df[y_col], 99))
        ]
        y_range = filtered_df[y_col].max() - filtered_df[y_col].min()
        y_pad = y_range * 0.1
        y_min = filtered_df[y_col].min() - y_pad
        y_max = filtered_df[y_col].max() + y_pad

    # Get max step value for x-axis limit
    max_step = plot_df[x_col].max()

    for i, val in enumerate(subplot_values):
        # Filter for current value if subplot_col exists
        curr_df = plot_df.copy()
        if subplot_col is not None:
            curr_df = plot_df[plot_df[subplot_col] == val].copy()
        
        # Bin data
        curr_df = bin_df(curr_df, n_bins=n_bins)
        curr_df = curr_df[curr_df[hue_col].notna()]  # Remove NaN values
        curr_df[hue_col] = curr_df[hue_col].astype(int)
        
        # return curr_df
        
        # Create subplot
        sns.lineplot(
            data = curr_df,
            x = x_col,
            y = y_col,
            hue = hue_col,
            palette = 'deep',
            errorbar = None,
            ax = axes[i]
        )
        
        # Customize subplot
        axes[i].grid(True, alpha=0.4)
        axes[i].set_xlim(0, max_step)
        if same_y_axis:
            axes[i].set_ylim(y_min, y_max)
        else:
            # Calculate y limits for this subplot
            filtered_curr_df = curr_df[
                (curr_df[y_col] >= np.percentile(curr_df[y_col], 1)) &
                (curr_df[y_col] <= np.percentile(curr_df[y_col], 99))
            ]
            y_range = filtered_curr_df[y_col].max() - filtered_curr_df[y_col].min()
            y_pad = y_range * 0.1
            axes[i].set_ylim(
                filtered_curr_df[y_col].min() - y_pad,
                filtered_curr_df[y_col].max() + y_pad
            )
        
        if subplot_col is not None:
            axes[i].set_title(f'{subplot_col_label if subplot_col_label else subplot_col} = {val}')
        axes[i].set_xlabel('step (binned)')
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axes[i].set_ylabel(y_label if y_label else y_col)
        
        # Only show legend on first subplot
        if i > 0:
            axes[i].get_legend().remove()
        elif legend_title is not None:
            axes[i].legend(title=legend_title)
    
    # Remove any extra subplots
    for i in range(len(subplot_values), len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title if provided
    if subplot_col_label and subplot_col is not None:
        plt.suptitle(f'Learning Curves for {subplot_col_label}')
    
    plt.tight_layout()