from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from phd.feature_search.analysis.analysis_utils import *


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
        pow_2_legend: bool = False,
    ):
    if metric_type.lower() == 'cumulative':
        final_step_df = run_df.groupby(id_col).agg({
            metric_col: 'sum',
            **({baseline_col: 'sum'} if baseline_col is not None else {}),
            **{col: 'last' for col in run_df.columns if col not in [metric_col, baseline_col, id_col]}
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

    # Create color mapping for consistent colors and sorted legend
    if hue_col is not None:
        # Try sorting with a cast to float first, then as str if float fails
        hue_unique = final_step_df[hue_col].unique()
        try:
            hue_values = sorted(hue_unique, key=lambda x: float(x))
        except Exception:
            hue_values = sorted(hue_unique)
        palette = sns.color_palette('deep', n_colors=len(hue_values))
        color_map = dict(zip(hue_values, palette))
    else:
        color_map = None

    # Create plot with evenly spaced x positions
    sns.lineplot(
        data = final_step_df,
        x = 'x_position',
        y = metric_col, 
        hue = hue_col,
        marker = 'o',
        palette = color_map,
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
    
    # Handle legend with sorted values
    if hue_col is not None and legend_title is not None:
        # Create custom legend handles with sorted hue values
        legend_elements = [plt.Line2D([0], [0], color=color_map[val], marker='o', label=val) 
                         for val in hue_values]
        
        if pow_2_legend:
            labels = [f'{"0" if val == 0 else f"$2^{{{int(np.log2(float(val)))}}}$"}' 
                     for val in hue_values]
            plt.legend(legend_elements, labels, title=legend_title)
        else:
            plt.legend(legend_elements, hue_values, title=legend_title)
    elif legend_title is not None:
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
        pow_2_legend = False,
        xlim = None,
        ylim = None,
        show_ci = False,
        show_individual_lines = False,
        line_alpha = 0.3,
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
        pow_2_legend: Whether to display legend values as powers of 2
        xlim: Optional tuple of (min, max) for x-axis limits
        ylim: Optional tuple of (min, max) for y-axis limits
        show_ci: Whether to show 95% confidence intervals
        show_individual_lines: Whether to plot all individual runs in pale colors (default: False)
        line_alpha: Transparency for individual lines (default: 0.3)
    """
    # Get full dataset
    plot_df = run_df
    if config_df is not None:
        plot_df = plot_df.merge(config_df, on='run_id', how='left')
    
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
    if same_y_axis and ylim is None:
        # Remove NaN and inf values for percentile calculation
        valid_y = plot_df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_y) > 0:
            filtered_df = plot_df[
                (plot_df[y_col] >= np.percentile(valid_y, 1)) &
                (plot_df[y_col] <= np.percentile(valid_y, 99))
            ]
            y_range = filtered_df[y_col].max() - filtered_df[y_col].min()
            y_pad = y_range * 0.1
            y_min = filtered_df[y_col].min() - y_pad
            y_max = filtered_df[y_col].max() + y_pad
        else:
            y_min, y_max = 0, 1  # Default if no valid data

    # Get max step value for x-axis limit
    max_step = plot_df[x_col].max()
    
    # Create color mapping for consistent colors across subplots
    if hue_col is not None:
        hue_values = sorted(plot_df[hue_col].unique())
        palette = sns.color_palette('deep', n_colors=len(hue_values))
        color_map = dict(zip(hue_values, palette))

    for i, val in enumerate(subplot_values):
        # Filter for current value if subplot_col exists
        curr_df = plot_df.copy()
        if subplot_col is not None:
            curr_df = plot_df[plot_df[subplot_col] == val].copy()
        
        # Bin data for mean plot
        binned_df = bin_df(curr_df, n_bins=n_bins)

        # Show individual runs in pale colors if requested
        if show_individual_lines and 'run_id' in curr_df.columns:
            if hue_col is not None:
                for hue_val in hue_values:
                    sub_df = curr_df[curr_df[hue_col] == hue_val]
                    color = color_map[hue_val]
                    # Group by run_id
                    for run_id, run_df_local in sub_df.groupby('run_id'):
                        axes[i].plot(run_df_local[x_col], run_df_local[y_col],
                                     color=color, alpha=line_alpha, linewidth=1, zorder=1)
            else:
                color = 'C0'
                for run_id, run_df_local in curr_df.groupby('run_id'):
                    axes[i].plot(run_df_local[x_col], run_df_local[y_col],
                                 color=color, alpha=line_alpha, linewidth=1, zorder=1)
        
        # Main mean line plot
        sns.lineplot(
            data = binned_df,
            x = x_col,
            y = y_col,
            hue = hue_col,
            palette = color_map if hue_col is not None else None,
            errorbar = ('ci', 95) if show_ci else None,
            ax = axes[i],
            legend = False  # Don't show legend for any subplot
        )
        
        # Customize subplot
        axes[i].grid(True, alpha=0.4)
        
        if xlim is not None:
            axes[i].set_xlim(*xlim)
        else:
            axes[i].set_xlim(0, max_step)
            
        if ylim is not None:
            axes[i].set_ylim(*ylim)
        elif same_y_axis:
            axes[i].set_ylim(y_min, y_max)
        else:
            # Calculate y limits for this subplot
            valid_y = binned_df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_y) > 0:
                filtered_curr_df = binned_df[
                    (binned_df[y_col] >= np.percentile(valid_y, 1)) &
                    (binned_df[y_col] <= np.percentile(valid_y, 99))
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
    
    # Add a single legend to the figure
    if hue_col is not None:
        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], color=color_map[val], label=val) 
                         for val in hue_values]
        
        # Add the legend to the figure
        if pow_2_legend:
            labels = [f'{"0" if val == 0 else f"$2^{{{int(np.log2(float(val)))}}}$"}' 
                     for val in hue_values]
            fig.legend(legend_elements, labels, title=legend_title,
                      bbox_to_anchor=(1.05, 0.5), loc='center left')
        else:
            fig.legend(legend_elements, hue_values, title=legend_title,
                      bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Remove any extra subplots
    for i in range(len(subplot_values), len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title if provided
    if subplot_col_label and subplot_col is not None:
        plt.suptitle(f'Learning Curves for {subplot_col_label}')
    
    plt.tight_layout()