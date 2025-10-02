from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from phd.feature_search.analysis.plotting_utils import *


def param_sensitivity_plot(
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

