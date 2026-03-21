from io import BytesIO
import itertools
import math
import os
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def t_distrib(vals):
    """Compute the 95% confidence interval for the mean of a set of values."""
    return st.t.interval(
        alpha=0.95, df=len(vals)-1, loc=np.mean(vals), scale=st.sem(vals))


def t_ci(vals):
    """
    Compute the 95% confidence interval for the mean of a set of values,
    and return the size of one side of the interval.
    """
    return t_distrib(vals)[1] - np.mean(vals)


def bin_df(df, n_bins=10, bin_var='step', zero_start=True, run_key='run_id', groupby_cols=None):
    if n_bins == -1:
        return df
    
    min_val = 0 if zero_start else df[bin_var].min()
    bin_size = (df[bin_var].max() - df[bin_var].min()) / n_bins
    bins = np.arange(min_val, df[bin_var].max() + bin_size, bin_size) # define bins
    labels = bins[:-1] # define labels as bin starting values
    df[bin_var] = pd.cut(df[bin_var], bins=bins, labels=labels)

    # Identify numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Use dictionary comprehension to create an aggregation dictionary
    agg_dict = {col: 'mean' if col in num_cols else 'first' for col in df.columns}

    # Determine grouping columns
    if groupby_cols is None:
        groupby_cols = [run_key, bin_var]
    else:
        # Ensure run_key and bin_var are included
        if run_key not in groupby_cols:
            groupby_cols = [run_key] + groupby_cols
        if bin_var not in groupby_cols:
            groupby_cols = groupby_cols + [bin_var]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        df = df.groupby(groupby_cols).agg(agg_dict)
    df[bin_var] = pd.to_numeric(df[bin_var])

    return df.reset_index(drop=True)

# COLOR_PALETTE = [
#     '#0086DA', '#CA3720', '#2BAF2A', '#9923DC',
#     '#F5C031', '#A4A4A3', '#EC82CB'
# ]


COLOR_PALETTE = [
    '#0086DA', '#CA3720', '#2BAF2A', '#9923DC',
    '#e67e22', '#34495e', '#1abc9c'
]


DEFAULT_COLOR_IDS = {
    'real': 0,
    'full method': 0,
    'distractor': 1,
    'adam + random recycling': 1,
    'non-zero weight init': 1,
    'no distractors': 2,
}


def get_color_palette(classes=None, n=None):
    """
    With no arguments, return the default color palette.
    With a list of classes, return a dict mapping each class to a color.
    With an integer n, return a list of n colors.
    """
    
    if classes is not None:
        if len(classes) > len(COLOR_PALETTE):
            raise ValueError(
                f'Only {len(COLOR_PALETTE)} colors available, but {len(classes)} '
                'classes were requested.'
            )
        # First set ids for classes in DEFAULT_COLOR_IDS
        # then do the rest alphabetically
        
        color_map = {}
        for c in classes:
            if c.lower() in DEFAULT_COLOR_IDS:
                color_map[c] = DEFAULT_COLOR_IDS[c.lower()]

        remaining_ids = [i for i in range(len(COLOR_PALETTE)) \
            if i not in color_map.values()]
        for c in sorted(classes, key=lambda x: x.lower()):
            if c not in color_map:
                color_map[c] = remaining_ids.pop(0)

        color_map = {c: COLOR_PALETTE[i] for c, i in color_map.items()}
        
        return color_map

    elif n is not None:
        if n > len(COLOR_PALETTE):
            raise ValueError(
                f'Only {len(COLOR_PALETTE)} colors available, but {n} were requested.'
            )
        return COLOR_PALETTE[:n]
    
    return COLOR_PALETTE
    
    
def standardize_env_name(name):
    name = name.lower() \
        .replace('-rand', '') \
        .replace('-stochastic', '') \
        .replace('-v0', '') \
        .replace('-fullobs', '') \
        .replace('-6x6', '') \
        .replace('-', ' ') \
        .replace('_', ' ')
    return name.title()


def set_fig_labels(xlabel, ylabel, xsci=False, ysci=False, ax=None):
    """
    Set the labels for the x and y axes of the current figure.
    If style is 'sci', use scientific notation for the y axis.
    """
    if ax is None:
        ax = plt.gca()

    if xsci:
        ax.ticklabel_format(
            style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.savefig(BytesIO())
        offset = ax.get_xaxis().get_offset_text()
        ax.set_xlabel(f'{xlabel} ({offset.get_text()})')
        offset.set_visible(False)
    else:
        ax.set_xlabel(xlabel)

    if ysci:
        ax.ticklabel_format(
            style='sci', axis='y', scilimits=(0,0), useMathText=True)
        plt.savefig(BytesIO())
        offset = ax.get_yaxis().get_offset_text()
        ax.set_ylabel(f'{ylabel} ({offset.get_text()})')
        offset.set_visible(False)
    else:
        ax.set_ylabel(ylabel)


def set_matplotlib_style(style='default'):
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    if style == 'default':
        plt.rcParams.update({
            'figure.figsize': (5, 3),
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'axes.titlepad': 10,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'lines.linewidth': 1.5
        })
    elif style == 'long':
        plt.rcParams.update({
            'figure.figsize': (12, 3),
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'axes.titlepad': 10,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'lines.linewidth': 1.5
        })
    elif style == '4-row':
        # Make plots 1.8x larger when there are 4 in a row
        plt.rcParams.update({
            'figure.figsize': (5, 4),
            'axes.labelsize': 23,
            'axes.titlesize': 25,
            'axes.titlepad': 21,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'lines.linewidth': 2.7
        })
    else:
        raise ValueError(f'Unknown style {style}')


def save_fig_versions(name, dir='../figures/svg/', type='svg', **kwargs):
    """
    Save the current figure with and with the legend.
    """
    if plt.gca().get_legend() is None:
        warnings.warn('No legend found')
    else:
        plt.gca().get_legend().set_visible(False)
    plt.savefig(
        f'{os.path.join(dir, name)}_nl.{type}', bbox_inches='tight', dpi=400, **kwargs)
    
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().set_visible(True)
    plt.savefig(
        f'{os.path.join(dir, name)}.{type}', bbox_inches='tight', dpi=400, **kwargs)
    
    if type == 'svg':
        save_fig_versions(name, '../figures/png/', type='png', **kwargs)


# Function to flatten nested dictionaries with dot notation
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Process each row for both config dataframes
def flatten_config_df(df):
    flattened_rows = []
    for _, row in df.iterrows():
        flat_row = {}
        for col, val in row.items():
            if isinstance(val, str) and val.startswith('{'):
                try:
                    # Try to evaluate string as dict
                    dict_val = eval(val)
                    if isinstance(dict_val, dict):
                        flat_row.update(flatten_dict(dict_val, parent_key=col))
                    else:
                        flat_row[col] = val
                except:
                    flat_row[col] = val
            else:
                flat_row[col] = val
        flattened_rows.append(flat_row)
    return pd.DataFrame(flattened_rows)


def select_experiment_runs(
    run_df: pd.DataFrame,
    config_df: Optional[pd.DataFrame],
    chosen_values: dict,
    id_col: str = 'run_id',
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Select runs from a run and config dataframe based on chosen values.
    
    Args:
        run_df: DataFrame containing run data.
        config_df: DataFrame containing config data. Not needed if only selecting from values
            column values that are in the run dataframe.
        chosen_values: Map of column names to values to select runs by.
        id_col: Column name to use for matching runs in run and config dataframes.
        
    Returns:
        Tuple of selected run and config dataframes. If no config df is provided, the second
        dataframe will be None.
    """
    
    if config_df is not None:
        # Build mask for config df based on chosen values
        mask = pd.Series(True, index=config_df.index)
        for col, val in chosen_values.items():
            mask = mask & (config_df[col] == val)
        chosen_config_df = config_df[mask]
        
        # Select matching runs
        chosen_run_df = run_df[run_df[id_col].isin(chosen_config_df[id_col])]
        return chosen_run_df, chosen_config_df
    else:
        # If no config df, just filter run_df directly
        mask = pd.Series(True, index=run_df.index)
        for col, val in chosen_values.items():
            mask = mask & (run_df[col] == val)
        chosen_run_df = run_df[mask]
        return chosen_run_df, None
    

def load_experiment_data(
    config_data_path: str,
    run_data_path: str,
    drop_duplicated: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load experiment data and split it by sweep name."""
    all_sweeps_config_df = pd.read_csv(config_data_path, index_col=0)
    all_sweeps_run_df = pd.read_csv(run_data_path, index_col=0)

    # Remove duplicate config rows, ignoring run_id column
    if drop_duplicated:
        all_sweeps_config_df = all_sweeps_config_df.drop_duplicates(
            subset = [col for col in all_sweeps_config_df.columns if col != 'run_id'],
            keep = 'last',
        )
        valid_run_ids = set(all_sweeps_config_df['run_id'].unique())
        all_sweeps_run_df = all_sweeps_run_df[all_sweeps_run_df['run_id'].isin(valid_run_ids)]
        
    all_sweeps_run_df = all_sweeps_run_df.dropna(subset=['step'])

    sweep_names = list(all_sweeps_config_df['sweep_name'].unique())

    # Split up dfs by sweep name
    run_dfs = {}
    config_dfs = {}
    for sweep_name in sweep_names:
        config_dfs[sweep_name] = all_sweeps_config_df[all_sweeps_config_df['sweep_name'] == sweep_name].reset_index(drop=True)
        # Drop experiments that did not complete
        max_step = config_dfs[sweep_name]['curr_step'].max()
        config_dfs[sweep_name] = config_dfs[sweep_name][config_dfs[sweep_name]['curr_step'] == max_step]
        
        run_ids = set(config_dfs[sweep_name]['run_id'].unique())
        run_dfs[sweep_name] = all_sweeps_run_df[all_sweeps_run_df['run_id'].isin(run_ids)].reset_index(drop=True)
        
    return config_dfs, run_dfs


def infer_sweep_vars(config_df: pd.DataFrame) -> list:
    """Given a df of config values from a single sweep, infer what the sweep variables are."""
    return [
        col for col in config_df.columns
        if (
            col != 'run_id' and
            col != 'seed' and
            '.' not in col and
            config_df[col].nunique() > 1
        )
    ]


def float_is_power_of_2(val, tol=1e-8):
    if isinstance(val, float) and val > 0.0:
        exp = math.log2(val)
        return abs(exp - round(exp)) < tol
    return False


def format_val_for_report(val):
    if isinstance(val, float) and float_is_power_of_2(val):
        exp = int(round(math.log2(val)))
        return f"2**{exp}"
    elif isinstance(val, float):
        if (abs(val) > 1e5 or (abs(val) < 1e-5 and val != 0)):
            return f"{val:.2e}"
        else:
            return f"{val}"
    else:
        return f"{val}"


def build_sorted_split_key(sweep_name, split_vars, split_vals):
    """Return key with split_vars/vals sorted by var name."""
    if len(split_vars) == 0:
        return sweep_name
    # Pair vars and vals, sort by var name
    sorted_pairs = sorted(zip(split_vars, split_vals), key=lambda x: x[0])
    key_parts = [sweep_name]
    for var, val in sorted_pairs:
        key_parts.append(f"{var}={format_val_for_report(val)}")
    return "&".join(key_parts)


def get_best_ablation_values(
    config_dfs: Dict[str, pd.DataFrame],
    run_dfs: Dict[str, pd.DataFrame],
    sweep_ablation_vars: Optional[Dict[str, List[str]]] = None,
    sweep_split_vars: Optional[Dict[str, List[str]]] = None,
    print_best_values: bool = True,
    metric_col: str = 'loss',
    metric_direction: str = 'min', # {'min', 'max'}
    metric_type: str = 'final_avg', # {'cumulative', 'final_avg'}
    step_col: str = 'step',
    split_dfs_by_split_vars: bool = False,
    nan_policy: str = 'propagate', # {'propagate', 'omit'}
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, any]]]:
    """Get the best ablation values for each sweep.
    
    Args:
        config_dfs: Dictionary mapping sweep names to config DataFrames.
        run_dfs: Dictionary mapping sweep names to run DataFrames.
        sweep_ablation_vars: Dictionary mapping sweep names to lists of ablation variables.
        sweep_split_vars: Dictionary mapping sweep names to lists of split variables.
        print_best_values: Whether to print the best values.
        metric_col: Name of the column containing the metric to use.
        metric_direction: Direction of the metric to use. 'min' uses the minimum value,
                         'max' uses the maximum value.
        metric_type: Type of metric to use. 'cumulative' uses mean loss over all steps,
                     'final_avg' uses mean loss over final 5% of steps.
        step_col: Name of the column containing time step information.
        split_dfs_by_split_vars: If True, return separate DataFrames for each split variable
                            combination with keys like 'sweep_name|var1=val1|var2=val2'.
                            If False, return one DataFrame per sweep containing all split
                            combinations (default behavior).
        nan_policy: How to handle NaN metric values when computing the mean.
                    'propagate' (default): any NaN in a group makes the group mean NaN,
                    preventing diverged runs from being selected as best.
                    'omit': ignore NaN values (previous behavior), computing the mean
                    over only the non-NaN values.
    
    Returns:
        Tuple of (best_config_dfs, best_run_dfs, best_var_vals) dictionaries.
        best_var_vals: dict mapping sweep/split key to dict of best ablation var->value.
    """
    best_config_dfs = {}
    best_run_dfs = {}
    best_var_vals = {}

    report_str = ""
    for sweep_name in config_dfs.keys():
        # Skip sweeps where we aren't doing a step-size ablation
        if sweep_ablation_vars is not None and sweep_name not in sweep_ablation_vars:
            continue

        config_df = config_dfs[sweep_name]
        run_df = run_dfs[sweep_name]

        split_vars = sweep_split_vars.get(sweep_name, []) if sweep_split_vars is not None else []
        
        if sweep_ablation_vars is not None:
            ablation_vars = sweep_ablation_vars[sweep_name]
        else:
            ablation_vars = infer_sweep_vars(config_df)
            ablation_vars = list(set(ablation_vars) - set(split_vars))

        if isinstance(ablation_vars, str):
            ablation_vars = [ablation_vars]

        # Merge config info into run df
        merge_cols = ['run_id'] + split_vars + ablation_vars
        run_df_with_ablation = run_df.merge(
            config_df[merge_cols],
            on = 'run_id',
            how = 'left'
        )

        # Filter data based on metric_type
        if metric_type == 'final_avg':
            # Get final 5% of time steps
            max_step = run_df_with_ablation[step_col].max()
            min_step = run_df_with_ablation[step_col].min()
            step_threshold = max_step - 0.05 * (max_step - min_step)
            run_df_filtered = run_df_with_ablation[run_df_with_ablation[step_col] >= step_threshold]
        elif metric_type == 'cumulative':
            run_df_filtered = run_df_with_ablation
        else:
            raise ValueError(f"Invalid metric_type: {metric_type}. Must be 'cumulative' or 'final_avg'")

        # Calculate mean loss grouped by ablation vars and split vars
        skipna = nan_policy == 'omit'
        groupby_cols = ablation_vars + split_vars
        if len(groupby_cols) == 0:
            # No groupby columns; every row is its own group
            mean_loss = run_df_filtered[[metric_col]].copy()
            mean_loss.index = pd.RangeIndex(len(mean_loss))
        else:
            if skipna:
                mean_loss = run_df_filtered.groupby(groupby_cols)[metric_col].mean()
            else:
                mean_loss = run_df_filtered.groupby(groupby_cols)[metric_col].agg(
                    lambda x: x.mean(skipna=False)
                )

        report_str += f"\n=== {sweep_name} ===\n"

        # Handle case where split_vars is an empty list
        if len(split_vars) == 0:
            split_combinations = [()]  # Single combination: ()
        else:
            split_var_values = [sorted(config_df[var].unique()) for var in split_vars]
            split_combinations = list(itertools.product(*split_var_values))

        # Find best ablation values for each split combination
        best_run_ids_all = []  # For collecting all run_ids when split_by_split_vars=False
        for split_vals in split_combinations:
            config_str = ""
            vals_str = ""
            if len(split_vars) == 0:
                config_str += "\nConfiguration: (no split vars)"
            else:
                config_str += "\nConfiguration: "
                config_str += " | ".join(
                    f"{var}={format_val_for_report(val)}"
                    for var, val in zip(split_vars, split_vals)
                )

            # Build query for this split combination
            split_query = pd.Series(True, index=mean_loss.index)
            for var, val in zip(split_vars, split_vals):
                split_query &= (mean_loss.index.get_level_values(var) == val)

            # Get losses for this split combination
            split_losses = mean_loss[split_query]

            # Skip this split if no runs are present or all losses are NaN
            if split_losses.empty or split_losses.isna().all():
                reason = "No runs" if split_losses.empty else "All runs have NaN metric"
                vals_str += f"  [WARNING] {reason} for split combination in '{sweep_name}': {config_str.strip()} Skipping.\n"
                report_str += vals_str
                continue

            if metric_direction == 'min':
                best_loss_idx = split_losses.idxmin()
            elif metric_direction == 'max':
                best_loss_idx = split_losses.idxmax()
            else:
                raise ValueError(f"Invalid metric_direction: {metric_direction}. Must be 'min' or 'max'")

            # Format idx as tuple of ablation var values even if there is only one ablation var
            best_loss_idx = best_loss_idx if isinstance(best_loss_idx, tuple) else (best_loss_idx,)

            # Add best ablation values to report and save for output
            best_ablation_vals = {}
            for i, var in enumerate(ablation_vars):
                val = best_loss_idx[i]
                best_ablation_vals[var] = val
                vals_str += f"  Best {var}={format_val_for_report(val)}\n"

            # Get run_ids for this best configuration
            query = pd.Series(True, index=config_df.index)
            for var, val in zip(split_vars, split_vals):
                query &= (config_df[var] == val)

            for var, val in zip(ablation_vars, best_loss_idx[:len(ablation_vars)]):
                query &= (config_df[var] == val)

            num_in_bucket = query.sum()
            config_str += f" | # runs: {num_in_bucket}\n"

            matching_runs = config_df[query]
            if len(matching_runs) == 0:
                report_str += f"  [WARNING] No matching runs found for configuration in '{sweep_name}': {config_str.strip()} ablation={best_loss_idx[:len(ablation_vars)]}\n"
                continue

            best_run_ids = matching_runs['run_id'].tolist()

            report_str += config_str + vals_str

            if split_dfs_by_split_vars:
                # Create key for this split combination, with sorted split vars
                split_key = build_sorted_split_key(sweep_name, split_vars, split_vals)

                best_config_dfs[split_key] = config_df[config_df['run_id'].isin(best_run_ids)]
                best_run_dfs[split_key] = run_df[run_df['run_id'].isin(best_run_ids)]
                best_var_vals[split_key] = best_ablation_vals
            else:
                # Collect all run_ids to store under original sweep name
                best_run_ids_all.extend(best_run_ids)
                # For the sweep key, save just last (if more than 1 split, will be last split combination value)
                # Instead: If multiple splits, accumulate best var vals per split
                if len(split_combinations) == 1:  # No split vars
                    best_var_vals[sweep_name] = best_ablation_vals
                else:
                    # Use a joined key like var1=val1|var2=val2 (sorted by var name)
                    split_key = build_sorted_split_key(sweep_name, split_vars, split_vals)
                    best_var_vals[split_key] = best_ablation_vals

        # If not splitting by split_vars, store all results under original sweep name
        if not split_dfs_by_split_vars:
            best_config_dfs[sweep_name] = config_df[config_df['run_id'].isin(best_run_ids_all)]
            best_run_dfs[sweep_name] = run_df[run_df['run_id'].isin(best_run_ids_all)]

        if print_best_values:
            print(report_str)
        report_str = ""
        
    return best_config_dfs, best_run_dfs, best_var_vals