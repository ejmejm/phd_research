"""Download run data from Wandb and convert it into a CSV file."""

# Source: https://docs.wandb.ai/guides/track/public-api-guide

import argparse
from functools import reduce
from requests.exceptions import HTTPError
from itertools import chain
import json
import os
import pandas as pd 
import shutil
import wandb
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--entity', type=str, default='ejmejm')
parser.add_argument('--project', type=str, default='discrete-representations-discrete_mbrl')
# Options are run and table
parser.add_argument('--data_type', type=str, default='run')
parser.add_argument('--sweeps', nargs='*', type=str, default=[])
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--history_x', type=str, default='step')
parser.add_argument('--history_vars', nargs='*', type=str, default=['*'])
parser.add_argument('--include_crashed', action='store_true')
parser.set_defaults(include_crashed=False)

TABLE_TMP_DIR = 'data/table_tmp'
TABLE_EXTENSION = '.table.json'
TABLE_NAME = 'n-step_stats'
ARTIFACT_TABLE_NAME = TABLE_NAME.replace('-', '')

api = wandb.Api()
args = parser.parse_args()

api = wandb.Api()
args = parser.parse_args()
PROJECT_PATH = f'{args.entity}/{args.project}'
runs = []
if args.tags:
    runs.append(api.runs(PROJECT_PATH, filters={'tags': {'$in': args.tags}}))
    print(f'Found {len(runs[-1])} runs with tags in {args.tags}')
if args.sweeps:
    for sweep in args.sweeps:
        runs.append(api.sweep(f'{PROJECT_PATH}/{sweep}').runs)
        print(f'Found {len(runs[-1])} runs with sweep {sweep}')
if runs:
    runs = chain(*runs)

if not args.sweeps and not args.tags:
    runs = api.runs(PROJECT_PATH) 


BACKUP_PROJECTS = ['discrete-mbrl-full', 'discrete-representations-discrete_mbrl_sweep_configs']


data_list, config_list, name_list, id_list, sweep_list = [], [], [], [], []
for run in runs:
    if run.state == 'crashed' and not args.include_crashed:
        continue
    if run.id in id_list:
        continue

    # .history contains the output keys/values for metrics like reward.
    #  We call ._json_dict to omit large files

    if args.data_type == 'run':
        if len(args.history_vars) == 0 or (
                len(args.history_vars) == 1 \
                and args.history_vars[0].lower() in ('*', 'all')
            ):
            rows = run.scan_history()
            df = pd.DataFrame(rows)
        elif args.history_x is None:
            df = run.history(keys=args.history_vars)
        else:
            var_dfs = []
            empty_vars = []
            for var in args.history_vars:
                df = run.history(keys=[args.history_x, var])
                if df.empty:
                    empty_vars.append(var)
                    continue
                if '_step' in df.columns:
                    df = df.drop('_step', axis=1)
                var_dfs.append(df)
            if empty_vars:
                if len(empty_vars) == len(args.history_vars):
                    warnings.warn(f'No data in {run.id} for all vars, skipping...')
                    continue
                else:
                    warnings.warn(f'No data in {run.id} for vars: {empty_vars}')
            try:
                df = reduce(lambda left, right: pd.merge(
                    left, right, on=args.history_x, how='outer'), var_dfs)
            except KeyError as e:
                print(f'Error: {e}')
                print(f'Run {run.name} is missing a value for {args.history_x}')
                continue
        if '_step' in df.columns:
            df = df.drop('_step', axis=1)
    elif args.data_type == 'table':
        artifact = None
        for project in [args.project] + BACKUP_PROJECTS:
            try:
                artifact = api.artifact(
                    f'{args.entity}/{project}/run-{run.id}-{ARTIFACT_TABLE_NAME}:v0')
                break
            except wandb.errors.CommError:
                continue
        
        if artifact is None:
            print(f'Could not find artifact for run {run.id}, skipping run!')
            continue

        try:
            artifact.download(TABLE_TMP_DIR)
        except HTTPError:
            print(f'Download error, retrying...')
            try:
                artifact.download(TABLE_TMP_DIR)
            except:
                print(f'Download error, skipping run {run.name} ({run.id})!')
        file_path = os.path.join(TABLE_TMP_DIR, TABLE_NAME + TABLE_EXTENSION)
        with open(file_path, 'r') as f:
            table_data = json.load(f)
        df = pd.DataFrame(columns=table_data['columns'], data=table_data['data'])
    df['run_id'] = run.id
    data_list.append(df)

    print(run.name, run.id) #[key for key in run.history().to_dict().keys() if 'gradient' not in key])

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    id_list.append(run.id)

    if run.sweep:
        # .sweep contains the sweep id.
        sweep_list.append(run.sweep.id)
    else:
        sweep_list.append(None)

# Delete table_tmp directory
if os.path.exists(TABLE_TMP_DIR):
    shutil.rmtree(TABLE_TMP_DIR)

# Create config dataframe
config_df_data = {k: [] for k in config_list[0].keys()}
for config in config_list:
    for k in config_df_data.keys():
        config_df_data[k].append(config.get(k))
run_df_data = {
    'name': name_list,
    'id': id_list,
    'sweep': sweep_list,
    **config_df_data
}
run_df = pd.DataFrame(run_df_data)
os.makedirs('data', exist_ok=True)
run_df.to_csv(f'data/{args.project}_config_data.csv')

# Create logged data dataframe
log_df = pd.concat(data_list)
log_df.reset_index(drop=True, inplace=True)
log_df.to_csv(f'data/{args.project}_{args.data_type}_data.csv')

print('{} runs saved.'.format(len(name_list)))