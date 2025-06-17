import ast
import argparse
import copy
import logging
import os
import signal
import subprocess
import time
import yaml

import comet_ml
from comet_ml import Optimizer


logger = logging.getLogger(__name__)


# Create args
parser = argparse.ArgumentParser()

parser.add_argument(
    '-s', '--sweep_id', default=None,
    help='Will perform run(s) for the sweep with the given ID.',
)
parser.add_argument(
    '-n', '--count', type=int, default=1,
    help='Number of runs to perform for the sweep.',
)
parser.add_argument(
    '-c', '--config', type=str, nargs='*', default=None,
    help='Path to a YAML or JSON file containing sweep configuration. '
         'A sweep will be created for each file provided.',
)
parser.add_argument(
    '--offline', action='store_true',
    help='Run the sweeps in offline mode. Note that this flag must be '
         'enabled when you run sweeps, not when you create them.',
)


def run_sweep(sweep_id, offline=False) -> str:
    """Run a sweep and returns the status of the optimizer after the run."""
    if 'COMET_OPTIMIZER_ID' in os.environ:
        del os.environ['COMET_OPTIMIZER_ID']
    experiment_class = comet_ml.OfflineExperiment if offline else comet_ml.Experiment
    opt = Optimizer(sweep_id, verbose=0, experiment_class=experiment_class)
    config = opt.status()
    if config['status'] == 'completed':
        return 'completed'
    
    command = config['parameters']['sweep_command']['values'][0].split()
    
    environ = os.environ.copy()
    environ['COMET_OPTIMIZER_ID'] = opt.id
    environ['COMET_MODE'] = 'offline' if offline else 'online'

    # This is adapted from comet_optimize.py

    proc = subprocess.Popen(command, env=environ, stderr=subprocess.STDOUT)
    try:
        proc.wait()
        if proc.returncode != 0:
                print('There was an error running the script!')
                print('Exit code:', proc.returncode)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)

        # Check that all subprocesses exit cleanly
        i = 0
        while i < 60:
            proc.poll()
            dead = proc.returncode is not None
            if dead:
                break

            i += 1
            time.sleep(1)

            # Timeout, hard-kill all the remaining subprocess
            if i >= 60:
                proc.poll()
                if proc.returncode is None:
                    proc.kill()

    print()
    results = opt.status()
    for key in ['algorithm', 'status']:
        print('     ', '%s:' % key, results[key])
    if isinstance(results['endTime'], float) and \
         isinstance(results['startTime'], float):
        print(
            '     ',
            'time:',
            (results['endTime'] - results['startTime']) / 1000.0,
            'seconds',
        )
    
    opt_status = opt.status()['status']
    return opt_status


def create_sweep(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yml', '.yaml')):
            config = yaml.safe_load(f)
        else:
            config = ast.literal_eval(f.read())

    if 'project' in config:
        config['parameters']['sweep_project'] = config['project']
        del config['project']
    if 'command' not in config:
        raise ValueError(f"'command' field not found in config file: {config_path}")
    config['parameters']['sweep_command'] = config['command']
    del config['command']
    if 'name' in config:
        config['parameters']['sweep_name'] = config['name']

    final_configs = []
    config_stack = [config]
    while len(config_stack) > 0:
        tmp_config = config_stack.pop()
        for key, entry in tmp_config['parameters'].items():
            if isinstance(entry, dict) and 'dependents' in entry:
                # Make a copy of the tmp_config
                # Add one of the depent values as a value to the parameters
                # Add the copy to the stack
                n_vals = len(entry['values'])
                dependents = entry['dependents']
                for i in range(n_vals):
                    new_config = copy.deepcopy(tmp_config)
                    for dependent in dependents:
                        new_config['parameters'][key]['values'] = [entry['values'][i]]

                        new_config['parameters'][dependent] = {}
                        new_config['parameters'][dependent]['type'] = \
                            dependents[dependent]['type']
                        new_config['parameters'][dependent]['values'] = \
                            [dependents[dependent]['values'][i]]
                    del new_config['parameters'][key]['dependents']
                    config_stack.append(new_config)
                break
        else:
            final_configs.append(tmp_config)
    # print(len(final_configs))
    # import sys; sys.exit()

    if 'COMET_OPTIMIZER_ID' in os.environ:
        del os.environ['COMET_OPTIMIZER_ID']
    opts = []
    combinations = []
    for config in final_configs:
        opt = Optimizer(config)
        n_combinations = str(opt.status().get('configSpaceSize', '?'))
        opts.append(opt.id)
        combinations.append(n_combinations)
        print('Created sweep with id:\n', opt.id, f'({n_combinations} combinations)')
    return opts, combinations


def main():
    """Main entry point for the comet_sweep command-line tool."""
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if args.sweep_id is None and args.config is None:
        print('You must provide either a sweep id or a config file.')
        parser.print_help()
        return

    if args.config:
        config_ids = []
        combination_counts = []
        config_names = []
        for config in args.config:
            # Get file name from path
            config_names.append(config.split('/')[-1])
            new_ids, new_combination_counts = create_sweep(config)
            config_ids.extend(new_ids)
            combination_counts.extend(new_combination_counts)
        
        print('\n===== Created sweeps with ids =====')
        for name, count, sweep_id in zip(config_names, combination_counts, config_ids):
            print(f'{name} ({count}): {sweep_id}')

    if args.sweep_id is not None:
        if args.sweep_id == 'new':
            args.sweep_id = new_ids[-1]
        for _ in range(args.count):
            status = run_sweep(args.sweep_id, offline=args.offline)
            if status == 'completed':
                print('Sweep completed!')
                break


if __name__ == '__main__':
    main()