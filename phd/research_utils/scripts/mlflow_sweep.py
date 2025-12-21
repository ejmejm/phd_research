from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import logging
import os
import signal
import time
import subprocess
from typing import Any, Dict, List, Tuple

from filelock import FileLock
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna

# TODO:
# - [x] Implement grid sweep from config with run command and parameters
# - [x] Only have a single parent MLFlow run for all trials even when distributed
# - [ ] Make a sampler that uses a lock to avoid duplicate runs
# - [ ] Change sweep run # that is printed to be the # run in this sweep, not in the whole experiment
# - [ ] Don't mark parent MLFlow run as complete until all trials are done
# - [ ] Make sure filelocks are deleted after use
# - [x] Add an option to delete a sweep, and remove it from both MLFlow and Optuna storages
# - [ ] Better handle failed runs, ideally retrying or just overwritting with the same param set
# - [ ] Delete failed runs when being replaced
# - [ ] Implement random sweep from config with run command and parameters
# - [ ] Implement hyperparameter sensitivity from config with run command and parameters

# 1. Get config from parser that describes the sweep (run command, parameters, potential parent)
# 2. Create a new local study in optuna with SQLite storage


# Workflow for a single sweep:
# 1. Create a config that specifies the sweep name, the run command, and sweep parameters
# 2. SSH to compute cluster
# 3. Launch and array job that each call this script with the same config


# If the sweep name is included in the config, then the sweep name will need to be included in the CC run command.
# I want is so that locally, the whole sweep can be run purely from the config.
# I don't want to specify hardware-wise how the sweep should be run in this script.

CONFIG_REQUIRED_FIELDS = [
    'experiment', 'sweep_name', 'command', 'algorithm', 'parameters',
    'optuna_storage', 'mlflow_storage', 'output_dir',
]
VALID_ALGORITHMS = ['grid']
MLFLOW_DEFAULT_TRACKING_URI = 'sqlite:///mlruns.db'


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        logging.DEBUG: '\033[90m',     # Grey
        logging.INFO: '\033[38;5;141m',  # Purple/Blue
        logging.WARNING: '\033[33m',   # Yellow
        logging.ERROR: '\033[31m',     # Red
        logging.CRITICAL: '\033[1;31m',  # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', type=str, nargs='+',
        help='Path(s) to YAML or JSON file(s) containing sweep configuration. '
             'A sweep will be run for each config file provided.',
    )
    parser.add_argument(
        '-n', '--n_trials', type=int, default=None,
        help='Number of trials to perform for the sweep. If None, '
             'will perform until the study is marked as complete.',
    )
    parser.add_argument(
        '-j', '--n_jobs', type=int, default=1,
        help='Number of parallel jobs in this call of the script.',
    )
    parser.add_argument(
        '--delete', action='store_true',
        help='Delete all data associated with the MLFlow sweep and Optuna study.',
    )
    return parser.parse_args()


def validate_config(config: DictConfig):
    for field in CONFIG_REQUIRED_FIELDS:
        assert field in config, f"'{field}' field is required in the config!"
        
    assert config.algorithm in VALID_ALGORITHMS, f"'algorithm' must be one of {VALID_ALGORITHMS}!"
    
    if 'spec' in config and 'direction' in config.spec:
        assert config.spec.direction in ['minimize', 'maximize'], f"'direction' must be one of ['minimize', 'maximize']!"


@dataclass
class ParamSpec(ABC):
    name: str
    
    @abstractmethod
    def to_categorical(self) -> 'CategoricalParam':
        """Convert parameter to a CategoricalParam if possible."""
        pass
    
    def to_list(self) -> List[Any]:
        return self.to_categorical().values
    
    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggest a value for the parameter for the given Optuna trial."""
        pass


@dataclass
class AtomicParam(ParamSpec):
    value: Any

    def to_categorical(self) -> 'CategoricalParam':
        return CategoricalParam(self.name, [self.value])

    def suggest(self, trial: optuna.Trial) -> Any:
        return self.value


@dataclass
class CategoricalParam(ParamSpec):
    values: List[Any]

    def to_categorical(self) -> 'CategoricalParam':
        return self

    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_categorical(self.name, self.values)


@dataclass
class IntRangeParam(ParamSpec):
    low: int
    high: int
    log: bool = False
    step: int = 1
    
    def to_categorical(self) -> CategoricalParam:
        if self.log:
            raise ValueError(f"Integer ranges in log space cannot be converted to CategoricalParam!")
        return CategoricalParam(self.name, list(range(self.low, self.high, self.step)))
    
    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclass
class FloatRangeParam(ParamSpec):
    low: float
    high: float
    log: bool = False
    step: float | None = None
    
    def to_categorical(self) -> CategoricalParam:
        if self.log:
            raise ValueError(f"Float ranges in log space cannot be converted to CategoricalParam!")
        if self.step is None:
            raise ValueError(f"FloatRangeParam '{self.name}' must specify 'step' when converting to CategoricalParam!")
        return CategoricalParam(self.name, list(np.linspace(self.low, self.high, num=self.step)))
    
    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


def validate_param_dict_entry(name: str, entry: Dict[str, Any]):
    """Raises a ValueError if the specification does not follow the required format.

    Args:
        name (str): Name of the parameter.
        spec (Dict[str, Any]): Specification dictionary for the parameter.

    Supported types:
        - 'atomic': requires 'type' and 'value'
        - 'int_range'/'float_range': requires 'type', 'low', and 'high'; supports optional 'log' and 'step'
        - 'categorical': requires 'type' and 'values'

    Example of passing ("OK"):
        >>> validate_param_dict_spec("seed", {"type": "atomic", "value": 0})
        # passes (no error)

    Example of raising error ("ERROR"):
        >>> validate_param_dict_spec("hidden_dim", {"type": "atomic", "val": 128})
        Traceback (most recent call last):
            ...
        ValueError: Atomic parameter 'hidden_dim' missing required fields: ['value']
    """
    if 'type' not in entry:
        raise ValueError(f"Parameter must specify 'type' in the config!")
    
    if entry['type'] not in ['atomic', 'int_range', 'float_range', 'categorical']:
        raise ValueError(f"Invalid parameter type: {entry['type']}, but be one of ['atomic', 'int_range', 'float_range', 'categorical']!")
    
    if entry['type'] == 'atomic':
        allowed_keys = {'type', 'value'}
        missing = [k for k in ['type', 'value'] if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(f"Atomic parameter '{name}' missing required fields: {missing}")
        if extraneous:
            raise ValueError(f"Atomic parameter '{name}' has extraneous fields: {extraneous}")
    
    elif entry['type'] in ['int_range', 'float_range']:
        required_keys = {'type', 'low', 'high'}
        optional_keys = {'log', 'step'}
        allowed_keys = required_keys.union(optional_keys)
        missing = [k for k in required_keys if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(f"{entry['type'].capitalize()} parameter '{name}' missing required fields: {missing}")
        if extraneous:
            raise ValueError(f"{entry['type'].capitalize()} parameter '{name}' has extraneous fields: {extraneous}")
    
    elif entry['type'] == 'categorical':
        allowed_keys = {'type', 'values'}
        missing = [k for k in ['type', 'values'] if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(f"Categorical parameter '{name}' missing required fields: {missing}")
        if extraneous:
            raise ValueError(f"Categorical parameter '{name}' has extraneous fields: {extraneous}")


def param_from_config_entry(name: str, value: Any) -> ParamSpec:
    """
    Create a ParamSpec (AtomicParam, CategoricalParam, IntRangeParam, or FloatRangeParam) from a name and value.

    Args:
        name (str): The parameter name.
        value (Any): The parameter value or specification.

    Returns:
        ParamSpec: Corresponding parameter specification dataclass.

    Examples:
        >>> param_from_value("dropout", {"type": "float_range", "low": 0.1, "high": 0.5})
        FloatRangeParam(name='dropout', low=0.1, high=0.5, log=False, step=None)

        >>> param_from_value("hidden_dim", [128, 256, 512])
        CategoricalParam(name='hidden_dim', values=[128, 256, 512])

        >>> param_from_value("seed", 123)
        AtomicParam(name='seed', value=123)
    """
    if isinstance(value, Dict):
        validate_param_dict_entry(name, value)

        if value['type'] == 'atomic':
            return AtomicParam(name, value['value'])
        elif value['type'] == 'int_range':
            return IntRangeParam(
                name, value['low'], value['high'], value.get('log', False), value.get('step', 1))
        elif value['type'] == 'float_range':
            return FloatRangeParam(
                name, value['low'], value['high'], value.get('log', False), value.get('step', None))
        elif value['type'] == 'categorical':
            return CategoricalParam(name, value['values'])

    if isinstance(value, list):
        return CategoricalParam(name, value)

    else:
        return AtomicParam(name, value)
    

def config_params_to_spec_dict(config: DictConfig) -> Dict[str, ParamSpec]:
    """Convert config.parameters into a dict of parameter specifications."""
    params_dict = OmegaConf.to_container(config.parameters, throw_on_missing=True)
    param_specs = {}
    for name, value in params_dict.items():
        param_specs[name] = param_from_config_entry(name, value)
    return param_specs


def get_study_name(config: DictConfig) -> str:
    return f'{config.experiment}/{config.sweep_name}'


def make_sampler(config: DictConfig, param_specs: Dict[str, ParamSpec]) -> optuna.samplers.BaseSampler:
    if config.algorithm == 'grid':
        search_space = {
            name: param_spec.to_list() for name, param_spec in param_specs.items()
        }
        return optuna.samplers.GridSampler(search_space)
    else:
        raise ValueError(f"Invalid sweep algorithm: {config.algorithm}")


def _optuna_study_lock_path(config: DictConfig) -> str:
    # Compute a unique filename based on (study_name, storage) tuple
    study_name = get_study_name(config)
    storage = str(config.optuna_storage)
    lock_id = hashlib.md5(f'{study_name}:{storage}'.encode('utf-8')).hexdigest()
    return os.path.join(config.output_dir, f'study_{lock_id}.lock')


def init_study(config: DictConfig) -> Tuple[optuna.Study, Dict[str, ParamSpec]]:
    study_name = get_study_name(config)
    param_specs = config_params_to_spec_dict(config)

    with FileLock(_optuna_study_lock_path(config)):
        study = optuna.create_study(
            study_name = study_name,
            sampler = make_sampler(config, param_specs),
            storage = config.optuna_storage,
            direction = config.get('spec', {}).get('direction'),
            load_if_exists = True,
        )
    return study, param_specs


def get_param_values_for_trial(trial: optuna.Trial, param_specs: Dict[str, ParamSpec]) -> Dict[str, Any]:
    param_values = {}
    for name, param_spec in param_specs.items():
        param_values[name] = param_spec.suggest(trial)
    return param_values


def run_experiment(
    trial: optuna.Trial,
    config: DictConfig,
    param_specs: Dict[str, ParamSpec],
    mlflow_client: MlflowClient,
) -> float:
    param_values = get_param_values_for_trial(trial, param_specs)
    params_str_list = [f'{name}={value}' for name, value in param_values.items()]
    command_list = config.command.split() + params_str_list
    full_command_str = ' '.join(command_list)
    
    logger.debug(f"Active MLFlow run: {mlflow.active_run().info.run_id}.")
    # TODO: Create a name based on the index of the run in the sweep, and use a lock to avoid duplicate runs
    trial_run = mlflow.start_run(
        tags = {'sweep_name': config.sweep_name},
        nested = True,
    )
    trial_run_id = trial_run.info.run_id
    mlflow.log_param('full_command', full_command_str)
    
    environ = os.environ.copy()
    environ['MLFLOW_RUN_ID'] = trial_run_id
    
    logger.info(f"Sweep run #{trial._trial_id}.")
    logger.info(f"Created trial MLFlow run: {trial_run_id}.")
    logger.info(f"Running new trial with command: `{full_command_str}`")
    
    proc = subprocess.Popen(
        command_list,
        env=environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Print subprocess output with faded color
    FADED = '\033[2;37m'  # Dim grey
    RESET = '\033[0m'
    try:
        for line in proc.stdout:
            print(f"{FADED}{line}{RESET}", end='')
        proc.wait()
        if proc.returncode != 0:
            logger.error('There was an error running the script!')
            logger.error(f'Exit code: {proc.returncode}')
            raise ValueError(f"Trial run {trial_run_id} failed with exit code {proc.returncode}!")
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
    
    # TODO: Check to make sure the trial run is ended at this point
    mlflow.end_run()
    logger.info(f"Trial run {trial_run_id} finished with exit code {proc.returncode}.")
    
    optimization_metric = config.get('spec', {}).get('metric')
    trial_run = mlflow_client.get_run(trial_run_id)
    summary_metrics = trial_run.data.metrics
    
    if optimization_metric is not None:
        trial_run = mlflow_client.get_run(trial_run_id)
        summary_metrics = trial_run.data.metrics
        if optimization_metric not in summary_metrics:
            raise ValueError(f"Optimization metric {optimization_metric} not found in trial run {trial_run_id}!")
        return summary_metrics[optimization_metric]
    
    return 0


def start_mlflow_parent_run(client: MlflowClient, config: DictConfig, optuna_study_name: str) -> mlflow.ActiveRun:
    # Make lock for this study
    lock_id = hashlib.md5(f'{config.mlflow_storage}:{optuna_study_name}'.encode('utf-8')).hexdigest()
    lock_path = os.path.join(config.output_dir, f'mlflow_sweep_parent_{lock_id}.lock')
    
    with FileLock(lock_path):
        # First search for a run to get if one already exists
        print(config.sweep_name, type(config.sweep_name), optuna_study_name, type(optuna_study_name))
        runs = mlflow.search_runs(
            experiment_names = [config.experiment],
            filter_string = f'tags.sweep_name = "{config.sweep_name}" AND tags.optuna_study_name = "{optuna_study_name}"',
            output_format = 'list',
        )
        assert len(runs) <= 1, f"Multiple parent runs found for sweep {config.sweep_name} and study {optuna_study_name}!"
        
        # Create a new parent run if one doesn't exist
        if len(runs) == 0:
            parent_run = mlflow.start_run(
                run_name = config.sweep_name,
                tags = {
                    'sweep_name': config.sweep_name,
                    'optuna_study_name': optuna_study_name,
                },
            )
            logger.info(f"Creating new parent MLFlow run: {config.sweep_name}.")
            return parent_run
    
        # Otherwise, return the existing parent run
        run_id = runs[0].info.run_id
        logger.info(f"Using existing parent MLFlow run: {run_id}.")
        return mlflow.start_run(run_id=run_id)
    

def run_sweep(args: argparse.Namespace, config: DictConfig):
    os.makedirs(config.output_dir, exist_ok=True)
        
    # Init Optuna study
    study, param_specs = init_study(config)
    
    # Setup MLFlow tracking
    logger.info(f"Running sweep: {config.experiment}/{config.sweep_name}")
    mlflow.set_tracking_uri(config.mlflow_storage)
    mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    mlflow.set_experiment(config.experiment)
    parent_run = start_mlflow_parent_run(mlflow_client, config, study.study_name)
    
    dict_config = OmegaConf.to_container(config, throw_on_missing=True)
    mlflow.log_params(dict_config)
    
    # Run the Optuna study
    run_fn = partial(
        run_experiment,
        config = config,
        param_specs = param_specs,
        mlflow_client = mlflow_client,
    )
    study.optimize(run_fn, n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    # TODO: Don't mark run as complete if there are more trials remaining.
    #       Check if all combinations have successful completions, otherwise pass different status arguments to end_run.
    mlflow.end_run()
    logger.info(f"Sweep {config.sweep_name} completed.")


# Format of MLFlow runs and Optuna studies that I can use to query for deletion:
# - Parent run:
#   - `experiment={config.experiment}`
#   - `run_name={config.sweep_name}`
#   - `tags.optuna_study_name={config.experiment}/{config.sweep_name}`
#   - `tags.sweep_name={config.sweep_name}`
# - Child runs:
#   - `experiment={config.experiment}`
#   - `tags.sweep_name={config.sweep_name}`
# - Optuna study name: `{config.experiment}/{config.sweep_name}`

def delete_sweep(config: DictConfig):
    # First delete the Optuna study
    study_name = get_study_name(config)
    try:
        optuna.delete_study(study_name=study_name, storage=config.optuna_storage)
        logger.info(f"Deleted Optuna study: {study_name}.")
    except KeyError:
        logger.warning(f"Could not find Optuna study: {study_name}.")
    
    # Then delete the MLFlow runs
    mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    experiment_id = mlflow_client.get_experiment_by_name(config.experiment).experiment_id
    runs = mlflow_client.search_runs(
        experiment_ids = [experiment_id],
        filter_string = f'tags.sweep_name = "{config.sweep_name}"',
    )
    
    for run in runs:
        mlflow_client.delete_run(run.info.run_id)
    logger.info(f"Deleted {len(runs)} associated MLFlow runs.")


def configure_logging():
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main():
    configure_logging()
    
    # Parse args
    args = parse_args()
    
    # Load configs
    configs = []
    for config_path in args.config:
        config = OmegaConf.load(config_path)
        configs.append(config)
    
    # If delete flag is set, delete the sweeps then exit
    if args.delete:
        for config in configs:
            logger.info(f"Deleting sweep: {config.experiment}/{config.sweep_name}...")
            delete_sweep(config)
        return
    
    # Validate configs
    for config in configs:
        logger.info(f"Validating config: {config_path}")
        validate_config(config)

    # Run sweeps
    for config in configs:
        run_sweep(args, config)


if __name__ == '__main__':
    main()