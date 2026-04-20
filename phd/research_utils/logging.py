from typing import Dict, Any, List, Optional, Union
import io
import logging
import os
import random
import string
import sys
import tempfile
from typing import Tuple

import numpy as np
import omegaconf
from omegaconf import DictConfig


COMET_META_PARAMS = ['config-name', 'sweep_command', 'sweep_name', 'sweep_project']
MLFLOW_DEFAULT_TRACKING_URI = 'sqlite:///mlruns.db'


wandb = None
comet_ml = None
mlflow = None
experiment_module_name: Optional[str] = None
_mlflow_child_run_ids: List[str] = []
_mlflow_run_id: Optional[str] = None
_mlflow_client = None


logger = logging.getLogger(__name__)


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Add non-dict items
            items.append((new_key, v))
    return dict(items)


def init_experiment(project: str, config: Optional[DictConfig],
                    run_name: Optional[str] = None) -> Optional[DictConfig]:
    """Initialize experiment logging with W&B or Comet ML.

    Args:
        project: Name of the project for experiment tracking.
        config: Hydra configuration object containing experiment parameters.
        run_name: Optional name for the run (MLflow only).

    Returns:
        Updated configuration object, or None if no config provided.
    """
    global wandb, comet_ml, mlflow, experiment_module_name
    global _mlflow_run_id, _mlflow_client

    if config and config.get('mlflow', False):
        import mlflow

        if os.environ.get('MLFLOW_RUN_ID') and os.environ.get('MLFLOW_TRACKING_URI'):
            # When both are passed this is a sweep, so use the environment variable
            # provided by the sweep script.
            mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
        elif config.get('mlflow_tracking_uri'):
            if os.environ.get('MLFLOW_TRACKING_URI') is not None:
                logger.warning(
                    f"MLFLOW_TRACKING_URI is set in both the environment and the config, "
                    f"using the config value instead of the environment value"
                )
            mlflow.set_tracking_uri(config.get('mlflow_tracking_uri'))
        elif os.environ.get('MLFLOW_TRACKING_URI'):
            mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
        else:
            mlflow.set_tracking_uri(config.get('mlflow_tracking_uri', MLFLOW_DEFAULT_TRACKING_URI))

        print(f'MLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}')

        if os.environ.get('MLFLOW_RUN_ID') is None:
            mlflow.set_experiment(project)
        mlflow.start_run(run_name=run_name)

        # Cache run ID and client for thread-safe logging (mlflow.log_metrics
        # uses thread-local active run which breaks from background threads)
        _mlflow_run_id = mlflow.active_run().info.run_id
        _mlflow_client = mlflow.tracking.MlflowClient()

        raw_dict_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        flat_config = flatten_dict(raw_dict_config)
        mlflow.log_params(flat_config)
        mlflow.set_tag('mlflow.note.content', f'```\npython {" ".join(sys.argv)}\n```')
    
    if config and config.get('wandb', False):
        import wandb
    
        raw_dict_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        wandb.init(
            project=project, config=raw_dict_config, tags=config.get('tags', None),
            settings=wandb.Settings(start_method='thread'), allow_val_change=True,
        )
        
        # config = wandb.config # TODO: Make sure this is still a DictConfig and has the right values

    comet_sweep_id = get_comet_sweep_id()
    if comet_sweep_id or (config and config.get('comet_ml', False)):
        import comet_ml

        # Used for capturing log output
        if experiment_module_name is None:
            from comet_ml import experiment
            experiment_module_name = experiment.__name__

        log_capture_string = io.StringIO()
        stream_handler = logging.StreamHandler(log_capture_string)
        stream_handler.setLevel(logging.ERROR)
        logger = logging.getLogger(experiment_module_name)
        logger.addHandler(stream_handler)
            
        api = comet_ml.api.API()
        workspace = config.get('comet_ml_workspace', None)
        if workspace is None:
            workspace = api.get_default_workspace()
            logger.info(f'CometML workspace not specified, using retrieved default: {workspace}')
        
        # Determine whether experiments should be run online or offline
        experiment_cls = comet_ml.Experiment
        if os.environ.get('COMET_MODE', 'online') == 'offline':
            experiment_cls = comet_ml.OfflineExperiment
            logger.info('\n========== Running Comet in offline mode! ==========\n')

        if comet_sweep_id:
            opt = comet_ml.Optimizer(comet_sweep_id, verbose=0, experiment_class=experiment_cls)
            project = opt.status()['parameters'].get('sweep_project')
            if project is not None:
                project = project['values'][0]
            
            experiment = opt.next(project_name=project, workspace=workspace)
            
            error_log = log_capture_string.getvalue()

            if 'run will not be logged' in error_log.lower():
                print('Error captured in experiment setup!')
                if 'was already uploaded' in error_log.lower():
                    print('Creating an `ExistingExperiment` after error')
                    new_experiment = comet_ml.ExistingExperiment(
                        project_name=project, workspace=workspace,
                        experiment_key=experiment.get_key())
                else:
                    print('Creating an `OfflineExperiment` after error')
                    new_experiment = comet_ml.OfflineExperiment(
                        project_name=project, workspace=workspace)

                # Get parameters from original experiment
                api_exp = api.get_experiment_by_id(experiment.id)
                param_summary = api_exp.get_parameters_summary()
                params = {x['name']: x['valueCurrent'] for x in param_summary}
                new_experiment.params = params
                experiment = new_experiment

            comet_ml.config.set_global_experiment(experiment)
            if experiment is None:
                print('No more experiments to run in sweep!')
                sys.exit(0)
            
            sweep_overrides_list = [f'{k}={v}' for k, v in experiment.params.items()]
            sweep_overrides = omegaconf.OmegaConf.from_dotlist(sweep_overrides_list)
            
            # Add fields for meta parameters used for running comet sweeps
            config_dict = omegaconf.OmegaConf.to_container(config)
            for param in COMET_META_PARAMS:
                if param not in config_dict:
                    config_dict[param] = ''
            config = omegaconf.OmegaConf.create(config_dict)
            
            # Combine new sweep-sepecific overrides with base config
            config = omegaconf.OmegaConf.unsafe_merge(config, sweep_overrides)
            
            # Increment the seed for each trial
            n_trials = opt.status().get('trials', 1)
            seed = config.get('seed', None)
            if seed is not None and n_trials > 1:
                config.seed = seed + n_trials - 1
            
            # Log the parameters to the experiment
            raw_dict_config = omegaconf.OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True)
            
            # comet_config = process_args(comet_config)
            experiment.log_parameters(raw_dict_config)

            # Pretty print chosen args for sweep
            print('Sweep args:')
            for k, v in experiment.params.items():
                print(f'    {k}: {v}')

            if config.get('tags', None) is not None:
                for tag in config.tags:
                    experiment.add_tag(tag)

        else:
            experiment = experiment_cls(project_name=project, workspace=workspace)
            error_log = log_capture_string.getvalue()

            if 'run will not be logged' in error_log.lower():
                print('Error captured in experiment setup!')
                if 'was already uploaded' in error_log.lower():
                    print('Creating an `ExistingExperiment` after error')
                    experiment = comet_ml.ExistingExperiment(
                        project_name=project, workspace=workspace)
                else:
                    print('Creating an `OfflineExperiment` after error')
                    experiment = comet_ml.OfflineExperiment(
                        project_name=project, workspace=workspace)

            comet_ml.config.set_global_experiment(experiment)
            raw_dict_config = omegaconf.OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True)
            experiment.log_parameters(raw_dict_config)
            if config.get('tags', None) is not None:
                for tag in config.tags:
                    experiment.add_tag(tag)

        log_capture_string.close()
        
    return config


def init_child_runs(seeds: list, config: DictConfig) -> List[str]:
    """Create child MLflow runs for each seed in a multi-seed experiment.

    Also tags the parent run as 'multi_seed' or 'single_seed'.

    Args:
        seeds: List of random seeds.
        config: Hydra configuration object.

    Returns:
        List of child run IDs (empty if not using MLflow, single seed,
        or log_individual_seeds is disabled).
    """
    global _mlflow_child_run_ids
    _mlflow_child_run_ids = []

    if not config.get('mlflow', False):
        return []

    is_multi_seed = len(seeds) > 1

    # Tag parent run
    mlflow.set_tag('seed_mode', 'multi_seed' if is_multi_seed else 'single_seed')

    if not is_multi_seed or not config.get('log_individual_seeds', True):
        return []

    from mlflow.entities import Param

    parent_run = mlflow.active_run()
    client = mlflow.tracking.MlflowClient()

    raw_dict_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True)
    flat_config = flatten_dict(raw_dict_config)

    for seed in seeds:
        child_run = client.create_run(
            experiment_id=parent_run.info.experiment_id,
            tags={
                'mlflow.parentRunId': parent_run.info.run_id,
                'mlflow.runName': f'seed_{seed}',
                'seed_mode': 'single_seed',
            },
        )
        params = [Param(str(k), str(v)) for k, v in flat_config.items()]
        params.append(Param('child_seed', str(seed)))
        for i in range(0, len(params), 100):
            client.log_batch(child_run.info.run_id, params=params[i:i+100])

        _mlflow_child_run_ids.append(child_run.info.run_id)

    return _mlflow_child_run_ids


def update_params(config: DictConfig) -> None:
    """Update experiment parameters in the active logging framework.
    
    Args:
        config: Hydra configuration object containing experiment parameters.
    """
    if config.get('mlflow', False):
        raw_dict_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        mlflow.log_params(raw_dict_config)
    if config.get('wandb', False):
        wandb.config.update(config, allow_val_change=True)
    elif config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()
        raw_dict_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        experiment.log_parameters(raw_dict_config)


def get_comet_sweep_id() -> Optional[str]:
    """Get Comet ML sweep ID from environment variables.
    
    Returns:
        Sweep ID if available, None otherwise.
    """
    return os.environ.get('COMET_OPTIMIZER_ID', None)


def import_logger(config: DictConfig) -> None:
    """Import the appropriate logging framework based on configuration.

    Args:
        config: Hydra configuration object specifying which logger to use.
    """
    global wandb, comet_ml, mlflow

    if config.get('mlflow', False):
        import mlflow
    if config.get('wandb', False):
        import wandb
    elif config.get('comet_ml', False):
        import comet_ml


def bind_to_active_run(config: DictConfig) -> None:
    """Populate module-level MLflow globals from the currently-active run.

    For sweep drivers (e.g. mlflow-sweeper) that start the trial's MLflow
    run themselves, callees that skip ``init_experiment`` still need the
    module-level ``mlflow`` / ``_mlflow_run_id`` / ``_mlflow_client``
    globals populated — otherwise ``log_metrics`` / ``init_child_runs``
    will fail on ``None.set_tag(...)`` / ``None.log_metric(...)``.
    Assumes ``mlflow.active_run()`` returns the trial run.
    """
    global mlflow, _mlflow_run_id, _mlflow_client
    if not config.get('mlflow', False):
        return
    import mlflow as _mlflow
    mlflow = _mlflow
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            'bind_to_active_run() called but no MLflow run is active. '
            'Either call init_experiment() first, or start a run before '
            'invoking run_config().'
        )
    _mlflow_run_id = active.info.run_id
    _mlflow_client = mlflow.tracking.MlflowClient()


def log_metrics(metrics: Dict[str, Union[int, float]], config: DictConfig, 
                prefix: Optional[str] = None, step: Optional[int] = None) -> None:
    """Log metrics to the active experiment tracking framework.
    
    Args:
        metrics: Dictionary of metric names and values to log.
        config: Hydra configuration object specifying logging framework.
        prefix: Optional prefix to prepend to metric names.
        step: Optional step number for the metrics.
    """
    if config.get('mlflow', False):
        if 'step' in metrics:
            if step is None:
                step = metrics['step']
            else:
                assert metrics['step'] == step, 'Step mismatch in metrics and explicit step argument!'
        prefix = prefix + '/' if prefix else ''
        prepped_metrics = {}
        for k, v in metrics.items():
            if k == 'step':
                continue
            if isinstance(v, np.ndarray):
                v = v.tolist()
            prepped_metrics[f'{prefix}{k}'] = v
        step = int(step) if step is not None else None
        # Use cached client + run ID for thread safety
        for k, v in prepped_metrics.items():
            _mlflow_client.log_metric(_mlflow_run_id, k, v, step=step)
    
    if config.get('wandb', False):
        if step is not None:
            metrics['step'] = step
        prefix = prefix + '/' if prefix else ''
        wandb.log({f'{prefix}{k}': v for k, v in metrics.items()})
    
    if config.get('comet_ml', False):
        if 'step' in metrics:
            if step is None:
                step = metrics['step']
            else:
                assert metrics['step'] == step, 'Step mismatch in metrics and explicit step argument!'
        
        experiment = comet_ml.get_global_experiment()
        experiment.log_metrics(metrics, prefix=prefix, step=step)


def log_child_metrics(per_seed_metrics: Dict[str, list], config: DictConfig,
                      step: Optional[int] = None) -> None:
    """Log per-seed metrics to child MLflow runs.

    Args:
        per_seed_metrics: Dict mapping metric names to lists of values (one per seed).
        config: Hydra configuration object.
        step: Optional step number.
    """
    if not config.get('mlflow', False) or not _mlflow_child_run_ids:
        return

    import time
    from mlflow.entities import Metric

    client = mlflow.tracking.MlflowClient()
    timestamp = int(time.time() * 1000)
    step = int(step) if step is not None else 0

    for i, run_id in enumerate(_mlflow_child_run_ids):
        metrics = [
            Metric(k, float(v[i]), timestamp, step)
            for k, v in per_seed_metrics.items()
        ]
        client.log_batch(run_id, metrics=metrics)


def log_images(images: Dict[str, List[np.ndarray]], config: DictConfig, 
               prefix: Optional[str] = None, step: Optional[int] = None) -> None:
    """Log images to the active experiment tracking framework.
    
    Args:
        images: Dictionary of image names and lists of image arrays.
        config: Hydra configuration object specifying logging framework.
        prefix: Optional prefix to prepend to image names.
        step: Optional step number for the images.
    """
    prefix = prefix + '/' if prefix else ''
    if config.get('mlflow', False):
        for name, image in images.items():
            mlflow.log_image(image, key=f'{prefix}{name}', step=step)
    if config.get('wandb', False):
        formatted_imgs = {f'{prefix}{k}': [wandb.Image(img) for img in v] \
            for k, v in images.items()}
        wandb.log(formatted_imgs) #, step=step)
    if config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()
        for k, v in images.items():
            for image in v:
                experiment.log_image(image, name=f'{prefix}{k}', step=step)


def log_figures(figures: Dict[str, List[Any]], config: DictConfig, 
                prefix: Optional[str] = None, step: Optional[int] = None) -> None:
    """Log matplotlib figures to the active experiment tracking framework.
    
    Args:
        figures: Dictionary of figure names and lists of matplotlib figures.
        config: Hydra configuration object specifying logging framework.
        prefix: Optional prefix to prepend to figure names.
        step: Optional step number for the figures.
    """
    prefix = prefix + '/' if prefix else ''
    if config.get('mlflow', False):
        for name, figure in figures.items():
            mlflow.log_figure(figure, artifact_file=f'{prefix}{name}.png')
    if config.get('wandb', False):
        wandb.log({f'{prefix}{k}': [wandb.Image(figure) for figure in v] \
            for k, v in figures.items()}) #, step=step)
    if config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()
        for k, v in figures.items():
            for figure in v:
                experiment.log_figure(f'{prefix}{k}', figure, step=step)


def log_videos(videos: Dict[str, List[List[np.ndarray]]], config: DictConfig, 
               prefix: Optional[str] = None, step: Optional[int] = None) -> None:
    """Log videos to the active experiment tracking framework.
    
    Args:
        videos: Dictionary of video names and lists of frame sequences.
        config: Hydra configuration object specifying logging framework.
        prefix: Optional prefix to prepend to video names.
        step: Optional step number for the videos.
    """
    prefix = prefix + '/' if prefix else ''
    if config.get('mlflow', False):
        raise NotImplementedError('Logging videos to MLflow is not supported.')
    if config.get('wandb', False):
        formatted_vids = {
            f'{prefix}{k}': [wandb.Video(frames, fps=4, format='gif') for frames in v] \
            for k, v in videos.items()
        }
        wandb.log(formatted_vids) #, step=step)
    if config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()
        for k, v in videos.items():
            for frames in v:
                image = np.concatenate(frames, axis=1)
                experiment.log_image(
                    image, name=f'{prefix}{k}', step=step, image_channels='first')


def finish_experiment(config: DictConfig) -> None:
    """Finish and clean up the active experiment.
    
    Args:
        config: Hydra configuration object specifying logging framework.
    """
    global _mlflow_run_id, _mlflow_client
    if config.get('mlflow', False):
        mlflow.end_run()
        _mlflow_run_id = None
        _mlflow_client = None
    if config.get('wandb', False):
        wandb.finish()
    if config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()
        experiment.end()


def finish_child_runs(config: DictConfig) -> None:
    """End child MLflow runs."""
    global _mlflow_child_run_ids
    if not config.get('mlflow', False) or not _mlflow_child_run_ids:
        return

    client = mlflow.tracking.MlflowClient()
    for run_id in _mlflow_child_run_ids:
        client.set_terminated(run_id)
    _mlflow_child_run_ids = []


def track_model(model: Any, config: DictConfig) -> None:
    """Track a model with the active experiment tracking framework.
    
    Args:
        model: The model to track (typically a PyTorch or TensorFlow model).
        config: Hydra configuration object specifying logging framework.
    """
    if config.get('mlflow', False):
        raise NotImplementedError('Tracking models with MLflow is not supported.')
    if config.get('wandb', False):
        wandb.watch(model)
    if config.get('comet_ml', False):
        raise NotImplementedError('Tracking models with Comet ML is not supported.')


def log_np_array(arr: np.ndarray, name: str, config: DictConfig) -> None:
    """Log a numpy array as an artifact to the active experiment tracking framework.
    
    Args:
        arr: The numpy array to log.
        name: Name for the array artifact.
        config: Hydra configuration object specifying logging framework.
    """
    if config.get('mlflow', False):
        with tempfile.NamedTemporaryFile() as f:
            np.save(f, arr)
            mlflow.log_artifact(f.name, artifact_path=f'{name}.npy')
    
    if config.get('wandb', False):
        artifact = wandb.Artifact(name, type='np_array')
        # Generate a temporary file to write the array to
        with tempfile.NamedTemporaryFile() as f:
            np.save(f, arr)
            artifact.add_file(f.name)
            wandb.log_artifact(artifact)

    if config.get('comet_ml', False):
        experiment = comet_ml.get_global_experiment()

        # Generate a random string to use as the temporary file name
        tmp_file_name = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        tmp_file_name = f'{tmp_file_name}.npy'

        # Save the array to the temporary file
        with open(tmp_file_name, 'wb') as f:
            np.save(f, arr)

        # Print size of the file in KB
        print(f'\nSize of {name}: {os.path.getsize(tmp_file_name) / 1000:.2f} KB')

        def success_callback(*args: Any, **kwargs: Any) -> None:
            print(f'\nSuccessfully uploaded {name}')
            print('args:', args)
            print('kwargs:', kwargs)
            # Delete the temporary file
            os.remove(tmp_file_name)

        def failure_callback(*args: Any, **kwargs: Any) -> None:
            print(f'\nFailed to upload {name}')
            print('args:', args)
            print('kwargs:', kwargs)
            # Delete the temporary file
            os.remove(tmp_file_name)

        experiment._log_asset(
            tmp_file_name, file_name=f'{name}.npy',
            on_asset_upload=success_callback,
            on_failed_asset_upload=failure_callback)