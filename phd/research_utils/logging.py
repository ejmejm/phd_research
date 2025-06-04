from typing import Dict, Any, List, Optional, Union
import io
import logging
import os
import random
import string
import sys
import tempfile
import warnings

import numpy as np
import omegaconf
from omegaconf import DictConfig


COMET_META_PARAMS = ['config-name', 'sweep_command', 'sweep_name', 'sweep_project']


wandb = None
comet_ml = None
experiment_module_name: Optional[str] = None


def update_params(config: DictConfig) -> None:
    """Update experiment parameters in the active logging framework.
    
    Args:
        config: Hydra configuration object containing experiment parameters.
    """
    if config.wandb:
        wandb.config.update(config, allow_val_change=True)
    elif config.comet_ml:
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


def init_experiment(project: str, config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Initialize experiment logging with W&B or Comet ML.
    
    Args:
        project: Name of the project for experiment tracking.
        config: Hydra configuration object containing experiment parameters.
        
    Returns:
        Updated configuration object, or None if no config provided.
    """
    global wandb, comet_ml, experiment_module_name
    
    if config and config.wandb:
        import wandb
    
        raw_dict_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        wandb.init(
            project=project, config=raw_dict_config, tags=config.get('tags', None),
            settings=wandb.Settings(start_method='thread'), allow_val_change=True,
        )
        
        # config = wandb.config # TODO: Make sure this is still a DictConfig and has the right values

    comet_sweep_id = get_comet_sweep_id()
    if comet_sweep_id or (config and config.comet_ml):
        # import os
        # os.environ['COMET_LOGGING_FILE_LEVEL'] = 'DEBUG'
        # os.environ['COMET_LOGGING_FILE'] = './comet.log'

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
            logger.log(f'CometML workspace not specified, using retrieved default: {workspace}')
                
        if comet_sweep_id:
            opt = comet_ml.Optimizer(comet_sweep_id, verbose=0)
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
            experiment = comet_ml.Experiment(project_name=project, workspace=workspace)
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


def import_logger(config: DictConfig) -> None:
    """Import the appropriate logging framework based on configuration.
    
    Args:
        config: Hydra configuration object specifying which logger to use.
    """
    global wandb, comet_ml
    
    if config.wandb:
        import wandb
    elif config.comet_ml:
        import comet_ml


def log_metrics(metrics: Dict[str, Union[int, float]], config: DictConfig, 
                prefix: Optional[str] = None, step: Optional[int] = None) -> None:
    """Log metrics to the active experiment tracking framework.
    
    Args:
        metrics: Dictionary of metric names and values to log.
        config: Hydra configuration object specifying logging framework.
        prefix: Optional prefix to prepend to metric names.
        step: Optional step number for the metrics.
    """
    if config.wandb:
        if step is not None:
            metrics['step'] = step
        prefix = prefix + '/' if prefix else ''
        wandb.log({f'{prefix}{k}': v for k, v in metrics.items()}) #, step=step)
    
    if config.comet_ml:
        if 'step' in metrics:
            if step is None:
                step = metrics['step']
            else:
                assert metrics['step'] == step, 'Step mismatch in metrics and explicit step argument!'
        
        experiment = comet_ml.get_global_experiment()
        experiment.log_metrics(metrics, prefix=prefix, step=step)


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
    if config.wandb:
        formatted_imgs = {f'{prefix}{k}': [wandb.Image(img) for img in v] \
            for k, v in images.items()}
        wandb.log(formatted_imgs) #, step=step)
    if config.comet_ml:
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
    if config.wandb:
        wandb.log({f'{prefix}{k}': [wandb.Image(figure) for figure in v] \
            for k, v in figures.items()}) #, step=step)
    if config.comet_ml:
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
    if config.wandb:
        formatted_vids = {
            f'{prefix}{k}': [wandb.Video(frames, fps=4, format='gif') for frames in v] \
            for k, v in videos.items()
        }
        wandb.log(formatted_vids) #, step=step)
    if config.comet_ml:
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
    if config.wandb:
        wandb.finish()
    if config.comet_ml:
        experiment = comet_ml.get_global_experiment()
        experiment.end()


def track_model(model: Any, config: DictConfig) -> None:
    """Track a model with the active experiment tracking framework.
    
    Args:
        model: The model to track (typically a PyTorch or TensorFlow model).
        config: Hydra configuration object specifying logging framework.
    """
    if config.wandb:
        wandb.watch(model)


def log_np_array(arr: np.ndarray, name: str, config: DictConfig) -> None:
    """Log a numpy array as an artifact to the active experiment tracking framework.
    
    Args:
        arr: The numpy array to log.
        name: Name for the array artifact.
        config: Hydra configuration object specifying logging framework.
    """
    if config.wandb:
        artifact = wandb.Artifact(name, type='np_array')
        # Generate a temporary file to write the array to
        with tempfile.NamedTemporaryFile() as f:
            np.save(f, arr)
            artifact.add_file(f.name)
            wandb.log_artifact(artifact)

    if config.comet_ml:
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