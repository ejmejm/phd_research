import logging

import equinox as eqx
import jax.numpy as jnp
from omegaconf import DictConfig

from phd.jax_core.models import MLP
from phd.jax_core.tasks.geoff import NonlinearGEOFFTask, CoreTransientBinaryTask
from phd.feature_search.jax_core.feature_recycling import CBPTracker


logger = logging.getLogger(__name__)


def _nonlinear_geoff_create_perfect_model(
    model: MLP,
    task: NonlinearGEOFFTask,
    cfg: DictConfig,
) -> MLP:
    """Return a new version of the model that has the same features as the target network in the task."""
    has_bias = cfg.model.get('use_bias', True)
    model_hidden_dim = model.layers[0].weight.shape[0]
    
    if has_bias:
        model_hidden_dim -= 1
        
    task_feature_weights = task.weights[0].T # -> (hidden_dim, in_features)
    if has_bias:
        input_dim = task_feature_weights.shape[1]
        task_feature_weights = jnp.concatenate(
            [jnp.zeros((1, input_dim)), task_feature_weights],
            axis = 0,
        )
        
    model_weights = model.layers[0].weight
    new_weights = jnp.concatenate(
        [task_feature_weights, model_weights[task_feature_weights.shape[0]:]],
        axis = 0,
    )
    assert new_weights.shape == model_weights.shape, \
        f"Bug in code: new weights shape ({new_weights.shape}) does not match model weights shape ({model_weights.shape})!"
    
    new_model = eqx.tree_at(
        lambda m: m.layers[0].weight,
        model,
        new_weights,
    )
    
    return new_model


def _core_transient_binary_create_perfect_model(
    model: MLP,
    task: CoreTransientBinaryTask,
    cfg: DictConfig,
) -> MLP:
    """Return a new version of the model that has the same features as the target network in the task."""
    if len(model.layers) < task.n_core_layers + 1:
        logger.warning(
            "Model does not have enough layers to represent the core features of the task."
            "Only part of the representations of the core layers will be used."
        )
    
    if cfg.model.get('use_bias', True):
        raise NotImplementedError("Bias is not supported for core transient binary task!")
    
    for layer_idx in range(min(task.n_core_layers, len(model.layers) - 1)):
        task_feature_weights = task.weights[layer_idx].T # -> (hidden_dim, in_features/hidden_dim)
            
        model_weights = model.layers[layer_idx].weight
        
        new_weights = jnp.zeros_like(model_weights)
        new_weights = new_weights.at[:task_feature_weights.shape[0], :task_feature_weights.shape[1]].set(task_feature_weights)
        
        assert new_weights.shape == model_weights.shape, \
            f"Bug in code: new weights shape ({new_weights.shape}) does not match model weights shape ({model_weights.shape})!"

        model = eqx.tree_at(
            lambda m: m.layers[layer_idx].weight,
            model,
            new_weights,
        )
    
    return model


def create_model_with_perfect_features(
    model: MLP,
    task: NonlinearGEOFFTask | CoreTransientBinaryTask,
    cfg: DictConfig,
) -> MLP:
    """Return a new version of the model that has the same features as the target network in the task."""
    
    if isinstance(task, CoreTransientBinaryTask):
        return _core_transient_binary_create_perfect_model(model, task, cfg)
    elif isinstance(task, NonlinearGEOFFTask):
        return _nonlinear_geoff_create_perfect_model(model, task, cfg)
    else:
        raise ValueError(f"Unsupported task type: {type(task)}")


def _nonlinear_geoff_make_perfect_features_irreplacable(
    cbp_tracker: CBPTracker,
    task: NonlinearGEOFFTask,
    cfg: DictConfig,
) -> CBPTracker:
    """Make the perfect features irreplacable by setting their utility to infinity."""
    use_bias = cfg.model.get('use_bias', True)
    n_task_features = task.n_features + int(use_bias)
    utilities = cbp_tracker.all_feature_stats[0].utility
    cbp_tracker = eqx.tree_at(
        lambda x: x.all_feature_stats[0].utility,
        cbp_tracker,
        utilities.at[:n_task_features].set(jnp.inf),
    )
    return cbp_tracker


def _core_transient_binary_make_perfect_features_irreplacable(
    cbp_tracker: CBPTracker,
    task: CoreTransientBinaryTask,
    cfg: DictConfig,
) -> CBPTracker:
    """Make the perfect features irreplacable by setting their utility to infinity."""
    use_bias = cfg.model.get('use_bias', True)
    n_task_features = task.core_hidden_dim + int(use_bias)
    for layer_idx in range(min(task.n_core_layers, len(cbp_tracker.all_feature_stats))):
        utilities = cbp_tracker.all_feature_stats[layer_idx].utility
        cbp_tracker = eqx.tree_at(
            lambda x: x.all_feature_stats[layer_idx].utility,
            cbp_tracker,
            utilities.at[:n_task_features].set(jnp.inf),
        )
        return cbp_tracker


def make_perfect_features_irreplacable(
    cbp_tracker: CBPTracker,
    task: NonlinearGEOFFTask | CoreTransientBinaryTask,
    cfg: DictConfig,
) -> CBPTracker:
    """Make the perfect features irreplacable by setting their utility to infinity."""
    if (cfg.model.hidden_dim != cfg.task.hidden_dim and
        cfg.model.hidden_dim <= 2 *cfg.task.hidden_dim and
        cfg.feature_recycling.utility_reset_mode.lower() == 'median' and
        cfg.feature_recycling.recycle_rate > 0.0
    ):
        raise ValueError(
            f"Setting perfect features to be frozen but less than 2x the task hidden dim is not supported when utility reset mode is median! "
            f"This would cause the median utility to be infinity, and no features would be prunable."
        )
    
    if isinstance(task, CoreTransientBinaryTask):
        return _core_transient_binary_make_perfect_features_irreplacable(cbp_tracker, task, cfg)
    elif isinstance(task, NonlinearGEOFFTask):
        return _nonlinear_geoff_make_perfect_features_irreplacable(cbp_tracker, task, cfg)
    else:
        raise ValueError(f"Unsupported task type: {type(task)}")


def validate_config(cfg: DictConfig):
    assert cfg.task.name != 'nonlinear_geoff' or cfg.model.n_layers == 2, \
        "Only 2-layer models are supported for nonlinear GEOFF task!"
    
    if cfg.model.hidden_dim != cfg.task.hidden_dim:
        logger.warning(
            f"Model hidden dim ({cfg.model.hidden_dim}) does not match task hidden dim ({cfg.task.hidden_dim})! "
            f"This means that the learner will have features in addition to the perfect features."
        )
        
    if cfg.model.n_frozen_layers == 0:
        logger.warning(
            f"Model has {cfg.model.n_frozen_layers} frozen layers, so perfect features may be changed!")
    
    if cfg.feature_recycling.recycle_rate != 0.0:
        logger.warning(f"Recycle rate is {cfg.feature_recycling.recycle_rate}, but it must be 0 if you want to maintain perfect features.")

