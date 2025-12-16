from functools import partial
import logging
from typing import Tuple, Callable, Optional

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
import optax
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import standardize_targets
from phd.feature_search.jax_core.feature_recycling import CBPTracker
from phd.feature_search.jax_core.metrics import *
from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.optimizers import EqxOptimizer
from phd.feature_search.jax_core.tasks.geoff import NonlinearGEOFFTask
from phd.feature_search.jax_core.utils import tree_replace
from phd.research_utils.logging import *

from phd.feature_search.scripts.jax_full_feature_search import (
    prepare_ltu_geoff_experiment,
    TrainState as OriginalTrainState,
    StepStats,
    MetricsBuffer,
    compute_baseline_loss,
    compute_metrics,
    log_task_output_weights,
    train_multi_step,
)


TRAIN_LOOP_UNROLL = 2


logger = logging.getLogger(__name__)


class TrainState(OriginalTrainState):
    learning_feature_idx: Int[Array, '']
    learning_layer_idx: Int[Array, '']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_feature_idx = jnp.array(0)
        self.learning_layer_idx = jnp.array(0)


def filter_grads_by_feature_and_layer(grads, train_state):
    """Filter gradients by which feature and layer are currently being learned."""
    assert len(grads.layers) == 2, "Only 2-layer models are supported!"
    
    layer_switch_interval = train_state.cfg.train.get('learning_layer_switch_interval', -1)
    feature_switch_interval = train_state.cfg.train.get('learning_feature_switch_interval', -1)
    
    learning_layer_idx = train_state.learning_layer_idx
    learning_feature_idx = train_state.learning_feature_idx
    
    # If no filtering is needed, return gradients as-is
    if feature_switch_interval == -1 and layer_switch_interval == -1:
        return grads
    
    # Apply layer mask if specified using numerical masking
    if layer_switch_interval != -1:
        # Create masks for each layer based on whether it's the learning layer
        def mask_layer(layer_idx, layer):
            is_learning_layer = (layer_idx == learning_layer_idx)
            return jax.tree.map(
                lambda g: jnp.where(is_learning_layer, g, jnp.zeros_like(g)),
                layer,
            )
        
        masked_layers = [mask_layer(i, layer) for i, layer in enumerate(grads.layers)]
        grads = eqx.tree_at(lambda g: g.layers, grads, masked_layers)
    
    # Apply feature mask if specified using numerical masking
    if feature_switch_interval != -1:
        # Mask layer 0 weights: only keep row at learning_feature_idx
        layer0_weight = grads.layers[0].weight
        feature_mask_0 = (jnp.arange(layer0_weight.shape[0]) == learning_feature_idx)[:, None]
        masked_weight_0 = jnp.where(feature_mask_0, layer0_weight, jnp.zeros_like(layer0_weight))
        
        # Mask layer 1 weights: only keep column at learning_feature_idx
        layer1_weight = grads.layers[1].weight
        feature_mask_1 = (jnp.arange(layer1_weight.shape[1]) == learning_feature_idx)[None, :]
        masked_weight_1 = jnp.where(feature_mask_1, layer1_weight, jnp.zeros_like(layer1_weight))
        
        # Update the grads with masked weights
        grads = eqx.tree_at(lambda g: g.layers[0].weight, grads, masked_weight_0)
        grads = eqx.tree_at(lambda g: g.layers[1].weight, grads, masked_weight_1)
    
    return grads


def maybe_update_learning_indices(
    train_state: TrainState,
    feature_idx_key: PRNGKeyArray,
) -> TrainState:
    """Update the learning feature and layer indices based on configuration intervals."""
    cfg = train_state.cfg
    feature_switch_interval = cfg.train.get('learning_feature_switch_interval', -1)
    layer_switch_interval = cfg.train.get('learning_layer_switch_interval', -1)

    updated_state = train_state

    if feature_switch_interval > 0:
        n_features = train_state.model.layers[0].weight.shape[0]
        new_feature_idx = jax.random.randint(feature_idx_key, (1,), 0, n_features)[0]
        new_feature_idx = jnp.where(
            train_state.step % feature_switch_interval == 0,
            new_feature_idx,
            train_state.learning_feature_idx,
        )
        updated_state = tree_replace(updated_state, learning_feature_idx=new_feature_idx)

    if layer_switch_interval > 0:
        new_layer_idx = 1 - train_state.learning_layer_idx
        new_layer_idx = jnp.where(
            train_state.step % layer_switch_interval == 0,
            new_layer_idx,
            train_state.learning_layer_idx,
        )
        updated_state = tree_replace(updated_state, learning_layer_idx=new_layer_idx)

    return updated_state


def log_stuff(train_state, grads):
    jax.debug.print('\n==================================')
    
    jax.debug.print('Feature idx: {x} | Layer idx: {y}\n', x=train_state.learning_feature_idx, y=train_state.learning_layer_idx, ordered=True)
    
    jax.debug.print('Output grads: {x}', x=grads.layers[1].weight.squeeze(0), ordered=True)
    jax.debug.print('Input grads: {x}\n', x=jnp.abs(grads.layers[0].weight).mean(axis=1), ordered=True)
    
    jax.debug.print('Output params: {x}', x=train_state.model.layers[1].weight.squeeze(0), ordered=True)
    jax.debug.print('Input params: {x}\n', x=jnp.abs(train_state.model.layers[0].weight).mean(axis=1), ordered=True)
    
    optim_state = train_state.optimizer.state
    
    jax.debug.print('Output β: {x}', x=optim_state.beta.layers[1].weight.squeeze(0), ordered=True)
    jax.debug.print('Input β: {x}\n', x=jnp.abs(optim_state.beta.layers[0].weight).mean(axis=1), ordered=True)
    
    jax.debug.print('Output h: {x}', x=optim_state.h.layers[1].weight.squeeze(0), ordered=True)
    jax.debug.print('Input h: {x}\n', x=jnp.abs(optim_state.h.layers[0].weight).mean(axis=1), ordered=True)
    
    jax.debug.print('Output v: {x}', x=optim_state.v.layers[1].weight.squeeze(0), ordered=True)
    jax.debug.print('Input v: {x}\n', x=jnp.abs(optim_state.v.layers[0].weight).mean(axis=1), ordered=True)


def train_step(
    train_state: TrainState,
    data: Tuple[Float[Array, 'batch_size n_inputs'], Float[Array, 'batch_size n_outputs']],
    task: NonlinearGEOFFTask | None = None,
    do_prune: bool = False,
) -> Tuple[TrainState, StepStats]:
    inputs, targets = data
    cfg, model, optimizer, repr_optimizer = \
        train_state.cfg, train_state.model, train_state.optimizer, train_state.repr_optimizer
    cbp_tracker, distractor_tracker = \
        train_state.cbp_tracker, train_state.distractor_tracker

    use_bias = cfg.model.get('use_bias', True)
    rng, noise_key, model_key, cbp_key, feature_idx_key  = jax.random.split(train_state.rng, 5)

    # Update learning feature and layer indices using the helper function
    train_state = maybe_update_learning_indices(
        train_state, feature_idx_key)
    
    # Add noise to targets
    if cfg.task.noise_std > 0:
        targets += jax.random.normal(noise_key, targets.shape) * cfg.task.noise_std
        
    standardized_targets, cumulant_stats = standardize_targets(targets, train_state.cumulant_stats)
    
    if cfg.train.standardize_cumulants:
        targets = standardized_targets

    def compute_loss(model, inputs, targets):
        outputs, param_inputs = jax.vmap(partial(
            model, set_first_element_to_one=use_bias, key=model_key))(inputs)
        loss = train_state.criterion(outputs, targets)
        return loss, (outputs, param_inputs)
    
    # Backward pass
    (loss, (outputs, param_inputs)), grads = jax.value_and_grad(
        compute_loss, has_aux=True)(model, inputs, targets)
    
    # Compute loss for a naive baseline prediction
    baseline_loss = compute_baseline_loss(cfg, train_state, targets)
    
    # If using IDBD we also need the prediction gradients
    if cfg.optimizer.name == 'idbd':
        output_grads = jax.grad(
            lambda m, x: jax.vmap(partial(
                m, set_first_element_to_one=use_bias, key=model_key
            ))(x)[0].mean(axis=0).sum())(model, inputs) # TODO: Divide the results of this by the batch size (or maybe mean instead of sum?)
        grads = filter_grads_by_feature_and_layer(grads, train_state)
        output_grads = filter_grads_by_feature_and_layer(output_grads, train_state)
        optimizer_grads = (grads, output_grads)
    else:
        grads = filter_grads_by_feature_and_layer(grads, train_state)
        optimizer_grads = grads
    updates, optimizer = optimizer.with_update(optimizer_grads, model)
    
    if repr_optimizer is not None:
        repr_updates, repr_optimizer = repr_optimizer.with_update(grads, model)
        updates = eqx.combine(updates, repr_updates)
    
    # TODO: Test with breakpoints to make sure only given feature and layer are updated.
    #       Check param updates and optimizer state changes.
    model = eqx.apply_updates(model, updates)
    if cfg.model.get('use_normalize_and_project', False):
        model = model.with_projected_weights()
    
    # CBP resets
    n_pruned = 0
    n_best_features_pruned = 0
    if train_state.cbp_tracker is not None:
        if do_prune:
            pre_prune_model = model
            if repr_optimizer is not None:
                cbp_tracker, model, (optimizer, repr_optimizer), prune_masks = train_state.cbp_tracker.prune_features(
                    model, param_inputs, (optimizer, repr_optimizer), rng=cbp_key)
            else:
                cbp_tracker, model, optimizer, prune_masks = train_state.cbp_tracker.prune_features(
                    model, param_inputs, optimizer, rng=cbp_key)
                
            if cfg.train.get('log_pruning_stats', False):
                assert task is not None, "Task is required for logging pruning stats!"
                assert len(prune_masks) == 1, "There should only be one prune mask!"
                prune_mask = prune_masks[0]
                n_pruned = prune_mask.sum()
                n_best_features_pruned = compute_n_best_features_pruned(pre_prune_model, prune_mask, task)
                
        else:
            cbp_tracker = train_state.cbp_tracker.update_feature_stats(model, param_inputs)
    
    # Update state
    train_state_updates = dict(
        model = model,
        optimizer = optimizer,
        repr_optimizer = repr_optimizer,
        cbp_tracker = cbp_tracker,
        distractor_tracker = distractor_tracker,
        step = train_state.step + 1,
        cumulant_stats = cumulant_stats,
        rng = rng,
    )
    train_state_updates = {k: v for k, v in train_state_updates.items() if v is not None}
    train_state = tree_replace(train_state, **train_state_updates)
    step_stats = StepStats(loss, targets, baseline_loss, n_pruned, n_best_features_pruned)
    
    # log_stuff(train_state, grads)
    
    return train_state, step_stats


def run_experiment(
        cfg: DictConfig,
        train_fn: Callable,
        metrics_fn: Callable,
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: Callable,
        optimizer: EqxOptimizer,
        repr_optimizer: Optional[EqxOptimizer],
        cbp_tracker, # : Optional[CBPTracker],
        distractor_tracker, # : DistractorTracker,
        rng: PRNGKeyArray,
        show_progress: bool = True,
    ) -> Tuple[TrainState, NonlinearGEOFFTask, Dict[str, Array]]:
    
    train_state = TrainState(
        model = model,
        optimizer = optimizer,
        repr_optimizer = repr_optimizer,
        cbp_tracker = cbp_tracker,
        distractor_tracker = distractor_tracker,
        cfg = cfg,
        criterion = criterion,
        rng = rng,
    )
    log_task_output_weights(task, cfg)
    metrics_buffer = MetricsBuffer()
    all_metrics = []
    
    sequence_length = cfg.train.log_freq
    train_cycles = cfg.train.total_steps // sequence_length
    
    # Warmup
    train_fn(train_state, task, sequence_length)

    if show_progress:
        pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    else:
        pbar = None
    
    # Training loop
    for _ in range(train_cycles):
        
        # Train
        train_state, task, step_stats = train_fn(train_state, task, sequence_length)
        
        # Metrics
        metrics_buffer, metrics = metrics_fn(train_state, task, metrics_buffer, step_stats, cfg)
        metrics = {k: np.asarray(v) for k, v in metrics.items()}
        all_metrics.append(metrics)
        log_metrics(metrics, cfg, step=train_state.step) # Consider making logging async
        
        if pbar is not None:
            pbar.set_postfix(loss=f"{metrics['loss']:.5f}")
            pbar.update(sequence_length)
        
    if pbar is not None:
        pbar.close()
    
    all_metrics = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics)
    return train_state, task, all_metrics


def validate_config(cfg: DictConfig):
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"


@hydra.main(config_path='../conf', config_name='full_feature_search', version_base='1.1')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_persistent_cache_enable_xla_caches', 'xla_gpu_per_fusion_autotune_cache_dir')
    
    jax.config.update('jax_platform_name', cfg.device)
    print(f"JAX is using device: {jax.devices(cfg.device)[0]}")
    
    cfg = init_experiment(cfg.project, cfg)
    validate_config(cfg)

    task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
        prepare_ltu_geoff_experiment(cfg)
    
    distractor_tracker = None
    
    train_step_fn = jax.jit(train_step, static_argnums=(3,))
    train_fn = jax.jit(
        partial(train_multi_step, train_step_fn=train_step_fn),
        static_argnames = ('n_steps', 'train_step_fn'),
    )
    metrics_fn = jax.jit(compute_metrics, static_argnums=(4,))
    
    # train_step_fn = train_step
    # train_fn = partial(train_multi_step, train_step_fn=train_step_fn)
    # metrics_fn = compute_metrics
    
    run_experiment(
        cfg, train_fn, metrics_fn, task, model, criterion, optimizer,
        repr_optimizer, cbp_tracker, distractor_tracker, rng,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
