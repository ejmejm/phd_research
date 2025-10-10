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

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_task,
    prepare_optimizer,
    seed_from_string,
    set_seed,
    standardize_targets,
    StandardizationStats,
    rng_from_string,
)
from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.optimizers import EqxOptimizer
from phd.feature_search.jax_core.feature_recycling import CBPTracker
from phd.feature_search.jax_core.tasks.geoff import NonlinearGEOFFTask
from phd.feature_search.jax_core.utils import tree_replace
from phd.research_utils.logging import *


TRAIN_LOOP_UNROLL = 2


logger = logging.getLogger(__name__)


def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    use_bias = cfg.model.get('use_bias', True)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim = cfg.task.n_features,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim + int(use_bias),
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        key = rng_from_string(rng, 'model'),
    )
    
    criterion = (optax.softmax_cross_entropy if cfg.task.type == 'classification'
                else lambda x, y: jnp.square(y - x).mean())
    optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
    
    # Determine if we need separate optimizers for the intermediate and output layers
    repr_optimizer_name = cfg.get('representation_optimizer', {}).get('name')
    assert repr_optimizer_name != 'idbd', "IDBD is not supported for the representation optimizer!"
    n_repr_trainable_layers = max(0, len(model.layers) - 1 - model.n_frozen_layers)
    
    if repr_optimizer_name is not None and n_repr_trainable_layers > 0:
        base_filter_spec = jax.tree.map(lambda _: False, model)
        repr_filter_spec = eqx.tree_at(
            lambda x: x.layers[model.n_frozen_layers:len(model.layers) - 1],
            base_filter_spec,
            jax.tree.map(lambda _: True, model.layers[model.n_frozen_layers:len(model.layers) - 1]),
        )
        output_filter_spec = eqx.tree_at(
            lambda x: x.layers[-1],
            base_filter_spec,
            jax.tree.map(lambda _: True, model.layers[-1]),
        )
        
        # Use separate optimizers for the intermediate and output layers
        repr_optimizer = prepare_optimizer(model, repr_optimizer_name, cfg.representation_optimizer, filter_spec=repr_filter_spec)
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer, filter_spec=output_filter_spec)
        logger.info(f"Using separate optimizers for the intermediate and output layers: {repr_optimizer_name} and {cfg.optimizer.name}")
    else:
        # Only use one optimizer
        repr_optimizer = None
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
        logger.info(f"Using single optimizer: {cfg.optimizer.name}")
    
    # Initialize CBP tracker
    if cfg.feature_recycling.use_cbp_utility:
        cbp_tracker = CBPTracker(
            model = model,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate = cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.feature_recycling.feature_protection_steps,
            initial_step_size_method = cfg.feature_recycling.initial_step_size_method,
            incoming_weight_init = 'binary',
            filter_spec = None, # Don't forget to add if doing more than 2 layers
            rng = rng_from_string(rng, 'cbp_tracker'),
        )
    else:
        cbp_tracker = None
        
    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker


def prepare_ltu_geoff_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    
    task, model, criterion, optimizer, repr_optimizer, cbp_tracker = \
        prepare_components(cfg)

    assert isinstance(task, NonlinearGEOFFTask)
    
    assert cfg.model.weight_init_method == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.weight_init == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.model.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"
    assert cbp_tracker.incoming_weight_init == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.train.log_freq % cfg.feature_recycling.get('prune_frequency', 1) == 0, \
        "Log frequency must be a multiple of prune frequency!"
    
    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_key = rng_from_string(rng, 'task_init_key')
    task.weights[-1] = jax.random.uniform(
        task_init_key,
        task.weights[-1].shape,
        minval = -jnp.sqrt(6 / task.weights[-1].shape[0]),
        maxval = jnp.sqrt(6 / task.weights[-1].shape[0]),
    )
    model = eqx.tree_at(lambda m: m.layers[-1].weight, model, jnp.zeros_like(model.layers[-1].weight))
    
    # Note: LTU thresholds are all set to 0.0 by default

    set_seed(seed_from_string(base_seed, 'experiment_setup'))

    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng


class TrainState(eqx.Module):
    # Static
    cfg: DictConfig = eqx.field(static=True)
    criterion: Callable = eqx.field(static=True)
    log_utility_stats: bool = eqx.field(static=True)
    log_pruning_stats: bool = eqx.field(static=True)
    log_model_stats: bool = eqx.field(static=True)
    log_optimizer_stats: bool = eqx.field(static=True)
    
    # Non-static
    model: MLP
    optimizer: EqxOptimizer
    repr_optimizer: Optional[EqxOptimizer]
    cbp_tracker: Optional[CBPTracker]
    distractor_tracker: Any # Optional[DistractorTracker]
    
    step: Int[Array, '']
    cumulant_stats: StandardizationStats
    rng: PRNGKeyArray
    
    def __init__(self, cfg, criterion, model, optimizer, repr_optimizer, cbp_tracker, distractor_tracker, rng):
        self.cfg = cfg
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.repr_optimizer = repr_optimizer
        self.cbp_tracker = cbp_tracker
        self.distractor_tracker = distractor_tracker
        self.step = jnp.int32(0)
        self.cumulant_stats = StandardizationStats(gamma=0.99)
        self.rng = rng
        
        self.log_utility_stats = self.cfg.train.get('log_utility_stats', False)
        self.log_pruning_stats = self.cfg.train.get('log_pruning_stats', False)
        self.log_model_stats = self.cfg.train.get('log_model_stats', False)
        self.log_optimizer_stats = self.cfg.train.get('log_optimizer_stats', False)


class StepStats(eqx.Module):
    """Stats for a single step."""
    loss: Float[Array, '']
    targets: Float[Array, 'batch_size n_outputs']
    baseline_loss: Float[Array, '']


class MetricsBuffer(eqx.Module):
    """Buffer for accumulating metrics over a set of steps."""
    cumulative_loss: Float[Array, ''] = eqx.field(default=0.0, converter=jnp.asarray)
    total_samples: Int[Array, ''] = eqx.field(default=0, converter=jnp.asarray)
    prior_log_step: Int[Array, ''] = eqx.field(default=0, converter=jnp.asarray)


def train_step(
    train_state: TrainState,
    data: Tuple[Float[Array, 'batch_size n_inputs'], Float[Array, 'batch_size n_outputs']],
    do_prune: bool = False,
) -> Tuple[TrainState, StepStats]:
    inputs, targets = data
    cfg, model, optimizer, repr_optimizer = \
        train_state.cfg, train_state.model, train_state.optimizer, train_state.repr_optimizer
    cbp_tracker, distractor_tracker = \
        train_state.cbp_tracker, train_state.distractor_tracker
    
    use_bias = cfg.model.get('use_bias', True)
    rng, noise_key, model_key, cbp_key = jax.random.split(train_state.rng, 4)
    
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
            lambda m, x: jax.vmap(partial(m, key=model_key))(x)[0].sum())(model, inputs)
        updates, optimizer = optimizer.with_update((grads, output_grads), model)
    else:
        updates, optimizer = optimizer.with_update(grads, model)
    
    if repr_optimizer is not None:
        # TODO: Set breakpoint to make sure updates are combined correctly
        repr_updates, repr_optimizer = repr_optimizer.with_update(grads, model)
        updates = eqx.combine(updates, repr_updates)
    
    model = eqx.apply_updates(model, updates)
    
    # CBP resets
    if train_state.cbp_tracker is not None:
        if do_prune:
            cbp_tracker, model, optimizer, prune_masks = train_state.cbp_tracker.prune_features(
                model, param_inputs, optimizer, rng=cbp_key)
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
    step_stats = StepStats(loss, targets, baseline_loss)
    
    return train_state, step_stats


def train_multi_step(
    train_state: TrainState,
    task: NonlinearGEOFFTask,
    n_steps: int,
) -> Tuple[TrainState, StepStats]:
    train_step_fn = jax.jit(train_step, static_argnums=(2,))
    prune_frequency = train_state.cfg.feature_recycling.get('prune_frequency', 1)
    batch_size = train_state.cfg.train.batch_size

    def _inner_step(state, _):
        train_state, task = state
        
        all_step_stats = []
        for _ in range(prune_frequency - 1):
            task, data = task.generate_batch(batch_size)
            train_state, step_stats = train_step_fn(train_state, data, False)
            all_step_stats.append(step_stats)
        
        task, data = task.generate_batch(batch_size)
        do_prune = train_state.cfg.feature_recycling.recycle_rate > 0
        train_state, step_stats = train_step_fn(train_state, data, do_prune)
        all_step_stats.append(step_stats)
        
        step_stats = jax.tree.map(lambda *args: jnp.stack(args), *all_step_stats)
        
        return (train_state, task), step_stats
    
    (train_state, task), step_stats = jax.lax.scan(
        _inner_step,
        init = (train_state, task),
        xs = None,
        length = n_steps // prune_frequency,
        unroll = TRAIN_LOOP_UNROLL,
    )
    
    return train_state, task, step_stats


def compute_metrics(
        metrics_buffer: MetricsBuffer,
        step_stats: StepStats,
        cfg: DictConfig,
        step: int,
    ) -> Tuple[MetricsBuffer, Dict[str, Any]]:
    cycle_loss = step_stats.loss.sum()
    steps_since_log = step - metrics_buffer.prior_log_step
    
    # Update metrics buffer
    metrics_buffer = tree_replace(
        metrics_buffer,
        cumulative_loss = metrics_buffer.cumulative_loss + cycle_loss,
        total_samples = metrics_buffer.total_samples + step_stats.loss.shape[0],
        prior_log_step = step,
    )
    
    # Compute metrics
    metrics = {
        'step': step,
        'samples': metrics_buffer.total_samples,
        'loss': cycle_loss / steps_since_log,
        'cumulative_loss': metrics_buffer.cumulative_loss,
        'squared_targets': jnp.square(step_stats.targets).mean(),
        'baseline_loss': step_stats.baseline_loss.sum() / steps_since_log,
    }
    
    if cfg.train.get('log_pruning_stats', False):
        raise NotImplementedError("Pruning stats are not implemented yet!")
    if cfg.train.get('log_utility_stats', False):
        raise NotImplementedError("Utility stats are not implemented yet!")
    if cfg.train.get('log_model_stats', False):
        raise NotImplementedError("Model stats are not implemented yet!")
    if cfg.train.get('log_optimizer_stats', False):
        raise NotImplementedError("Optimizer stats are not implemented yet!")
    
    metrics = {k: np.asarray(v) for k, v in metrics.items()}
    
    return metrics_buffer, metrics


def compute_baseline_loss(
    cfg: DictConfig,
    train_state: TrainState,
    targets: Float[Array, 'batch_size n_outputs'],
) -> Float[Array, '']:
    """Loss for a naive baseline prediction that always predicts the mean of the targets."""
    if cfg.train.standardize_cumulants:
        baseline_preds = jnp.zeros_like(targets)
    else:
        baseline_preds = jnp.ones_like(targets) * train_state.cumulant_stats.running_mean  
    loss = train_state.criterion(baseline_preds, targets)
    return loss


def run_experiment(
        cfg: DictConfig,
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
    metrics_buffer = MetricsBuffer()
    all_metrics = []
    
    train_fn = jax.jit(train_multi_step, static_argnums=(2,))
    
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
        metrics_buffer, metrics = compute_metrics(metrics_buffer, step_stats, cfg, step=train_state.step)
        all_metrics.append(metrics)
        log_metrics(metrics, cfg, step=train_state.step)
        
        if pbar is not None:
            pbar.set_postfix(loss=f"{metrics['loss']:.5f}")
            pbar.update(sequence_length)
        
    if pbar is not None:
        pbar.close()
    
    all_metrics = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics)
    return train_state, task, all_metrics


@hydra.main(config_path='../conf', config_name='full_feature_search')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"

    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_persistent_cache_enable_xla_caches', 'xla_gpu_per_fusion_autotune_cache_dir')
    
    jax.config.update('jax_platform_name', cfg.device)
    print(f"JAX is using device: {jax.devices(cfg.device)[0]}")
    
    cfg = init_experiment(cfg.project, cfg)

    task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
        prepare_ltu_geoff_experiment(cfg)
    
    distractor_tracker = None
    
    run_experiment(
        cfg, task, model, criterion, optimizer, repr_optimizer,
        cbp_tracker, distractor_tracker, rng,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
