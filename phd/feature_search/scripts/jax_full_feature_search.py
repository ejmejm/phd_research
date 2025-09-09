from functools import partial
import logging
from typing import Iterator, Tuple, Callable, List, Optional

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
    # get_model_statistics,
    prepare_task,
    prepare_optimizer,
    seed_from_string,
    set_seed,
    standardize_targets,
    StandardizationStats,
    rng_from_string,
)
# from phd.feature_search.jax_core.idbd import IDBD
from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.optimizer import EqxOptimizer
# from phd.feature_search.jax_core.feature_recycling import CBPTracker
from phd.feature_search.jax_core.tasks.geoff import NonlinearGEOFFTask
from phd.feature_search.jax_core.utils import tree_replace
from phd.research_utils.logging import *

# TODO:
# 1. Add full train loop
# 2. Add logging
# 3. Add bias
# 4. Add compute CBP
# 5. Add distractors?

TRAIN_LOOP_UNROLL = 5


logger = logging.getLogger(__name__)


# TODO: Implement distractors


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
                else lambda x, y: optax.l2_loss(x, y).mean())
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
    
    # # TODO: Implement CBP tracker in JAX
    # # Initialize CBP tracker
    # if cfg.feature_recycling.use_cbp_utility:
    #     cbp_cls = SignedCBPTracker if cfg.feature_recycling.use_signed_utility else CBPTracker
    #     cbp_tracker = cbp_cls(
    #         optimizer = optimizer,
    #         replace_rate = cfg.feature_recycling.recycle_rate,
    #         decay_rate = cfg.feature_recycling.utility_decay,
    #         maturity_threshold = cfg.feature_recycling.feature_protection_steps,
    #         initial_step_size_method = cfg.feature_recycling.initial_step_size_method,
    #         seed = seed_from_string(base_seed, 'cbp_tracker'),
    #     )
    #     cbp_tracker.track_sequential(model.layers)
    # else:
    #     cbp_tracker = None
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

    if cbp_tracker is not None:
        cbp_tracker.incoming_weight_init = 'binary'
    
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



# General structure for training loop:
# 1. Do setup (setup function and holder class)
# 2. Scan over training steps for a set number of steps (individual step, and multiple steps via scan functions)
# 3. Log (log function)
# 4. Repeat from step 2


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
    cbp_tracker: Any # Optional[CBPTracker]
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
    loss: Float[Array, '']
    targets: Float[Array, 'batch_size n_outputs']
    # cumulant_loss: Float[Array, ''] = eqx.field(default=jnp.zeros(1))
    # mean_pred_loss: Float[Array, '']
    # effective_lr: Float[Array, '']
    # pruned_accum: Int[Array, '']
    # pruned_newest_feature_accum: Int[Array, '']
    # n_steps_since_log: Int[Array, '']


def train_step(
    train_state: TrainState,
    data: Tuple[Float[Array, 'batch_size n_inputs'], Float[Array, 'batch_size n_outputs']],
) -> Tuple[TrainState, StepStats]:
    inputs, targets = data
    cfg, model, optimizer, repr_optimizer = \
        train_state.cfg, train_state.model, train_state.optimizer, train_state.repr_optimizer
    cbp_tracker, distractor_tracker = \
        train_state.cbp_tracker, train_state.distractor_tracker
    
    rng, noise_key, model_key = jax.random.split(train_state.rng, 3)
    
    # Add noise to targets
    if cfg.task.noise_std > 0:
        targets += jax.random.normal(noise_key, targets.shape) * cfg.task.noise_std
        
    standardized_targets, cumulant_stats = standardize_targets(targets, train_state.cumulant_stats)
    
    if cfg.train.standardize_cumulants:
        targets = standardized_targets
    
    if train_state.cbp_tracker is not None:
        # TODO: Implement CBP tracker
        pass

    def compute_loss(model, inputs, targets):
        outputs, param_inputs = jax.vmap(partial(model, key=model_key))(inputs)
        loss = train_state.criterion(outputs, targets)
        return loss, (outputs, param_inputs)
    
    # Backward pass
    (loss, (outputs, param_inputs)), grads = jax.value_and_grad(
        compute_loss, has_aux=True)(model, inputs, targets)
    
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
    step_stats = StepStats(loss, targets)
    
    return train_state, step_stats


def train_multi_step(
    train_state: TrainState,
    task: NonlinearGEOFFTask,
    n_steps: int,
) -> Tuple[TrainState, StepStats]:

    def _step(state, _):
        train_state, task = state
        task, data = task.generate_batch(1)
        train_state, step_stats = train_step(train_state, data)
        return (train_state, task), step_stats
    
    (train_state, task), step_stats = jax.lax.scan(
        _step,
        init = (train_state, task),
        xs = None,
        length = n_steps,
        unroll = TRAIN_LOOP_UNROLL,
    )
    
    return train_state, task, step_stats


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
    ):
    use_bias = cfg.model.get('use_bias', True) # TODO: Add bias to model inputs
    
    # TODO: Implement distractor
    # # Distractor setup
    # n_hidden_units = model.layers[-1].in_features
    # first_feature_idx = 1 if use_bias else 0 # First feature is bias if enabled
    # distractor_tracker.process_new_features(list(range(first_feature_idx, n_hidden_units)))
    
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
    
    train_fn = jax.jit(train_multi_step, static_argnums=(2,))
    
    sequence_length = cfg.train.log_freq
    train_cycles = cfg.train.total_steps // sequence_length
    
    train_state, task, step_stats = train_fn(train_state, task, sequence_length)
        
    import time
    start_time = time.time()
    for cycle in range(train_cycles):
        train_state, task, step_stats = train_fn(train_state, task, sequence_length)
        # print(f"Cycle {cycle}")
    print(f"Time taken: {time.time() - start_time:.2f}s")


@hydra.main(config_path='../conf', config_name='full_feature_search')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"
    
    jax.config.update('jax_platform_name', cfg.device)
    print(f"JAX is using device: {jax.devices(cfg.device)[0]}")
    
    cfg = init_experiment(cfg.project, cfg)

    task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
        prepare_ltu_geoff_experiment(cfg)
        
    # TODO: Implement distractor tracking
    # model.forward = model_distractor_forward_pass.__get__(model)
    
    # distractor_tracker = DistractorTracker(
    #     model,
    #     cfg.task.distractor_chance,
    #     tuple(cfg.task.distractor_mean_range),
    #     tuple(cfg.task.distractor_std_range),
    #     seed = seed_from_string(cfg.seed, 'distractor_tracker'),
    # )
    distractor_tracker = None
    
    run_experiment(
        cfg, task, model, criterion, optimizer, repr_optimizer,
        cbp_tracker, distractor_tracker, rng,
    )
    
    # finish_experiment(cfg)


if __name__ == '__main__':
    main()
