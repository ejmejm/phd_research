import logging
from typing import Iterator, Tuple, Callable, List, Optional

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
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
# from phd.feature_search.jax_core.feature_recycling import CBPTracker
from phd.feature_search.jax_core.tasks.geoff import NonlinearGEOFFTask
from phd.research_utils.logging import *


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
                else optax.l2_loss)
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

    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker


@hydra.main(config_path='../conf', config_name='full_feature_search')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"
    
    jax.config.update('jax_platform_name', cfg.device)
    print(f"JAX is using device: {jax.devices(cfg.device)[0]}")
    
    cfg = init_experiment(cfg.project, cfg)

    task, model, criterion, optimizer, repr_optimizer, cbp_tracker = \
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
    
    # run_experiment(
    #     cfg, task, task_iterator, model, criterion, optimizer,
    #     repr_optimizer, cbp_tracker, distractor_tracker,
    # )
    
    # finish_experiment(cfg)


if __name__ == '__main__':
    main()
