from jax_full_feature_search import *


def create_model_with_perfect_features(model: MLP, task: NonlinearGEOFFTask, cfg: DictConfig) -> MLP:
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


def validate_config(cfg: DictConfig):
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"
    
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


@hydra.main(config_path='../conf', config_name='perfect_features_baseline')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_persistent_cache_enable_xla_caches', 'xla_gpu_per_fusion_autotune_cache_dir')
    
    jax.config.update('jax_platform_name', cfg.device)
    print(f"JAX is using device: {jax.devices(cfg.device)[0]}")
    
    # Option to freeze only perfect features: feature_recycling.freeze_perfect_features
    
    cfg = init_experiment(cfg.project, cfg)
    validate_config(cfg)
    
    task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
        prepare_ltu_geoff_experiment(cfg)
    model = create_model_with_perfect_features(model, task, cfg)
    
    # Set true feature uilities to infinity so they are never replaced
    if cfg.feature_recycling.get('freeze_perfect_features', False):
        if (cfg.model.hidden_dim != cfg.task.hidden_dim and
            cfg.model.hidden_dim <= 2 *cfg.task.hidden_dim and
            cfg.feature_recycling.utility_reset_mode.lower() == 'median'
        ):
            raise ValueError(
                f"Setting perfect features to be frozen but less than 2x the task hidden dim is not supported when utility reset mode is median! "
                f"This would cause the median utility to be infinity, and no features would be prunable."
            )
        use_bias = cfg.model.get('use_bias', True)
        n_task_features = cfg.task.n_features + int(use_bias)
        utilities = cbp_tracker.all_feature_stats[0].utility
        cbp_tracker = eqx.tree_at(
            lambda x: x.all_feature_stats[0].utility,
            cbp_tracker,
            utilities.at[:n_task_features].set(jnp.inf),
        )
    
    distractor_tracker = None
    
    train_step_fn = jax.jit(train_step, static_argnums=(3,))
    train_fn = jax.jit(
        partial(train_multi_step, train_step_fn=train_step_fn),
        static_argnames = ('n_steps', 'train_step_fn'),
    )
    metrics_fn = jax.jit(compute_metrics, static_argnums=(4,))
    run_experiment(
        cfg, train_fn, metrics_fn, task, model, criterion, optimizer,
        repr_optimizer, cbp_tracker, distractor_tracker, rng,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
