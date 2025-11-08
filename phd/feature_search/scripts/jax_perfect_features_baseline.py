from jax_full_feature_search import *


def create_model_with_perfect_features(model: MLP, task: NonlinearGEOFFTask, cfg: DictConfig) -> MLP:
    """Return a new version of the model that has the same features as the target network in the task."""
    has_bias = cfg.model.get('use_bias', True)
    model_hidden_dim = model.layers[0].weight.shape[0]
    task_hidden_dim = task.weights[0].shape[1]
    
    if has_bias:
        model_hidden_dim -= 1
    
    assert model_hidden_dim == task_hidden_dim, \
        f"Model hidden dim ({model_hidden_dim}) must match task hidden dim ({task_hidden_dim})!"
        
    task_feature_weights = task.weights[0].T
    if has_bias:
        input_dim = task_feature_weights.shape[1]
        task_feature_weights = jnp.concatenate(
            [jnp.zeros((1, input_dim)), task_feature_weights],
            axis = 0,
        )
    
    new_model = eqx.tree_at(
        lambda m: m.layers[0].weight,
        model,
        task_feature_weights,
    )
    
    return new_model


def validate_config(cfg: DictConfig):
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"
    assert cfg.model.hidden_dim == cfg.task.hidden_dim, (
        f"Model hidden dim must match task hidden dim! "
        f"Got model hidden dim: {cfg.model.hidden_dim}, task hidden dim: {cfg.task.hidden_dim}"
    )
    
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
    
    cfg = init_experiment(cfg.project, cfg)
    validate_config(cfg)
    
    task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
        prepare_ltu_geoff_experiment(cfg)
    model = create_model_with_perfect_features(model, task, cfg)
    
    distractor_tracker = None
    
    run_experiment(
        cfg, task, model, criterion, optimizer, repr_optimizer,
        cbp_tracker, distractor_tracker, rng,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
