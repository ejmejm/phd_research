from phd.feature_search.scripts.jax_full_feature_search import *


def run_multiseed_experiment(
    cfg: DictConfig,
    train_fn: Callable,
    metrics_fn: Callable,
    task: NonlinearGEOFFTask,
    model: MLP,
    criterion: Callable,
    optimizer: EqxOptimizer,
    repr_optimizer: Optional[EqxOptimizer],
    cbp_tracker, # : Optional[CBPTracker],
    rng: PRNGKeyArray,
    show_progress: bool = True,
) -> Tuple[TrainState, NonlinearGEOFFTask, Dict[str, Array]]:
    train_state = jax.vmap(TrainState, in_axes=(None, None, 0, 0, 0, 0, None, 0))(
        cfg, criterion, model, optimizer, repr_optimizer, cbp_tracker, None, rng)
    
    # log_task_output_weights(task, cfg)
    metrics_buffers = jax.tree.map(lambda *args: jnp.stack(args), *[MetricsBuffer() for _ in range(len(rng))])
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
        metrics_buffers, metrics = metrics_fn(train_state, task, metrics_buffers, step_stats, cfg)
        metrics = {k: np.asarray(v).mean(axis=0) for k, v in metrics.items()}
        all_metrics.append(metrics)
        log_metrics(metrics, cfg, step=train_state.step[0]) # Consider making logging async
        
        if pbar is not None:
            pbar.set_postfix(loss=f"{metrics['loss']:.5f}")
            pbar.update(sequence_length)
        
    if pbar is not None:
        pbar.close()
    
    all_metrics = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics)
    return train_state, task, all_metrics


@hydra.main(config_path='../conf', config_name='full_feature_search', version_base='1.1')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.seed is None or not isinstance(cfg.seed, list), "Seed must be a list of integers!"
    
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)
    validate_config(cfg)
    
    seeds = cfg.seed
    run_vars = []
    
    for seed in seeds:
        cfg.seed = seed
        task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
            prepare_ltu_geoff_experiment(cfg)
            
        if cfg.model.get('use_perfect_features', False):
            model = create_model_with_perfect_features(model, task, cfg)
            
            if cfg.feature_recycling.get('perfect_features_irreplaceable', False):
                cbp_tracker = make_perfect_features_irreplacable(cbp_tracker, task, cfg)
            
        run_vars.append([task, model, optimizer, repr_optimizer, cbp_tracker, rng])
    
    tasks, models, optimizers, repr_optimizers, cbp_trackers, rngs = \
        [stack_pytrees(pytrees) for pytrees in zip(*run_vars)]
    
    train_step_fn = jax.jit(train_step, static_argnums=(3,))
    train_fn = jax.jit(
        jax.vmap(partial(train_multi_step, train_step_fn=train_step_fn), in_axes=(0, 0, None)),
        static_argnames = ('n_steps', 'train_step_fn'),
    )
    metrics_fn = jax.jit(jax.vmap(compute_metrics, in_axes=(0, 0, 0, 0, None)), static_argnums=(4,))
    
    run_multiseed_experiment(
        cfg, train_fn, metrics_fn, tasks, models, criterion, optimizers,
        repr_optimizers, cbp_trackers, rngs,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
