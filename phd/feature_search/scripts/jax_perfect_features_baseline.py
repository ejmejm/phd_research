from jax_full_feature_search import *



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
            lambda m, x: jax.vmap(partial(
                m, set_first_element_to_one=use_bias, key=model_key
            ))(x)[0].sum())(model, inputs)
        updates, optimizer = optimizer.with_update((grads, output_grads), model)
    else:
        updates, optimizer = optimizer.with_update(grads, model)
    
    if repr_optimizer is not None:
        # TODO: Set breakpoint to make sure updates are combined correctly
        repr_updates, repr_optimizer = repr_optimizer.with_update(grads, model)
        updates = eqx.combine(updates, repr_updates)
    
    model = eqx.apply_updates(model, updates)
    if cfg.model.get('use_normalize_and_project', False):
        model = model.with_projected_weights()
    
    # CBP resets
    if train_state.cbp_tracker is not None:
        if do_prune:
            if repr_optimizer is not None:
                cbp_tracker, model, (optimizer, repr_optimizer), prune_masks = train_state.cbp_tracker.prune_features(
                    model, param_inputs, (optimizer, repr_optimizer), rng=cbp_key)
            else:
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


def create_model_with_perfect_features(model: MLP, task: NonlinearGEOFFTask, cfg: DictConfig) -> MLP:
    """Return a new version of the model that has the same features as the target network in the task."""
    has_bias = cfg.model.get('use_bias', True)
    model_hidden_dim = model.layers[0].weight.shape[0]
    task_hidden_dim = task.weights[0].shape[1]
    
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
    # assert cfg.model.hidden_dim == cfg.task.hidden_dim, (
    #     f"Model hidden dim must match task hidden dim! "
    #     f"Got model hidden dim: {cfg.model.hidden_dim}, task hidden dim: {cfg.task.hidden_dim}"
    # )
    
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
    
    distractor_tracker = None
    
    train_step_fn = jax.jit(train_step, static_argnums=(2,))
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
