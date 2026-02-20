from phd.feature_search.scripts.jax_full_feature_search import *
from phd.sandbox.nethack_benchmarking.nethack import NethackPredictionTask
from .models import ConvNet


def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    task = NethackPredictionTask(batch_size=cfg.batch_size * cfg.train.log_freq)
    use_bias = cfg.model.get('use_bias', True)
    
    model = ConvNet(
        input_dim = # (3, 224, 224), 
        embedding_dim = 8,
        conv_configs = [(5, 2, 32), (3, 1, 32), (3, 1, 64)],
        fc_hidden_dims = [256, 128],
        output_dim = # 1,
        activation = 'relu',
        use_bias = use_bias,
        key = rng_from_string(rng, 'model'),
    )
    
    # # Initialize model and optimizer
    # model = MLP(
    #     input_dim = cfg.task.n_features,
    #     output_dim = cfg.task.get('n_outputs', 1),
    #     n_layers = cfg.model.n_layers,
    #     hidden_dim = cfg.model.hidden_dim + int(use_bias),
    #     weight_init_method = cfg.model.weight_init_method,
    #     activation = cfg.model.activation,
    #     n_frozen_layers = cfg.model.n_frozen_layers,
    #     use_normalize_and_project = cfg.model.get('use_normalize_and_project', False),
    #     key = rng_from_string(rng, 'model'),
    # )
    
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
            incoming_weight_init = cfg.feature_recycling.incoming_weight_init,
            filter_spec = None, # Don't forget to add if doing more than 2 layers
            rng = rng_from_string(rng, 'cbp_tracker'),
        )
    else:
        cbp_tracker = None
        
    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker


def prepare_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = jax.random.key(base_seed)
    task, model, criterion, optimizer, repr_optimizer, cbp_tracker = \
        prepare_components(cfg)
    model = eqx.tree_at(lambda m: m.layers[-1].weight, model, jnp.zeros_like(model.layers[-1].weight))
    set_seed(seed_from_string(base_seed, 'experiment_setup'))
    
    return task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng


def train_step(
    train_state: TrainState,
    data: Tuple[Float[Array, 'batch_size ...'], Float[Array, 'batch_size ...']],
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
            ))(x)[0].mean(axis=0).sum())(model, inputs)
        updates, optimizer = optimizer.with_update((grads, output_grads), model)
    else:
        updates, optimizer = optimizer.with_update(grads, model)
    
    if repr_optimizer is not None:
        # TODO: Set breakpoint to make sure updates are combined correctly
        repr_updates, repr_optimizer = repr_optimizer.with_update(grads, model)
        updates = eqx.combine(updates, repr_updates)
    
    # Compute mean L1 of updates per layer (always computed, flag controls logging)
    # Use float64 for precision in accumulation
    update_weights = [layer.weight for layer in updates.layers if layer.weight is not None]
    update_mean_l1s = jnp.array(
        [jnp.mean(jnp.abs(w)) for w in update_weights], dtype=jnp.float64)
    
    model = eqx.apply_updates(model, updates)
    if cfg.model.get('use_normalize_and_project', False):
        model = model.with_projected_weights()
    
    # CBP resets
    n_pruned = 0
    n_best_features_pruned = 0
    if train_state.cbp_tracker is not None:
        if do_prune:
            if repr_optimizer is not None:
                cbp_tracker, model, (optimizer, repr_optimizer), prune_masks = train_state.cbp_tracker.prune_features(
                    model, param_inputs, (optimizer, repr_optimizer), rng=cbp_key)
            else:
                cbp_tracker, model, optimizer, prune_masks = train_state.cbp_tracker.prune_features(
                    model, param_inputs, optimizer, rng=cbp_key)
                
            if cfg.train.get('log_pruning_stats', False):
                assert len(prune_masks) == 1, "There should only be one prune mask!"
                prune_mask = prune_masks[0]
                n_pruned = prune_mask.sum()
                # Note: n_best_features_pruned computation requires task, so we skip it in jitted function
                n_best_features_pruned = 0
                
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
    step_stats = StepStats(
        loss, targets, baseline_loss, n_pruned, n_best_features_pruned,
        update_mean_l1s,
    )
    
    return train_state, step_stats


def train_multi_step(
    train_state: TrainState,
    data_batches: Tuple[Float[Array, 'n_steps batch_size n_inputs'], Float[Array, 'n_steps batch_size n_outputs']],
    train_step_fn: Callable,
) -> Tuple[TrainState, StepStats]:
    """Train for multiple steps using pre-generated data batches.
    
    Args:
        train_state: Current training state
        data_batches: Pre-generated data batches of shape (n_steps, batch_size, ...)
        train_step_fn: Function to execute a single training step
        
    Returns:
        Updated train_state and step_stats
    """
    prune_frequency = train_state.cfg.feature_recycling.get('prune_frequency', 1)
    input_batch, target_batch = data_batches
    
    samples_per_inner_step = prune_frequency
    n_outer_steps = input_batch.shape[0] // samples_per_inner_step
    batch_size = input_batch.shape[1]

    def _inner_step(train_state, data_batch):
        inputs, targets = data_batch['inputs'], data_batch['targets'] # (batch_size, ...)
        
        all_step_stats = []
        for i in range(prune_frequency - 1):
            step_data = (inputs[i], targets[i])
            train_state, step_stats = train_step_fn(train_state, step_data, False)
            all_step_stats.append(step_stats)
        
        step_data = (inputs[samples_per_inner_step - 1], targets[samples_per_inner_step - 1])
        do_prune = train_state.cfg.feature_recycling.recycle_rate > 0
        train_state, step_stats = train_step_fn(train_state, step_data, do_prune)
        all_step_stats.append(step_stats)
        
        step_stats = jax.tree.map(lambda *args: jnp.stack(args), *all_step_stats)
        
        return train_state, step_stats
    
    train_data = {
        'inputs': input_batch.reshape(
            n_outer_steps, samples_per_inner_step, batch_size, *input_batch.shape[2:]),
        'targets': target_batch.reshape(
            n_outer_steps, samples_per_inner_step, batch_size, *target_batch.shape[2:]),
    }
    
    train_state, step_stats = jax.lax.scan(
        _inner_step,
        init = train_state,
        xs = train_data,
        unroll = TRAIN_LOOP_UNROLL,
    )
    
    step_stats = jax.tree.map(lambda arr: arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:]), step_stats)
    
    return train_state, step_stats


def compute_metrics_simple(
    train_state: TrainState,
    metrics_buffer: MetricsBuffer,
    step_stats: StepStats,
    cfg: DictConfig,
) -> Tuple[MetricsBuffer, Dict[str, Any]]:
    """Compute simplified metrics: step, samples, loss, and cumulative_loss."""
    step = train_state.step
    cycle_loss = step_stats.loss.sum()
    steps_since_log = step - metrics_buffer.prior_log_step
    
    # Update metrics buffer
    metrics_buffer = tree_replace(
        metrics_buffer,
        cumulative_loss = metrics_buffer.cumulative_loss + cycle_loss,
        total_samples = metrics_buffer.total_samples + step_stats.targets.shape[0],
        prior_log_step = step,
    )
    
    # Compute only the essential metrics
    metrics = {
        'step': step,
        'samples': metrics_buffer.total_samples,
        'loss': cycle_loss / steps_since_log,
        'cumulative_loss': metrics_buffer.cumulative_loss,
    }
    
    return metrics_buffer, metrics


def run_multiseed_experiment(
    cfg: DictConfig,
    train_fn: Callable,
    metrics_fn: Callable,
    task: NethackPredictionTask,
    model: MLP,
    criterion: Callable,
    optimizer: EqxOptimizer,
    repr_optimizer: Optional[EqxOptimizer],
    cbp_tracker, # : Optional[CBPTracker],
    rng: PRNGKeyArray,
    show_progress: bool = True,
) -> Tuple[TrainState, NethackPredictionTask, Dict[str, Array]]:
    train_state = jax.vmap(TrainState, in_axes=(None, None, 0, 0, 0, 0, None, 0))(
        cfg, criterion, model, optimizer, repr_optimizer, cbp_tracker, None, rng)
    
    metrics_buffers = jax.tree.map(lambda *args: jnp.stack(args), *[MetricsBuffer() for _ in range(len(rng))])
    all_metrics = []
    
    assert cfg.train.total_steps % sequence_length == 0, "Total steps must be a multiple of train.log_freq!"
    sequence_length = cfg.train.log_freq
    train_cycles = cfg.train.total_steps // sequence_length
    batch_size = cfg.train.batch_size

    if show_progress:
        pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    else:
        pbar = None
    
    # Training loop
    for cycle_idx in range(train_cycles):
        
        # Generate data for this cycle - task generates sequence_length * batch_size samples
        task, (inputs, targets) = task.generate_batch()
        
        # Reshape to (sequence_length, batch_size, ...) for training
        inputs = inputs.reshape(sequence_length, batch_size, *inputs.shape[1:])
        targets = targets.reshape(sequence_length, batch_size, *targets.shape[1:])
        cycle_data = (inputs, targets)
        
        # Train
        train_state, step_stats = train_fn(train_state, cycle_data)
        
        # Metrics
        metrics_buffers, metrics = metrics_fn(train_state, metrics_buffers, step_stats, cfg)
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


def stack_pytrees(pytrees: List[Any]) -> Any:
    treedef = jax.tree.structure(pytrees[0])
    all_leaves = [jax.tree.leaves(pytree) for pytree in pytrees]
    new_pytrees = [jax.tree.unflatten(treedef, leaves) for leaves in all_leaves]
    stacked_pytrees = jax.tree.map(lambda *args: jnp.stack(args), *new_pytrees)
    return stacked_pytrees


@hydra.main(config_path='./conf', config_name='defaults', version_base='1.1')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.seed is None or not isinstance(cfg.seed, list), "Seed must be a list of integers!"
    
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)
    
    seeds = cfg.seed
    run_vars = []
    
    for seed in seeds:
        cfg.seed = seed
        task, model, criterion, optimizer, repr_optimizer, cbp_tracker, rng = \
            prepare_experiment(cfg)
            
        if cfg.model.get('use_perfect_features', False):
            model = create_model_with_perfect_features(model, task, cfg)
            
            if cfg.feature_recycling.get('perfect_features_irreplaceable', False):
                cbp_tracker = make_perfect_features_irreplacable(cbp_tracker, task, cfg)
            
        run_vars.append([task, model, optimizer, repr_optimizer, cbp_tracker, rng])
    
    tasks, models, optimizers, repr_optimizers, cbp_trackers, rngs = \
        [stack_pytrees(pytrees) for pytrees in zip(*run_vars)]
    
    train_step_fn = jax.jit(train_step, static_argnums=(2,))
    train_fn = jax.jit(
        jax.vmap(partial(train_multi_step, train_step_fn=train_step_fn), in_axes=(0, None)),
    )
    
    metrics_fn = jax.jit(jax.vmap(compute_metrics_simple, in_axes=(0, 0, 0, None)), static_argnums=(3,))
    
    run_multiseed_experiment(
        cfg, train_fn, metrics_fn, tasks, models, criterion, optimizers,
        repr_optimizers, cbp_trackers, rngs,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
