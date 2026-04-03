from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Optional, Tuple, Union

# Resolve relative MLflow tracking URI to absolute before Hydra changes CWD
_mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
if _mlflow_uri.startswith('sqlite:///') and not os.path.isabs(_mlflow_uri[len('sqlite:///'):]):
    os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{os.path.abspath(_mlflow_uri[len("sqlite:///"):])}'

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer,
    set_seed,
    rng_from_string,
)
from phd.jax_core.models import MLP
from phd.jax_core.optimizers import EqxOptimizer
from phd.jax_core.utils import tree_replace
from phd.research_utils.logging import (
    init_experiment,
    init_child_runs,
    log_metrics,
    log_child_metrics,
    finish_child_runs,
    finish_experiment,
)
from phd.structure_search.block_sparse_mlp import (
    BlockSparseMLP, compute_hidden_dim_for_params,
)
from phd.structure_search.data import load_dataset, DataStream, ParallelMNISTStream
from phd.structure_search.connectivity_manager import (
    ConnectivityManagerBase, ConnectivityManager,
    ConnectionConnectivityManager,
    full_input_generate,
    contribution_utility, upgd_utility, si_utility, loo_utility,
)
from phd.structure_search.dynamic_network import (
    DynamicNetwork, sync_outgoing_weights, build_outgoing_indices,
    init_random_dynamic_network,
    count_active_connections, count_active_units,
)
from phd.structure_search.metrics import StepMetrics, compute_structure_metrics


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Dummy structure tracker (placeholder for future structure search)
# ---------------------------------------------------------------------------

class DummyStructureTracker(eqx.Module):
    """Placeholder for future structure/feature search algorithm.

    Mirrors the role of CBPTracker in feature_search: sits in TrainState,
    gets called each step to update statistics, and periodically modifies
    the network structure.
    """
    rng: PRNGKeyArray

    def update_stats(self, model, param_inputs, grads=None, updates=None,
                      targets=None, predictions=None):
        """Update feature/structure statistics after a training step."""
        return self

    def modify_structure(self, model, optimizer, *, rng):
        """Modify network structure (e.g. add/remove neurons). No-op for now.

        Returns:
            (tracker, model, optimizer) — potentially modified.
        """
        return self, model, optimizer


# ---------------------------------------------------------------------------
# Train state and metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: Union[MLP, DynamicNetwork, BlockSparseMLP]
    optimizer: EqxOptimizer
    structure_tracker: Union[DummyStructureTracker, ConnectivityManagerBase]
    step: jax.Array
    rng: PRNGKeyArray


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _make_dynamic_filter_spec(model: DynamicNetwork):
    """Build optimizer filter_spec for DynamicNetwork: only weights and output_weights."""
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda n: (n.weights, n.output_weights), spec, (True, True),
    )


def _make_output_only_filter_spec(model: DynamicNetwork):
    """Build optimizer filter_spec for DynamicNetwork: only output_weights are trainable."""
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(lambda n: n.output_weights, spec, True)


def prepare_experiment(
    cfg: DictConfig,
) -> Tuple[TrainState, list, int, int, int, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Initialize per-seed models, optimizers, data streams.

    Returns:
        batched_state: Vmapped TrainState across seeds.
        streams: List of DataStream or ParallelMNISTStream per seed.
        n_params: Parameter count of the model.
        num_classes: Number of classes per task (e.g. 10 for MNIST).
        n_tasks: Number of parallel tasks (1 for single-task).
        test_data: (test_images, test_labels) or None if eval_freq == 0.
    """
    seeds = cfg.seed
    model_type = cfg.model.get('type', 'mlp')
    dataset_name = cfg.dataset.name
    n_tasks = cfg.dataset.get('n_tasks', 1)
    permute_period = cfg.dataset.get('permute_period', 0)
    eval_freq = cfg.train.get('eval_freq', 0)
    use_bias = cfg.model.get('use_bias', False)

    # --- Load data ---
    if dataset_name == 'parallel_mnist':
        base_images, base_labels, num_classes, input_dim_per_task = load_dataset('mnist', split='train')
        input_dim = n_tasks * input_dim_per_task
        output_dim = n_tasks * num_classes

        test_data = None
        test_images_raw = test_labels_raw = None
        if eval_freq > 0:
            test_images_raw, test_labels_raw, _, _ = load_dataset('mnist', split='test')
    else:
        images, labels, num_classes, input_dim = load_dataset(dataset_name)
        output_dim = num_classes
        n_tasks = 1

        test_data = None
        if eval_freq > 0:
            test_img, test_lbl, _, _ = load_dataset(dataset_name, split='test')
            test_data = (test_img, test_lbl)

    # --- Resolve hidden_dim ---
    hidden_dim = cfg.model.hidden_dim
    init_strategy = cfg.model.get('init_strategy', 'linear')
    if init_strategy == 'empty' and hidden_dim != 0:
        raise ValueError(
            f"init_strategy='empty' requires hidden_dim=0, got hidden_dim={hidden_dim}"
        )
    target_params = cfg.model.get('target_params', None)
    if target_params is not None:
        hidden_dim = compute_hidden_dim_for_params(
            target_params, model_type, cfg.model.n_layers, n_tasks,
        )
        print(f'target_params={target_params} → hidden_dim={hidden_dim}')

    # --- Build per-seed state ---
    streams = []
    train_states = []
    for seed in seeds:
        rng = jax.random.key(seed)
        model_key = rng_from_string(rng, 'model')

        # Data stream
        if dataset_name == 'parallel_mnist':
            streams.append(ParallelMNISTStream(
                images=base_images, labels=base_labels,
                n_tasks=n_tasks, batch_size=cfg.train.batch_size,
                seed=seed, permute_period=permute_period,
                test_images=test_images_raw, test_labels=test_labels_raw,
            ))
        else:
            streams.append(DataStream(
                images=images, labels=labels,
                num_classes=num_classes,
                batch_size=cfg.train.batch_size, seed=seed,
            ))

        # Model
        if model_type == 'dynamic':
            model = init_random_dynamic_network(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=cfg.model.n_layers,
                units_per_layer=hidden_dim,
                max_units_per_layer=cfg.model.get('max_units_per_layer', None),
                max_connections_per_unit=cfg.model.get('max_connections_per_unit', None),
                activations=(cfg.model.activation,),
                max_fan_out=cfg.model.get('max_fan_out', None),
                connect_all_to_output=cfg.model.get('connect_all_to_output', False),
                init_strategy=cfg.model.get('init_strategy', 'linear'),
                key=model_key,
            )
            if cfg.model.get('freeze_hidden_weights', False):
                filter_spec = _make_output_only_filter_spec(model)
            else:
                filter_spec = _make_dynamic_filter_spec(model)
            optimizer = prepare_optimizer(
                model, cfg.optimizer.name, cfg.optimizer,
                filter_spec=filter_spec,
            )
        elif model_type == 'block_sparse':
            model = BlockSparseMLP(
                n_tasks=n_tasks,
                input_dim_per_task=input_dim // n_tasks,
                output_dim_per_task=num_classes,
                n_layers=cfg.model.n_layers,
                hidden_dim=hidden_dim,
                weight_init_method=cfg.model.weight_init_method,
                activation=cfg.model.activation,
                n_frozen_layers=cfg.model.get('n_frozen_layers', 0),
                key=model_key,
            )
            optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
        else:
            model = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=cfg.model.n_layers,
                hidden_dim=hidden_dim + int(use_bias),
                weight_init_method=cfg.model.weight_init_method,
                activation=cfg.model.activation,
                n_frozen_layers=cfg.model.get('n_frozen_layers', 0),
                key=model_key,
            )
            optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)

        # Structure tracker
        if model_type == 'dynamic' and cfg.structure_tracker.get('enabled', False):
            generate_strategy = cfg.structure_tracker.get('generate_strategy', 'random')
            generate_fn = full_input_generate if generate_strategy == 'full_input' else None
            tracker_mode = cfg.structure_tracker.get('mode', 'unit')

            if tracker_mode == 'connection':
                tracker = ConnectionConnectivityManager(
                    model=model,
                    prune_rate=cfg.structure_tracker.prune_rate,
                    connection_budget=cfg.structure_tracker.connection_budget,
                    decay_rate=cfg.structure_tracker.decay_rate,
                    max_new_units_per_step=cfg.structure_tracker.get(
                        'max_new_units_per_step', 512),
                    output_connect_strategy=cfg.structure_tracker.get(
                        'output_connect_strategy', 'all'),
                    output_weight_init=cfg.structure_tracker.get(
                        'output_weight_init', 'zero'),
                    generate_fn=generate_fn,
                    rng=rng_from_string(rng, 'tracker'),
                )
            else:
                utility_fn_map = {
                    'contribution': contribution_utility,
                    'upgd': upgd_utility,
                    'si': si_utility,
                    'loo': loo_utility,
                }
                utility_fn = utility_fn_map.get(
                    cfg.structure_tracker.get('utility_fn', 'contribution'))
                tracker = ConnectivityManager(
                    model=model,
                    prune_rate=cfg.structure_tracker.prune_rate,
                    connection_budget=cfg.structure_tracker.connection_budget,
                    decay_rate=cfg.structure_tracker.decay_rate,
                    maturity_threshold=cfg.structure_tracker.maturity_threshold,
                    max_new_units_per_step=cfg.structure_tracker.get(
                        'max_new_units_per_step', 512),
                    output_connect_strategy=cfg.structure_tracker.get(
                        'output_connect_strategy', 'all'),
                    output_weight_init=cfg.structure_tracker.get(
                        'output_weight_init', 'zero'),
                    utility_fn=utility_fn,
                    generate_fn=generate_fn,
                    rng=rng_from_string(rng, 'tracker'),
                )
        else:
            tracker = DummyStructureTracker(rng=rng_from_string(rng, 'tracker'))

        train_states.append(TrainState(
            model=model, optimizer=optimizer,
            structure_tracker=tracker,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    if model_type == 'dynamic':
        net = train_states[0].model
        n_units = count_active_units(net)
        n_conns = count_active_connections(net)
        max_conns = net.weights.size + net.output_weights.size
        print(f'Model: DynamicNetwork, Params: {n_params}, '
              f'Units: {n_units}, Connections: {n_conns}/{max_conns}, '
              f'Seeds: {seeds}')
    elif model_type == 'block_sparse':
        print(f'Model: BlockSparseMLP, Params: {n_params}, '
              f'hidden_dim={hidden_dim}, n_tasks={n_tasks}, Seeds: {seeds}')
    else:
        print(f'Model: MLP, Params: {n_params}, hidden_dim={hidden_dim}, Seeds: {seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params, num_classes, n_tasks, test_data


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _mse_loss(logits, one_hot):
    return jnp.mean(jnp.square(logits - one_hot))


def _cross_entropy_loss(logits, one_hot):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


LOSS_FNS = {
    'mse': _mse_loss,
    'cross_entropy': _cross_entropy_loss,
}


def train_step(
    train_state: TrainState,
    data,
    do_restructure: bool = False,
    loss_name: str = 'mse',
    num_classes: int = 10,
    n_tasks: int = 1,
) -> Tuple[TrainState, StepMetrics]:
    """Single training step.

    Args:
        train_state: Current training state.
        data: Tuple of (images, labels).
        do_restructure: If True and structure_tracker is a ConnectivityManager,
            call modify_structure after the utility update. This parameter is
            static (Python bool) so JAX traces separate paths for True/False.
        loss_name: Loss function to use ('mse' or 'cross_entropy'). Static.
        num_classes: Classes per task. Static.
        n_tasks: Number of parallel tasks. Static.
    """
    images, labels = data
    loss_fn_impl = LOSS_FNS[loss_name]

    if n_tasks > 1:
        # labels: (batch_size, K)
        one_hot = jax.nn.one_hot(labels, num_classes)  # (batch_size, K, num_classes)

        def loss_fn(model):
            raw_outputs, param_inputs = jax.vmap(model)(images)  # (batch_size, K*num_classes)
            outputs = raw_outputs.reshape(-1, n_tasks, num_classes)
            loss = loss_fn_impl(outputs, one_hot)
            return loss, (raw_outputs, param_inputs)

        (loss, (raw_outputs, param_inputs)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(train_state.model)

        # Accuracy: per-task argmax
        outputs_reshaped = raw_outputs.reshape(-1, n_tasks, num_classes)
        predicted = jnp.argmax(outputs_reshaped, axis=-1)  # (batch_size, K)
        correct = (predicted == labels).astype(jnp.float32).mean()
    else:
        # labels: (batch_size,)
        one_hot = jax.nn.one_hot(labels, num_classes)  # (batch_size, num_classes)

        def loss_fn(model):
            outputs, param_inputs = jax.vmap(model)(images)
            loss = loss_fn_impl(outputs, one_hot)
            return loss, (outputs, param_inputs)

        (loss, (raw_outputs, param_inputs)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(train_state.model)

        predicted = jnp.argmax(raw_outputs, axis=-1)
        correct = (predicted == labels).astype(jnp.float32).mean()

    # Optimizer update (IDBD needs prediction gradients in addition to loss gradients)
    if train_state.optimizer.name == 'idbd':
        output_grads = eqx.filter_grad(
            lambda m: jax.vmap(m)(images)[0].mean(axis=0).sum()
        )(train_state.model)
        updates, new_optimizer = train_state.optimizer.with_update(
            (grads, output_grads), train_state.model)
    else:
        updates, new_optimizer = train_state.optimizer.with_update(
            grads, train_state.model)
    new_model = eqx.apply_updates(train_state.model, updates)

    # Sync outgoing weights after optimizer update (for custom backward gather)
    if isinstance(new_model, DynamicNetwork):
        new_model = sync_outgoing_weights(new_model)

    # Structure tracker: always update stats
    new_tracker = train_state.structure_tracker.update_stats(
        new_model, param_inputs, grads=grads, updates=updates,
        targets=one_hot.reshape(raw_outputs.shape), predictions=raw_outputs)

    # Restructure (only when do_restructure=True and using ConnectivityManager)
    n_model_layers = new_model.max_layers if hasattr(new_model, 'max_layers') else 0
    pruned_per_layer = jnp.zeros(n_model_layers, dtype=jnp.int32)
    generated_per_layer = jnp.zeros(n_model_layers, dtype=jnp.int32)
    if do_restructure and isinstance(new_tracker, ConnectivityManagerBase):
        rng, restructure_rng = jax.random.split(train_state.rng)
        new_tracker, new_model, new_optimizer, pruned_per_layer, generated_per_layer = (
            new_tracker.modify_structure(new_model, new_optimizer, rng=restructure_rng))

    new_state = tree_replace(
        train_state,
        model=new_model,
        optimizer=new_optimizer,
        structure_tracker=new_tracker,
        step=train_state.step + 1,
    )

    return new_state, StepMetrics(
        loss=loss, correct=correct,
        pruned_per_layer=pruned_per_layer,
        generated_per_layer=generated_per_layer,
    )


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _eval_forward(model, images, labels, loss_fn_impl, num_classes, n_tasks):
    """Evaluate model on a batch. Designed to be vmapped over seeds."""
    outputs, _ = jax.vmap(model)(images)
    if n_tasks > 1:
        one_hot = jax.nn.one_hot(labels, num_classes)
        outputs_r = outputs.reshape(-1, n_tasks, num_classes)
        loss = loss_fn_impl(outputs_r, one_hot)
        predicted = jnp.argmax(outputs_r, axis=-1)
        correct = (predicted == labels).astype(jnp.float32).mean()
    else:
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = loss_fn_impl(outputs, one_hot)
        predicted = jnp.argmax(outputs, axis=-1)
        correct = (predicted == labels).astype(jnp.float32).mean()
    return loss, correct


def evaluate_test(
    batched_model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    loss_fn_impl,
    num_classes: int,
    n_tasks: int,
    batch_size: int = 512,
):
    """Evaluate batched model on test set, chunked to manage memory.

    Args:
        batched_model: Model vmapped over seeds (leading dimension = n_seeds).
        test_images: (N_test, input_dim)
        test_labels: (N_test,) or (N_test, K)
        loss_fn_impl: Loss function.
        num_classes: Classes per task.
        n_tasks: Number of parallel tasks.
        batch_size: Chunk size for test evaluation.

    Returns:
        per_seed_loss: (n_seeds,)
        per_seed_acc: (n_seeds,)
    """
    # Build jitted vmapped eval for a single chunk
    @jax.jit
    def _eval_chunk(model, imgs, lbls):
        return jax.vmap(
            lambda m: _eval_forward(m, imgs, lbls, loss_fn_impl, num_classes, n_tasks)
        )(model)

    n_test = test_images.shape[0]
    total_loss = None
    total_acc = None
    n_chunks = 0

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        chunk_imgs = jnp.array(test_images[start:end])
        chunk_lbls = jnp.array(test_labels[start:end])
        chunk_loss, chunk_acc = _eval_chunk(batched_model, chunk_imgs, chunk_lbls)
        if total_loss is None:
            total_loss = chunk_loss
            total_acc = chunk_acc
        else:
            total_loss = total_loss + chunk_loss
            total_acc = total_acc + chunk_acc
        n_chunks += 1

    return total_loss / n_chunks, total_acc / n_chunks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: DictConfig,
    train_state: TrainState,
    streams: list,
    num_classes: int,
    n_tasks: int,
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[TrainState, list, list, list, list, list, list]:
    """Outer training loop: pre-sample data on CPU, train on GPU via vmapped scan.

    When structure_tracker is enabled, the scan body uses a Python for-loop
    that unrolls prune_frequency steps: (prune_frequency - 1) normal steps
    followed by 1 restructure step. This avoids a runtime conditional inside
    jax.lax.scan — JAX traces two static paths at compile time.

    Returns:
        train_state, all_losses, all_accuracies, all_per_seed_losses,
        all_per_seed_accuracies, all_test_losses, all_test_accuracies
    """
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq
    prune_frequency = cfg.structure_tracker.get('prune_frequency', log_freq)
    use_restructure = cfg.structure_tracker.get('enabled', False)
    loss_name = cfg.train.get('loss', 'mse')
    eval_freq = cfg.train.get('eval_freq', 0)
    loss_fn_impl = LOSS_FNS[loss_name]

    if use_restructure:
        assert log_freq % prune_frequency == 0, \
            f'log_freq ({log_freq}) must be divisible by prune_frequency ({prune_frequency})'
        n_inner_blocks = log_freq // prune_frequency

        def _normal_step(state, data):
            return train_step(
                state, data, do_restructure=False, loss_name=loss_name,
                num_classes=num_classes, n_tasks=n_tasks,
            )

        def _inner_step(state, data_block):
            """One restructure cycle: (prune_frequency-1) normal steps + 1 restructure step.

            data_block: (prune_frequency, batch_size, ...) slice of the data.
            """
            normal_data = (data_block[0][:-1], data_block[1][:-1])
            state, normal_metrics = jax.lax.scan(_normal_step, state, normal_data, unroll=SCAN_UNROLL)

            state, restructure_metrics = train_step(
                state, (data_block[0][-1], data_block[1][-1]),
                do_restructure=True, loss_name=loss_name,
                num_classes=num_classes, n_tasks=n_tasks,
            )
            stacked = jax.tree.map(
                lambda a, b: jnp.concatenate([a, b[None]]),
                normal_metrics, restructure_metrics,
            )
            return state, stacked

        def scan_steps(state, data):
            # data: (log_freq, batch_size, ...) — reshape into (n_inner_blocks, prune_frequency, ...)
            images, labels = data
            images = images.reshape(n_inner_blocks, prune_frequency, *images.shape[1:])
            labels = labels.reshape(n_inner_blocks, prune_frequency, *labels.shape[1:])
            state, metrics = jax.lax.scan(_inner_step, state, (images, labels))
            # metrics leaves have shape (n_inner_blocks, prune_frequency) — flatten
            metrics = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), metrics)
            return state, metrics
    else:
        def _step(state, data):
            return train_step(
                state, data, loss_name=loss_name,
                num_classes=num_classes, n_tasks=n_tasks,
            )

        def scan_steps(state, data):
            return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_accuracies = []
    all_per_seed_losses = []
    all_per_seed_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []

    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    is_parallel_mnist = cfg.dataset.name == 'parallel_mnist'

    for scan_idx in range(num_scans):
        # Pre-sample one cycle of data on CPU per seed
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))  # (n_seeds, log_freq, batch_size, input_dim)
        labels = jnp.array(np.stack([b[1] for b in batch]))  # (n_seeds, log_freq, batch_size[, K])

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        # metrics.loss / metrics.correct: (n_seeds, log_freq)
        per_seed_loss = metrics.loss.mean(axis=1)  # (n_seeds,)
        per_seed_acc = metrics.correct.mean(axis=1)  # (n_seeds,)
        mean_loss = float(per_seed_loss.mean())
        mean_acc = float(per_seed_acc.mean())
        std_loss = float(per_seed_loss.std())
        std_acc = float(per_seed_acc.std())

        step = int(train_state.step[0].item())

        # Structure metrics (computed outside JIT, snapshot of current state)
        structure_metrics = compute_structure_metrics(train_state)

        # Prune/gen counts: sum over log_freq steps, average across seeds
        # metrics.pruned_per_layer: (n_seeds, log_freq, max_layers)
        if metrics.pruned_per_layer.size > 0:
            pruned = np.array(metrics.pruned_per_layer.sum(axis=1))  # (n_seeds, max_layers)
            generated = np.array(metrics.generated_per_layer.sum(axis=1))
            n_layers = pruned.shape[1]
            structure_metrics['total_pruned'] = float(pruned.sum(axis=1).mean())
            structure_metrics['total_generated'] = float(generated.sum(axis=1).mean())
            for l in range(n_layers):
                structure_metrics[f'layer_{l}/pruned'] = float(pruned[:, l].mean())
                structure_metrics[f'layer_{l}/generated'] = float(generated[:, l].mean())

        # --- Test evaluation ---
        test_metrics_dict = {}
        if eval_freq > 0 and step % eval_freq == 0:
            if is_parallel_mnist:
                # Get test batch from the first stream (permutations are per-seed
                # but we use the first seed's permutation state for simplicity)
                t_imgs, t_lbls = streams[0].get_test_batch()
            else:
                t_imgs, t_lbls = test_data

            test_loss, test_acc = evaluate_test(
                train_state.model, t_imgs, t_lbls,
                loss_fn_impl, num_classes, n_tasks,
            )
            mean_test_loss = float(test_loss.mean())
            mean_test_acc = float(test_acc.mean())
            all_test_losses.append(mean_test_loss)
            all_test_accuracies.append(mean_test_acc)
            test_metrics_dict = {
                'test_loss': mean_test_loss,
                'test_accuracy': mean_test_acc,
            }

        # Background logging
        if logging_active:
            def _log_step(mean_loss, std_loss, mean_acc, std_acc,
                          per_seed_loss, per_seed_acc, structure_metrics,
                          test_metrics_dict, step):
                base_metrics = {
                    'loss': mean_loss,
                    'loss_std': std_loss,
                    'accuracy': mean_acc,
                    'accuracy_std': std_acc,
                }
                base_metrics.update(structure_metrics)
                base_metrics.update(test_metrics_dict)
                log_metrics(base_metrics, cfg, step=step)
                log_child_metrics(
                    {'loss': per_seed_loss, 'accuracy': per_seed_acc},
                    cfg, step=step,
                )

            log_futures.append(log_executor.submit(
                _log_step, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(), structure_metrics,
                test_metrics_dict, step,
            ))

        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accuracies.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        postfix = {'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'}
        if test_metrics_dict:
            postfix['t_loss'] = f'{test_metrics_dict["test_loss"]:.4f}'
            postfix['t_acc'] = f'{test_metrics_dict["test_accuracy"]:.4f}'
        pbar.set_postfix(postfix)

    # Wait for all logging to finish
    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)

    pbar.close()
    return (train_state, all_losses, all_accuracies,
            all_per_seed_losses, all_per_seed_accuracies,
            all_test_losses, all_test_accuracies)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    # Normalize seeds
    if cfg.seed is None:
        cfg.seed = [np.random.randint(0, 1_000_000_000)]
    elif isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    if cfg.get('log_individual_seeds', False) and not cfg.get('mlflow', False):
        raise ValueError(
            'log_individual_seeds requires mlflow logging. '
            'Set mlflow=true or disable log_individual_seeds.')

    set_seed(cfg.seed[0])
    init_child_runs(cfg.seed, cfg)

    train_state, streams, n_params, num_classes, n_tasks, test_data = prepare_experiment(cfg)

    (train_state, all_losses, all_accuracies,
     all_per_seed_losses, all_per_seed_accuracies,
     all_test_losses, all_test_accuracies) = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks, test_data)

    # Final summary
    average_loss = float(np.mean(all_losses))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))
    asymptotic_accuracy = float(np.mean(all_accuracies[-n_tail:]))

    print(f'Average loss: {average_loss:.4f}')
    print(f'Asymptotic loss: {asymptotic_loss:.4f}')
    print(f'Asymptotic accuracy: {asymptotic_accuracy:.4f}')

    summary = {
        'average_loss': average_loss,
        'asymptotic_loss': asymptotic_loss,
        'asymptotic_accuracy': asymptotic_accuracy,
        'num_params': n_params,
    }

    if all_test_losses:
        n_test_tail = max(1, len(all_test_losses) // 10)
        summary['asymptotic_test_loss'] = float(np.mean(all_test_losses[-n_test_tail:]))
        summary['asymptotic_test_accuracy'] = float(np.mean(all_test_accuracies[-n_test_tail:]))
        print(f'Asymptotic test loss: {summary["asymptotic_test_loss"]:.4f}')
        print(f'Asymptotic test accuracy: {summary["asymptotic_test_accuracy"]:.4f}')

    log_metrics(summary, cfg)

    # Per-seed summary to child runs
    if all_per_seed_losses:
        per_seed_losses = np.stack(all_per_seed_losses)  # (num_scans, n_seeds)
        per_seed_accs = np.stack(all_per_seed_accuracies)
        log_child_metrics({
            'average_loss': per_seed_losses.mean(axis=0).tolist(),
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
            'num_params': [n_params] * len(cfg.seed),
        }, cfg)

    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
