from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Tuple, Union

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
from phd.structure_search.data import load_dataset, DataStream
from phd.structure_search.connectivity_manager import ConnectivityManager, full_input_generate
from phd.structure_search.dynamic_network import (
    DynamicNetwork, sync_outgoing_weights, build_outgoing_indices,
    init_random_dynamic_network,
    count_active_connections, count_active_units,
)
from phd.structure_search.metrics import StepMetrics, compute_structure_metrics


SCAN_UNROLL = 4
NUM_CLASSES = 10  # CIFAR-10


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

    def update_stats(self, model, param_inputs):
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
    model: Union[MLP, DynamicNetwork]
    optimizer: EqxOptimizer
    structure_tracker: Union[DummyStructureTracker, ConnectivityManager]
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
) -> Tuple[TrainState, List[DataStream], int]:
    """Initialize per-seed models, optimizers, data streams."""
    seeds = cfg.seed
    model_type = cfg.model.get('type', 'mlp')

    images, labels, num_classes, input_dim = load_dataset(cfg.dataset.name)

    use_bias = cfg.model.get('use_bias', False)

    streams = []
    train_states = []
    for seed in seeds:
        rng = jax.random.key(seed)

        streams.append(DataStream(
            images=images,
            labels=labels,
            num_classes=num_classes,
            batch_size=cfg.train.batch_size,
            seed=seed,
        ))

        model_key = rng_from_string(rng, 'model')

        if model_type == 'dynamic':
            model = init_random_dynamic_network(
                input_dim=input_dim,
                output_dim=num_classes,
                n_layers=cfg.model.n_layers,
                units_per_layer=cfg.model.hidden_dim,
                max_units_per_layer=cfg.model.get('max_units_per_layer', None),
                max_connections_per_unit=cfg.model.get('max_connections_per_unit', None),
                activations=(cfg.model.activation,),
                max_fan_out=cfg.model.get('max_fan_out', None),
                connect_all_to_output=cfg.model.get('connect_all_to_output', False),
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
        else:
            model = MLP(
                input_dim=input_dim,
                output_dim=num_classes,
                n_layers=cfg.model.n_layers,
                hidden_dim=cfg.model.hidden_dim + int(use_bias),
                weight_init_method=cfg.model.weight_init_method,
                activation=cfg.model.activation,
                n_frozen_layers=cfg.model.get('n_frozen_layers', 0),
                key=model_key,
            )
            optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)

        if model_type == 'dynamic' and cfg.structure_tracker.get('enabled', False):
            generate_strategy = cfg.structure_tracker.get('generate_strategy', 'random')
            generate_fn = full_input_generate if generate_strategy == 'full_input' else None
            tracker = ConnectivityManager(
                model=model,
                prune_rate=cfg.structure_tracker.prune_rate,
                connection_budget=cfg.structure_tracker.connection_budget,
                decay_rate=cfg.structure_tracker.decay_rate,
                maturity_threshold=cfg.structure_tracker.maturity_threshold,
                generate_fn=generate_fn,
                rng=rng_from_string(rng, 'tracker'),
            )
        else:
            tracker = DummyStructureTracker(rng=rng_from_string(rng, 'tracker'))

        train_states.append(TrainState(
            model=model,
            optimizer=optimizer,
            structure_tracker=tracker,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    if model_type == 'dynamic':
        net = train_states[0].model
        n_units = count_active_units(net)
        n_conns = count_active_connections(net)
        print(f'Model: DynamicNetwork, Params: {n_params}, '
              f'Units: {n_units}, Connections: {n_conns}, Seeds: {seeds}')
    else:
        print(f'Model: MLP, Params: {n_params}, Seeds: {seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params


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
) -> Tuple[TrainState, StepMetrics]:
    """Single training step.

    Args:
        train_state: Current training state.
        data: Tuple of (images, labels).
        do_restructure: If True and structure_tracker is a ConnectivityManager,
            call modify_structure after the utility update. This parameter is
            static (Python bool) so JAX traces separate paths for True/False.
        loss_name: Loss function to use ('mse' or 'cross_entropy'). Static.
    """
    images, labels = data  # (batch_size, input_dim), (batch_size,)

    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)  # (batch_size, 10)
    loss_fn_impl = LOSS_FNS[loss_name]

    def loss_fn(model):
        outputs, param_inputs = jax.vmap(model)(images)  # (batch_size, 10)
        loss = loss_fn_impl(outputs, one_hot)
        return loss, (outputs, param_inputs)

    (loss, (outputs, param_inputs)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(train_state.model)

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
        new_model, param_inputs)

    # Restructure (only when do_restructure=True and using ConnectivityManager)
    n_layers = new_model.max_layers if hasattr(new_model, 'max_layers') else 0
    pruned_per_layer = jnp.zeros(n_layers, dtype=jnp.int32)
    generated_per_layer = jnp.zeros(n_layers, dtype=jnp.int32)
    if do_restructure and isinstance(new_tracker, ConnectivityManager):
        rng, restructure_rng = jax.random.split(train_state.rng)
        new_tracker, new_model, new_optimizer, pruned_per_layer, generated_per_layer = (
            new_tracker.modify_structure(new_model, new_optimizer, rng=restructure_rng))

    # Accuracy (pre-update predictions)
    predicted = jnp.argmax(outputs, axis=-1)  # (batch_size,)
    correct = (predicted == labels).astype(jnp.float32).mean()

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
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: DictConfig,
    train_state: TrainState,
    streams: List[DataStream],
) -> Tuple[TrainState, list, list, list, list]:
    """Outer training loop: pre-sample data on CPU, train on GPU via vmapped scan.

    When structure_tracker is enabled, the scan body uses a Python for-loop
    that unrolls prune_frequency steps: (prune_frequency - 1) normal steps
    followed by 1 restructure step. This avoids a runtime conditional inside
    jax.lax.scan — JAX traces two static paths at compile time.
    """
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq
    prune_frequency = cfg.structure_tracker.get('prune_frequency', log_freq)
    use_restructure = cfg.structure_tracker.get('enabled', False)
    loss_name = cfg.train.get('loss', 'mse')

    if use_restructure:
        assert log_freq % prune_frequency == 0, \
            f'log_freq ({log_freq}) must be divisible by prune_frequency ({prune_frequency})'
        n_inner_blocks = log_freq // prune_frequency

        def _inner_step(state, data_block):
            """One restructure cycle: (prune_frequency-1) normal steps + 1 restructure step.

            data_block: (prune_frequency, batch_size, ...) slice of the data.
            """
            all_metrics = []
            for i in range(prune_frequency - 1):
                state, step_metrics = train_step(state, (data_block[0][i], data_block[1][i]), do_restructure=False, loss_name=loss_name)
                all_metrics.append(step_metrics)
            state, step_metrics = train_step(state, (data_block[0][-1], data_block[1][-1]), do_restructure=True, loss_name=loss_name)
            all_metrics.append(step_metrics)
            stacked = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics)
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
            return train_step(state, data, loss_name=loss_name)

        def scan_steps(state, data):
            return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_accuracies = []
    all_per_seed_losses = []
    all_per_seed_accuracies = []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []

    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for _ in range(num_scans):
        # Pre-sample one cycle of data on CPU per seed
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))  # (n_seeds, log_freq, batch_size, input_dim)
        labels = jnp.array(np.stack([b[1] for b in batch]))  # (n_seeds, log_freq, batch_size)

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

        # Background logging
        if logging_active:
            def _log_step(mean_loss, std_loss, mean_acc, std_acc,
                          per_seed_loss, per_seed_acc, structure_metrics, step):
                base_metrics = {
                    'loss': mean_loss,
                    'loss_std': std_loss,
                    'accuracy': mean_acc,
                    'accuracy_std': std_acc,
                }
                base_metrics.update(structure_metrics)
                log_metrics(base_metrics, cfg, step=step)
                log_child_metrics(
                    {'loss': per_seed_loss, 'accuracy': per_seed_acc},
                    cfg, step=step,
                )

            log_futures.append(log_executor.submit(
                _log_step, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(), structure_metrics, step,
            ))

        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accuracies.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    # Wait for all logging to finish
    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)

    pbar.close()
    return train_state, all_losses, all_accuracies, all_per_seed_losses, all_per_seed_accuracies


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

    train_state, streams, n_params = prepare_experiment(cfg)

    (train_state, all_losses, all_accuracies,
     all_per_seed_losses, all_per_seed_accuracies) = run_experiment(
        cfg, train_state, streams)

    # Final summary
    average_loss = float(np.mean(all_losses))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))
    asymptotic_accuracy = float(np.mean(all_accuracies[-n_tail:]))

    print(f'Average loss: {average_loss:.4f}')
    print(f'Asymptotic loss: {asymptotic_loss:.4f}')
    print(f'Asymptotic accuracy: {asymptotic_accuracy:.4f}')

    log_metrics({
        'average_loss': average_loss,
        'asymptotic_loss': asymptotic_loss,
        'asymptotic_accuracy': asymptotic_accuracy,
        'num_params': n_params,
    }, cfg)

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
