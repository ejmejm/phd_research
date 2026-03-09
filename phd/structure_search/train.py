from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import List, Tuple

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
from phd.structure_search.dynamic_network import (
    DynamicNetwork, sync_outgoing_weights, init_random_dynamic_network,
    count_active_connections, count_active_units,
)


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
    model: MLP
    optimizer: EqxOptimizer
    structure_tracker: DummyStructureTracker
    step: jax.Array
    rng: PRNGKeyArray


class StepMetrics(eqx.Module):
    loss: jax.Array
    correct: jax.Array


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _make_dynamic_filter_spec(model: DynamicNetwork):
    """Build optimizer filter_spec for DynamicNetwork: only weights and output_weights."""
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda n: (n.weights, n.output_weights), spec, (True, True),
    )


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

def train_step(train_state: TrainState, data) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan."""
    images, labels = data  # (batch_size, input_dim), (batch_size,)

    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)  # (batch_size, 10)

    def loss_fn(model):
        outputs, param_inputs = jax.vmap(model)(images)  # (batch_size, 10)
        loss = jnp.mean(jnp.square(outputs - one_hot))
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

    # Structure tracker (no-op for now)
    new_tracker = train_state.structure_tracker.update_stats(
        new_model, param_inputs)

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

    return new_state, StepMetrics(loss=loss, correct=correct)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: DictConfig,
    train_state: TrainState,
    streams: List[DataStream],
) -> Tuple[TrainState, list, list, list, list]:
    """Outer training loop: pre-sample data on CPU, train on GPU via vmapped scan."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    def scan_steps(state, data):
        return jax.lax.scan(train_step, state, data, unroll=SCAN_UNROLL)

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

        # Background logging
        if logging_active:
            def _log_step(mean_loss, std_loss, mean_acc, std_acc,
                          per_seed_loss, per_seed_acc, step):
                log_metrics({
                    'loss': mean_loss,
                    'loss_std': std_loss,
                    'accuracy': mean_acc,
                    'accuracy_std': std_acc,
                }, cfg, step=step)
                log_child_metrics(
                    {'loss': per_seed_loss, 'accuracy': per_seed_acc},
                    cfg, step=step,
                )

            log_futures.append(log_executor.submit(
                _log_step, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(), step,
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
