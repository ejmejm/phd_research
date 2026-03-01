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
    log_metrics,
    finish_experiment,
)
from phd.structure_search.data import load_dataset, DataStream


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
# JAX configuration
# ---------------------------------------------------------------------------

def configure_jax(cfg: DictConfig):
    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update(
        'jax_persistent_cache_enable_xla_caches',
        'xla_gpu_per_fusion_autotune_cache_dir',
    )
    jax.config.update('jax_platform_name', cfg.device)
    print(f'JAX device: {jax.devices(cfg.device)[0]}')


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree.leaves(params))


def stack_pytrees(pytrees):
    treedef = jax.tree.structure(pytrees[0])
    all_leaves = [jax.tree.leaves(pt) for pt in pytrees]
    stacked = [jnp.stack(xs) for xs in zip(*all_leaves)]
    return jax.tree.unflatten(treedef, stacked)


def prepare_experiment(
    cfg: DictConfig,
) -> Tuple[TrainState, List[DataStream], int]:
    """Initialize per-seed models, optimizers, data streams."""
    seeds = cfg.seed

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

        model = MLP(
            input_dim=input_dim,
            output_dim=num_classes,
            n_layers=cfg.model.n_layers,
            hidden_dim=cfg.model.hidden_dim + int(use_bias),
            weight_init_method=cfg.model.weight_init_method,
            activation=cfg.model.activation,
            n_frozen_layers=cfg.model.get('n_frozen_layers', 0),
            key=rng_from_string(rng, 'model'),
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
        output_grads = jax.grad(
            lambda m: jax.vmap(m)(images)[0].mean(axis=0).sum()
        )(train_state.model)
        updates, new_optimizer = train_state.optimizer.with_update(
            (grads, output_grads), train_state.model)
    else:
        updates, new_optimizer = train_state.optimizer.with_update(
            grads, train_state.model)
    new_model = eqx.apply_updates(train_state.model, updates)

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
) -> Tuple[TrainState, list, list]:
    """Outer training loop: pre-sample data on CPU, train on GPU via vmapped scan."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    def scan_steps(state, data):
        return jax.lax.scan(train_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_accuracies = []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    for _ in range(num_scans):
        # Pre-sample one cycle of data on CPU per seed
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))  # (n_seeds, log_freq, batch_size, input_dim)
        labels = jnp.array(np.stack([b[1] for b in batch]))  # (n_seeds, log_freq, batch_size)

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        # metrics.loss / metrics.correct: (n_seeds, log_freq)
        mean_loss = float(metrics.loss.mean())
        mean_acc = float(metrics.correct.mean())

        step = int(train_state.step[0].item())
        log_metrics({'loss': mean_loss, 'accuracy': mean_acc}, cfg, step=step)

        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)

        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    pbar.close()
    return train_state, all_losses, all_accuracies


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

    set_seed(cfg.seed[0])

    train_state, streams, n_params = prepare_experiment(cfg)

    train_state, all_losses, all_accuracies = run_experiment(
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

    finish_experiment(cfg)


if __name__ == '__main__':
    main()
