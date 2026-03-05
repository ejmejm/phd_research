from concurrent.futures import ThreadPoolExecutor
import os
from typing import Tuple

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
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer,
    set_seed,
    rng_from_string,
)
from phd.jax_core.models import MLP
from phd.jax_core.optimizers import EqxOptimizer
from phd.sandbox.atari_prediction_generalization.resnet import ResNet
from phd.jax_core.utils import tree_replace
from phd.research_utils.logging import (
    init_experiment,
    init_child_runs,
    log_metrics,
    log_child_metrics,
    finish_child_runs,
    finish_experiment,
)
from phd.sandbox.atari_prediction_generalization.data import (
    BackgroundDataLoader,
    ContinualAtariStream,
    compute_input_dim,
    load_atari_data,
)


SCAN_UNROLL = 4
OUTPUT_DIM = 1  # Predicting scalar return


# ---------------------------------------------------------------------------
# Train state and metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: eqx.Module
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray


class StepMetrics(eqx.Module):
    loss: jax.Array


# ---------------------------------------------------------------------------
# JAX configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def create_model(cfg: DictConfig, input_dim: int, *, key: PRNGKeyArray) -> eqx.Module:
    """Create a model based on config."""
    model_type = cfg.model.get('type', 'mlp')
    if model_type == 'mlp':
        return MLP(
            input_dim=input_dim,
            output_dim=OUTPUT_DIM,
            n_layers=cfg.model.n_layers,
            hidden_dim=cfg.model.hidden_dim,
            weight_init_method=cfg.model.weight_init_method,
            activation=cfg.model.activation,
            n_frozen_layers=cfg.model.get('n_frozen_layers', 0),
            key=key,
        )
    elif model_type == 'resnet':
        pp = cfg.preprocessing
        return ResNet(
            in_channels=pp.frame_stack if pp.grayscale else pp.frame_stack * 3,
            output_dim=OUTPUT_DIM,
            width_scale=cfg.model.width_scale,
            n_conv_sequences=cfg.model.n_conv_sequences,
            key=key,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _create_train_state(cfg: DictConfig, input_dim: int, rng: PRNGKeyArray,
                        step: jax.Array = None) -> TrainState:
    """Create a single seed's TrainState with fresh model and optimizer."""
    rng, model_key = jax.random.split(rng)
    model = create_model(cfg, input_dim, key=model_key)

    if cfg.model.get('type', 'mlp') == 'resnet':
        filter_spec = jax.tree.map(eqx.is_array, model)
    else:
        filter_spec = None
    optimizer = prepare_optimizer(
        model, cfg.optimizer.name, cfg.optimizer, filter_spec=filter_spec)

    return TrainState(
        model=model,
        optimizer=optimizer,
        step=step if step is not None else jnp.array(0),
        rng=rng,
    )


def reinit_train_state(train_state: TrainState, cfg: DictConfig,
                       input_dim: int) -> TrainState:
    """Reinitialize model weights and optimizer state for all seeds.

    Uses each seed's current RNG to generate new weights, so each reinit
    produces different (but deterministic) parameters.
    """
    n_seeds = train_state.step.shape[0]
    new_states = []
    for i in range(n_seeds):
        new_states.append(_create_train_state(
            cfg, input_dim, train_state.rng[i], step=train_state.step[i]))
    return stack_pytrees(new_states)


def prepare_experiment(cfg: DictConfig) -> Tuple[TrainState, ContinualAtariStream, int]:
    """Initialize per-seed models, optimizers, and data stream."""
    seeds = cfg.seed
    input_dim = compute_input_dim(cfg)

    stream = load_atari_data(cfg)

    train_states = []
    for seed in seeds:
        rng = rng_from_string(jax.random.key(seed), 'train')
        train_states.append(_create_train_state(cfg, input_dim, rng))

    n_params = count_params(train_states[0].model)
    model_type = cfg.model.get('type', 'mlp')
    print(f'Model: {model_type}, Params: {n_params}, Seeds: {seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, stream, n_params


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(train_state: TrainState, data) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan."""
    observations, targets = data  # (batch_size, C, H, W), (batch_size, 1)

    def loss_fn(model):
        predictions = jax.vmap(lambda x: model(x)[0])(observations)  # (batch_size, 1)
        loss = jnp.mean(jnp.square(predictions - targets))
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)

    updates, new_optimizer = train_state.optimizer.with_update(
        grads, train_state.model)
    new_model = eqx.apply_updates(train_state.model, updates)

    new_state = tree_replace(
        train_state,
        model=new_model,
        optimizer=new_optimizer,
        step=train_state.step + 1,
    )

    return new_state, StepMetrics(loss=loss)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: DictConfig,
    train_state: TrainState,
    stream: ContinualAtariStream,
) -> Tuple[TrainState, list, list]:
    """Outer training loop with background data preloading."""
    log_freq = cfg.train.log_freq
    batch_size = cfg.train.get('batch_size', 1)
    updates_per_scan = log_freq // batch_size
    num_scans = cfg.train.total_steps // log_freq
    reinit = cfg.train.get('reinit_at_game_boundary', False)
    steps_per_game = cfg.dataset.steps_per_game
    input_dim = compute_input_dim(cfg)

    if reinit:
        assert steps_per_game % log_freq == 0, (
            f"steps_per_game ({steps_per_game}) must be divisible by "
            f"log_freq ({log_freq}) when reinit_at_game_boundary is enabled"
        )

    def scan_steps(state, data):
        return jax.lax.scan(train_step, state, data, unroll=SCAN_UNROLL)

    # vmap over train_state (axis 0) but not over data (None) — all seeds
    # share the same observation/return sequence
    vmapped_scan = jax.jit(jax.vmap(scan_steps, in_axes=(0, None)))

    # Background data loading
    loader = BackgroundDataLoader(stream, chunk_size=log_freq)
    loader.start_preload()

    all_losses = []
    all_per_seed_losses = []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []

    # Capture parent run ID so background thread can log to the correct run
    import mlflow
    parent_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    mlflow_client = mlflow.tracking.MlflowClient() if parent_run_id else None

    for _ in range(num_scans):
        # Get preloaded batch (next preload starts automatically)
        obs, returns = loader.get()

        # Reshape into (n_updates, batch_size, ...) for scan over batched steps
        obs_jax = jnp.array(obs).reshape(updates_per_scan, batch_size, *obs.shape[1:])
        returns_jax = jnp.array(returns).reshape(updates_per_scan, batch_size, *returns.shape[1:])

        # Data is shared across seeds (not vmapped) — only train_state is vmapped
        train_state, metrics = vmapped_scan(
            train_state, (obs_jax, returns_jax))

        # metrics.loss: (n_seeds, updates_per_scan)
        per_seed_loss = metrics.loss.mean(axis=1)  # (n_seeds,)
        mean_loss = float(per_seed_loss.mean())
        update_step = int(train_state.step[0].item())
        env_step = update_step * batch_size

        # Background logging
        if parent_run_id:
            def _log_step(mean_loss, per_seed_loss, env_step, update_step):
                mlflow_client.log_metric(parent_run_id, 'loss', mean_loss, step=env_step)
                mlflow_client.log_metric(parent_run_id, 'update_step', update_step, step=env_step)
                log_child_metrics({'loss': per_seed_loss}, cfg, step=env_step)

            log_futures.append(log_executor.submit(
                _log_step, mean_loss, per_seed_loss.tolist(), env_step, update_step))

        all_losses.append(mean_loss)
        all_per_seed_losses.append(np.array(per_seed_loss))

        # Show current game in progress bar
        game_name = stream.dataset.get_game_name(
            min(stream.current_step, stream.total_steps - 1))
        pbar.update(log_freq)
        pbar.set_postfix({
            'loss': f'{mean_loss:.4f}', 'game': game_name,
        })

        # Reinitialize at game boundaries if enabled
        if (reinit
                and stream.current_step % steps_per_game == 0
                and stream.current_step < stream.total_steps):
            print(f'\nReinitializing network at step {stream.current_step} '
                  f'(game boundary)')
            train_state = reinit_train_state(train_state, cfg, input_dim)

    # Wait for all logging to finish
    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    loader.shutdown()

    pbar.close()
    return train_state, all_losses, all_per_seed_losses


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
    init_child_runs(cfg.seed, cfg)

    train_state, stream, n_params = prepare_experiment(cfg)

    train_state, all_losses, all_per_seed_losses = run_experiment(
        cfg, train_state, stream)

    # Final summary
    average_loss = float(np.mean(all_losses))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))

    print(f'Average loss: {average_loss:.4f}')
    print(f'Asymptotic loss: {asymptotic_loss:.4f}')

    log_metrics({
        'average_loss': average_loss,
        'asymptotic_loss': asymptotic_loss,
        'num_params': n_params,
    }, cfg)

    # Per-seed summary to child runs
    if all_per_seed_losses:
        per_seed_losses = np.stack(all_per_seed_losses)  # (num_scans, n_seeds)
        log_child_metrics({
            'average_loss': per_seed_losses.mean(axis=0).tolist(),
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'num_params': [n_params] * len(cfg.seed),
        }, cfg)

    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
