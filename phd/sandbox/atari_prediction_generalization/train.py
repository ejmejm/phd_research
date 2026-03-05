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


def prepare_experiment(cfg: DictConfig) -> Tuple[TrainState, ContinualAtariStream, int]:
    """Initialize per-seed models, optimizers, and data stream."""
    seeds = cfg.seed
    input_dim = compute_input_dim(cfg)

    stream = load_atari_data(cfg)

    train_states = []
    for seed in seeds:
        rng = jax.random.key(seed)

        model = create_model(cfg, input_dim, key=rng_from_string(rng, 'model'))

        # ResNet has no .layers/.n_frozen_layers; pass a filter_spec
        # that marks all array leaves as trainable.
        if cfg.model.get('type', 'mlp') == 'resnet':
            filter_spec = jax.tree.map(eqx.is_array, model)
        else:
            filter_spec = None
        optimizer = prepare_optimizer(
            model, cfg.optimizer.name, cfg.optimizer, filter_spec=filter_spec)

        train_states.append(TrainState(
            model=model,
            optimizer=optimizer,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

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
    observations, targets = data  # (input_dim,) or (C, H, W), (1,)

    def loss_fn(model):
        predictions, _ = model(observations)  # (1,)
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
) -> Tuple[TrainState, list, list, list]:
    """Outer training loop with background data preloading."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

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

        obs_jax = jnp.array(obs)
        returns_jax = jnp.array(returns)

        # Data is shared across seeds (not vmapped) — only train_state is vmapped
        train_state, metrics = vmapped_scan(
            train_state, (obs_jax, returns_jax))

        # metrics.loss: (n_seeds, log_freq)
        per_seed_loss = metrics.loss.mean(axis=1)  # (n_seeds,)
        mean_loss = float(per_seed_loss.mean())
        step = int(train_state.step[0].item())

        # Background logging
        if parent_run_id:
            def _log_step(mean_loss, per_seed_loss, step):
                mlflow_client.log_metric(parent_run_id, 'loss', mean_loss, step=step)
                log_child_metrics({'loss': per_seed_loss}, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log_step, mean_loss, per_seed_loss.tolist(), step))

        all_losses.append(mean_loss)
        all_per_seed_losses.append(np.array(per_seed_loss))

        # Show current game in progress bar
        game_name = stream.dataset.get_game_name(
            min(stream.current_step, stream.total_steps - 1))
        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'game': game_name})

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
