from typing import Callable, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import set_seed
from phd.jax_core.utils import configure_jax, count_params, tree_replace
from phd.research_utils.logging import init_experiment, log_metrics, finish_experiment


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, key: random.PRNGKey):
        k1, k2 = random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)


class TrainState(eqx.Module):
    # Static
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    # Dynamic
    model: MLP
    optimizer_state: optax.OptState
    rng: random.PRNGKey
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def prepare_experiment(cfg: DictConfig) -> TrainState:
    """Initialize model, optimizer, and training state."""
    set_seed(cfg.seed)
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 1_000_000_000)
    key = random.PRNGKey(seed)

    key, model_key = random.split(key)
    model = MLP(in_dim=1, hidden_dim=cfg.model.hidden_dim, out_dim=1, key=model_key)

    optimizer = optax.sgd(cfg.optimizer.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    return TrainState(
        optimizer=optimizer,
        batch_size=cfg.train.batch_size,
        model=model,
        optimizer_state=optimizer_state,
        rng=key,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(train_state: TrainState, _) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan."""
    key, batch_key = random.split(train_state.rng)

    # Sample batch
    x = random.uniform(batch_key, (train_state.batch_size, 1), minval=-jnp.pi, maxval=jnp.pi)
    y = jnp.sin(x)

    # Forward + backward
    def loss_fn(model):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)

    # Apply gradients
    updates, new_opt_state = train_state.optimizer.update(grads, train_state.optimizer_state)
    new_model = eqx.apply_updates(train_state.model, updates)

    new_state = tree_replace(
        train_state,
        model=new_model,
        optimizer_state=new_opt_state,
        rng=key,
        step=train_state.step + 1,
    )
    return new_state, StepMetrics(loss=loss)


def train_multi_step(train_state: TrainState, n_steps: int) -> Tuple[TrainState, StepMetrics]:
    """Run `n_steps` of training via a single jax.lax.scan.

    Returns the final state and per-step metrics stacked along a leading axis
    of length `n_steps`.
    """
    return jax.lax.scan(train_step, train_state, length=n_steps, unroll=SCAN_UNROLL)


def compute_metrics(step_metrics: StepMetrics) -> dict:
    """Aggregate per-step metrics from a logged cycle into scalars."""
    return {
        'loss': float(step_metrics.loss.mean()),
        'loss_std': float(step_metrics.loss.std()),
    }


def run_experiment(cfg: DictConfig, train_state: TrainState, train_fn: Callable) -> TrainState:
    """Outer Python loop calling `train_fn` once per logged cycle."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    for _ in range(num_scans):
        train_state, step_metrics = train_fn(train_state, log_freq)

        step = int(train_state.step)
        metrics = compute_metrics(step_metrics)
        log_metrics(metrics, cfg, step=step)

        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{metrics["loss"]:.4f}'})

    pbar.close()
    return train_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    try:
        train_state = prepare_experiment(cfg)
        train_fn = jax.jit(train_multi_step, static_argnames=('n_steps',))
        train_state = run_experiment(cfg, train_state, train_fn)
    finally:
        finish_experiment(cfg)

    print('Run complete!')


if __name__ == '__main__':
    main()
