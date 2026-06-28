"""JAX/Equinox port of the sequential linear stock-price predictor.

Same experiment as ``linear_predictor.py`` -- an online linear model that, at every
timestep, predicts the closing prices of the loaded stocks from the previous step's
closes, with both inputs and targets standardized by an exponentially-weighted Welford
normalizer (w/ outlier clip).

The difference is purely structural: one timestep is a ``train_step`` compatible with
``jax.lax.scan``, and ``train_multi_step`` scans it over a block of ``log_freq`` steps.
The outer Python loop in ``run_experiment`` runs one such scan per logging threshold,
aggregates the per-step metrics, and logs them -- i.e. the data is streamed through the
scan in cycles of ``log_freq`` steps between log points.

Notes:
  * The model is a full cross-stock linear map (``W`` is ``num_stocks x num_stocks``),
    zero-initialized, trained with plain SGD on the summed (not meaned) squared error so
    each stock's row gets its full gradient -- matching the numpy version.
  * ``optimizer.update`` on loss ``0.5 * sum(err**2)`` reproduces ``W -= lr*outer(err,x)``.
  * Stable ``lr`` scales ~``1/num_stocks`` (the effective LMS step is ``lr*||x||^2``), so the
    default is tuned for the default 200 stocks; raise/lower it with the stock count.
  * x64 is on by default for parity with the float64 numpy version (the normalizer numerics
    matter); turn off with ``x64=false``.
"""

from __future__ import annotations

from collections import deque
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

from data import load_stock_data
from phd.feature_search.jax_core.experiment_helpers import set_seed
from phd.jax_core.utils import configure_jax, count_params, tree_replace
from phd.research_utils.logging import init_experiment, log_metrics, finish_experiment


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Exponentially-weighted Welford normalizer (port of linear_predictor.py)
# ---------------------------------------------------------------------------

class WelfordNormalizer(eqx.Module):
    """Vectorized online normalizer with exponentially-weighted mean/variance.

    Functional JAX port: ``update`` returns a new normalizer rather than mutating in place,
    so it lives inside the scan carry. Guard (``clip``) matches
    ``linear_predictor.WelfordNormalizer`` exactly.
    """

    # Static config
    alpha: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    clip: float = eqx.field(static=True)

    # Dynamic state
    count: jax.Array
    mean: jax.Array
    var: jax.Array  # exponentially-weighted variance

    def __init__(
        self,
        dim: int,
        alpha: float = 0.001,
        eps: float = 1e-8,
        clip: float = 6.0,
    ):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.clip = float(clip)
        self.count = jnp.array(0)
        self.mean = jnp.zeros(dim)
        self.var = jnp.zeros(dim)

    @property
    def std(self) -> jax.Array:
        return jnp.sqrt(self.var)

    def normalize(self, x: jax.Array) -> jax.Array:
        std = self.std
        z = (x - self.mean) / (std + self.eps)
        # Clip to a one-in-a-billion N(0, 1) tail to avoid outlier problems due to innacurate variance estimates.
        z = jnp.clip(z, -self.clip, self.clip)
        return z

    def denormalize(self, z: jax.Array) -> jax.Array:
        return z * (self.std + self.eps) + self.mean

    def update(self, x: jax.Array) -> "WelfordNormalizer":
        """Exponentially-weighted Welford update."""
        count = self.count + 1
        a = jnp.maximum(self.alpha, 1.0 / count)  # equal-weight until ~1/alpha samples seen
        delta = x - self.mean
        incr = a * delta
        mean = self.mean + incr
        var = (1.0 - a) * (self.var + delta * incr)
        return tree_replace(self, count=count, mean=mean, var=var)


# ---------------------------------------------------------------------------
# Agent state: produces the model's input features from the persistent state
# ---------------------------------------------------------------------------

def features_from_state(state: jax.Array) -> jax.Array:
    """Map the persistent state to the model's input features (identity, NaN-safe)."""
    return jnp.nan_to_num(state, nan=0.0)


def agent_state_update(state: jax.Array, close_t: jax.Array) -> jax.Array:
    """Update the persistent state at the end of a step. Starter: store this step's close,
    so next step's input is this step's close (predict close_t from close_{t-1})."""
    return close_t


# ---------------------------------------------------------------------------
# Train state / metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    # Static
    optimizer: optax.GradientTransformation = eqx.field(static=True)

    # Dynamic
    model: eqx.nn.Linear
    optimizer_state: optax.OptState
    in_norm: WelfordNormalizer
    tgt_norm: WelfordNormalizer
    agent_state: jax.Array  # persistent state (previous close), shape (num_stocks,)
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    """Per-step metrics (each a scalar already meaned across stocks)."""
    norm_se: jax.Array  # mean over stocks of normalized squared error
    raw_se: jax.Array   # mean over stocks of de-normalized (raw price) squared error
    # "Predict the running mean" baseline: the mean normalizes to 0, so this is just the
    # same metric with pred replaced by 0 (always computed; logging gated by train.log_baseline).
    baseline_norm_se: jax.Array
    baseline_raw_se: jax.Array


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def prepare_experiment(cfg: DictConfig, num_stocks: int) -> TrainState:
    """Initialize the zero-init linear model, SGD optimizer, normalizers, and state."""
    set_seed(cfg.seed)
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 1_000_000_000)
    key = random.PRNGKey(seed)

    # Full cross-stock linear map W @ x + b, zero-initialized to match the numpy version.
    model = eqx.nn.Linear(num_stocks, num_stocks, use_bias=True, key=key)
    model = tree_replace(
        model, weight=jnp.zeros_like(model.weight), bias=jnp.zeros_like(model.bias)
    )

    optimizer = optax.sgd(cfg.optimizer.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    norm_kwargs = dict(alpha=cfg.norm.alpha, eps=cfg.norm.eps, clip=cfg.norm.clip)
    return TrainState(
        optimizer=optimizer,
        model=model,
        optimizer_state=optimizer_state,
        in_norm=WelfordNormalizer(num_stocks, **norm_kwargs),
        tgt_norm=WelfordNormalizer(num_stocks, **norm_kwargs),
        agent_state=jnp.zeros(num_stocks),
        step=jnp.array(0),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(train_state: TrainState, close_t: jax.Array) -> Tuple[TrainState, StepMetrics]:
    """One online step for jax.lax.scan. ``close_t`` is this step's close, shape (num_stocks,)."""
    state = train_state

    # Inputs come from the persistent state (set at the end of the previous step).
    x_raw = features_from_state(state.agent_state)
    y_raw = close_t

    # Normalize with statistics from prior steps only (no look-ahead on the target).
    x = state.in_norm.normalize(x_raw)
    y = state.tgt_norm.normalize(y_raw)

    # Loss is the SUMMED squared error across stocks (each output row gets its full gradient),
    # so optax.sgd reproduces W -= lr*outer(err, x), b -= lr*err.
    def loss_fn(model: eqx.nn.Linear) -> Tuple[jax.Array, jax.Array]:
        pred = model(x)
        return 0.5 * jnp.sum((pred - y) ** 2), pred

    (_, pred), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)
    updates, new_opt_state = state.optimizer.update(grads, state.optimizer_state)
    new_model = eqx.apply_updates(state.model, updates)

    # Metrics, using the pre-update normalizer stats (leakage-free, matches numpy version).
    err = pred - y
    norm_se = jnp.mean(err ** 2)
    pred_raw = state.tgt_norm.denormalize(pred)
    raw_se = jnp.mean((pred_raw - y_raw) ** 2)

    # Baseline: predict the running mean. The mean normalizes to 0, so substitute pred -> 0
    # and de-normalize back to the raw mean -- the same metrics for a trivial predictor.
    baseline_pred = jnp.zeros_like(pred)
    baseline_norm_se = jnp.mean((baseline_pred - y) ** 2)  # == mean(y**2)
    baseline_pred_raw = state.tgt_norm.denormalize(baseline_pred)  # == tgt running mean
    baseline_raw_se = jnp.mean((baseline_pred_raw - y_raw) ** 2)

    # Now reveal this step's observation: update the normalizers and the persistent state.
    new_state = tree_replace(
        state,
        model=new_model,
        optimizer_state=new_opt_state,
        in_norm=state.in_norm.update(x_raw),
        tgt_norm=state.tgt_norm.update(y_raw),
        agent_state=agent_state_update(state.agent_state, close_t),
        step=state.step + 1,
    )
    return new_state, StepMetrics(
        norm_se=norm_se, raw_se=raw_se,
        baseline_norm_se=baseline_norm_se, baseline_raw_se=baseline_raw_se,
    )


def train_multi_step(
    train_state: TrainState, close_block: jax.Array
) -> Tuple[TrainState, StepMetrics]:
    """Scan ``train_step`` over a block of steps. ``close_block`` is (n_steps, num_stocks)."""
    return jax.lax.scan(train_step, train_state, close_block, unroll=SCAN_UNROLL)


def compute_metrics(step_metrics: StepMetrics) -> dict:
    """Aggregate a scan block of per-step metrics: mean over steps (already meaned over stocks)."""
    return {
        'norm_mse': float(step_metrics.norm_se.mean()),
        'raw_rmse': float(jnp.sqrt(step_metrics.raw_se.mean())),
        'baseline_norm_mse': float(step_metrics.baseline_norm_se.mean()),
        'baseline_raw_rmse': float(jnp.sqrt(step_metrics.baseline_raw_se.mean())),
    }


def run_experiment(
    cfg: DictConfig, train_state: TrainState, train_fn: Callable, close_host: np.ndarray
) -> TrainState:
    """Outer Python loop: stream the data through ``train_fn`` one ``log_freq`` block at a time.

    The full series stays in host memory; each ``log_freq`` block is copied to the device only
    when it is about to be trained on, so the device never holds more than ``prefetch + 1``
    blocks at once. With ``train.prefetch > 0`` the copies for upcoming blocks are dispatched
    before the current block is synced on, so the host->device transfer overlaps GPU compute.
    """
    log_freq = cfg.train.log_freq
    log_baseline = cfg.train.get('log_baseline', True)
    prefetch = max(int(cfg.train.get('prefetch', 2)), 0)
    avail = close_host.shape[0]
    total_steps = cfg.train.total_steps if cfg.train.total_steps is not None else avail
    total_steps = min(total_steps, avail)
    num_scans = total_steps // log_freq

    def to_device(i: int) -> jax.Array:
        # Contiguous host slice -> async host->device copy (returns before the copy finishes).
        return jax.device_put(close_host[i * log_freq:(i + 1) * log_freq])

    # Prime the pipeline: dispatch transfers for the first ``prefetch + 1`` blocks.
    pending = deque(to_device(i) for i in range(min(prefetch + 1, num_scans)))
    next_fetch = len(pending)

    pbar = tqdm(total=num_scans * log_freq, desc='Training')
    for i in range(num_scans):
        block = pending.popleft()
        train_state, step_metrics = train_fn(train_state, block)  # async dispatch

        # Kick off the next transfer now so it overlaps this block's compute.
        if next_fetch < num_scans:
            pending.append(to_device(next_fetch))
            next_fetch += 1

        metrics = compute_metrics(step_metrics)  # syncs on this block's compute
        if not log_baseline:
            metrics = {k: v for k, v in metrics.items() if not k.startswith('baseline_')}
        log_metrics(metrics, cfg, step=int(train_state.step))

        pbar.update(log_freq)
        postfix = {
            'norm_mse': f'{metrics["norm_mse"]:.5f}', 'raw_rmse': f'{metrics["raw_rmse"]:.4f}',
        }
        if log_baseline:
            postfix['base_rmse'] = f'{metrics["baseline_raw_rmse"]:.4f}'
        pbar.set_postfix(postfix)
    pbar.close()
    return train_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    if cfg.get('x64', True):
        jax.config.update('jax_enable_x64', True)
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    try:
        data = load_stock_data(
            fields=['close'], num_stocks=cfg.data.num_stocks,
            num_steps=cfg.data.num_steps, full=cfg.data.full,
        )
        # Keep the full series in host memory; run_experiment copies it to the device one
        # block at a time. Contiguous (num_steps, num_stocks) layout so each block slice is a
        # clean, contiguous host->device transfer.
        close_host = np.ascontiguousarray(data['close'].T)  # (num_steps, num_stocks), host
        num_steps, num_stocks = close_host.shape

        train_state = prepare_experiment(cfg, num_stocks)
        print(
            f'Training: {num_stocks} stocks, {num_steps} steps, '
            f'lr={cfg.optimizer.learning_rate}, params={count_params(train_state.model)}'
        )

        train_fn = jax.jit(train_multi_step)
        train_state = run_experiment(cfg, train_state, train_fn, close_host)
    finally:
        finish_experiment(cfg)

    print('Run complete!')


if __name__ == '__main__':
    main()
