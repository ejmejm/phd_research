"""JAX/Equinox sequential GVF predictor over multi-stock market data.

An online linear model that, at every 15-min step, predicts a configurable set of General
Value Functions (GVFs) for every loaded stock. A GVF is a (cumulant, gamma) pair: the
value of a state estimates the discounted sum of future cumulants,

    v(s_t) ~= E[ c_{t+1} + gamma * c_{t+2} + ... ].

Each GVF is *per stock*, so listing ``k`` GVFs in the config yields ``k * num_stocks``
prediction targets. With ``gamma = 0`` a GVF is a one-step prediction of the next cumulant
-- so ``gvfs: [{field: close, gamma: 0}]`` reproduces the original next-close predictor.

GVFs are learned online with semi-gradient TD(0): the target for ``v(s_{t-1})`` is
``c_t + gamma * v(s_t)`` (the bootstrap value is treated as a constant). At ``gamma = 0``
this is exactly the supervised one-step MSE update of the original predictor.

Structure (unchanged from the next-close port): one timestep is a ``train_step`` compatible
with ``jax.lax.scan``; ``train_multi_step`` scans it over a block of ``log_freq`` steps; the
outer ``run_experiment`` loop streams the data through the scan one block at a time.

Notes:
  * Inputs are the previous step's (normalized) closes -- a full cross-stock linear map
    ``W`` of shape ``(n_targets, num_stocks)``, zero-initialized. The agent state holds the
    whole latest observation so cumulants can read any field; only ``close`` feeds the
    input features for now (extend ``features_from_state`` to add more inputs).
  * Inputs and cumulants are standardized by exponentially-weighted Welford normalizers
    (with outlier clip). The TD target lives in normalized-cumulant space, so the bootstrap
    ``gamma * v(s_t)`` is consistent with the normalized ``c_t``.
  * Loss is the summed (not meaned) squared TD error, so ``optax.sgd`` reproduces
    ``W -= lr * outer(err, x)``; stable ``lr`` scales ~``1/num_stocks``.
  * x64 is on by default for parity with the float64 numpy version; turn off with ``x64=false``.
"""

from __future__ import annotations

import functools
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
from gvfs import build_gvfs, cumulant_vector, gamma_vector, required_fields
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
# Agent state: the latest observation, and the model's input features from it
# ---------------------------------------------------------------------------
# The agent state is a ``{field: (num_stocks,) array}`` dict holding the most recent
# observation. GVF cumulants are defined in terms of it (see gvfs.py); the input features
# are derived from it here.

def features_from_state(state: dict) -> jax.Array:
    """Map the persistent state to the model's input features.

    Currently the previous close (NaN-safe). Extend here (e.g. concat more fields) to give
    the model richer inputs without touching the rest of the pipeline."""
    return jnp.nan_to_num(state["close"], nan=0.0)


def agent_state_update(state: dict, obs: dict) -> dict:
    """Fold the current observation into the persistent state.

    Starter version simply stores the (NaN-safe) observation, so next step's input is this
    step's close. ``obs`` is ``{field: (num_stocks,)}`` for the current step."""
    return {f: jnp.nan_to_num(obs[f], nan=0.0) for f in obs}


# ---------------------------------------------------------------------------
# Train state / metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    # Static
    optimizer: optax.GradientTransformation = eqx.field(static=True)

    # Dynamic
    model: eqx.nn.Linear
    optimizer_state: optax.OptState
    in_norm: WelfordNormalizer    # over input features (num_stocks)
    tgt_norm: WelfordNormalizer   # over cumulants (n_targets = n_gvfs * num_stocks)
    agent_state: dict             # latest observation, {field: (num_stocks,)}
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    """Per-step, per-GVF metrics (each shape ``(n_gvfs,)``, already meaned across stocks)."""
    norm_se: jax.Array      # mean over stocks of normalized squared TD error
    raw_se: jax.Array       # mean over stocks of de-normalized squared error vs the cumulant
    # "Predict the running mean" baseline: the mean normalizes to 0, so substitute pred -> 0.
    baseline_norm_se: jax.Array
    baseline_raw_se: jax.Array


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def prepare_experiment(
    cfg: DictConfig, gvfs: list, num_stocks: int, fields: list[str]
) -> TrainState:
    """Initialize the zero-init linear model, SGD optimizer, normalizers, and agent state."""
    set_seed(cfg.seed)
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 1_000_000_000)
    key = random.PRNGKey(seed)

    n_targets = len(gvfs) * num_stocks

    # Full cross-stock linear map W @ x + b: num_stocks inputs -> n_targets outputs,
    # zero-initialized to match the numpy version.
    model = eqx.nn.Linear(num_stocks, n_targets, use_bias=True, key=key)
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
        tgt_norm=WelfordNormalizer(n_targets, **norm_kwargs),
        agent_state={f: jnp.zeros(num_stocks) for f in fields},
        step=jnp.array(0),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(
    state: TrainState, obs: dict, *, gvfs: list, gammas: jax.Array
) -> Tuple[TrainState, StepMetrics]:
    """One online TD(0) step for jax.lax.scan.

    ``obs`` is this step's observation ``{field: (num_stocks,)}``. The model input is the
    *previous* agent state (lagged by one step); the cumulant and bootstrap value come from
    the current observation, so we update ``v(s_{t-1})`` toward ``c_t + gamma * v(s_t)``.
    """
    # Model input: features of the previous agent state (s_{t-1}).
    x_raw = features_from_state(state.agent_state)

    # Fold in the current observation -> current agent state (s_t); read cumulants from it.
    new_agent_state = agent_state_update(state.agent_state, obs)
    c_raw = cumulant_vector(gvfs, new_agent_state)         # c_t, (n_targets,)
    xnext_raw = features_from_state(new_agent_state)        # s_t features, for bootstrap

    # Normalize with statistics from prior steps only (no look-ahead on the cumulant).
    x = state.in_norm.normalize(x_raw)
    xnext = state.in_norm.normalize(xnext_raw)
    c = state.tgt_norm.normalize(c_raw)

    # Semi-gradient TD(0): bootstrap value is a constant (no gradient flows through it).
    v_next = state.model(xnext)
    target = c + gammas * v_next

    # Loss is the SUMMED squared TD error (each output row gets its full gradient),
    # so optax.sgd reproduces W -= lr*outer(err, x), b -= lr*err.
    def loss_fn(model: eqx.nn.Linear) -> Tuple[jax.Array, jax.Array]:
        v = model(x)
        return 0.5 * jnp.sum((v - target) ** 2), v

    (_, v), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)
    updates, new_opt_state = state.optimizer.update(grads, state.optimizer_state)
    new_model = eqx.apply_updates(state.model, updates)

    # Per-GVF metrics (mean over stocks), using pre-update model + normalizer stats.
    n_gvfs = len(gvfs)
    num_stocks = x_raw.shape[0]

    def per_gvf(vec: jax.Array) -> jax.Array:  # (n_targets,) -> (n_gvfs,)
        return vec.reshape(n_gvfs, num_stocks).mean(axis=1)

    err = v - target
    norm_se = per_gvf(err ** 2)
    pred_raw = state.tgt_norm.denormalize(v)
    raw_se = per_gvf((pred_raw - c_raw) ** 2)  # a true raw error only when gamma == 0

    # Baseline: predict the running mean. The mean normalizes to 0, so substitute pred -> 0.
    baseline_norm_se = per_gvf(target ** 2)  # (0 - target)^2
    baseline_pred_raw = state.tgt_norm.denormalize(jnp.zeros_like(v))  # == running mean
    baseline_raw_se = per_gvf((baseline_pred_raw - c_raw) ** 2)

    # Now reveal this step's observation: update the normalizers and the agent state.
    new_state = tree_replace(
        state,
        model=new_model,
        optimizer_state=new_opt_state,
        in_norm=state.in_norm.update(x_raw),
        tgt_norm=state.tgt_norm.update(c_raw),
        agent_state=new_agent_state,
        step=state.step + 1,
    )
    return new_state, StepMetrics(
        norm_se=norm_se, raw_se=raw_se,
        baseline_norm_se=baseline_norm_se, baseline_raw_se=baseline_raw_se,
    )


def make_train_fn(gvfs: list, gammas: jax.Array) -> Callable:
    """Build the jitted scan over ``train_step`` with ``gvfs``/``gammas`` bound in."""
    step_fn = functools.partial(train_step, gvfs=gvfs, gammas=gammas)

    @jax.jit
    def train_fn(state: TrainState, obs_block: dict) -> Tuple[TrainState, StepMetrics]:
        """Scan one step over a block. ``obs_block`` is {field: (n_steps, num_stocks)}."""
        return jax.lax.scan(step_fn, state, obs_block, unroll=SCAN_UNROLL)

    return train_fn


def compute_metrics(step_metrics: StepMetrics, gvfs: list) -> dict:
    """Aggregate a scan block: mean over steps, per GVF, plus an overall (mean across GVFs)."""
    norm_mse = step_metrics.norm_se.mean(axis=0)            # (n_gvfs,)
    raw_rmse = jnp.sqrt(step_metrics.raw_se.mean(axis=0))
    base_norm_mse = step_metrics.baseline_norm_se.mean(axis=0)
    base_raw_rmse = jnp.sqrt(step_metrics.baseline_raw_se.mean(axis=0))

    metrics: dict = {
        'norm_mse': float(norm_mse.mean()),
        'raw_rmse': float(raw_rmse.mean()),
        'baseline_norm_mse': float(base_norm_mse.mean()),
        'baseline_raw_rmse': float(base_raw_rmse.mean()),
    }
    for i, g in enumerate(gvfs):
        metrics[f'{g.name}/norm_mse'] = float(norm_mse[i])
        metrics[f'{g.name}/raw_rmse'] = float(raw_rmse[i])
        metrics[f'{g.name}/baseline_norm_mse'] = float(base_norm_mse[i])
        metrics[f'{g.name}/baseline_raw_rmse'] = float(base_raw_rmse[i])
    return metrics


def run_experiment(
    cfg: DictConfig, train_state: TrainState, train_fn: Callable,
    obs_host: dict, gvfs: list,
) -> TrainState:
    """Outer Python loop: stream the data through ``train_fn`` one ``log_freq`` block at a time.

    The full series stays in host memory; each ``log_freq`` block is copied to the device only
    when it is about to be trained on, so the device never holds more than ``prefetch + 1``
    blocks at once. With ``train.prefetch > 0`` the copies for upcoming blocks are dispatched
    before the current block is synced on, so the host->device transfer overlaps GPU compute.
    ``obs_host`` is ``{field: (num_steps, num_stocks)}`` on the host.
    """
    log_freq = cfg.train.log_freq
    log_baseline = cfg.train.get('log_baseline', True)
    prefetch = max(int(cfg.train.get('prefetch', 2)), 0)
    fields = list(obs_host)
    avail = obs_host['close'].shape[0]
    total_steps = cfg.train.total_steps if cfg.train.total_steps is not None else avail
    total_steps = min(total_steps, avail)
    num_scans = total_steps // log_freq

    def to_device(i: int) -> dict:
        # Contiguous host slice per field -> async host->device copy.
        sl = slice(i * log_freq, (i + 1) * log_freq)
        return {f: jax.device_put(obs_host[f][sl]) for f in fields}

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

        metrics = compute_metrics(step_metrics, gvfs)  # syncs on this block's compute
        if not log_baseline:
            metrics = {k: v for k, v in metrics.items() if 'baseline' not in k}
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
        gvfs = build_gvfs(cfg.gvfs)
        fields = required_fields(gvfs)  # cumulant fields + 'close' (input features)

        data = load_stock_data(
            fields=fields, num_stocks=cfg.data.num_stocks,
            num_steps=cfg.data.num_steps, full=cfg.data.full,
        )
        # Keep the full series in host memory; run_experiment copies it to the device one
        # block at a time. Contiguous (num_steps, num_stocks) layout per field so each block
        # slice is a clean, contiguous host->device transfer.
        obs_host = {f: np.ascontiguousarray(data[f].T) for f in fields}  # (num_steps, num_stocks)
        num_steps, num_stocks = obs_host['close'].shape

        train_state = prepare_experiment(cfg, gvfs, num_stocks, fields)
        gammas = gamma_vector(gvfs, num_stocks)
        gvf_desc = ', '.join(f'{g.name}(gamma={g.gamma:g})' for g in gvfs)
        print(
            f'Training: {num_stocks} stocks, {num_steps} steps, '
            f'{len(gvfs)} GVF(s) x {num_stocks} = {len(gvfs) * num_stocks} targets '
            f'[{gvf_desc}], lr={cfg.optimizer.learning_rate}, '
            f'params={count_params(train_state.model)}'
        )

        train_fn = make_train_fn(gvfs, gammas)
        train_state = run_experiment(cfg, train_state, train_fn, obs_host, gvfs)
    finally:
        finish_experiment(cfg)

    print('Run complete!')


if __name__ == '__main__':
    main()
