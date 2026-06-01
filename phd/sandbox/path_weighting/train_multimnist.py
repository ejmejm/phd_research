"""
8-task non-stationary parallel-MNIST with learned backward path-gates.
======================================================================

Setup
-----
- Input: K MNIST digits concatenated -> [B, K*784] (K=8 default).
- Per-task labels collapsed into a single K*10 = 80-dim binary vector.
- Loss: per-output sigmoid + binary cross-entropy (no softmax coupling).
- Net: 1 hidden layer of `hidden_dim` (default 32) leaky-ReLU; single linear
  "head" with K*10 outputs.
- Non-stationarity: every `permute_period` steps, ParallelMNISTStream permutes
  the label mapping of ONE randomly chosen task.

Implementation
--------------
Follows the lab jax-experiment skill: `TrainState` carries the path-gates
`Params` / `MetaState` and per-step metrics are emitted by a `jax.lax.scan`
over `train_step`. Data for a `permute_period`-sized chunk is pre-sampled on
CPU (so permutations align with chunk boundaries) then handed to the scanned
inner loop.

CLI (Hydra-style; compatible with mlflow-sweeper)
-------------------------------------------------
    python train_multimnist.py variant=honest alpha=0.0625 theta=64 ...
"""
from __future__ import annotations

import os
import sys
from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Repo-local imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from phd.structure_search.data import load_dataset, ParallelMNISTStream  # noqa: E402
import path_gates as pg  # noqa: E402


UNROLL_STEPS = 4


# ----------------------------------------------------------------------------- #
#  Variant table
# ----------------------------------------------------------------------------- #
VARIANTS = {
    "baseline":     dict(learn_gates=False, use_proxy=False, gamma=0.0, beta_init=8.0),
    "fixed":        dict(learn_gates=False, use_proxy=False, gamma=0.0, beta_init=0.0),
    "honest":       dict(learn_gates=True,  use_proxy=False, gamma=0.0, beta_init=2.0),
    "proxy":        dict(learn_gates=True,  use_proxy=True,  gamma=0.0, beta_init=2.0),
    "honest_trace": dict(learn_gates=True,  use_proxy=False, gamma=0.9, beta_init=2.0),
}


DEFAULT_CONFIG = {
    "variant": "honest",
    "seed": None,
    "alpha": 0.0625,
    "theta": 8.0,
    "gamma": None,
    "beta_init": None,
    "leaky_slope": 0.01,
    "hidden_dim": 32,
    "n_tasks": 8,
    "n_classes_per_task": 10,
    "permute_period": 200,
    "activation": "leaky_relu",
    "batch_size": 1,
    "num_steps": 20_000,
    "log_interval": 200,
    "dataset": "mnist",
    "mlflow": True,
}


def parse_args():
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    if len(sys.argv) > 1:
        overrides = OmegaConf.from_dotlist(sys.argv[1:])
        cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


# ----------------------------------------------------------------------------- #
#  State and metrics
# ----------------------------------------------------------------------------- #
class TrainState(eqx.Module):
    # static (recompiles if changed)
    learn_gates: bool = eqx.field(static=True)
    use_proxy: bool = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    leaky_slope: float = eqx.field(static=True)
    n_tasks: int = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)
    log_interval: int = eqx.field(static=True)

    # dynamic
    params: pg.Params
    meta_state: pg.MetaState
    alpha: jax.Array
    theta: jax.Array
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    loss: jax.Array
    online_acc: jax.Array
    gate_mean_head: jax.Array
    gate_std_head: jax.Array


# ----------------------------------------------------------------------------- #
#  Data helpers
# ----------------------------------------------------------------------------- #
def labels_to_binary_block(labels_np: np.ndarray, n_tasks: int, n_classes: int) -> np.ndarray:
    """labels_np: [T, B, n_tasks] int -> [T, B, n_tasks*n_classes] binary."""
    T, B, K = labels_np.shape
    out = np.zeros((T, B, K * n_classes), dtype=np.float32)
    t_idx = np.repeat(np.arange(T), B * K)
    b_idx = np.tile(np.repeat(np.arange(B), K), T)
    cols = (np.arange(K)[None, None, :] * n_classes + labels_np).reshape(-1)
    out[t_idx, b_idx, cols] = 1.0
    return out


# ----------------------------------------------------------------------------- #
#  Init
# ----------------------------------------------------------------------------- #
def init_experiment(cfg) -> Tuple[TrainState, ParallelMNISTStream, dict]:
    variant_cfg = dict(VARIANTS[cfg.variant])
    if cfg.gamma is not None:
        variant_cfg["gamma"] = float(cfg.gamma)
    if cfg.beta_init is not None:
        variant_cfg["beta_init"] = float(cfg.beta_init)

    seed = int(cfg.seed) if cfg.seed is not None else int(np.random.randint(0, 10**9))
    key = jax.random.PRNGKey(seed)

    images, labels, num_classes_real, input_dim_per_task = load_dataset(cfg.dataset, split="train")
    assert num_classes_real == cfg.n_classes_per_task

    n_tasks = int(cfg.n_tasks)
    in_dim = n_tasks * input_dim_per_task
    out_dim = n_tasks * num_classes_real

    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=n_tasks, batch_size=int(cfg.batch_size),
        seed=seed, permute_period=int(cfg.permute_period),
    )

    params = pg.init_params(
        key, in_dim=in_dim, hidden_dims=[int(cfg.hidden_dim)],
        n_classes=out_dim, n_heads=1,
        beta_init=float(variant_cfg["beta_init"]),
    )
    meta = pg.zeros_like_gates(params)

    state = TrainState(
        learn_gates=bool(variant_cfg["learn_gates"]),
        use_proxy=bool(variant_cfg["use_proxy"]),
        gamma=float(variant_cfg["gamma"]),
        activation=str(cfg.activation),
        leaky_slope=float(cfg.leaky_slope),
        n_tasks=n_tasks,
        n_classes=int(cfg.n_classes_per_task),
        log_interval=int(cfg.log_interval),
        params=params,
        meta_state=meta,
        alpha=jnp.array(float(cfg.alpha)),
        theta=jnp.array(float(cfg.theta)),
        step=jnp.array(0, dtype=jnp.int32),
    )
    return state, stream, {"seed": seed, "variant_cfg": variant_cfg,
                          "in_dim": in_dim, "out_dim": out_dim}


# ----------------------------------------------------------------------------- #
#  Train step (scan-compatible)
# ----------------------------------------------------------------------------- #
def train_step(state: TrainState,
               batch: Tuple[jax.Array, jax.Array, jax.Array]) -> Tuple[TrainState, StepMetrics]:
    """Single training step: prequential predict -> pg.step weight & gate update.

    batch = (x [B, in_dim], y_binary [B, K*C], y_labels [B, K])
    """
    x, y_bin, y_lbl = batch
    if state.activation == "leaky_relu":
        f, fprime = pg.act("leaky_relu", leaky_slope=state.leaky_slope)
    else:
        f, fprime = pg.act(state.activation)

    # Prequential predict (uses CURRENT params, before update)
    logits, _, _ = pg.forward(state.params, x, f)
    logits_per_task = logits[0].reshape(x.shape[0], state.n_tasks, state.n_classes)
    preds = jnp.argmax(logits_per_task, axis=-1)         # [B, K]
    online_acc = jnp.mean(preds == y_lbl).astype(jnp.float32)

    # Path-gates step (forward + modulated backward + meta update)
    new_params, new_meta, loss = pg.step(
        state.params, state.meta_state, x, [y_bin], [1.0],
        f=f, fprime=fprime,
        alpha=state.alpha, theta=state.theta,
        gamma=state.gamma, learn_gates=state.learn_gates,
        use_proxy=state.use_proxy, loss_kind="sigmoid_bce",
    )

    g_head = jax.nn.sigmoid(new_params.head_beta[0])
    metrics = StepMetrics(
        loss=loss,
        online_acc=online_acc,
        gate_mean_head=jnp.mean(g_head),
        gate_std_head=jnp.std(g_head),
    )

    new_state = TrainState(
        learn_gates=state.learn_gates,
        use_proxy=state.use_proxy,
        gamma=state.gamma,
        activation=state.activation,
        leaky_slope=state.leaky_slope,
        n_tasks=state.n_tasks,
        n_classes=state.n_classes,
        log_interval=state.log_interval,
        params=new_params,
        meta_state=new_meta,
        alpha=state.alpha,
        theta=state.theta,
        step=state.step + 1,
    )
    return new_state, metrics


# ----------------------------------------------------------------------------- #
#  Metric logging
# ----------------------------------------------------------------------------- #
def log_metrics(metrics: StepMetrics, step: int, use_mlflow: bool) -> dict:
    out = {
        "online_acc": float(metrics.online_acc.mean()),
        "loss": float(metrics.loss.mean()),
        "gate_mean_head": float(metrics.gate_mean_head.mean()),
        "gate_std_head": float(metrics.gate_std_head.mean()),
    }
    if use_mlflow:
        mlflow.log_metrics(out, step=step)
    return out


# ----------------------------------------------------------------------------- #
#  Outer training loop
# ----------------------------------------------------------------------------- #
def train(state: TrainState, stream: ParallelMNISTStream, cfg) -> Tuple[TrainState, list]:
    num_steps = int(cfg.num_steps)
    chunk_size = int(cfg.permute_period)  # align chunks with permutation events
    assert num_steps % chunk_size == 0, (
        f"num_steps={num_steps} must be a multiple of permute_period={chunk_size}"
    )
    log_interval = int(cfg.log_interval)
    assert chunk_size == log_interval or chunk_size % log_interval == 0 \
        or log_interval % chunk_size == 0, (
            "log_interval and permute_period must align"
        )

    n_chunks = num_steps // chunk_size

    @jax.jit
    def scan_chunk(state: TrainState, batch) -> Tuple[TrainState, StepMetrics]:
        return jax.lax.scan(train_step, state, batch, unroll=UNROLL_STEPS)

    pbar = tqdm(total=num_steps, desc=str(cfg.variant))
    window_accs = []
    n_tasks = int(cfg.n_tasks)
    n_classes = int(cfg.n_classes_per_task)

    steps_done = 0
    for _ in range(n_chunks):
        imgs_np, lbls_np = stream.sample_batch(chunk_size)   # [chunk, B, in_dim], [chunk, B, K]
        ybin_np = labels_to_binary_block(lbls_np, n_tasks, n_classes)
        x = jnp.asarray(imgs_np)
        y_bin = jnp.asarray(ybin_np)
        y_lbl = jnp.asarray(lbls_np)

        state, metrics = scan_chunk(state, (x, y_bin, y_lbl))
        # metrics fields each have leading dim = chunk_size
        # Log per-log_interval window
        if log_interval == chunk_size:
            steps_done += chunk_size
            m = log_metrics(metrics, step=steps_done, use_mlflow=cfg.mlflow)
            window_accs.append(m["online_acc"])
        else:
            # chunk_size is multiple of log_interval: split metrics
            num_log = chunk_size // log_interval
            for i in range(num_log):
                sl = slice(i * log_interval, (i + 1) * log_interval)
                sub = jax.tree_util.tree_map(lambda a: a[sl], metrics)
                steps_done += log_interval
                m = log_metrics(sub, step=steps_done, use_mlflow=cfg.mlflow)
                window_accs.append(m["online_acc"])

        pbar.update(chunk_size)
        pbar.set_postfix({"online": f"{window_accs[-1]:.3f}",
                          "g_head": f"{m['gate_mean_head']:.3f}"})
    pbar.close()
    return state, window_accs


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #
def main():
    cfg = parse_args()
    state, stream, info = init_experiment(cfg)

    if cfg.mlflow:
        mlflow.start_run()
        mlflow.log_params({
            **OmegaConf.to_container(cfg, resolve=True),
            **{f"cfg_{k}": v for k, v in info["variant_cfg"].items()},
            "seed": info["seed"],
            "in_dim": info["in_dim"],
            "out_dim": info["out_dim"],
        })

    state, window_accs = train(state, stream, cfg)

    final_online_acc = float(np.mean(window_accs)) if window_accs else 0.0
    if window_accs:
        cut = len(window_accs) // 2
        asymp = float(np.mean(window_accs[cut:]))
    else:
        asymp = 0.0
    if cfg.mlflow:
        mlflow.log_metrics({"final_online_acc": final_online_acc,
                            "asymptotic_online_acc": asymp})
        mlflow.end_run()
    print(f"done: variant={cfg.variant} alpha={float(cfg.alpha)} theta={float(cfg.theta)} "
          f"final_online_acc={final_online_acc:.4f} asymp={asymp:.4f}")
    return final_online_acc


if __name__ == "__main__":
    main()
