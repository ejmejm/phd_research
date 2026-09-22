"""The training loop, shared by every model and connectivity algorithm.

Structure of a run, outermost first:

    for each log period (``train.log_freq`` steps)
        for each structure event (``algorithm.event_period`` steps)   [if any]
            scan ``event_period`` training steps
            apply the algorithm's structure event
        apply the algorithm's host-side hook
        log

Everything down to the host-side hook is inside one jitted, vmapped function,
so a multi-seed run is a single dispatch per log period rather than one per
step. Batches for the next period are prepared on a worker thread while the
current one runs on the accelerator.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig
from tqdm import tqdm

from .algorithms.base import ConnectivityAlgorithm
from .metrics import DIVERGENCE_LOSS_THRESHOLD
from .optim import EqxOptimizer
from .utils import tree_replace


# Unrolling the inner scan trades compile time for step throughput; 4 was the
# best of {1, 2, 4, 8} on the paper's configurations.
SCAN_UNROLL = 4


class TrainState(eqx.Module):
    model: eqx.Module
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray
    #: Connectivity-algorithm state; ``None`` for algorithms that are stateless.
    algo: Any = None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def per_task_loss(outputs_r: jax.Array, one_hot: jax.Array, loss_type: str) -> jax.Array:
    """Per-example, per-task loss of shape (B, K).

    ``softmax_ce`` applies a softmax over each task's ``num_classes`` outputs
    and takes categorical cross-entropy -- this is what the paper uses.
    ``sigmoid_bce`` treats each output independently.
    """
    if loss_type == 'softmax_ce':
        log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
        return -jnp.sum(one_hot * log_probs, axis=-1)
    if loss_type == 'sigmoid_bce':
        z = outputs_r
        per_class = jnp.maximum(z, 0) - z * one_hot + jnp.log1p(jnp.exp(-jnp.abs(z)))
        return jnp.sum(per_class, axis=-1)
    raise ValueError(f'Unknown loss: {loss_type}')


def train_step(
    state: TrainState, data, *,
    algorithm: ConnectivityAlgorithm,
    num_classes: int, n_tasks: int, loss_type: str,
):
    """One online update. Scanned over, so it must be shape-stable."""
    images, labels = data                           # (B, K*784), (B, K)
    one_hot = jax.nn.one_hot(labels, num_classes)   # (B, K, C)

    def loss_fn(model):
        outputs, h = jax.vmap(model)(images)
        outputs_r = outputs.reshape(-1, n_tasks, num_classes)
        loss_per_batch = per_task_loss(outputs_r, one_hot, loss_type)   # (B, K)
        # Sum over tasks, mean over the batch: the multi-MNIST objective is
        # the sum of the per-task losses.
        loss = jnp.mean(jnp.sum(loss_per_batch, axis=1), axis=0)
        return loss, (outputs_r, h)

    (loss, (outputs_r, h)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(state.model)
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()

    updates, new_optimizer = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    # Per-step model modification, for algorithms whose weight dynamics run
    # faster than their structural bookkeeping (DEEP-R's L1 + Langevin terms).
    # Runs before post_update so the model mirrors whatever it leaves behind.
    new_rng = state.rng
    if algorithm.needs_step_key:
        new_rng, step_key = jax.random.split(state.rng)
    else:
        step_key = None
    new_model, new_algo = algorithm.step_update(new_model, state.algo, key=step_key)

    # Models with derived state (DynamicNetwork mirrors its incoming weights
    # into an outgoing-indexed copy for the next backward pass) reconcile it
    # here, once, rather than in every algorithm that uses them.
    post_update = getattr(new_model, 'post_update', None)
    if post_update is not None:
        new_model = post_update()

    # Hand the algorithm what the forward pass already produced, so an
    # algorithm tracking contribution utility needs no extra compute.
    new_algo = algorithm.step_hook(new_algo, state.model, new_model, (images, h))

    new_state = tree_replace(
        state,
        model=new_model, optimizer=new_optimizer, algo=new_algo,
        rng=new_rng, step=state.step + 1,
    )
    return new_state, jnp.stack([loss, correct])


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _eval_forward(model, images, labels, num_classes, n_tasks, loss_type):
    outputs, _ = jax.vmap(model)(images)
    one_hot = jax.nn.one_hot(labels, num_classes)
    outputs_r = outputs.reshape(-1, n_tasks, num_classes)
    loss = jnp.mean(jnp.sum(per_task_loss(outputs_r, one_hot, loss_type), axis=1))
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()
    return loss, correct


def evaluate_test(batched_model, test_images, test_labels,
                  num_classes: int, n_tasks: int, loss_type: str,
                  batch_size: int = 512):
    """Evaluate a seed-vmapped model on the test pool, in chunks."""
    @jax.jit
    def _eval_chunk(model, imgs, lbls):
        return jax.vmap(
            lambda m: _eval_forward(m, imgs, lbls, num_classes, n_tasks, loss_type),
        )(model)

    n_test = test_images.shape[0]
    total_loss = total_acc = None
    n_chunks = 0
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        cl, ca = _eval_chunk(
            batched_model, jnp.array(test_images[start:end]), jnp.array(test_labels[start:end]))
        total_loss = cl if total_loss is None else total_loss + cl
        total_acc = ca if total_acc is None else total_acc + ca
        n_chunks += 1
    return total_loss / n_chunks, total_acc / n_chunks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_scan_log_period(
    algorithm: ConnectivityAlgorithm, train_step_fn, log_freq: int,
):
    """Build the jitted function that advances one log period.

    When the algorithm has no in-scan event, a log period is a single scan.
    When it does, the period is split into ``log_freq // event_period``
    cycles, each a scan followed by one structure event.
    """
    if not algorithm.event_period:
        def scan_log_period(state, data):
            state, metrics = jax.lax.scan(
                train_step_fn, state, data, unroll=SCAN_UNROLL)
            return state, metrics, {}
        return scan_log_period

    event_period = int(algorithm.event_period)
    assert log_freq % event_period == 0, (
        f'train.log_freq={log_freq} must be divisible by the algorithm event '
        f'period {event_period}, so that events align with log boundaries'
    )

    if event_period == 1:
        # An event on every step. Nesting a length-1 scan inside the outer one
        # would pay the scan machinery per step for nothing, so fuse the event
        # into the step and scan as usual. DEEP-R is the case this serves.
        def fused_step(state, data):
            state, metrics = train_step_fn(state, data)
            next_rng, event_key = jax.random.split(state.rng)
            model, optimizer, algo_state, info = algorithm.event(
                state.model, state.optimizer, state.algo, key=event_key)
            state = tree_replace(
                state, model=model, optimizer=optimizer, algo=algo_state, rng=next_rng)
            return state, (metrics, info)

        def scan_log_period(state, data):
            state, (metrics, info) = jax.lax.scan(
                fused_step, state, data, unroll=SCAN_UNROLL)
            return state, metrics, info

        return scan_log_period

    cycles_per_log = log_freq // event_period

    def event_cycle(state, cycle_data):
        state, metrics = jax.lax.scan(
            train_step_fn, state, cycle_data, unroll=SCAN_UNROLL)
        next_rng, event_key = jax.random.split(state.rng)
        model, optimizer, algo_state, info = algorithm.event(
            state.model, state.optimizer, state.algo, key=event_key)
        state = tree_replace(
            state, model=model, optimizer=optimizer, algo=algo_state, rng=next_rng)
        return state, (metrics, info)

    def scan_log_period(state, data):
        data = jax.tree.map(
            lambda x: x.reshape(cycles_per_log, event_period, *x.shape[1:]), data)
        state, (metrics, info) = jax.lax.scan(event_cycle, state, data)
        return state, metrics.reshape(-1, *metrics.shape[2:]), info

    return scan_log_period


def run_experiment(
    cfg: DictConfig,
    train_state: TrainState,
    streams,
    algorithm: ConnectivityAlgorithm,
    num_classes: int,
    n_tasks: int,
    diagnostics_fn=None,
    log_fn=None,
):
    """Run training to ``train.total_steps``, returning per-log-period curves.

    Args:
        diagnostics_fn: ``(model, n_tasks) -> {name: per-seed array}`` for the
            model type in use. Called once up front when connectivity is
            static, and once per log period otherwise.
        log_fn: ``(metrics, per_seed_metrics, cfg, step) -> None``, called on
            a worker thread so logging never blocks the accelerator.
    """
    log_freq = int(cfg.train.log_freq)
    num_log_periods = int(cfg.train.total_steps) // log_freq
    eval_freq = int(cfg.train.get('eval_freq', 0))
    n_test_samples_cfg = cfg.train.get('n_test_samples', None)
    n_test_samples = int(n_test_samples_cfg) if n_test_samples_cfg is not None else None
    loss_type = str(cfg.train.get('loss', 'softmax_ce'))

    train_step_fn = partial(
        train_step, algorithm=algorithm,
        num_classes=num_classes, n_tasks=n_tasks, loss_type=loss_type,
    )
    vmapped_scan = jax.jit(jax.vmap(build_scan_log_period(
        algorithm, train_step_fn, log_freq)))

    curves = {k: [] for k in (
        'loss', 'accuracy', 'per_seed_loss', 'per_seed_accuracy',
        'test_loss', 'test_accuracy', 'per_seed_test_loss', 'per_seed_test_accuracy',
    )}

    logging_active = bool(
        cfg.get('mlflow', False) or cfg.get('wandb', False) or cfg.get('comet_ml', False))
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    cumulative_info: Dict[str, float] = {}

    # When connectivity is fixed for the whole run its diagnostics are
    # constant, so compute them once rather than syncing the accelerator
    # every log period.
    def _diagnostics() -> Dict[str, float]:
        if diagnostics_fn is None:
            return {}
        return {k: float(v.mean())
                for k, v in diagnostics_fn(train_state.model, n_tasks).items()}

    static_diagnostics = _diagnostics() if algorithm.structure_is_static else None

    # Prepare the next batch on a worker thread while the accelerator runs the
    # current scan. One worker, so the streams' RNG state is only ever touched
    # from a single thread.
    def _prepare_batch():
        batch = [s.sample_batch(log_freq) for s in streams]
        return (np.stack([b[0] for b in batch]), np.stack([b[1] for b in batch]))

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    next_batch_future = prefetch_executor.submit(_prepare_batch)

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    for _ in range(num_log_periods):
        imgs_np, lbls_np = next_batch_future.result()
        next_batch_future = prefetch_executor.submit(_prepare_batch)

        train_state, metrics, info = vmapped_scan(
            train_state, (jnp.array(imgs_np), jnp.array(lbls_np)))

        # metrics: (n_seeds, log_freq, 2)
        per_seed_loss = metrics[..., 0].mean(axis=1)
        per_seed_acc = metrics[..., 1].mean(axis=1)
        mean_loss, mean_acc = float(per_seed_loss.mean()), float(per_seed_acc.mean())
        std_loss, std_acc = float(per_seed_loss.std()), float(per_seed_acc.std())
        step = int(train_state.step[0].item())

        # Host-side structure hook (e.g. the dense transition).
        model, optimizer, algo_state, host_info = algorithm.on_log_period(
            train_state.model, train_state.optimizer, train_state.algo, step)
        train_state = tree_replace(
            train_state, model=model, optimizer=optimizer, algo=algo_state)

        structure_metrics = (
            dict(static_diagnostics) if static_diagnostics is not None
            else _diagnostics()
        )
        structure_metrics.update(algorithm.diagnostics(train_state.model, n_tasks))
        structure_metrics.update(host_info)
        for key, value in info.items():
            cumulative_info[key] = cumulative_info.get(key, 0.0) + float(
                np.asarray(value).sum(axis=-1).mean())
            structure_metrics[f'cumulative_{key}'] = cumulative_info[key]

        test_metrics, per_seed_test_metrics = {}, {}
        if eval_freq > 0 and step % eval_freq == 0:
            t_imgs, t_lbls = streams[0].get_test_batch()
            if n_test_samples is not None and n_test_samples < t_imgs.shape[0]:
                t_imgs, t_lbls = t_imgs[:n_test_samples], t_lbls[:n_test_samples]
            test_loss, test_acc = evaluate_test(
                train_state.model, t_imgs, t_lbls, num_classes, n_tasks, loss_type)
            per_seed_test_loss = np.array(test_loss)
            per_seed_test_acc = np.array(test_acc)
            curves['test_loss'].append(float(per_seed_test_loss.mean()))
            curves['test_accuracy'].append(float(per_seed_test_acc.mean()))
            curves['per_seed_test_loss'].append(per_seed_test_loss)
            curves['per_seed_test_accuracy'].append(per_seed_test_acc)
            test_metrics = {
                'test_loss': curves['test_loss'][-1],
                'test_accuracy': curves['test_accuracy'][-1],
            }
            per_seed_test_metrics = {
                'test_loss': per_seed_test_loss.tolist(),
                'test_accuracy': per_seed_test_acc.tolist(),
            }

        if logging_active and log_fn is not None:
            log_futures.append(log_executor.submit(
                log_fn,
                {'loss': mean_loss, 'loss_std': std_loss,
                 'accuracy': mean_acc, 'accuracy_std': std_acc,
                 **structure_metrics, **test_metrics},
                {'loss': per_seed_loss.tolist(), 'accuracy': per_seed_acc.tolist(),
                 **per_seed_test_metrics},
                cfg, step,
            ))

        curves['loss'].append(mean_loss)
        curves['accuracy'].append(mean_acc)
        curves['per_seed_loss'].append(np.array(per_seed_loss))
        curves['per_seed_accuracy'].append(np.array(per_seed_acc))

        pbar.update(log_freq)
        postfix = {'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'}
        for key in ('active_units', 'active_connections'):
            if key in structure_metrics:
                postfix[key[7:11]] = f'{structure_metrics[key]:.0f}'
        if test_metrics:
            postfix['t_acc'] = f'{test_metrics["test_accuracy"]:.4f}'
        pbar.set_postfix(postfix)

        # A diverged run is a valid measurement of a bad hyperparameter; stop
        # early rather than burning the remaining budget on infinities.
        if not np.isfinite(mean_loss) or mean_loss > DIVERGENCE_LOSS_THRESHOLD:
            print(f'\n[diverged] train loss {mean_loss:.4g} exceeded threshold '
                  f'{DIVERGENCE_LOSS_THRESHOLD:.4g} at step {step}; stopping early.')
            break

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    prefetch_executor.shutdown(wait=False)
    pbar.close()
    return train_state, curves
