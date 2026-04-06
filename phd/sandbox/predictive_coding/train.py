from concurrent.futures import ThreadPoolExecutor
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

from phd.jax_core.models import MLP, ACTIVATION_MAP
from phd.jax_core.optimizers import EqxOptimizer
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees, tree_replace
from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer,
    set_seed,
    rng_from_string,
)
from phd.research_utils.logging import (
    init_experiment,
    init_child_runs,
    log_metrics,
    log_child_metrics,
    finish_child_runs,
    finish_experiment,
)
from phd.sandbox.predictive_coding.data import (
    load_mnist, RotatingMNISTStream, ShuffledMNISTStream,
)
from phd.sandbox.predictive_coding.models import (
    PCNetwork, init_pc_network, pc_forward_pass, ipc_step, ACTIVATION_DERIV_MAP,
)


SCAN_UNROLL = 4
NUM_CLASSES = 10
INPUT_DIM = 784


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

class PCStepMetrics(eqx.Module):
    """Per-step metrics from the PC training loop."""
    online_correct: jax.Array   # scalar: 1.0 if prediction correct before update
    total_energy: jax.Array     # scalar: sum of squared errors
    layer_errors: jax.Array     # (max_layers,): per-layer squared error norms
    weight_update_norms: jax.Array  # (max_layers,): per-layer weight update norms


class BPStepMetrics(eqx.Module):
    """Per-step metrics from the backprop baseline."""
    loss: jax.Array
    correct: jax.Array


# ---------------------------------------------------------------------------
# iPC Train State and step
# ---------------------------------------------------------------------------

class PCTrainState(eqx.Module):
    network: PCNetwork
    value_nodes: list     # persistent across observations
    output_node: jax.Array  # persistent output node (layer 0)
    step: jax.Array
    rng: PRNGKeyArray


def streaming_ipc_step(state, observation, *, T, gamma, alpha, variant, ema_beta):
    """One observation step for streaming iPC.

    All keyword arguments are static (traced once per unique combination).
    """
    x_input, label = observation
    y_target = jax.nn.one_hot(label, NUM_CLASSES)

    network = state.network
    value_nodes = state.value_nodes
    output_node = state.output_node

    # Online accuracy: predict BEFORE any update
    predicted = jnp.argmax(output_node)
    online_correct = (predicted == label).astype(jnp.float32)

    # Optionally reinitialize value nodes
    if variant == 'forward_init':
        value_nodes = pc_forward_pass(network, x_input)
        f = ACTIVATION_MAP[network.activation_name]
        output_node = network.weights[0] @ f(value_nodes[0])
    elif variant == 'ema':
        fwd_nodes = pc_forward_pass(network, x_input)
        f = ACTIVATION_MAP[network.activation_name]
        fwd_output = network.weights[0] @ f(fwd_nodes[0])
        value_nodes = [
            ema_beta * vn + (1.0 - ema_beta) * fn
            for vn, fn in zip(value_nodes, fwd_nodes)
        ]
        output_node = ema_beta * output_node + (1.0 - ema_beta) * fwd_output
    # else 'streaming': keep value_nodes and output_node as-is

    # Run T iPC steps (Python for-loop, unrolled at trace time)
    info = None
    for _t in range(T):
        network, value_nodes, output_node, info = ipc_step(
            network, value_nodes, output_node, x_input, gamma, alpha,
            has_target=True, y_target=y_target,
        )

    metrics = PCStepMetrics(
        online_correct=online_correct,
        total_energy=info['total_energy'],
        layer_errors=info['layer_errors'],
        weight_update_norms=info['weight_update_norms'],
    )

    new_state = tree_replace(
        state,
        network=network,
        value_nodes=value_nodes,
        output_node=output_node,
        step=state.step + 1,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Prequential iPC step
# ---------------------------------------------------------------------------

class PrequentialPCStepMetrics(eqx.Module):
    """Per-sample metrics from the prequential PC training loop."""
    correct: jax.Array            # scalar: 1.0 if prediction correct after inference
    infer_loss: jax.Array         # scalar: MSE between output_node and label after inference
    total_energy: jax.Array       # scalar: energy after final learning step
    layer_errors: jax.Array       # (max_layers,): per-layer squared error norms
    weight_update_norms: jax.Array  # (max_layers,): per-layer weight update norms


def prequential_ipc_step(state, observation, *, T_infer, T_learn, T_rest, gamma,
                          alpha, variant, ema_beta, infer_weight_update):
    """One prequential sample step: inference, learning, then rest.

    Per-sample sequence:
      1. T_infer steps: input clamped, no label (layer 0 free)
      2. Measure prediction (prequential accuracy/loss)
      3. T_learn steps: input + label clamped
      4. T_rest steps: blank input (zeros), no label (layer 0 free)

    All keyword arguments are static (traced once per unique combination).
    """
    x_input, label = observation
    y_target = jax.nn.one_hot(label, NUM_CLASSES)
    blank_input = jnp.zeros_like(x_input)

    network = state.network
    value_nodes = state.value_nodes
    output_node = state.output_node

    # Optionally reinitialize value nodes for the new sample
    if variant == 'forward_init':
        value_nodes = pc_forward_pass(network, x_input)
        f = ACTIVATION_MAP[network.activation_name]
        output_node = network.weights[0] @ f(value_nodes[0])
    elif variant == 'ema':
        fwd_nodes = pc_forward_pass(network, x_input)
        f = ACTIVATION_MAP[network.activation_name]
        fwd_output = network.weights[0] @ f(fwd_nodes[0])
        value_nodes = [
            ema_beta * vn + (1.0 - ema_beta) * fn
            for vn, fn in zip(value_nodes, fwd_nodes)
        ]
        output_node = ema_beta * output_node + (1.0 - ema_beta) * fwd_output
    # else 'streaming': keep as-is

    # Phase 1: Inference — present input without label (layer 0 free)
    unsupervised_alpha = alpha if infer_weight_update else 0.0
    for _ in range(T_infer):
        network, value_nodes, output_node, _ = ipc_step(
            network, value_nodes, output_node, x_input, gamma, unsupervised_alpha,
            has_target=False,
        )

    # Measure prediction at end of inference phase
    predicted = jnp.argmax(output_node)
    correct = (predicted == label).astype(jnp.float32)
    infer_loss = jnp.mean(jnp.square(output_node - y_target))

    # Phase 2: Learning — present input with label (layer 0 clamped)
    info = None
    for _ in range(T_learn):
        network, value_nodes, output_node, info = ipc_step(
            network, value_nodes, output_node, x_input, gamma, alpha,
            has_target=True, y_target=y_target,
        )

    # Phase 3: Rest — no input, no label (blank input, layer 0 free)
    for _ in range(T_rest):
        network, value_nodes, output_node, info = ipc_step(
            network, value_nodes, output_node, blank_input, gamma, unsupervised_alpha,
            has_target=False,
        )

    metrics = PrequentialPCStepMetrics(
        correct=correct,
        infer_loss=infer_loss,
        total_energy=info['total_energy'],
        layer_errors=info['layer_errors'],
        weight_update_norms=info['weight_update_norms'],
    )

    new_state = tree_replace(
        state,
        network=network,
        value_nodes=value_nodes,
        output_node=output_node,
        step=state.step + 1,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# BP baseline Train State and step
# ---------------------------------------------------------------------------

class BPTrainState(eqx.Module):
    model: MLP
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray


def bp_train_step(state, observation):
    """One SGD/BP training step on a single observation."""
    x_input, label = observation
    one_hot = jax.nn.one_hot(label, NUM_CLASSES)

    def loss_fn(model):
        x = jnp.reshape(x_input, (-1,))
        logits, _ = model(x)
        loss = jnp.mean(jnp.square(logits - one_hot))
        return loss, logits

    (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)

    updates, new_opt = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    predicted = jnp.argmax(logits)
    correct = (predicted == label).astype(jnp.float32)

    new_state = tree_replace(state, model=new_model, optimizer=new_opt, step=state.step + 1)
    return new_state, BPStepMetrics(loss=loss, correct=correct)


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def prepare_ipc_experiment(cfg):
    """Create per-seed PCTrainState and data streams."""
    seeds = cfg.seed
    all_images, samples_per_class, num_classes, input_dim = load_mnist()

    # Layer dims: (output, hidden..., input) = (10, hidden_dim, ..., 784)
    n_hidden = cfg.model.num_layers - 2  # num_layers includes input + output
    layer_dims = (
        [NUM_CLASSES]
        + [cfg.model.hidden_dim] * n_hidden
        + [INPUT_DIM]
    )

    streams = []
    train_states = []

    for seed in seeds:
        rng = jax.random.key(seed)

        if cfg.data.ordering == 'shuffled':
            stream = ShuffledMNISTStream(
                all_images, samples_per_class, num_classes, seed=seed,
            )
        else:
            stream = RotatingMNISTStream(
                all_images, samples_per_class, num_classes,
                delta_degrees=cfg.data.delta_degrees, seed=seed,
            )
        streams.append(stream)

        network = init_pc_network(
            layer_dims=layer_dims,
            activation=cfg.model.activation,
            key=rng_from_string(rng, 'model'),
        )

        # Initialize value nodes and output node to zeros
        L = network.num_layers
        value_nodes = [jnp.zeros(layer_dims[l]) for l in range(1, L)]
        output_node = jnp.zeros(layer_dims[0])

        train_states.append(PCTrainState(
            network=network,
            value_nodes=value_nodes,
            output_node=output_node,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = sum(w.size for w in train_states[0].network.weights)
    print(f'PCNetwork: layers={layer_dims}, params={n_params}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params


def prepare_bp_experiment(cfg):
    """Create per-seed BPTrainState and data streams."""
    seeds = cfg.seed
    all_images, samples_per_class, num_classes, input_dim = load_mnist()

    streams = []
    train_states = []

    for seed in seeds:
        rng = jax.random.key(seed)

        if cfg.data.ordering == 'shuffled':
            stream = ShuffledMNISTStream(
                all_images, samples_per_class, num_classes, seed=seed,
            )
        else:
            stream = RotatingMNISTStream(
                all_images, samples_per_class, num_classes,
                delta_degrees=cfg.data.delta_degrees, seed=seed,
            )
        streams.append(stream)

        model = MLP(
            input_dim=INPUT_DIM,
            output_dim=NUM_CLASSES,
            n_layers=cfg.model.num_layers - 1,  # MLP n_layers = weight layers
            hidden_dim=cfg.model.hidden_dim,
            weight_init_method='lecun_uniform',
            activation=cfg.model.activation,
            key=rng_from_string(rng, 'model'),
        )
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)

        train_states.append(BPTrainState(
            model=model,
            optimizer=optimizer,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    print(f'MLP (BP): params={n_params}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_ipc_experiment(cfg, train_state, streams):
    """Outer training loop for iPC variants."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq
    T = cfg.ipc.T
    gamma = cfg.ipc.gamma
    alpha = cfg.ipc.alpha
    variant = cfg.ipc.variant
    ema_beta = cfg.ipc.ema_beta

    def _step(state, observation):
        return streaming_ipc_step(
            state, observation,
            T=T, gamma=gamma, alpha=alpha, variant=variant, ema_beta=ema_beta,
        )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_energies = []
    all_accuracies = []
    pbar = tqdm(total=cfg.train.total_steps, desc='iPC Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for _ in range(num_scans):
        # Pre-sample data on CPU
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))   # (n_seeds, log_freq, 784)
        labels = jnp.array(np.stack([b[1] for b in batch]))   # (n_seeds, log_freq)

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        # metrics: PCStepMetrics with shapes (n_seeds, log_freq, ...)
        per_seed_acc = metrics.online_correct.mean(axis=1)     # (n_seeds,)
        per_seed_energy = metrics.total_energy.mean(axis=1)    # (n_seeds,)
        mean_acc = float(per_seed_acc.mean())
        mean_energy = float(per_seed_energy.mean())

        step = int(train_state.step[0].item())

        if logging_active:
            def _log(mean_acc, mean_energy, per_seed_acc, per_seed_energy, step):
                log_metrics({
                    'accuracy': mean_acc,
                    'energy': mean_energy,
                    'accuracy_std': float(per_seed_acc.std()),
                }, cfg, step=step)
                log_child_metrics({
                    'accuracy': per_seed_acc.tolist(),
                    'energy': per_seed_energy.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, mean_acc, mean_energy,
                np.array(per_seed_acc), np.array(per_seed_energy), step,
            ))

        all_energies.append(mean_energy)
        all_accuracies.append(mean_acc)
        pbar.update(log_freq)
        pbar.set_postfix({'energy': f'{mean_energy:.2f}', 'acc': f'{mean_acc:.4f}'})

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_energies, all_accuracies


def run_prequential_experiment(cfg, train_state, streams):
    """Outer training loop for prequential iPC evaluation."""
    log_freq = cfg.train.log_freq
    total_samples = cfg.train.total_samples
    num_scans = total_samples // log_freq
    T_infer = cfg.ipc.T_infer
    T_learn = cfg.ipc.T_learn
    T_rest = cfg.ipc.T_rest
    gamma = cfg.ipc.gamma
    alpha = cfg.ipc.alpha
    variant = cfg.ipc.variant
    ema_beta = cfg.ipc.ema_beta
    infer_weight_update = cfg.ipc.infer_weight_update

    def _step(state, observation):
        return prequential_ipc_step(
            state, observation,
            T_infer=T_infer, T_learn=T_learn, T_rest=T_rest,
            gamma=gamma, alpha=alpha, variant=variant, ema_beta=ema_beta,
            infer_weight_update=infer_weight_update,
        )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_energies = []
    all_accuracies = []
    pbar = tqdm(total=total_samples, desc='Prequential iPC Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for _ in range(num_scans):
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))
        labels = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        per_seed_acc = metrics.correct.mean(axis=1)
        per_seed_loss = metrics.infer_loss.mean(axis=1)
        per_seed_energy = metrics.total_energy.mean(axis=1)
        mean_acc = float(per_seed_acc.mean())
        mean_loss = float(per_seed_loss.mean())
        mean_energy = float(per_seed_energy.mean())

        step = int(train_state.step[0].item())

        if logging_active:
            def _log(mean_acc, mean_loss, mean_energy,
                     per_seed_acc, per_seed_loss, per_seed_energy, step):
                log_metrics({
                    'accuracy': mean_acc,
                    'infer_loss': mean_loss,
                    'energy': mean_energy,
                    'accuracy_std': float(per_seed_acc.std()),
                }, cfg, step=step)
                log_child_metrics({
                    'accuracy': per_seed_acc.tolist(),
                    'infer_loss': per_seed_loss.tolist(),
                    'energy': per_seed_energy.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, mean_acc, mean_loss, mean_energy,
                np.array(per_seed_acc), np.array(per_seed_loss),
                np.array(per_seed_energy), step,
            ))

        all_losses.append(mean_loss)
        all_energies.append(mean_energy)
        all_accuracies.append(mean_acc)
        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_losses, all_energies, all_accuracies


def run_bp_experiment(cfg, train_state, streams):
    """Outer training loop for BP baseline."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    def scan_steps(state, data):
        return jax.lax.scan(bp_train_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_accuracies = []
    pbar = tqdm(total=cfg.train.total_steps, desc='BP Training')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for _ in range(num_scans):
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))
        labels = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        per_seed_loss = metrics.loss.mean(axis=1)
        per_seed_acc = metrics.correct.mean(axis=1)
        mean_loss = float(per_seed_loss.mean())
        mean_acc = float(per_seed_acc.mean())

        step = int(train_state.step[0].item())

        if logging_active:
            def _log(mean_loss, mean_acc, per_seed_loss, per_seed_acc, step):
                log_metrics({
                    'loss': mean_loss,
                    'accuracy': mean_acc,
                }, cfg, step=step)
                log_child_metrics({
                    'loss': per_seed_loss.tolist(),
                    'accuracy': per_seed_acc.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, mean_loss, mean_acc,
                np.array(per_seed_loss), np.array(per_seed_acc), step,
            ))

        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)
        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
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
    init_child_runs(cfg.seed, cfg)

    algorithm = cfg.get('algorithm', 'ipc')

    if algorithm == 'bp':
        train_state, streams, n_params = prepare_bp_experiment(cfg)
        train_state, all_losses, all_accuracies = run_bp_experiment(
            cfg, train_state, streams)

        n_tail = max(1, len(all_losses) // 10)
        summary = {
            'average_loss': float(np.mean(all_losses)),
            'asymptotic_loss': float(np.mean(all_losses[-n_tail:])),
            'asymptotic_accuracy': float(np.mean(all_accuracies[-n_tail:])),
            'num_params': n_params,
        }
    elif algorithm == 'prequential_ipc':
        train_state, streams, n_params = prepare_ipc_experiment(cfg)
        train_state, all_losses, all_energies, all_accuracies = run_prequential_experiment(
            cfg, train_state, streams)

        n_tail = max(1, len(all_losses) // 10)
        summary = {
            'average_infer_loss': float(np.mean(all_losses)),
            'asymptotic_infer_loss': float(np.mean(all_losses[-n_tail:])),
            'asymptotic_accuracy': float(np.mean(all_accuracies[-n_tail:])),
            'num_params': n_params,
        }
    else:
        train_state, streams, n_params = prepare_ipc_experiment(cfg)
        train_state, all_energies, all_accuracies = run_ipc_experiment(
            cfg, train_state, streams)

        n_tail = max(1, len(all_energies) // 10)
        summary = {
            'average_energy': float(np.mean(all_energies)),
            'asymptotic_energy': float(np.mean(all_energies[-n_tail:])),
            'asymptotic_accuracy': float(np.mean(all_accuracies[-n_tail:])),
            'num_params': n_params,
        }

    for k, v in summary.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')

    log_metrics(summary, cfg)
    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
