"""TD(lambda) value prediction with streaming iPC on the audio prediction benchmark.

Trains a streaming iPC network (or BP baseline) to predict discounted returns
from binary audio observations using TD(lambda) with eligibility traces.

Key design:
- The iPC weight gradient at equilibrium equals delta * nabla_V, so we compute
  nabla_V directly from the settled value nodes (closed-form for 2-layer networks)
  and maintain eligibility traces separately from the iPC inference.
- Value nodes settle via ipc_step_grads (no weight update during inference).
- Weight updates use: theta += alpha * delta * trace.
- Sequential formulation: V(s_t) comes from settling, V_old from previous step.
"""

from concurrent.futures import ThreadPoolExecutor
import os

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
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees, tree_replace
from phd.feature_search.jax_core.experiment_helpers import set_seed, rng_from_string
from phd.research_utils.logging import (
    init_experiment,
    init_child_runs,
    log_metrics,
    log_child_metrics,
    finish_child_runs,
    finish_experiment,
)
from phd.sandbox.predictive_coding.audio_data import load_audio_data, compute_true_returns
from phd.sandbox.predictive_coding.models import (
    PCNetwork, init_pc_network, ipc_step_grads, ACTIVATION_DERIV_MAP,
)


SCAN_UNROLL = 4
OUTPUT_DIM = 1
INPUT_DIM = 2500  # 50 freq bins * 50 mag bins


# ---------------------------------------------------------------------------
# Metrics containers
# ---------------------------------------------------------------------------

class TDiPCStepMetrics(eqx.Module):
    td_error_sq: jax.Array      # scalar
    msve: jax.Array             # scalar: (V_hat(s_t) - G_t)^2
    total_energy: jax.Array     # scalar
    v_prediction: jax.Array     # scalar: V_hat(s_t) after settling


class TDBPStepMetrics(eqx.Module):
    td_error_sq: jax.Array
    msve: jax.Array             # scalar: (V_hat(s_t) - G_t)^2
    v_prediction: jax.Array


# ---------------------------------------------------------------------------
# iPC Train State and TD(lambda) step
# ---------------------------------------------------------------------------

class PCTrainState(eqx.Module):
    network: PCNetwork
    value_nodes: list           # persistent across observations
    traces: list                # per-layer eligibility traces (same shapes as weights)
    v_old: jax.Array            # scalar: V from previous step
    grad_v_old: list            # per-layer nabla_V from previous step
    step: jax.Array
    rng: PRNGKeyArray


def _read_value_prediction(network, value_nodes):
    """Read scalar value prediction: V = weights[0] @ f(value_nodes[0])."""
    f = ACTIVATION_MAP[network.activation_name]
    return (network.weights[0] @ f(value_nodes[0]))[0]


def _compute_grad_v(network, value_nodes, s_t):
    """Compute nabla_theta V from settled value nodes (2-layer closed form).

    Layer 0: nabla_{theta^0} V = f(x^1)^T  (shape matches weights[0])
    Layer 1: nabla_{theta^1} V = [f'(x^1) * theta^0^T] @ f(s_t)^T
    """
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]

    L = network.num_layers
    fx1 = f(value_nodes[0])  # f(x^(1))
    fs = f(s_t)              # f(s_t) = f(x^(L))

    grads = []

    # Layer 0: nabla_{theta^0} V = outer(1, f(x^1)) = f(x^1) reshaped
    # weights[0] shape: (1, d1), so grad shape: (1, d1)
    grads.append(fx1.reshape(1, -1))

    # Layers 1..L-1: backprop through settled value nodes
    # For 2-layer (L=2): nabla_{theta^1} V = [f'(x^1) * theta^0^T] outer f(s)
    # General: propagate backward signal through value node chain
    #   b^(1) = f'(x^1) * (theta^0)^T  (shape d1,)
    #   nabla_{theta^1} V = outer(b^(1), f(x^2))
    # For deeper networks, continue the chain.
    backward_signal = f_prime(value_nodes[0]) * network.weights[0].T.reshape(-1)

    if L >= 2:
        # Layer 1 gradient
        grads.append(jnp.outer(backward_signal, fs))

    # For layers 2..L-1 (deeper networks)
    for l in range(2, L):
        backward_signal = (
            f_prime(value_nodes[l - 1])
            * (network.weights[l - 1].T @ backward_signal)
        )
        f_above = f(value_nodes[l]) if l < L - 1 else fs
        grads.append(jnp.outer(backward_signal, f_above))

    return grads


def td_ipc_lambda_step(state, observation, *, T, gamma_inf, alpha, gamma_td, td_lambda):
    """One TD(lambda) step for streaming iPC.

    Sequential algorithm:
        (a) Update trace with stored nabla_V from previous step
        (b) Clamp s_t, run T inference steps (value nodes only)
        (c) Read V_t from settled state
        (d) TD error: delta = r_{t-1} + gamma * V_t - V_old
        (e) Update weights: theta += alpha * delta * trace
        (f) Compute nabla_V_t from settled state
        (g) Store V_old, nabla_V_old
    """
    s_t, r_prev, g_t = observation  # g_t = true return G_t for state s_t

    network = state.network
    value_nodes = state.value_nodes

    # (a) Update eligibility traces with stored grad_v_old
    new_traces = [
        gamma_td * td_lambda * e + g
        for e, g in zip(state.traces, state.grad_v_old)
    ]

    # (b) Clamp s_t, run T inference steps (value nodes only)
    # Clamp v_old at layer 0 so value nodes settle toward the new input
    y_clamp = jnp.array([state.v_old])
    for _t in range(T):
        value_nodes, _weight_grads, info = ipc_step_grads(
            network, value_nodes, s_t, y_clamp, gamma_inf,
        )

    # (c) Read V_t from settled state
    v_t = _read_value_prediction(network, value_nodes)

    # (d) TD error
    delta = r_prev + gamma_td * v_t - state.v_old

    # (e) Update weights: theta += alpha * delta * trace
    new_weights = [
        w + alpha * delta * e
        for w, e in zip(network.weights, new_traces)
    ]
    new_network = PCNetwork(
        layer_dims=network.layer_dims,
        num_layers=network.num_layers,
        activation_name=network.activation_name,
        weights=new_weights,
    )

    # (f) Compute nabla_V_t from settled state (uses updated weights)
    grad_v_new = _compute_grad_v(new_network, value_nodes, s_t)

    # Metrics
    metrics = TDiPCStepMetrics(
        td_error_sq=jnp.square(delta),
        msve=jnp.square(v_t - g_t),
        total_energy=info['total_energy'],
        v_prediction=v_t,
    )

    new_state = PCTrainState(
        network=new_network,
        value_nodes=value_nodes,
        traces=new_traces,
        v_old=v_t,
        grad_v_old=grad_v_new,
        step=state.step + 1,
        rng=state.rng,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# BP baseline Train State and TD(lambda) step
# ---------------------------------------------------------------------------

class BPTrainState(eqx.Module):
    model: MLP
    trace: MLP          # eligibility trace (same structure as model, zero-init)
    v_old: jax.Array
    grad_v_old: MLP     # stored nabla_V from previous step
    step: jax.Array
    rng: PRNGKeyArray


def _bp_predict(model, s):
    """Forward pass returning scalar value prediction."""
    v, _ = model(s)
    return v[0]


def _bp_compute_grad_v(model, s):
    """Compute nabla_theta V(s) via backprop."""
    return eqx.filter_grad(lambda m: _bp_predict(m, s))(model)


def td_bp_lambda_step(state, observation, *, alpha, gamma_td, td_lambda):
    """One TD(lambda) step for BP baseline."""
    s_t, r_prev, g_t = observation

    # (a) Update trace with stored grad_v_old
    new_trace = jax.tree.map(
        lambda e, g: gamma_td * td_lambda * e + g,
        state.trace, state.grad_v_old,
    )

    # (b) Forward pass
    v_t = _bp_predict(state.model, s_t)

    # (c) TD error
    delta = r_prev + gamma_td * v_t - state.v_old

    # (d) Update weights: theta += alpha * delta * trace
    new_model = jax.tree.map(
        lambda w, e: w + alpha * delta * e,
        state.model, new_trace,
    )

    # (e) Compute nabla_V_t (uses updated model)
    grad_v_new = _bp_compute_grad_v(new_model, s_t)

    metrics = TDBPStepMetrics(
        td_error_sq=jnp.square(delta),
        msve=jnp.square(v_t - g_t),
        v_prediction=v_t,
    )

    new_state = BPTrainState(
        model=new_model,
        trace=new_trace,
        v_old=v_t,
        grad_v_old=grad_v_new,
        step=state.step + 1,
        rng=state.rng,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _init_ipc_step0(network, value_nodes, s_0, gamma_inf, T):
    """Process s_0: run inference and compute initial V_0 and nabla_V_0."""
    f = ACTIVATION_MAP[network.activation_name]

    # Run inference with a zero target (arbitrary, just to settle nodes)
    y_clamp = jnp.zeros(OUTPUT_DIM)
    for _t in range(T):
        value_nodes, _, _ = ipc_step_grads(
            network, value_nodes, s_0, y_clamp, gamma_inf,
        )

    v_0 = _read_value_prediction(network, value_nodes)
    grad_v_0 = _compute_grad_v(network, value_nodes, s_0)

    return value_nodes, v_0, grad_v_0


def prepare_ipc_experiment(cfg, s_0_per_seed):
    """Create batched PCTrainState for all seeds.

    Args:
        cfg: Hydra config.
        s_0_per_seed: (n_seeds, INPUT_DIM) float32 array of initial observations.
            All seeds see the same s_0 so this is broadcast, but we pass per-seed
            for the vmap initialization.
    """
    seeds = cfg.seed
    n_hidden = cfg.model.num_layers - 2
    layer_dims = (
        [OUTPUT_DIM]
        + [cfg.model.hidden_dim] * n_hidden
        + [INPUT_DIM]
    )

    T = cfg.ipc.T
    gamma_inf = cfg.ipc.gamma

    train_states = []
    for i, seed in enumerate(seeds):
        rng = jax.random.key(seed)

        network = init_pc_network(
            layer_dims=layer_dims,
            activation=cfg.model.activation,
            key=rng_from_string(rng, 'model'),
        )

        L = network.num_layers
        value_nodes = [jnp.zeros(layer_dims[l]) for l in range(1, L)]

        # Process s_0 to initialize V_0 and nabla_V_0
        s_0 = jnp.array(s_0_per_seed, dtype=jnp.float32)
        value_nodes, v_0, grad_v_0 = _init_ipc_step0(
            network, value_nodes, s_0, gamma_inf, T,
        )

        # Initialize traces to zeros (same shapes as weights)
        traces = [jnp.zeros_like(w) for w in network.weights]

        train_states.append(PCTrainState(
            network=network,
            value_nodes=value_nodes,
            traces=traces,
            v_old=v_0,
            grad_v_old=grad_v_0,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = sum(w.size for w in train_states[0].network.weights)
    print(f'PCNetwork: layers={layer_dims}, params={n_params}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, n_params


def prepare_bp_experiment(cfg, s_0):
    """Create batched BPTrainState for all seeds."""
    seeds = cfg.seed
    train_states = []

    for seed in seeds:
        rng = jax.random.key(seed)

        model = MLP(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=cfg.model.num_layers - 1,  # MLP n_layers = weight layers
            hidden_dim=cfg.model.hidden_dim,
            weight_init_method='lecun_uniform',
            activation=cfg.model.activation,
            key=rng_from_string(rng, 'model'),
        )

        # Initialize trace to zeros (same pytree structure as model)
        trace = jax.tree.map(jnp.zeros_like, model)

        # Process s_0
        s_0_jax = jnp.array(s_0, dtype=jnp.float32)
        v_0 = _bp_predict(model, s_0_jax)
        grad_v_0 = _bp_compute_grad_v(model, s_0_jax)

        train_states.append(BPTrainState(
            model=model,
            trace=trace,
            v_old=v_0,
            grad_v_old=grad_v_0,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    print(f'MLP (BP): params={n_params}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, n_params


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def run_ipc_experiment(cfg, train_state, observations, rewards, true_returns):
    """Outer training loop for iPC TD(lambda)."""
    log_freq = cfg.train.log_freq
    n_data = len(rewards)
    n_seeds = len(cfg.seed)

    # Determine total steps
    n_usable = n_data - 1  # steps per epoch (s_0 processed in init)
    total_steps = cfg.train.total_steps
    if total_steps == -1:
        total_steps = n_usable
    num_scans = total_steps // log_freq

    T = cfg.ipc.T
    gamma_inf = cfg.ipc.gamma
    alpha = cfg.ipc.alpha
    gamma_td = cfg.env.gamma_td
    td_lambda = cfg.ipc.td_lambda

    def _step(state, observation):
        return td_ipc_lambda_step(
            state, observation,
            T=T, gamma_inf=gamma_inf, alpha=alpha,
            gamma_td=gamma_td, td_lambda=td_lambda,
        )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps, in_axes=(0, None)))

    all_td_errors = []
    all_msve = []
    pbar = tqdm(total=total_steps, desc='iPC TD(λ)')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for scan_idx in range(num_scans):
        # Data: s_t = observations[t+1], r_prev = rewards[t], g_t = true_returns[t+1]
        cursor = scan_idx * log_freq
        positions = np.arange(cursor, cursor + log_freq)
        s_t = jnp.array(observations[(positions + 1) % n_data], dtype=jnp.float32)
        r_prev = jnp.array(rewards[positions % n_data], dtype=jnp.float32)
        g_t = jnp.array(true_returns[(positions + 1) % n_data], dtype=jnp.float32)

        train_state, metrics = vmapped_scan(train_state, (s_t, r_prev, g_t))

        # Aggregate metrics
        per_seed_td_error = metrics.td_error_sq.mean(axis=1)  # (n_seeds,)
        per_seed_msve = metrics.msve.mean(axis=1)             # (n_seeds,)
        mean_td_error = float(per_seed_td_error.mean())
        mean_msve = float(per_seed_msve.mean())
        mean_v_pred = float(metrics.v_prediction.mean())
        mean_energy = float(metrics.total_energy.mean())

        step = int(train_state.step[0].item())
        epoch_idx = int(cursor // n_usable)

        if logging_active:
            def _log(mean_td_error, mean_msve, mean_v_pred, mean_energy,
                     epoch_idx, per_seed_td_error, per_seed_msve, step):
                log_metrics({
                    'td_error': mean_td_error,
                    'msve': mean_msve,
                    'mean_v_prediction': mean_v_pred,
                    'energy': mean_energy,
                    'epoch': epoch_idx,
                }, cfg, step=step)
                log_child_metrics({
                    'td_error': per_seed_td_error.tolist(),
                    'msve': per_seed_msve.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, mean_td_error, mean_msve, mean_v_pred, mean_energy,
                epoch_idx, np.array(per_seed_td_error), np.array(per_seed_msve), step,
            ))

        all_td_errors.append(mean_td_error)
        all_msve.append(mean_msve)
        pbar.update(log_freq)
        pbar.set_postfix({
            'epoch': epoch_idx, 'td_err': f'{mean_td_error:.4f}',
            'msve': f'{mean_msve:.4f}', 'energy': f'{mean_energy:.2f}',
            'V_mean': f'{mean_v_pred:.4f}',
        })

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_td_errors, all_msve


def run_bp_experiment(cfg, train_state, observations, rewards, true_returns):
    """Outer training loop for BP TD(lambda) baseline."""
    log_freq = cfg.train.log_freq
    n_data = len(rewards)
    n_seeds = len(cfg.seed)

    n_usable = n_data - 1
    total_steps = cfg.train.total_steps
    if total_steps == -1:
        total_steps = n_usable
    num_scans = total_steps // log_freq

    alpha = cfg.optimizer.learning_rate
    gamma_td = cfg.env.gamma_td
    td_lambda = cfg.optimizer.td_lambda

    def _step(state, observation):
        return td_bp_lambda_step(
            state, observation,
            alpha=alpha, gamma_td=gamma_td, td_lambda=td_lambda,
        )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps, in_axes=(0, None)))

    all_td_errors = []
    all_msve = []
    pbar = tqdm(total=total_steps, desc='BP TD(λ)')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for scan_idx in range(num_scans):
        cursor = scan_idx * log_freq
        positions = np.arange(cursor, cursor + log_freq)
        s_t = jnp.array(observations[(positions + 1) % n_data], dtype=jnp.float32)
        r_prev = jnp.array(rewards[positions % n_data], dtype=jnp.float32)
        g_t = jnp.array(true_returns[(positions + 1) % n_data], dtype=jnp.float32)

        train_state, metrics = vmapped_scan(train_state, (s_t, r_prev, g_t))

        per_seed_td_error = metrics.td_error_sq.mean(axis=1)
        per_seed_msve = metrics.msve.mean(axis=1)
        mean_td_error = float(per_seed_td_error.mean())
        mean_msve = float(per_seed_msve.mean())
        mean_v_pred = float(metrics.v_prediction.mean())

        step = int(train_state.step[0].item())
        epoch_idx = int(cursor // n_usable)

        if logging_active:
            def _log(mean_td_error, mean_msve, mean_v_pred,
                     epoch_idx, per_seed_td_error, per_seed_msve, step):
                log_metrics({
                    'td_error': mean_td_error,
                    'msve': mean_msve,
                    'mean_v_prediction': mean_v_pred,
                    'epoch': epoch_idx,
                }, cfg, step=step)
                log_child_metrics({
                    'td_error': per_seed_td_error.tolist(),
                    'msve': per_seed_msve.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, mean_td_error, mean_msve, mean_v_pred,
                epoch_idx, np.array(per_seed_td_error), np.array(per_seed_msve), step,
            ))

        all_td_errors.append(mean_td_error)
        all_msve.append(mean_msve)
        pbar.update(log_freq)
        pbar.set_postfix({
            'epoch': epoch_idx, 'td_err': f'{mean_td_error:.4f}',
            'msve': f'{mean_msve:.4f}', 'V_mean': f'{mean_v_pred:.4f}',
        })

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_td_errors, all_msve


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='audio_td_config', version_base='1.1')
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

    # Load and preprocess audio data
    observations, rewards, metadata = load_audio_data(cfg.data.data_dir)

    # Compute true returns for MSVE evaluation
    gamma_td = cfg.env.gamma_td
    print(f'Computing true returns (gamma={gamma_td})...')
    true_returns = compute_true_returns(rewards, gamma_td)
    print(f'True returns: mean={true_returns.mean():.6f}, '
          f'std={true_returns.std():.6f}, '
          f'max_abs={np.abs(true_returns).max():.6f}')

    s_0 = observations[0].astype(np.float32)

    algorithm = cfg.get('algorithm', 'ipc')

    if algorithm == 'bp':
        train_state, n_params = prepare_bp_experiment(cfg, s_0)
        train_state, all_td_errors, all_msve = run_bp_experiment(
            cfg, train_state, observations, rewards, true_returns,
        )
    else:
        train_state, n_params = prepare_ipc_experiment(cfg, s_0)
        train_state, all_td_errors, all_msve = run_ipc_experiment(
            cfg, train_state, observations, rewards, true_returns,
        )

    # Summary metrics
    n_tail = max(1, len(all_td_errors) // 10)
    n_msve_tail = max(1, len(all_msve) // 5)
    summary = {
        'asymptotic_td_error': float(np.mean(all_td_errors[-n_tail:])),
        'asymptotic_msve': float(np.mean(all_msve[-n_msve_tail:])) if all_msve else 0.0,
        'average_msve': float(np.mean(all_msve)) if all_msve else 0.0,
        'num_params': n_params,
    }

    for k, v in summary.items():
        print(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}')

    log_metrics(summary, cfg)
    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
