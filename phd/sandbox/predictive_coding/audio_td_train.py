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
    log_figures,
    finish_child_runs,
    finish_experiment,
)
from phd.sandbox.predictive_coding.audio_data import (
    load_audio_data, compute_true_returns, compute_observation_trace,
)
from phd.sandbox.predictive_coding.models import (
    PCNetwork, init_pc_network, pc_forward_pass, ipc_step_grads, ACTIVATION_DERIV_MAP,
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
    backward_signals: list      # persistent b[l] vectors for pipelined nabla_V
    obs_trace: jax.Array        # (INPUT_DIM,) observation feature trace
    step: jax.Array
    rng: PRNGKeyArray


def _read_value_prediction(network, value_nodes):
    """Read scalar value prediction: V = weights[0] @ f(value_nodes[0])."""
    f = ACTIVATION_MAP[network.activation_name]
    return (network.weights[0] @ f(value_nodes[0]))[0]


def _compute_grad_v(network, value_nodes, s_t):
    """Compute nabla_theta V from settled value nodes via backward signal.

    Propagates a unit signal backward from the output layer:
      b[1] = f'(x[1]) * theta[0]^T
      b[l] = f'(x[l]) * (theta[l-1]^T @ b[l-1])   for l = 2..L-1

    Then: nabla_{theta[l]} V = outer(b[l+1], f(x[l+1]))
    with convention b[0] = 1 so nabla_{theta[0]} V = f(x[1])^T.
    """
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    # Build x[l+1] activations for each weight layer: f(x[1]), f(x[2]), ..., f(x[L])
    # value_nodes[i] = x[i+1], so x[l+1] = value_nodes[l] for l < L-1, x[L] = s_t
    def get_f_presynaptic(l):
        """f(x[l+1]) — the presynaptic activation for weight layer l."""
        if l < L - 1:
            return f(value_nodes[l])
        else:
            return f(s_t)

    grads = []

    # Layer 0: nabla_{theta[0]} V = f(x[1])^T (b[0] = 1)
    grads.append(get_f_presynaptic(0).reshape(1, -1))

    if L < 2:
        return grads

    # Backward signal: b[1] = f'(x[1]) * theta[0]^T
    backward_signal = f_prime(value_nodes[0]) * network.weights[0].T.reshape(-1)

    # Layer 1: nabla_{theta[1]} V = outer(b[1], f(x[2]))
    grads.append(jnp.outer(backward_signal, get_f_presynaptic(1)))

    # Layers 2..L-1
    for l in range(2, L):
        backward_signal = (
            f_prime(value_nodes[l - 1])
            * (network.weights[l - 1].T @ backward_signal)
        )
        grads.append(jnp.outer(backward_signal, get_f_presynaptic(l)))

    return grads


def td_ipc_simultaneous_step(state, observation, *, T, gamma_inf, alpha, gamma_td,
                              td_lambda, use_obs_trace, obs_trace_decay):
    """One TD(lambda) step for streaming iPC with simultaneous updates.

    The TD error delta serves as the output error (epsilon[0] := delta),
    providing a bottom-up signal to value nodes — analogous to the
    supervised error in standard iPC.

    With T=1, all updates are truly simultaneous. With T>1, value nodes
    settle for T rounds (letting the delta signal penetrate deeper layers)
    before the trace and weight updates. Delta is computed once from the
    initial state and held fixed during settling.

    Layer convention:
      value_nodes[i] = x[i+1]  (0-indexed list -> 1-indexed layers)
      weights[l] shape (dim[l], dim[l+1]), predicts x[l] from f(x[l+1])
      Layer 0 = output (V), Layer L = input (s_t)
    """
    s_t, r_prev, g_t = observation

    network = state.network
    value_nodes = state.value_nodes
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    # --- Observation trace ---
    new_obs_trace = obs_trace_decay * state.obs_trace + s_t
    s_input = new_obs_trace * (1.0 - obs_trace_decay) if use_obs_trace else s_t

    # --- Compute V_t and delta from current state (once, before settling) ---
    v_t = (network.weights[0] @ f(value_nodes[0]))[0]
    delta = r_prev + gamma_td * v_t - state.v_old

    # --- Settle value nodes for T steps with eps[0] = delta ---
    for _t in range(T):
        # Recompute hidden errors from current value nodes
        errors = []
        for l in range(1, L):
            x_above = value_nodes[l] if l < L - 1 else s_input
            mu_l = network.weights[l] @ f(x_above)
            errors.append(value_nodes[l - 1] - mu_l)

        # Update value nodes
        new_value_nodes = []
        for i in range(len(value_nodes)):
            x_l = value_nodes[i]
            top_down = -errors[i] if i < len(errors) else jnp.zeros_like(x_l)
            eps_below = jnp.array([delta]) if i == 0 else errors[i - 1]
            bottom_up = f_prime(x_l) * (network.weights[i].T @ eps_below)
            new_value_nodes.append(x_l + gamma_inf * (top_down + bottom_up))
        value_nodes = new_value_nodes

    # Energy metric (from final errors)
    total_energy = jnp.square(delta) + sum(jnp.sum(e ** 2) for e in errors)

    # --- Trace and weight updates (once, after settling) ---

    # Eligibility traces
    new_traces = [
        gamma_td * td_lambda * e + g
        for e, g in zip(state.traces, state.grad_v_old)
    ]

    # Weights: theta += alpha * delta * trace
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

    # nabla_V for next step (from pre-update state)
    grad_v_new = _compute_grad_v(network, state.value_nodes, s_input)

    metrics = TDiPCStepMetrics(
        td_error_sq=jnp.square(delta),
        msve=jnp.square(v_t - g_t),
        total_energy=total_energy,
        v_prediction=v_t,
    )

    new_state = PCTrainState(
        network=new_network,
        value_nodes=new_value_nodes,
        traces=new_traces,
        v_old=v_t,
        grad_v_old=grad_v_new,
        backward_signals=state.backward_signals,
        obs_trace=new_obs_trace,
        step=state.step + 1,
        rng=state.rng,
    )
    return new_state, metrics


def td_ipc_pipelined_step(state, observation, *, gamma_inf, alpha, gamma_td,
                           td_lambda, use_obs_trace, obs_trace_decay):
    """One pipelined TD(lambda) step for streaming iPC.

    Fully local: each layer only communicates with its immediate neighbors.
    No inner loops, no full-depth backprop.

    The backward signal b[l] for nabla_V is persistent state that advances
    one layer per timestep:
      b[1] = f'(x[1]) * theta[0]^T              (fresh each step)
      b[l] = f'(x[l]) * theta[l-1]^T @ b_old[l-1]  (uses stored b from previous step)

    This means nabla_V at layer l is l-1 timesteps stale — exact for layer 0,
    one step stale for layer 1, etc. Under continuity this is a minor approximation.
    """
    s_t, r_prev, g_t = observation

    network = state.network
    value_nodes = state.value_nodes
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers

    # --- Observation trace ---
    new_obs_trace = obs_trace_decay * state.obs_trace + s_t
    s_input = new_obs_trace * (1.0 - obs_trace_decay) if use_obs_trace else s_t

    # --- Forward pass: predictions, errors, V_t, delta ---
    v_t = (network.weights[0] @ f(value_nodes[0]))[0]
    delta = r_prev + gamma_td * v_t - state.v_old

    errors = []
    for l in range(1, L):
        x_above = value_nodes[l] if l < L - 1 else s_input
        mu_l = network.weights[l] @ f(x_above)
        errors.append(value_nodes[l - 1] - mu_l)

    total_energy = jnp.square(delta) + sum(jnp.sum(e ** 2) for e in errors)

    # --- Value node update (one step, eps[0] = delta) ---
    new_value_nodes = []
    for i in range(len(value_nodes)):
        x_l = value_nodes[i]
        top_down = -errors[i] if i < len(errors) else jnp.zeros_like(x_l)
        eps_below = jnp.array([delta]) if i == 0 else errors[i - 1]
        bottom_up = f_prime(x_l) * (network.weights[i].T @ eps_below)
        new_value_nodes.append(x_l + gamma_inf * (top_down + bottom_up))

    # --- Eligibility trace update ---
    new_traces = [
        gamma_td * td_lambda * e + g
        for e, g in zip(state.traces, state.grad_v_old)
    ]

    # --- Weight update ---
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

    # --- Pipelined backward signal for nabla_V ---
    # b[1] is always fresh (only needs layer 0 info)
    # b[l] for l >= 2 uses stored b[l-1] from previous timestep
    old_b = state.backward_signals
    new_b = []
    grad_v_new = []

    # Layer 0: nabla_{theta[0]} V = f(x[1])^T (no backward signal needed)
    grad_v_new.append(f(value_nodes[0]).reshape(1, -1))

    if L >= 2:
        # b[1] = f'(x[1]) * theta[0]^T — always fresh
        b_1 = f_prime(value_nodes[0]) * network.weights[0].T.reshape(-1)
        new_b.append(b_1)

        def get_f_presynaptic(l):
            if l < L - 1:
                return f(value_nodes[l])
            else:
                return f(s_input)

        # nabla_{theta[1]} V = outer(b[1], f(x[2]))
        grad_v_new.append(jnp.outer(b_1, get_f_presynaptic(1)))

        # Layers 2..L-1: pipeline using stored b from previous step
        for l in range(2, L):
            b_l = f_prime(value_nodes[l - 1]) * (network.weights[l - 1].T @ old_b[l - 2])
            new_b.append(b_l)
            grad_v_new.append(jnp.outer(b_l, get_f_presynaptic(l)))

    metrics = TDiPCStepMetrics(
        td_error_sq=jnp.square(delta),
        msve=jnp.square(v_t - g_t),
        total_energy=total_energy,
        v_prediction=v_t,
    )

    new_state = PCTrainState(
        network=new_network,
        value_nodes=new_value_nodes,
        traces=new_traces,
        v_old=v_t,
        grad_v_old=grad_v_new,
        backward_signals=new_b,
        obs_trace=new_obs_trace,
        step=state.step + 1,
        rng=state.rng,
    )
    return new_state, metrics


def td_ipc_lambda_step(state, observation, *, T, gamma_inf, alpha, gamma_td,
                        td_lambda, use_obs_trace, obs_trace_decay):
    """One TD(lambda) step for streaming iPC.

    Sequential algorithm:
        (a) Update trace with stored nabla_V from previous step
        (b) Update observation trace, choose network input
        (c) Clamp s_input, run T inference steps (value nodes only)
        (d) Read V_t from settled state
        (e) TD error: delta = r_{t-1} + gamma * V_t - V_old
        (f) Update weights: theta += alpha * delta * trace
        (g) Compute nabla_V_t from settled state
        (h) Store V_old, nabla_V_old, obs_trace
    """
    s_t, r_prev, g_t = observation  # g_t = true return G_t for state s_t

    network = state.network
    value_nodes = state.value_nodes

    # (a) Update eligibility traces with stored grad_v_old
    new_traces = [
        gamma_td * td_lambda * e + g
        for e, g in zip(state.traces, state.grad_v_old)
    ]

    # (b) Update observation trace and choose network input
    new_obs_trace = obs_trace_decay * state.obs_trace + s_t
    if use_obs_trace:
        s_input = new_obs_trace * (1.0 - obs_trace_decay)  # normalize to ~[0, 1]
    else:
        s_input = s_t

    # (c) Clamp s_input, run T inference steps (value nodes only)
    # Clamp v_old at layer 0 so value nodes settle toward the new input
    y_clamp = jnp.array([state.v_old])
    for _t in range(T):
        value_nodes, _weight_grads, info = ipc_step_grads(
            network, value_nodes, s_input, y_clamp, gamma_inf,
        )

    # (d) Read V_t from settled state
    v_t = _read_value_prediction(network, value_nodes)

    # (e) TD error
    delta = r_prev + gamma_td * v_t - state.v_old

    # (f) Update weights: theta += alpha * delta * trace
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

    # (g) Compute nabla_V_t from settled state (uses updated weights)
    grad_v_new = _compute_grad_v(new_network, value_nodes, s_input)

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
        backward_signals=state.backward_signals,
        obs_trace=new_obs_trace,
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
    obs_trace: jax.Array  # (INPUT_DIM,) observation feature trace
    step: jax.Array
    rng: PRNGKeyArray


def _bp_predict(model, s):
    """Forward pass returning scalar value prediction."""
    v, _ = model(s)
    return v[0]


def _bp_compute_grad_v(model, s):
    """Compute nabla_theta V(s) via backprop."""
    return eqx.filter_grad(lambda m: _bp_predict(m, s))(model)


def td_bp_lambda_step(state, observation, *, alpha, gamma_td, td_lambda,
                       use_obs_trace, obs_trace_decay):
    """One TD(lambda) step for BP baseline."""
    s_t, r_prev, g_t = observation

    # (a) Update trace with stored grad_v_old
    new_trace = jax.tree.map(
        lambda e, g: gamma_td * td_lambda * e + g,
        state.trace, state.grad_v_old,
    )

    # (b) Update observation trace and choose network input
    new_obs_trace = obs_trace_decay * state.obs_trace + s_t
    if use_obs_trace:
        s_input = new_obs_trace * (1.0 - obs_trace_decay)  # normalize to ~[0, 1]
    else:
        s_input = s_t

    # (c) Forward pass
    v_t = _bp_predict(state.model, s_input)

    # (d) TD error
    delta = r_prev + gamma_td * v_t - state.v_old

    # (e) Update weights: theta += alpha * delta * trace
    new_model = jax.tree.map(
        lambda w, e: w + alpha * delta * e,
        state.model, new_trace,
    )

    # (f) Compute nabla_V_t (uses updated model)
    grad_v_new = _bp_compute_grad_v(new_model, s_input)

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
        obs_trace=new_obs_trace,
        step=state.step + 1,
        rng=state.rng,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _init_ipc_step0(network, value_nodes, s_0, gamma_inf, T,
                     use_obs_trace, obs_trace_decay):
    """Process s_0: run inference and compute initial V_0 and nabla_V_0."""
    # Initialize observation trace with s_0
    obs_trace = s_0.copy()
    if use_obs_trace:
        s_input = obs_trace * (1.0 - obs_trace_decay)
    else:
        s_input = s_0

    # Run inference with a zero target (arbitrary, just to settle nodes)
    y_clamp = jnp.zeros(OUTPUT_DIM)
    for _t in range(T):
        value_nodes, _, _ = ipc_step_grads(
            network, value_nodes, s_input, y_clamp, gamma_inf,
        )

    v_0 = _read_value_prediction(network, value_nodes)
    grad_v_0 = _compute_grad_v(network, value_nodes, s_input)

    # Initialize backward signals for pipelined nabla_V
    f = ACTIVATION_MAP[network.activation_name]
    f_prime = ACTIVATION_DERIV_MAP[network.activation_name]
    L = network.num_layers
    backward_signals = []
    if L >= 2:
        b = f_prime(value_nodes[0]) * network.weights[0].T.reshape(-1)
        backward_signals.append(b)
        for l in range(2, L):
            b = f_prime(value_nodes[l - 1]) * (network.weights[l - 1].T @ b)
            backward_signals.append(b)

    return value_nodes, v_0, grad_v_0, obs_trace, backward_signals


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
    use_obs_trace = cfg.data.get('observation_trace', False)
    obs_trace_decay = cfg.data.get('obs_trace_decay', 0.95)

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
        value_nodes, v_0, grad_v_0, obs_trace, backward_signals = _init_ipc_step0(
            network, value_nodes, s_0, gamma_inf, T,
            use_obs_trace, obs_trace_decay,
        )

        # Initialize traces to zeros (same shapes as weights)
        traces = [jnp.zeros_like(w) for w in network.weights]

        train_states.append(PCTrainState(
            network=network,
            value_nodes=value_nodes,
            traces=traces,
            v_old=v_0,
            grad_v_old=grad_v_0,
            backward_signals=backward_signals,
            obs_trace=obs_trace,
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
    use_obs_trace = cfg.data.get('observation_trace', False)
    obs_trace_decay = cfg.data.get('obs_trace_decay', 0.95)
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
        obs_trace = s_0_jax.copy()
        s_input = obs_trace * (1.0 - obs_trace_decay) if use_obs_trace else s_0_jax
        v_0 = _bp_predict(model, s_input)
        grad_v_0 = _bp_compute_grad_v(model, s_input)

        train_states.append(BPTrainState(
            model=model,
            trace=trace,
            v_old=v_0,
            grad_v_old=grad_v_0,
            obs_trace=obs_trace,
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
    use_obs_trace = bool(cfg.data.get('observation_trace', False))
    obs_trace_decay = float(cfg.data.get('obs_trace_decay', 0.95))
    algorithm = cfg.get('algorithm', 'ipc')

    if algorithm == 'ipc_pipelined':
        def _step(state, observation):
            return td_ipc_pipelined_step(
                state, observation,
                gamma_inf=gamma_inf, alpha=alpha,
                gamma_td=gamma_td, td_lambda=td_lambda,
                use_obs_trace=use_obs_trace, obs_trace_decay=obs_trace_decay,
            )
    elif algorithm == 'ipc_simultaneous':
        def _step(state, observation):
            return td_ipc_simultaneous_step(
                state, observation,
                T=T, gamma_inf=gamma_inf, alpha=alpha,
                gamma_td=gamma_td, td_lambda=td_lambda,
                use_obs_trace=use_obs_trace, obs_trace_decay=obs_trace_decay,
            )
    else:
        def _step(state, observation):
            return td_ipc_lambda_step(
                state, observation,
                T=T, gamma_inf=gamma_inf, alpha=alpha,
                gamma_td=gamma_td, td_lambda=td_lambda,
                use_obs_trace=use_obs_trace, obs_trace_decay=obs_trace_decay,
            )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps, in_axes=(0, None)))

    all_td_errors = []
    all_msve = []
    first_preds = None
    last_preds = None
    first_range = None
    last_range = None
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

        # Save online predictions from first and last chunks (seed 0)
        obs_start = int((positions[0] + 1) % n_data)
        obs_end = int((positions[-1] + 1) % n_data) + 1
        if scan_idx == 0:
            first_preds = np.array(metrics.v_prediction[0])  # (log_freq,)
            first_range = (obs_start, obs_end)
        last_preds = np.array(metrics.v_prediction[0])
        last_range = (obs_start, obs_end)

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

    return train_state, all_td_errors, all_msve, (first_preds, last_preds, first_range, last_range)


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
    use_obs_trace = bool(cfg.data.get('observation_trace', False))
    obs_trace_decay = float(cfg.data.get('obs_trace_decay', 0.95))

    def _step(state, observation):
        return td_bp_lambda_step(
            state, observation,
            alpha=alpha, gamma_td=gamma_td, td_lambda=td_lambda,
            use_obs_trace=use_obs_trace, obs_trace_decay=obs_trace_decay,
        )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps, in_axes=(0, None)))

    all_td_errors = []
    all_msve = []
    first_preds = None
    last_preds = None
    first_range = None
    last_range = None
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

        # Save online predictions from first and last chunks (seed 0)
        obs_start = int((positions[0] + 1) % n_data)
        obs_end = int((positions[-1] + 1) % n_data) + 1
        if scan_idx == 0:
            first_preds = np.array(metrics.v_prediction[0])
            first_range = (obs_start, obs_end)
        last_preds = np.array(metrics.v_prediction[0])
        last_range = (obs_start, obs_end)

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

    return train_state, all_td_errors, all_msve, (first_preds, last_preds, first_range, last_range)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _predict_ipc(network, observations_jax):
    """Compute V(s) for observations using PC forward pass (single seed)."""
    f = ACTIVATION_MAP[network.activation_name]

    def predict_one(s):
        fwd_nodes = pc_forward_pass(network, s)
        return (network.weights[0] @ f(fwd_nodes[0]))[0]

    return jax.vmap(predict_one)(observations_jax)


def _predict_bp(model, observations_jax):
    """Compute V(s) for observations using BP forward pass (single seed)."""
    def predict_one(s):
        v, _ = model(s)
        return v[0]

    return jax.vmap(predict_one)(observations_jax)


def plot_value_predictions(audio, metadata, rewards, true_returns, predictions,
                           step_range, step_size=640, sample_rate=16384):
    """Create a 2-subplot figure: waveform with reward markers + value predictions.

    Args:
        audio: Raw audio array (1D float64).
        metadata: Benchmark metadata dict with 'events' list.
        rewards: (n_steps,) reward array.
        true_returns: (n_steps,) true discounted returns.
        predictions: (n_steps_in_range,) predicted values for the step range.
        step_range: (start, end) step indices.
        step_size: Samples per timestep.
        sample_rate: Audio sample rate.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    start, end = step_range
    n_steps = end - start

    # Time axis in seconds
    times = np.arange(start, end) * step_size / sample_rate
    time_start = times[0]
    time_end = times[-1]

    fig, (ax_wave, ax_val) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 1.2]})

    # --- Top: Waveform with reward markers ---
    audio_start = start * step_size
    audio_end = min(end * step_size, len(audio))
    audio_times = np.arange(audio_start, audio_end) / sample_rate
    ax_wave.plot(audio_times, audio[audio_start:audio_end],
                 linewidth=0.3, color='steelblue')

    # Reward markers from metadata events
    reward_colors = {1.0: 'green', -1.0: 'red'}
    reward_labels = {1.0: '+1 reward', -1.0: '-1 reward'}
    labeled = set()
    for event in metadata.get('events', []):
        r = event['reward']
        if r == 0.0:
            continue
        rt = event['reward_time']
        if time_start <= rt <= time_end:
            color = reward_colors.get(r, 'gray')
            label = reward_labels.get(r) if r not in labeled else None
            ax_wave.axvline(rt, color=color, alpha=0.7, linewidth=1.0,
                            linestyle='--', label=label)
            labeled.add(r)

    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_title(f'Audio Waveform (steps {start}-{end})')
    if labeled:
        ax_wave.legend(loc='upper right', fontsize=8)
    ax_wave.set_xlim(time_start, time_end)

    # --- Bottom: True return vs predicted return ---
    ax_val.plot(times, true_returns[start:end], label='True return $G_t$',
                color='black', linewidth=0.8, alpha=0.8)
    ax_val.plot(times, predictions, label='Predicted $\\hat{V}(s_t)$',
                color='tab:blue', linewidth=0.8, alpha=0.8)

    # Reward markers (same as top plot)
    for event in metadata.get('events', []):
        r = event['reward']
        if r == 0.0:
            continue
        rt = event['reward_time']
        if time_start <= rt <= time_end:
            color = reward_colors.get(r, 'gray')
            ax_val.axvline(rt, color=color, alpha=0.5, linewidth=0.8, linestyle='--')

    ax_val.set_xlabel('Time (s)')
    ax_val.set_ylabel('Value')
    ax_val.set_title('True vs Predicted Return')
    ax_val.legend(loc='upper right', fontsize=8)
    ax_val.axhline(0, color='gray', linewidth=0.3)

    fig.tight_layout()
    return fig


def generate_visualizations(cfg, rewards, true_returns, metadata,
                            first_preds, last_preds, first_range, last_range):
    """Generate and log start/end value prediction figures.

    Uses online predictions recorded DURING training (not re-predicted
    with the final model), so the "start" figure shows how the model
    predicted while it was still learning.

    Args:
        first_preds: (n_steps,) predictions from the first scan chunk (seed 0).
        last_preds: (n_steps,) predictions from the last scan chunk (seed 0).
        first_range: (start, end) step indices for the first chunk.
        last_range: (start, end) step indices for the last chunk.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf

    data_dir = os.path.expanduser(cfg.data.data_dir)
    audio, sample_rate = sf.read(os.path.join(data_dir, 'audio.wav'), dtype='float64')
    step_size = metadata.get('step_size', 640) if metadata else 640

    fig_start = plot_value_predictions(
        audio, metadata, rewards, true_returns, first_preds,
        step_range=first_range, step_size=step_size, sample_rate=sample_rate,
    )
    fig_end = plot_value_predictions(
        audio, metadata, rewards, true_returns, last_preds,
        step_range=last_range, step_size=step_size, sample_rate=sample_rate,
    )

    log_figures({
        'value_predictions_start': fig_start,
        'value_predictions_end': fig_end,
    }, cfg)

    plt.close(fig_start)
    plt.close(fig_end)
    print('Logged value prediction visualizations.')


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

    # Optionally apply observation trace (EMA) for memory
    if cfg.data.get('observation_trace', False):
        decay = cfg.data.get('obs_trace_decay', 0.95)
        print(f'Computing observation trace (decay={decay})...')
        observations = compute_observation_trace(observations, decay)
        print(f'Observation trace: shape={observations.shape}, dtype={observations.dtype}')

    input_dim = observations.shape[1]

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
        train_state, all_td_errors, all_msve, vis_data = run_bp_experiment(
            cfg, train_state, observations, rewards, true_returns,
        )
    elif algorithm in ('ipc', 'ipc_simultaneous', 'ipc_pipelined'):
        train_state, n_params = prepare_ipc_experiment(cfg, s_0)
        train_state, all_td_errors, all_msve, vis_data = run_ipc_experiment(
            cfg, train_state, observations, rewards, true_returns,
        )
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

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

    # Generate and log visualizations
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))
    first_preds, last_preds, first_range, last_range = vis_data
    if logging_active and first_preds is not None:
        generate_visualizations(
            cfg, rewards, true_returns, metadata,
            first_preds, last_preds, first_range, last_range,
        )

    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
