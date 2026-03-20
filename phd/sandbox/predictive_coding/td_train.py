"""TD(0) value prediction with streaming iPC on a continuous grid world.

Entry point for the TD value prediction experiment. Uses the same PC network
machinery as the Rotating MNIST experiments (models.py) but adapted for:
- 2D continuous state input -> 1D scalar value output
- TD(0) bootstrap targets instead of one-hot labels
- Periodic MSVE evaluation against precomputed V*
- Barrier-crossing stratified metrics
"""

from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial

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
import optax
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
from phd.sandbox.predictive_coding.environment import GridWorld
from phd.sandbox.predictive_coding.value_function import (
    compute_true_value_function,
    evaluate_msve_numpy,
)
from phd.sandbox.predictive_coding.models import (
    PCNetwork, init_pc_network, pc_forward_pass, ipc_step, ipc_step_grads,
)


SCAN_UNROLL = 4
INPUT_DIM = 2
OUTPUT_DIM = 1


# ---------------------------------------------------------------------------
# Metrics containers
# ---------------------------------------------------------------------------

class TDiPCStepMetrics(eqx.Module):
    """Per-step metrics from the iPC TD training loop."""
    td_error_sq: jax.Array          # scalar
    total_energy: jax.Array         # scalar
    v_prediction: jax.Array         # scalar: V_hat(s_t) before update
    barrier_crossing: jax.Array     # scalar: 0.0 or 1.0


class TDBPStepMetrics(eqx.Module):
    """Per-step metrics from the BP TD training loop."""
    td_error_sq: jax.Array
    v_prediction: jax.Array
    barrier_crossing: jax.Array


# ---------------------------------------------------------------------------
# iPC Train State and TD step
# ---------------------------------------------------------------------------

class PCTrainState(eqx.Module):
    network: PCNetwork
    value_nodes: list     # persistent across observations
    step: jax.Array
    rng: PRNGKeyArray
    opt_state: list       # per-layer optax optimizer states (empty list if not using optimizer)


def _read_value_prediction(network, value_nodes):
    """Read scalar value prediction from PC network's output layer.

    mu[0] = weights[0] @ f(value_nodes[0]) is the generative prediction of x[0].
    """
    f = ACTIVATION_MAP[network.activation_name]
    mu_0 = network.weights[0] @ f(value_nodes[0])
    return mu_0[0]  # scalar (output dim is 1)


def _forward_pass_value(network, s):
    """Compute V_hat(s) via a full top-down forward pass (no streaming nodes)."""
    f = ACTIVATION_MAP[network.activation_name]
    fwd_nodes = pc_forward_pass(network, s)
    return (network.weights[0] @ f(fwd_nodes[0]))[0]


def td_ipc_step(state, observation, *, T, gamma_inf, alpha, gamma_td, variant, ema_beta):
    """One TD(0) step for streaming iPC.

    Args:
        state: PCTrainState
        observation: (s_t, s_next, reward, barrier_crossing) tuple
        T: number of iPC inference steps
        gamma_inf: inference learning rate
        alpha: weight learning rate
        gamma_td: TD discount factor
        variant: 'streaming', 'forward_init', or 'ema'
        ema_beta: EMA blending parameter (only used when variant='ema')
    """
    s_t, s_next, reward, is_barrier_crossing = observation

    network = state.network
    value_nodes = state.value_nodes

    # 1. Online prediction V_hat(s_t) BEFORE update
    v_pred = _read_value_prediction(network, value_nodes)

    # 2. Compute V_hat(s') via forward pass (method a)
    v_next = _forward_pass_value(network, s_next)

    # 3. TD target (stop gradient is implicit: we clamp this at layer 0)
    td_target = reward + gamma_td * jax.lax.stop_gradient(v_next)
    y_target = jnp.array([td_target])  # shape (1,) to match output dim

    # 4. Variant-specific value node handling
    if variant == 'forward_init':
        value_nodes = pc_forward_pass(network, s_t)
    elif variant == 'ema':
        fwd_nodes = pc_forward_pass(network, s_t)
        value_nodes = [
            ema_beta * vn + (1.0 - ema_beta) * fn
            for vn, fn in zip(value_nodes, fwd_nodes)
        ]
    # else 'streaming': keep value_nodes as-is

    # 5. Run T iPC steps with s_t clamped at input, y_target at output
    info = None
    for _t in range(T):
        network, value_nodes, info = ipc_step(
            network, value_nodes, s_t, y_target, gamma_inf, alpha,
        )

    td_error_sq = jnp.square(reward + gamma_td * jax.lax.stop_gradient(v_next) - v_pred)

    metrics = TDiPCStepMetrics(
        td_error_sq=td_error_sq,
        total_energy=info['total_energy'],
        v_prediction=v_pred,
        barrier_crossing=is_barrier_crossing,
    )

    new_state = tree_replace(
        state,
        network=network,
        value_nodes=value_nodes,
        step=state.step + 1,
    )
    return new_state, metrics


def td_ipc_optim_step(state, observation, *, T, gamma_inf, gamma_td, variant, ema_beta, optimizer):
    """One TD(0) step for iPC with an optax optimizer for weight updates.

    Same as td_ipc_step but uses an optimizer (e.g. Adam) instead of raw
    alpha * gradient. The learning rate is controlled by the optimizer, not alpha.
    """
    s_t, s_next, reward, is_barrier_crossing = observation

    network = state.network
    value_nodes = state.value_nodes
    opt_state = state.opt_state

    # 1. Online prediction V_hat(s_t) BEFORE update
    v_pred = _read_value_prediction(network, value_nodes)

    # 2. Compute V_hat(s') via forward pass (method a)
    v_next = _forward_pass_value(network, s_next)

    # 3. TD target
    td_target = reward + gamma_td * jax.lax.stop_gradient(v_next)
    y_target = jnp.array([td_target])

    # 4. Variant-specific value node handling
    if variant == 'forward_init':
        value_nodes = pc_forward_pass(network, s_t)
    elif variant == 'ema':
        fwd_nodes = pc_forward_pass(network, s_t)
        value_nodes = [
            ema_beta * vn + (1.0 - ema_beta) * fn
            for vn, fn in zip(value_nodes, fwd_nodes)
        ]

    # 5. Run T iPC steps: update value nodes and collect weight gradients
    info = None
    for _t in range(T):
        value_nodes, weight_grads, info = ipc_step_grads(
            network, value_nodes, s_t, y_target, gamma_inf,
        )

    # 6. Apply weight updates via optimizer
    new_weights = []
    new_opt_state = []
    for l in range(network.num_layers):
        updates, new_state_l = optimizer.update(weight_grads[l], opt_state[l], network.weights[l])
        new_weights.append(network.weights[l] + updates)
        new_opt_state.append(new_state_l)

    new_network = PCNetwork(
        layer_dims=network.layer_dims,
        num_layers=network.num_layers,
        activation_name=network.activation_name,
        weights=new_weights,
    )

    td_error_sq = jnp.square(reward + gamma_td * jax.lax.stop_gradient(v_next) - v_pred)

    metrics = TDiPCStepMetrics(
        td_error_sq=td_error_sq,
        total_energy=info['total_energy'],
        v_prediction=v_pred,
        barrier_crossing=is_barrier_crossing,
    )

    new_state = tree_replace(
        state,
        network=new_network,
        value_nodes=value_nodes,
        opt_state=new_opt_state,
        step=state.step + 1,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# BP baseline Train State and TD step
# ---------------------------------------------------------------------------

class BPTrainState(eqx.Module):
    model: MLP
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray


def td_bp_step(state, observation, *, gamma_td):
    """One semi-gradient TD(0) step with backprop."""
    s_t, s_next, reward, is_barrier_crossing = observation

    # Forward pass for V(s_t)
    v_pred_vec, _ = state.model(s_t)
    v_pred = v_pred_vec[0]

    # Forward pass for V(s') — stop gradient for semi-gradient TD
    v_next_vec, _ = state.model(s_next)
    v_next = jax.lax.stop_gradient(v_next_vec[0])

    td_target = reward + gamma_td * v_next

    def loss_fn(model):
        v, _ = model(s_t)
        return jnp.mean(jnp.square(v[0] - td_target))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model)
    updates, new_opt = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    new_state = tree_replace(state, model=new_model, optimizer=new_opt, step=state.step + 1)
    metrics = TDBPStepMetrics(
        td_error_sq=jnp.square(v_pred - td_target),
        v_prediction=v_pred,
        barrier_crossing=is_barrier_crossing,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _build_grid_world(cfg) -> GridWorld:
    """Build the grid world from config."""
    return GridWorld(
        sigma=cfg.env.sigma,
        gamma_td=cfg.env.gamma_td,
    )


def _build_ipc_optimizer(cfg):
    """Build optax optimizer for iPC weight updates, or None for raw SGD."""
    opt_name = cfg.ipc.get('optimizer', 'none')
    if opt_name == 'none':
        return None
    lr = cfg.ipc.alpha
    if opt_name == 'adam':
        return optax.adam(lr)
    elif opt_name == 'sgd':
        return optax.sgd(lr)
    else:
        raise ValueError(f"Unknown iPC optimizer: {opt_name}")


def prepare_ipc_experiment(cfg):
    """Create per-seed PCTrainState objects."""
    seeds = cfg.seed

    # Layer dims: (output=1, hidden..., input=2)
    n_hidden = cfg.model.num_layers - 2
    layer_dims = (
        [OUTPUT_DIM]
        + [cfg.model.hidden_dim] * n_hidden
        + [INPUT_DIM]
    )

    optimizer = _build_ipc_optimizer(cfg)

    train_states = []
    for seed in seeds:
        rng = jax.random.key(seed)

        network = init_pc_network(
            layer_dims=layer_dims,
            activation=cfg.model.activation,
            key=rng_from_string(rng, 'model'),
        )

        # Initialize value nodes to zeros
        L = network.num_layers
        value_nodes = [jnp.zeros(layer_dims[l]) for l in range(1, L)]

        # Initialize per-layer optimizer states
        if optimizer is not None:
            opt_state = [optimizer.init(w) for w in network.weights]
        else:
            opt_state = []

        train_states.append(PCTrainState(
            network=network,
            value_nodes=value_nodes,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
            opt_state=opt_state,
        ))

    n_params = sum(w.size for w in train_states[0].network.weights)
    opt_label = cfg.ipc.get('optimizer', 'none')
    print(f'PCNetwork: layers={layer_dims}, params={n_params}, optimizer={opt_label}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, n_params


def prepare_bp_experiment(cfg, linear=False):
    """Create per-seed BPTrainState objects."""
    seeds = cfg.seed
    train_states = []

    for seed in seeds:
        rng = jax.random.key(seed)

        if linear:
            # Linear model: 1 weight layer, linear activation
            model = MLP(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                n_layers=1,
                hidden_dim=1,  # unused for n_layers=1
                weight_init_method='lecun_uniform',
                activation='linear',
                key=rng_from_string(rng, 'model'),
            )
        else:
            model = MLP(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                n_layers=cfg.model.num_layers - 1,
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
    label = 'Linear' if linear else 'MLP (BP)'
    print(f'{label}: params={n_params}, seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, n_params


# ---------------------------------------------------------------------------
# MSVE evaluation
# ---------------------------------------------------------------------------

def _evaluate_msve_ipc(network, eval_grid_jax, n_eval):
    """Evaluate MSVE for a PC network using forward-pass predictions."""
    f = ACTIVATION_MAP[network.activation_name]

    def predict_one(s):
        fwd_nodes = pc_forward_pass(network, s)
        return (network.weights[0] @ f(fwd_nodes[0]))[0]

    predictions = jax.vmap(predict_one)(eval_grid_jax)
    return predictions


def _evaluate_msve_bp(model, eval_grid_jax, n_eval):
    """Evaluate MSVE for a BP model using forward passes."""
    def predict_one(s):
        v, _ = model(s)
        return v[0]

    predictions = jax.vmap(predict_one)(eval_grid_jax)
    return predictions


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def run_ipc_experiment(cfg, train_state, grid_world, eval_grid_np, v_star_interp):
    """Outer training loop for iPC TD variants."""
    log_freq = cfg.train.log_freq
    eval_freq = cfg.train.eval_freq
    num_scans = cfg.train.total_steps // log_freq
    T = cfg.ipc.T
    gamma_inf = cfg.ipc.gamma
    alpha = cfg.ipc.alpha
    gamma_td = cfg.env.gamma_td
    variant = cfg.ipc.variant
    ema_beta = cfg.ipc.ema_beta
    n_seeds = len(cfg.seed)
    use_optimizer = cfg.ipc.get('optimizer', 'none') != 'none'

    if use_optimizer:
        optimizer = _build_ipc_optimizer(cfg)
        def _step(state, observation):
            return td_ipc_optim_step(
                state, observation,
                T=T, gamma_inf=gamma_inf,
                gamma_td=gamma_td, variant=variant, ema_beta=ema_beta,
                optimizer=optimizer,
            )
    else:
        def _step(state, observation):
            return td_ipc_step(
                state, observation,
                T=T, gamma_inf=gamma_inf, alpha=alpha,
                gamma_td=gamma_td, variant=variant, ema_beta=ema_beta,
            )

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    # JIT the MSVE evaluation (extract one seed's network for eval)
    eval_grid_jax = jnp.array(eval_grid_np)
    n_eval = eval_grid_np.shape[0]

    @jax.jit
    def eval_msve_all_seeds(batched_network):
        return jax.vmap(lambda net: _evaluate_msve_ipc(net, eval_grid_jax, n_eval))(batched_network)

    # Per-seed trajectory generators
    trajectory_seeds = [
        np.random.default_rng(seed * 100_000) for seed in cfg.seed
    ]

    all_td_errors = []
    all_msve = []
    pbar = tqdm(total=cfg.train.total_steps, desc=f'iPC TD ({variant})')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for scan_idx in range(num_scans):
        # Pre-sample trajectory chunks on CPU (per seed)
        s_t_list, s_next_list, reward_list, bc_list = [], [], [], []
        for i in range(n_seeds):
            traj_seed = int(trajectory_seeds[i].integers(0, 2**31))
            positions, rewards, barrier_crossings = grid_world.sample_trajectory(
                log_freq, seed=traj_seed,
            )
            s_t_list.append(positions[:-1].astype(np.float32))      # (log_freq, 2)
            s_next_list.append(positions[1:].astype(np.float32))     # (log_freq, 2)
            reward_list.append(rewards.astype(np.float32))           # (log_freq,)
            bc_list.append(barrier_crossings.astype(np.float32))     # (log_freq,)

        # Stack and ship to JAX: (n_seeds, log_freq, ...)
        obs = (
            jnp.array(np.stack(s_t_list)),
            jnp.array(np.stack(s_next_list)),
            jnp.array(np.stack(reward_list)),
            jnp.array(np.stack(bc_list)),
        )

        train_state, metrics = vmapped_scan(train_state, obs)

        # Aggregate metrics
        per_seed_td_error = metrics.td_error_sq.mean(axis=1)   # (n_seeds,)
        mean_td_error = float(per_seed_td_error.mean())

        # Stratified TD error by barrier crossing
        bc_mask = metrics.barrier_crossing  # (n_seeds, log_freq)
        n_barrier = float(bc_mask.sum())
        n_open = float((1.0 - bc_mask).sum())

        if n_barrier > 0:
            td_barrier = float((metrics.td_error_sq * bc_mask).sum() / n_barrier)
        else:
            td_barrier = 0.0

        if n_open > 0:
            td_open = float((metrics.td_error_sq * (1.0 - bc_mask)).sum() / n_open)
        else:
            td_open = 0.0

        mean_energy = float(metrics.total_energy.mean())
        bc_frac = float(bc_mask.mean())

        step = int(train_state.step[0].item())

        # Periodic MSVE evaluation
        msve_val = None
        if step % eval_freq == 0 or scan_idx == num_scans - 1:
            predictions = eval_msve_all_seeds(train_state.network)  # (n_seeds, n_eval)
            per_seed_msve = []
            for s in range(n_seeds):
                preds_np = np.array(predictions[s])
                msve = evaluate_msve_numpy(preds_np, eval_grid_np, v_star_interp)
                per_seed_msve.append(msve)
            msve_val = float(np.mean(per_seed_msve))
            all_msve.append(msve_val)

        if logging_active:
            log_dict = {
                'td_error': mean_td_error,
                'td_error_open': td_open,
                'td_error_barrier': td_barrier,
                'energy': mean_energy,
                'barrier_crossing_frac': bc_frac,
            }
            if msve_val is not None:
                log_dict['msve'] = msve_val

            def _log(log_dict, per_seed_td_error, step):
                log_metrics(log_dict, cfg, step=step)
                log_child_metrics({
                    'td_error': per_seed_td_error.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, log_dict, np.array(per_seed_td_error), step,
            ))

        all_td_errors.append(mean_td_error)
        pbar.update(log_freq)
        postfix = {'td_err': f'{mean_td_error:.4f}', 'energy': f'{mean_energy:.2f}'}
        if msve_val is not None:
            postfix['msve'] = f'{msve_val:.4f}'
        pbar.set_postfix(postfix)

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_td_errors, all_msve


def run_bp_experiment(cfg, train_state, grid_world, eval_grid_np, v_star_interp):
    """Outer training loop for BP TD baseline."""
    log_freq = cfg.train.log_freq
    eval_freq = cfg.train.eval_freq
    num_scans = cfg.train.total_steps // log_freq
    gamma_td = cfg.env.gamma_td
    n_seeds = len(cfg.seed)

    def _step(state, observation):
        return td_bp_step(state, observation, gamma_td=gamma_td)

    def scan_steps(state, data):
        return jax.lax.scan(_step, state, data, unroll=SCAN_UNROLL)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    eval_grid_jax = jnp.array(eval_grid_np)
    n_eval = eval_grid_np.shape[0]

    @jax.jit
    def eval_msve_all_seeds(batched_model):
        return jax.vmap(lambda m: _evaluate_msve_bp(m, eval_grid_jax, n_eval))(batched_model)

    trajectory_seeds = [
        np.random.default_rng(seed * 100_000) for seed in cfg.seed
    ]

    all_td_errors = []
    all_msve = []
    pbar = tqdm(total=cfg.train.total_steps, desc='BP TD')

    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))

    for scan_idx in range(num_scans):
        s_t_list, s_next_list, reward_list, bc_list = [], [], [], []
        for i in range(n_seeds):
            traj_seed = int(trajectory_seeds[i].integers(0, 2**31))
            positions, rewards, barrier_crossings = grid_world.sample_trajectory(
                log_freq, seed=traj_seed,
            )
            s_t_list.append(positions[:-1].astype(np.float32))
            s_next_list.append(positions[1:].astype(np.float32))
            reward_list.append(rewards.astype(np.float32))
            bc_list.append(barrier_crossings.astype(np.float32))

        obs = (
            jnp.array(np.stack(s_t_list)),
            jnp.array(np.stack(s_next_list)),
            jnp.array(np.stack(reward_list)),
            jnp.array(np.stack(bc_list)),
        )

        train_state, metrics = vmapped_scan(train_state, obs)

        per_seed_td_error = metrics.td_error_sq.mean(axis=1)
        mean_td_error = float(per_seed_td_error.mean())

        bc_mask = metrics.barrier_crossing
        n_barrier = float(bc_mask.sum())
        n_open = float((1.0 - bc_mask).sum())
        td_barrier = float((metrics.td_error_sq * bc_mask).sum() / max(n_barrier, 1.0))
        td_open = float((metrics.td_error_sq * (1.0 - bc_mask)).sum() / max(n_open, 1.0))
        bc_frac = float(bc_mask.mean())

        step = int(train_state.step[0].item())

        msve_val = None
        if step % eval_freq == 0 or scan_idx == num_scans - 1:
            predictions = eval_msve_all_seeds(train_state.model)
            per_seed_msve = []
            for s in range(n_seeds):
                preds_np = np.array(predictions[s])
                msve = evaluate_msve_numpy(preds_np, eval_grid_np, v_star_interp)
                per_seed_msve.append(msve)
            msve_val = float(np.mean(per_seed_msve))
            all_msve.append(msve_val)

        if logging_active:
            log_dict = {
                'td_error': mean_td_error,
                'td_error_open': td_open,
                'td_error_barrier': td_barrier,
                'barrier_crossing_frac': bc_frac,
            }
            if msve_val is not None:
                log_dict['msve'] = msve_val

            def _log(log_dict, per_seed_td_error, step):
                log_metrics(log_dict, cfg, step=step)
                log_child_metrics({
                    'td_error': per_seed_td_error.tolist(),
                }, cfg, step=step)

            log_futures.append(log_executor.submit(
                _log, log_dict, np.array(per_seed_td_error), step,
            ))

        all_td_errors.append(mean_td_error)
        pbar.update(log_freq)
        postfix = {'td_err': f'{mean_td_error:.4f}'}
        if msve_val is not None:
            postfix['msve'] = f'{msve_val:.4f}'
        pbar.set_postfix(postfix)

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return train_state, all_td_errors, all_msve


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='td_config', version_base='1.1')
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

    # Build environment and compute V*
    grid_world = _build_grid_world(cfg)
    print(f'Computing V* (resolution={cfg.env.v_star_resolution}, '
          f'samples={cfg.env.v_star_samples})...')
    v_star_grid, v_star_interp, _, _ = compute_true_value_function(
        grid_world,
        resolution=cfg.env.v_star_resolution,
        n_samples=cfg.env.v_star_samples,
        seed=0,
    )
    print('V* computed.')

    # Evaluation grid
    eval_grid_np, _ = grid_world.get_eval_grid(cfg.train.eval_resolution)
    eval_grid_np = eval_grid_np.astype(np.float32)
    print(f'Eval grid: {eval_grid_np.shape[0]} points')

    algorithm = cfg.get('algorithm', 'ipc')

    if algorithm in ('bp', 'linear_bp'):
        linear = (algorithm == 'linear_bp')
        train_state, n_params = prepare_bp_experiment(cfg, linear=linear)
        train_state, all_td_errors, all_msve = run_bp_experiment(
            cfg, train_state, grid_world, eval_grid_np, v_star_interp,
        )
    else:
        train_state, n_params = prepare_ipc_experiment(cfg)
        train_state, all_td_errors, all_msve = run_ipc_experiment(
            cfg, train_state, grid_world, eval_grid_np, v_star_interp,
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
