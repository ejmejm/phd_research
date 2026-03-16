"""Pruning experiment: evaluate utility functions by how well they identify
the least important hidden units for pruning.

For each seed, all methods start from the same initial model. Every prune_freq
steps, the active hidden unit with the lowest |utility| is pruned (masked out).
Methods are compared by their loss/accuracy trajectories and final performance.

NOTE: Currently uses CIFAR-10 with MSE loss (one-hot regression).
      This will be changed to a proper regression dataset later.
"""

import logging
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from phd.jax_core.models import MLP, ACTIVATION_MAP
from phd.jax_core.utils import configure_jax
from phd.structure_search.data import load_dataset, DataStream

from configurable_utility import (
    UtilityConfig,
    enumerate_valid_configs,
    make_utility_fn,
    contribution_utility,
    upgd_utility,
    si_utility,
    ACTIVATIONS,
)


NUM_CLASSES = 10


# ==============================================================================
# Method registry
# ==============================================================================

logger = logging.getLogger(__name__)


@dataclass
class MethodSpec:
    """Specification for a utility-based pruning method."""
    name: str
    utility_fn: Callable
    needs_grads: bool = False    # Whether the utility fn requires loss gradients
    needs_updates: bool = False  # Whether the utility fn requires optimizer updates
    config: Optional[UtilityConfig] = None  # None for baselines


def build_method_specs(activation: str = 'relu') -> List[MethodSpec]:
    """Build the full list of methods: 115 configurable + 3 baselines."""
    methods = []

    # Baselines
    methods.append(MethodSpec(
        name='Contribution',
        utility_fn=partial(contribution_utility, activation=activation),
    ))
    methods.append(MethodSpec(
        name='UPGD',
        utility_fn=partial(upgd_utility, activation=activation),
        needs_grads=True,  # UPGD needs loss gradients to extract dL/dz
    ))
    methods.append(MethodSpec(
        name='SI',
        utility_fn=partial(si_utility, activation=activation),
        needs_grads=True,
        needs_updates=True,  # SI needs both gradients and optimizer updates
    ))

    # Configurable utility functions
    for config in enumerate_valid_configs():
        fn = make_utility_fn(config, activation=activation)
        name = (f"loo_{config.output_normalization}_{config.pseudo_error}_"
                f"{config.hidden_normalization}_cap{config.cap}"
                if config.child_score_method == "loo"
                else f"{config.child_score_method}_{config.output_normalization}")
        methods.append(MethodSpec(
            name=name,
            utility_fn=fn,
            config=config,
        ))

    return methods


BASELINE_NAMES = {'Contribution', 'UPGD', 'SI'}


# ==============================================================================
# Masked forward pass and loss
# ==============================================================================

def masked_forward(model, x, masks, act_fn):
    """Forward pass with activation masking for pruned units.

    Args:
        model: MLP model
        x: input (N,)
        masks: tuple of (H_l,) arrays, one per hidden layer. 1=active, 0=pruned.
        act_fn: activation function

    Returns:
        output: (D,) model output
    """
    for l, layer in enumerate(model.layers[:-1]):
        x = layer(x)
        x = act_fn(x)
        x = x * masks[l]
    return model.layers[-1](x)


def masked_loss_fn(model, images, labels, masks, act_fn):
    """MSE loss with one-hot targets and activation masking.

    Args:
        model: MLP model
        images: (batch, N) input batch
        labels: (batch,) integer labels
        masks: tuple of (H_l,) mask arrays
        act_fn: activation function

    Returns:
        loss: scalar MSE loss
        outputs: (batch, D) predictions
    """
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)

    def single_forward(x):
        return masked_forward(model, x, masks, act_fn)

    outputs = jax.vmap(single_forward)(images)  # (batch, D)
    loss = jnp.mean(jnp.square(outputs - one_hot))
    return loss, outputs


# ==============================================================================
# Train state
# ==============================================================================

class PruneState(eqx.Module):
    """Training state for a single method."""
    model: MLP
    opt_state: eqx.Module  # EqxOptimizer
    masks: Tuple  # tuple of (H_l,) arrays
    step: jax.Array
    rng: PRNGKeyArray


# ==============================================================================
# Training and pruning steps
# ==============================================================================

def train_step(state, data, act_fn):
    """Single training step without pruning. Returns only scalar metrics."""
    images, labels = data

    loss_grad_fn = eqx.filter_value_and_grad(
        lambda m: masked_loss_fn(m, images, labels, state.masks, act_fn),
        has_aux=True,
    )
    (loss, outputs), grads = loss_grad_fn(state.model)

    updates, new_opt = state.opt_state.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    predicted = jnp.argmax(outputs, axis=-1)
    accuracy = (predicted == labels).astype(jnp.float32).mean()

    new_state = eqx.tree_at(
        lambda s: (s.model, s.opt_state, s.step),
        state,
        (new_model, new_opt, state.step + 1),
    )
    return new_state, (loss, accuracy)


def compute_hidden_utilities(model, x, y_star, y_hat, loss_grads, updates,
                             masks, utility_fn):
    """Compute per-hidden-unit utilities using the specified method.

    Args:
        loss_grads: model gradients, or None if not needed
        updates: optimizer updates, or None if not needed

    Returns a flat array of |utilities| across all hidden layers,
    with masked units set to +inf so they're never selected for pruning.
    """
    _, hidden_utils = utility_fn(model, x, y_star, y_hat, loss_grads, updates, masks=masks)

    # Concatenate utilities across all hidden layers
    all_utils = jnp.concatenate(list(hidden_utils))
    all_masks = jnp.concatenate(list(masks))

    # Use absolute utility for pruning decision; set masked units to +inf
    abs_utils = jnp.abs(all_utils)
    abs_utils = jnp.where(all_masks > 0.5, abs_utils, jnp.inf)

    return abs_utils


def prune_one_unit(masks, abs_utilities, n_hidden_per_layer):
    """Find the active unit with lowest |utility| and set its mask to 0.

    Args:
        masks: tuple of (H_l,) mask arrays
        abs_utilities: flat array of |utility| across all layers (inf for masked units)
        n_hidden_per_layer: tuple of ints, hidden dim per layer

    Returns:
        new_masks: updated masks with one unit pruned
    """
    idx = jnp.argmin(abs_utilities)

    # Convert flat index to (layer, unit) index
    new_masks = list(masks)
    offset = 0
    for l, h in enumerate(n_hidden_per_layer):
        in_layer = (idx >= offset) & (idx < offset + h)
        local_idx = idx - offset
        new_masks[l] = jnp.where(
            in_layer,
            masks[l].at[local_idx].set(0.0),
            masks[l],
        )
        offset += h

    return tuple(new_masks)


# ==============================================================================
# Full experiment: vmap seeds (outer) × vmap methods (inner)
# ==============================================================================

SCAN_UNROLL = 4


def run_experiment(cfg: DictConfig):
    """Run the full pruning experiment.

    State has shape (n_seeds, n_methods, ...).
    The scan is double-vmapped: outer over seeds, inner over methods.
    Data has shape (n_seeds, prune_freq, batch_size, input_dim) — one stream
    per seed, broadcast across methods.

    Pruning is done in a Python loop over methods (different utility functions),
    with an inner loop over seeds to update each seed's masks.
    """
    import optax
    from phd.jax_core.optimizers import EqxOptimizer
    from phd.jax_core.utils import stack_pytrees

    dataset_cache = load_dataset(cfg.dataset.name)
    images_np, labels_np, num_classes, input_dim = dataset_cache
    methods = build_method_specs(activation=cfg.model.activation)
    act_fn = ACTIVATION_MAP[cfg.model.activation]

    seeds = list(cfg.seed)
    n_seeds = len(seeds)
    n_methods = len(methods)
    print(f"Running {n_methods} methods × {n_seeds} seeds")
    print(f"Total steps: {cfg.train.total_steps}, prune every {cfg.train.prune_freq} steps")

    # --- Build initial state: (n_seeds, n_methods, ...) ---
    # One model per seed, replicated across methods
    if cfg.optimizer.name == 'adam':
        opt = optax.adam(cfg.optimizer.learning_rate)
    elif cfg.optimizer.name == 'sgd':
        opt = optax.sgd(cfg.optimizer.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    per_seed_states = []
    streams = []
    for seed in seeds:
        rng = jax.random.PRNGKey(seed)
        model_key, train_key = jax.random.split(rng)

        model = MLP(
            input_dim=input_dim, output_dim=num_classes,
            n_layers=cfg.model.n_layers, hidden_dim=cfg.model.hidden_dim,
            weight_init_method=cfg.model.weight_init_method,
            activation=cfg.model.activation, key=model_key,
        )
        optimizer = EqxOptimizer(opt, model, name=cfg.optimizer.name)
        masks = tuple(jnp.ones(layer.weight.shape[0]) for layer in model.layers[:-1])
        base_state = PruneState(
            model=model, opt_state=optimizer, masks=masks,
            step=jnp.array(0), rng=train_key,
        )
        # Replicate across methods → (n_methods, ...)
        per_seed_states.append(stack_pytrees([base_state] * n_methods))

        streams.append(DataStream(
            images=images_np, labels=labels_np, num_classes=num_classes,
            batch_size=cfg.train.batch_size, seed=seed,
        ))

    # Stack across seeds → (n_seeds, n_methods, ...)
    batched_state = stack_pytrees(per_seed_states)

    n_hidden_per_layer = tuple(
        model.layers[l].weight.shape[0] for l in range(len(model.layers) - 1)
    )

    # --- Double-vmapped scan ---
    # Inner: vmap over methods (axis 0 of state), broadcast data
    # Outer: vmap over seeds (axis 0 of state AND axis 0 of data)
    def scan_body(state, data):
        return train_step(state, data, act_fn)

    vmapped_scan = jax.jit(
        jax.vmap(  # outer: over seeds
            jax.vmap(  # inner: over methods
                lambda state, data: jax.lax.scan(
                    scan_body, state, data, unroll=SCAN_UNROLL),
                in_axes=(0, None),  # methods dim on state, broadcast data
            ),
            in_axes=(0, 0),  # seeds dim on both state and data
        )
    )

    # --- JIT-compiled pruning with EMA utility traces ---
    # Each method maintains an exponential moving average of |utility| per unit.
    # At each prune event, batch-averaged utilities update the trace, and the
    # trace (not instantaneous utility) determines which unit to prune.
    # This handles dead-ReLU degeneracy: a unit dead for one sample but alive
    # for others accumulates nonzero trace over time.
    n_hidden_layers = len(n_hidden_per_layer)
    total_hidden = sum(n_hidden_per_layer)
    prune_freq = cfg.train.prune_freq
    trace_decay = 0.999 ** prune_freq  # per-step decay → effective per-event decay

    def _prune_all_methods(batched_state, prune_imgs, prune_lbls, utility_traces):
        new_masks = list(batched_state.masks)
        new_traces = utility_traces

        for m, method in enumerate(methods):
            m_model = jax.tree.map(lambda a, _m=m: a[:, _m], batched_state.model)
            m_masks = tuple(mask[:, m] for mask in batched_state.masks)
            m_opt_st = jax.tree.map(
                lambda a, _m=m: a[:, _m], batched_state.opt_state.state)

            def _prune_seed(model, masks, opt_st, imgs, lbls, trace):
                # Compute grads/updates from batch if needed
                grads = None
                updates = None
                if method.needs_grads:
                    grad_fn = eqx.filter_value_and_grad(
                        lambda mdl: masked_loss_fn(mdl, imgs, lbls, masks, act_fn),
                        has_aux=True,
                    )
                    (_, _), grads = grad_fn(model)

                    if method.needs_updates:
                        updates_raw, _ = opt.update(
                            eqx.filter(grads, eqx.is_array), opt_st)
                        updates = jax.tree.map(
                            lambda g, u: u if eqx.is_array(g) else None,
                            grads, updates_raw)

                # Batch-averaged raw |utility| (no mask applied yet)
                one_hot = jax.nn.one_hot(lbls, NUM_CLASSES)

                def _single_sample_utility(x, y_star):
                    y_hat = masked_forward(model, x, masks, act_fn)
                    _, hidden_utils = method.utility_fn(
                        model, x, y_star, y_hat, grads, updates, masks=masks)
                    return jnp.abs(jnp.concatenate(list(hidden_utils)))

                all_abs_utils = jax.vmap(_single_sample_utility)(imgs, one_hot)
                avg_abs_utils = jnp.mean(all_abs_utils, axis=0)

                # Update EMA trace
                new_trace = trace_decay * trace + (1.0 - trace_decay) * avg_abs_utils

                # Apply mask: pruned units → inf so they're never re-selected
                all_masks_flat = jnp.concatenate(list(masks))
                score = jnp.where(all_masks_flat > 0.5, new_trace, jnp.inf)

                return prune_one_unit(masks, score, n_hidden_per_layer), new_trace

            m_traces = utility_traces[:, m]  # (n_seeds, total_hidden)
            new_m_masks, new_m_traces = jax.vmap(_prune_seed)(
                m_model, m_masks, m_opt_st,
                prune_imgs, prune_lbls, m_traces)

            for l in range(n_hidden_layers):
                new_masks[l] = new_masks[l].at[:, m].set(new_m_masks[l])
            new_traces = new_traces.at[:, m].set(new_m_traces)

        return tuple(new_masks), new_traces

    jitted_prune = jax.jit(_prune_all_methods)

    total_steps = cfg.train.total_steps
    log_freq = cfg.train.log_freq
    prune_freq = cfg.train.prune_freq
    scan_chunk = cfg.train.get('scan_chunk', 100)  # steps per scan call (controls GPU memory)
    assert prune_freq % scan_chunk == 0, \
        f'prune_freq ({prune_freq}) must be divisible by scan_chunk ({scan_chunk})'
    n_prune_events = total_steps // prune_freq
    n_chunks_per_prune = prune_freq // scan_chunk

    # EMA utility traces: (n_seeds, n_methods, total_hidden)
    utility_traces = jnp.zeros((n_seeds, n_methods, total_hidden))

    # Per-method accumulators (averaged across seeds at the end)
    all_loss_history = [[] for _ in methods]
    all_acc_history = [[] for _ in methods]

    # Accumulator for logging independently of prune_freq
    # Collects per-step losses/accs across prune blocks, flushes at log_freq
    log_accum_losses = [[] for _ in range(n_methods)]  # list of per-step floats
    log_accum_accs = [[] for _ in range(n_methods)]

    def flush_logs():
        """Flush accumulated per-step metrics into log_freq-sized chunks."""
        for m in range(n_methods):
            while len(log_accum_losses[m]) >= log_freq:
                chunk_l = log_accum_losses[m][:log_freq]
                chunk_a = log_accum_accs[m][:log_freq]
                all_loss_history[m].append(float(np.mean(chunk_l)))
                all_acc_history[m].append(float(np.mean(chunk_a)))
                log_accum_losses[m] = log_accum_losses[m][log_freq:]
                log_accum_accs[m] = log_accum_accs[m][log_freq:]

    for prune_idx in tqdm(range(n_prune_events), desc='Training'):
        # Train prune_freq steps in scan_chunk-sized pieces
        for chunk_idx in range(n_chunks_per_prune):
            batch = [stream.sample_batch(scan_chunk) for stream in streams]
            images_jnp = jnp.array(np.stack([b[0] for b in batch]))
            labels_jnp = jnp.array(np.stack([b[1] for b in batch]))

            batched_state, (losses, accuracies) = vmapped_scan(
                batched_state, (images_jnp, labels_jnp))

            # Accumulate per-step metrics (seed-averaged)
            losses_np = np.array(losses)    # (n_seeds, n_methods, scan_chunk)
            accs_np = np.array(accuracies)
            for m in range(n_methods):
                log_accum_losses[m].extend(losses_np[:, m, :].mean(axis=0).tolist())
                log_accum_accs[m].extend(accs_np[:, m, :].mean(axis=0).tolist())

        flush_logs()

        # --- Pruning: batch-averaged utility → EMA trace → prune min trace ---
        prune_imgs = images_jnp[:, -1]
        prune_lbls = labels_jnp[:, -1]

        new_masks, utility_traces = jitted_prune(
            batched_state, prune_imgs, prune_lbls, utility_traces)

        for l in range(n_hidden_layers):
            batched_state = eqx.tree_at(
                lambda st, _l=l: st.masks[_l],
                batched_state,
                new_masks[l],
            )

    # Flush any remaining accumulated metrics
    flush_logs()


    # --- Collect results (averaged across seeds) ---
    results = {}
    for m, method in enumerate(methods):
        remaining_per_seed = []
        for s in range(n_seeds):
            sm_masks = tuple(mask[s, m] for mask in batched_state.masks)
            remaining_per_seed.append(sum(float(mask.sum()) for mask in sm_masks))

        loss_hist = np.array(all_loss_history[m])
        acc_hist = np.array(all_acc_history[m])
        results[method.name] = {
            'loss_history': loss_hist,
            'accuracy_history': acc_hist,
            'remaining_units': float(np.mean(remaining_per_seed)),
            'avg_loss': float(np.nanmean(loss_hist)),
            'avg_accuracy': float(np.nanmean(acc_hist)),
            'method_name': method.name,
            'config': method.config,
        }

    return results


# ==============================================================================
# Entry point
# ==============================================================================

@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    """Run the pruning experiment."""
    configure_jax(cfg)

    # Normalize seeds
    if isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    start = time.time()
    results = run_experiment(cfg)
    elapsed = time.time() - start

    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Methods run: {len(results)}")

    # Save results
    output_path = cfg.output
    with open(output_path, 'wb') as f:
        pickle.dump({'results': results, 'config': dict(cfg)}, f)
    print(f"Results saved to {output_path}")

    # Print summary
    valid = {k: v for k, v in results.items() if np.isfinite(v['avg_loss'])}
    if valid:
        sorted_methods = sorted(valid.items(), key=lambda x: x[1]['avg_loss'])
        print(f"\nTop 5 methods (lowest avg loss):")
        for name, r in sorted_methods[:5]:
            print(f"  {name}: avg_loss={r['avg_loss']:.4f}, "
                  f"avg_acc={r['avg_accuracy']:.4f}, "
                  f"remaining_units={r['remaining_units']:.0f}")

        print(f"\nBaselines:")
        for name in BASELINE_NAMES:
            if name in results:
                r = results[name]
                print(f"  {name}: avg_loss={r['avg_loss']:.4f}, "
                      f"avg_acc={r['avg_accuracy']:.4f}, "
                      f"remaining_units={r['remaining_units']:.0f}")


if __name__ == '__main__':
    main()
