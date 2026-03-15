"""Pruning experiment: evaluate utility functions by how well they identify
the least important hidden units for pruning.

For each seed, all methods start from the same initial model. Every prune_freq
steps, the active hidden unit with the lowest |utility| is pruned (masked out).
Methods are compared by their loss/accuracy trajectories and final performance.

NOTE: Currently uses CIFAR-10 with MSE loss (one-hot regression).
      This will be changed to a proper regression dataset later.
"""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from phd.jax_core.models import MLP, ACTIVATION_MAP
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

@dataclass
class MethodSpec:
    """Specification for a utility-based pruning method."""
    name: str
    utility_fn: Callable
    needs_updates: bool  # Whether the utility fn requires optimizer updates
    config: Optional[UtilityConfig] = None  # None for baselines


def build_method_specs(activation: str = 'relu') -> List[MethodSpec]:
    """Build the full list of methods: 115 configurable + 3 baselines."""
    methods = []

    # Baselines
    methods.append(MethodSpec(
        name='Contribution',
        utility_fn=partial(contribution_utility, activation=activation),
        needs_updates=False,
    ))
    methods.append(MethodSpec(
        name='UPGD',
        utility_fn=partial(upgd_utility, activation=activation),
        needs_updates=False,
    ))
    methods.append(MethodSpec(
        name='SI',
        utility_fn=partial(si_utility, activation=activation),
        needs_updates=True,
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
            needs_updates=False,
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
    """Single training step without pruning."""
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
    return new_state, (loss, accuracy, grads, updates)


def compute_hidden_utilities(model, x, y_star, y_hat, loss_grads, updates,
                             masks, utility_fn, needs_updates):
    """Compute per-hidden-unit utilities using the specified method.

    Returns a flat array of utilities across all hidden layers,
    with masked units set to +inf so they're never selected for pruning.
    """
    if needs_updates:
        _, hidden_utils = utility_fn(model, x, y_star, y_hat, loss_grads, updates, masks=masks)
    else:
        _, hidden_utils = utility_fn(model, x, y_star, y_hat, None, None, masks=masks)

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
# Single method experiment
# ==============================================================================

def run_single_method(
    init_state: PruneState,
    streams: List[DataStream],
    method: MethodSpec,
    act_fn,
    total_steps: int,
    log_freq: int,
    prune_freq: int,
    prune_on_batch: bool = True,
) -> Dict:
    """Run the full training + pruning loop for a single method across seeds.

    This function is NOT jitted — it runs the outer loop in Python and uses
    jitted inner functions for the actual computation. This allows flexible
    pruning logic without complex JAX control flow.

    Args:
        init_state: vmapped PruneState across seeds (all fields have leading seed dim)
        streams: list of DataStream, one per seed
        method: the MethodSpec to use for utility-based pruning
        act_fn: activation function
        total_steps: total training steps
        log_freq: how often to log metrics
        prune_freq: how often to prune a unit
        prune_on_batch: if True, compute utility on the current training batch.
            If False, compute on the next batch (not implemented yet).

    Returns:
        dict with keys: loss_history, accuracy_history, remaining_units
    """
    n_seeds = len(streams)
    n_hidden_per_layer = tuple(
        model_layer.weight.shape[0]
        for model_layer in init_state.model.layers[:-1]
    )

    # JIT the core functions
    jit_train_step = jax.jit(jax.vmap(partial(train_step, act_fn=act_fn)))

    state = init_state
    loss_history = []
    accuracy_history = []
    step_losses = []
    step_accs = []

    for step in range(total_steps):
        # Sample one batch per seed
        batch = [stream.sample_batch(1) for stream in streams]
        images = jnp.array(np.stack([b[0][0] for b in batch]))  # (n_seeds, batch_size, input_dim)
        labels = jnp.array(np.stack([b[1][0] for b in batch]))  # (n_seeds, batch_size)

        state, (loss, accuracy, grads, updates) = jit_train_step(state, (images, labels))

        step_losses.append(float(loss.mean()))
        step_accs.append(float(accuracy.mean()))

        # Prune every prune_freq steps (after warmup of at least prune_freq steps)
        if (step + 1) % prune_freq == 0 and (step + 1) > 0:
            # Compute utility on the current batch for each seed independently
            for s in range(n_seeds):
                # Get single-seed data
                x_sample = images[s, 0]  # single input for utility
                one_hot = jax.nn.one_hot(labels[s, 0], NUM_CLASSES)

                # Extract single-seed model and masks
                s_model = jax.tree.map(lambda a: a[s], state.model)
                s_masks = tuple(m[s] for m in state.masks)

                y_hat = masked_forward(s_model, x_sample, s_masks, act_fn)

                # Extract grads/updates for this seed if needed
                s_grads = jax.tree.map(lambda a: a[s], grads) if method.needs_updates else None
                s_updates = jax.tree.map(lambda a: a[s], updates) if method.needs_updates else None

                abs_utils = compute_hidden_utilities(
                    s_model, x_sample, one_hot, y_hat,
                    s_grads, s_updates, s_masks,
                    method.utility_fn, method.needs_updates,
                )

                new_masks = prune_one_unit(s_masks, abs_utils, n_hidden_per_layer)

                # Update masks for this seed in the vmapped state
                for l in range(len(new_masks)):
                    state = eqx.tree_at(
                        lambda s, _l=l: s.masks[_l],
                        state,
                        state.masks[l].at[s].set(new_masks[l]),
                    )

        # Log metrics
        if (step + 1) % log_freq == 0:
            avg_loss = np.mean(step_losses[-log_freq:])
            avg_acc = np.mean(step_accs[-log_freq:])
            loss_history.append(avg_loss)
            accuracy_history.append(avg_acc)

    # Count remaining active units
    remaining = sum(
        float(m.sum(axis=-1).mean())  # average across seeds
        for m in state.masks
    )

    return {
        'loss_history': np.array(loss_history),
        'accuracy_history': np.array(accuracy_history),
        'remaining_units': remaining,
        'avg_loss': np.mean(loss_history),
        'avg_accuracy': np.mean(accuracy_history),
    }


# ==============================================================================
# Full experiment
# ==============================================================================

def prepare_experiment(cfg):
    """Initialize shared model + optimizer for all seeds."""
    images, labels, num_classes, input_dim = load_dataset(cfg['dataset_name'])
    act_fn = ACTIVATION_MAP[cfg['activation']]

    streams = []
    states = []
    for seed in cfg['seeds']:
        rng = jax.random.PRNGKey(seed)
        model_key, train_key = jax.random.split(rng)

        streams.append(DataStream(
            images=images, labels=labels, num_classes=num_classes,
            batch_size=cfg['batch_size'], seed=seed,
        ))

        model = MLP(
            input_dim=input_dim,
            output_dim=num_classes,
            n_layers=cfg['n_layers'],
            hidden_dim=cfg['hidden_dim'],
            weight_init_method=cfg['weight_init_method'],
            activation=cfg['activation'],
            key=model_key,
        )

        # Optimizer
        import optax
        from phd.jax_core.optimizers import EqxOptimizer
        if cfg['optimizer_name'] == 'adam':
            opt = optax.adam(cfg['learning_rate'])
        elif cfg['optimizer_name'] == 'sgd':
            opt = optax.sgd(cfg['learning_rate'])
        else:
            raise ValueError(f"Unsupported optimizer: {cfg['optimizer_name']}")
        optimizer = EqxOptimizer(opt, model, name=cfg['optimizer_name'])

        # Masks: all ones (all units active)
        masks = tuple(
            jnp.ones(layer.weight.shape[0])
            for layer in model.layers[:-1]
        )

        states.append(PruneState(
            model=model,
            opt_state=optimizer,
            masks=masks,
            step=jnp.array(0),
            rng=train_key,
        ))

    from phd.jax_core.utils import stack_pytrees
    batched_state = stack_pytrees(states)
    return batched_state, streams, act_fn


def run_experiment(cfg):
    """Run the full pruning experiment over all methods and seeds."""
    init_state, streams, act_fn = prepare_experiment(cfg)
    methods = build_method_specs(activation=cfg['activation'])

    print(f"Running {len(methods)} methods × {len(cfg['seeds'])} seeds")
    print(f"Total steps: {cfg['total_steps']}, prune every {cfg['prune_freq']} steps")

    results = {}
    for i, method in enumerate(tqdm(methods, desc='Methods')):
        # Clone the initial state for this method
        method_state = jax.tree.map(lambda x: x.copy(), init_state)

        try:
            result = run_single_method(
                init_state=method_state,
                streams=streams,
                method=method,
                act_fn=act_fn,
                total_steps=cfg['total_steps'],
                log_freq=cfg['log_freq'],
                prune_freq=cfg['prune_freq'],
            )
            result['method_name'] = method.name
            result['config'] = method.config
            results[method.name] = result
        except Exception as e:
            print(f"  Method {method.name} failed: {e}")
            results[method.name] = {
                'loss_history': np.array([np.nan]),
                'accuracy_history': np.array([np.nan]),
                'remaining_units': 0,
                'avg_loss': np.nan,
                'avg_accuracy': np.nan,
                'method_name': method.name,
                'config': method.config,
                'error': str(e),
            }

    return results


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    """Run the pruning experiment with default config."""
    cfg = {
        'dataset_name': 'cifar10',
        'n_layers': 3,
        'hidden_dim': 128,
        'activation': 'relu',
        'weight_init_method': 'lecun_uniform',
        'optimizer_name': 'adam',
        'learning_rate': 1e-3,
        'batch_size': 32,
        'total_steps': 50_000,
        'log_freq': 500,
        'prune_freq': 1000,
        'seeds': [1, 2, 3],
    }

    # Allow overrides from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=cfg['total_steps'])
    parser.add_argument('--prune_freq', type=int, default=cfg['prune_freq'])
    parser.add_argument('--log_freq', type=int, default=cfg['log_freq'])
    parser.add_argument('--hidden_dim', type=int, default=cfg['hidden_dim'])
    parser.add_argument('--n_layers', type=int, default=cfg['n_layers'])
    parser.add_argument('--batch_size', type=int, default=cfg['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=cfg['learning_rate'])
    parser.add_argument('--seeds', type=int, nargs='+', default=cfg['seeds'])
    parser.add_argument('--output', type=str, default='results.pkl')
    args = parser.parse_args()

    cfg.update({k: v for k, v in vars(args).items() if k != 'output'})

    start = time.time()
    results = run_experiment(cfg)
    elapsed = time.time() - start

    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Methods run: {len(results)}")

    # Save results
    output_path = args.output
    with open(output_path, 'wb') as f:
        pickle.dump({'results': results, 'config': cfg}, f)
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
