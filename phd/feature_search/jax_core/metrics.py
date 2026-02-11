from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from omegaconf import DictConfig
from torch import layer_norm

from .models import MLP
from .optimizers import EqxOptimizer
from .tasks.geoff import NonlinearGEOFFTask


def compute_feature_diff_matrix(
    learning_net_weights: Float[Array, 'hidden_dim in_features'],
    target_net_weights: Float[Array, 'true_hidden in_features'],
    include_negative_features: bool = False,
) -> Float[Array, 'hidden_dim true_hidden']:
    """Compute the matrix of feature differences between the learning network and target network."""
    # Normalize weights
    learning_net_weights = learning_net_weights / jnp.linalg.norm(learning_net_weights, axis=1, keepdims=True)
    target_net_weights = target_net_weights / jnp.linalg.norm(target_net_weights, axis=1, keepdims=True)
    
    # (hidden_dim, 1, in_features) vs (1, true_hidden, in_features)
    learning_expanded = learning_net_weights[:, None, :]
    target_expanded = target_net_weights[None, :, :]
    
    weight_diffs = jnp.abs(learning_expanded - target_expanded)
    feature_diffs = weight_diffs.sum(axis=2)
    
    if not include_negative_features:
        return feature_diffs
    
    negative_feature_diffs = jnp.abs(learning_expanded + target_expanded)
    negative_feature_diffs = negative_feature_diffs.sum(axis=2)
    feature_diffs = jnp.minimum(feature_diffs, negative_feature_diffs) # (hidden_dim, true_hidden)
    negative_mask = negative_feature_diffs < feature_diffs
    return feature_diffs, negative_mask # Negative mask is True where the closest feature match is the negative feature


# Only works if doing feature search where feature matches can be exact
def compute_best_feature_match_counts(
    learning_net_weights: Float[Array, 'hidden_dim in_features'],
    target_net_weights: Float[Array, 'true_hidden in_features'],
) -> Float[Array, 'true_hidden']:
    """Get, for each target feature, how closely the closest learning network hidden unit matches it."""
    # (hidden_dim, 1, in_features) vs (1, true_hidden, in_features)
    feature_matches = jnp.expand_dims(learning_net_weights, axis=1) == jnp.expand_dims(target_net_weights, axis=0)
    positive_feature_match_counts = feature_matches.sum(axis=2) # Num matches between each learning net and true feature
    negative_feature_match_counts = (~feature_matches).sum(axis=2)
    feature_match_counts = jnp.maximum(positive_feature_match_counts, negative_feature_match_counts)
    best_feature_match_counts = feature_match_counts.max(axis=0)
    return best_feature_match_counts


def compute_best_feature_match_distances(
    learning_net_weights: Float[Array, 'hidden_dim in_features'],
    target_net_weights: Float[Array, 'true_hidden in_features'],
    target_net_output_weights: Optional[Float[Array, 'true_hidden out_features']] = None,
) -> Float[Array, 'true_hidden']:
    """Get, for each target feature, how closely the closest learning network hidden unit matches it."""
    feature_diffs, _ = compute_feature_diff_matrix(learning_net_weights, target_net_weights, include_negative_features=True)
    
    # Normalize so each diff can be a maximum of 1 with binary weights
    best_feature_match_diffs = feature_diffs.min(axis=0) # (true_hidden,)
    normalized_target_net_weights = target_net_weights / jnp.linalg.norm(target_net_weights, axis=1, keepdims=True)
    baseline_feature_l1_vals = jnp.abs(normalized_target_net_weights).sum(axis=1)
    normalized_diffs = best_feature_match_diffs / baseline_feature_l1_vals
    
    # Also normalize by output weights so that the sum of diffs will at most equal 1
    output_weighted_diffs = None
    if target_net_output_weights is not None:
        output_weight_magnitudes = jnp.abs(target_net_output_weights).sum(axis=1)
        output_weighting = output_weight_magnitudes / output_weight_magnitudes.sum()
        output_weighted_diffs = normalized_diffs * output_weighting
    
    return normalized_diffs, output_weighted_diffs


def compute_feature_match_stats(
    model: MLP,
    task: NonlinearGEOFFTask,
    perfect_matches: bool,
) -> Dict[str, float]:
    """Get, for each target feature, how closely the closest learning network hidden unit matches it."""
    learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
    target_net_weights = task.weights[0].T # (true_hidden, in_features)
    if len(task.weights) == 2:
        target_net_output_weights = task.weights[1] # (true_hidden, out_features)
    
    metrics = {}
    
    if len(task.weights) == 2:
        best_feature_match_diffs, output_weighted_diffs = compute_best_feature_match_distances(
            learning_net_weights, target_net_weights, target_net_output_weights)
        metrics['feature_match/best_match_output_weighted_normalized_distance'] = output_weighted_diffs.sum()
    else:
        best_feature_match_diffs, _ = compute_best_feature_match_distances(learning_net_weights, target_net_weights)
    metrics['feature_match/average_best_match_normalized_distance'] = best_feature_match_diffs.mean()
    
    if perfect_matches:
        best_feature_match_counts = compute_best_feature_match_counts(learning_net_weights, target_net_weights)
        n_inputs = learning_net_weights.shape[1]
        perfect_match_count = jnp.sum(best_feature_match_counts == n_inputs)
        average_match_count = best_feature_match_counts.mean()
        metrics['feature_match/perfect_match_count'] = perfect_match_count
        metrics['feature_match/average_best_match_count'] = average_match_count
    
    return metrics
    

def compute_model_stats(
    model: MLP,
    task: NonlinearGEOFFTask,
) -> Dict[str, float]:
    """Compute statistics about the model's weights."""
    assert len(model.layers) == 2, "Only supports two-layer learning networks!"
    assert model.layers[1].weight.shape[0] == 1, "Only supports single-output networks!"
    
    metrics = {}
    
    # First make a mask to divide weights into perfect matches and non-perfect matches
    learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
    target_net_weights = task.weights[0].T # (true_hidden, in_features)
    feature_diffs, _ = compute_feature_diff_matrix(learning_net_weights, target_net_weights, include_negative_features=True) # (learner_dim, target_dim)
    best_match_diffs = feature_diffs.min(axis=1) # (learner_dim,)
    
    perfect_match_mask = best_match_diffs < 1e-7
    imperfect_match_mask = ~perfect_match_mask
    
    output_weights = model.layers[1].weight.squeeze(0) # (learner_dim,)
    for prefix, mask in [('perfect_', perfect_match_mask), ('imperfect_', imperfect_match_mask)]:
        n_features = mask.sum()
        masked_weights = output_weights * mask
        metrics[f'model/{prefix}output_weight_l1'] = jnp.linalg.norm(masked_weights, ord=1) / n_features
        # metrics[f'model/{prefix}output_weight_max_magnitude'] = jnp.max(jnp.abs(masked_weights))
        # metrics[f'model/{prefix}output_weight_min_magnitude'] = jnp.min(
        #     jnp.inf * ~mask + jnp.abs(masked_weights))
    
    # for layer_idx, layer in enumerate(model.layers):
    #     weight_l1 = jnp.linalg.norm(layer.weight, ord=1) / layer.weight.size
    #     metrics[f'layer_{layer_idx}/weight_l1'] = weight_l1
    #     metrics[f'layer_{layer_idx}/max_weight_magnitude'] = jnp.max(jnp.abs(layer.weight))
    #     metrics[f'layer_{layer_idx}/min_weight_magnitude'] = jnp.min(jnp.abs(layer.weight))
        
    return metrics


def compute_optimizer_stats(
    optimizer: EqxOptimizer
) -> Dict[str, float]:
    """Compute statistics about the model's weights."""
    assert optimizer.name == 'idbd', "Only supports IDBD optimizer!"
    metrics = {}

    # Get the beta parameters from each layer
    beta_leaves = jax.tree.leaves(optimizer.state.beta)
    n_frozen_layers = len(optimizer.state.beta.layers) - len(beta_leaves)
    for i, beta in enumerate(beta_leaves, start=n_frozen_layers):
        beta_flat = beta.ravel()
        metrics[f'optimizer/layer_{i}/beta_mean'] = jnp.mean(beta_flat)
        metrics[f'optimizer/layer_{i}/beta_std'] = jnp.std(beta_flat)

    return metrics


def compute_update_stats(
    model: MLP,
    update_mean_l1s: Float[Array, '... n_layers'],
    steps_since_log: int,
) -> Dict[str, float]:
    """Compute mean L1 of updates for each model layer.

    Args:
        model: The model, used to determine layer structure.
        update_mean_l1s: Accumulated mean L1 values per layer with shape
            (..., n_trainable_layers) where ... are step dimensions.
        steps_since_log: Number of steps to average over.

    Returns:
        Dictionary of metrics with mean L1 per layer.
    """
    metrics = {}

    n_frozen_layers = len(model.layers) - update_mean_l1s.shape[-1]
    for i in range(n_frozen_layers, len(model.layers)):
        metrics[f'updates/layer_{i}/mean_l1'] = update_mean_l1s[:, i - n_frozen_layers].mean()

    return metrics


def compute_feature_matched_optimizer_stats(
    optimizer: EqxOptimizer,
    model: MLP,
    task: NonlinearGEOFFTask,
) -> Dict[str, float]:
    """Compute statistics about the model's weights."""
    assert len(model.layers) == 2, "Only supports two-layer learning networks!"
    assert model.layers[1].weight.shape[0] == 1, "Only supports single-output networks!"
    assert optimizer.name == 'idbd', "Only supports IDBD optimizer!"
    
    metrics = {}
    
    # First make a mask to divide weights into perfect matches and non-perfect matches
    learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
    target_net_weights = task.weights[0].T # (true_hidden, in_features)
    feature_diffs, _ = compute_feature_diff_matrix(learning_net_weights, target_net_weights, include_negative_features=True) # (learner_dim, target_dim)
    best_match_diffs = feature_diffs.min(axis=1) # (learner_dim,)
    
    perfect_match_mask = best_match_diffs < 1e-7
    imperfect_match_mask = ~perfect_match_mask
    
    # Get output layer step-sizes
    output_betas = jax.tree.leaves(optimizer.state.beta)[-1].squeeze(0)
    output_step_sizes = jnp.exp(output_betas) # (learner_dim,)
    output_weights = model.layers[1].weight.squeeze(0) # (learner_dim,)
    assert output_step_sizes.shape == output_weights.shape, "Step sizes and output weights should have the same shape!"
    
    # Get input layer step-sizes
    input_layer_trainable = len(model.layers) == len(jax.tree.leaves(optimizer.state.beta))
    if input_layer_trainable:
        input_betas = jax.tree.leaves(optimizer.state.beta)[0]
        input_step_sizes = jnp.exp(input_betas) # (learner_dim, in_features)
        input_step_sizes = input_step_sizes.mean(axis=1) # (learner_dim,)
    
    for prefix, mask in [('perfect_', perfect_match_mask), ('imperfect_', imperfect_match_mask)]:
        n_features = mask.sum()
        masked_output_step_sizes = output_step_sizes * mask
        metrics[f'optimizer/{prefix}mean_output_step_size'] = jnp.sum(masked_output_step_sizes) / n_features
        if input_layer_trainable:
            masked_input_step_sizes = input_step_sizes * mask
            metrics[f'optimizer/{prefix}mean_input_step_size'] = jnp.sum(masked_input_step_sizes) / n_features
        
    return metrics


def compute_cbp_stats(
    model: MLP,
    task: NonlinearGEOFFTask,
    utility: Float[Array, 'n_features'],
    cost_trace: Optional[Float[Array, 'out_features n_features']] = None,
) -> Dict[str, float]:
    """Compute CBP utility and cost trace statistics, split by perfect/imperfect feature match."""
    assert len(model.layers) == 2, "Only supports two-layer learning networks!"

    metrics = {}

    # Feature matching to split perfect vs imperfect
    learning_net_weights = model.layers[0].weight
    target_net_weights = task.weights[0].T
    feature_diffs, _ = compute_feature_diff_matrix(
        learning_net_weights, target_net_weights, include_negative_features=True)
    best_match_diffs = feature_diffs.min(axis=1)
    perfect_match_mask = best_match_diffs < 1e-7
    imperfect_match_mask = ~perfect_match_mask
    n_perfect = jnp.maximum(jnp.sum(perfect_match_mask), 1)
    n_imperfect = jnp.maximum(jnp.sum(imperfect_match_mask), 1)

    # Utility stats
    metrics['cbp/utility_mean'] = jnp.mean(utility)
    metrics['cbp/utility_perfect_mean'] = jnp.sum(utility * perfect_match_mask) / n_perfect
    metrics['cbp/utility_imperfect_mean'] = jnp.sum(utility * imperfect_match_mask) / n_imperfect

    # Cost trace stats (bias-corrected, aggregated per-feature)
    if cost_trace is not None:
        per_feature_cost = jnp.mean(cost_trace, axis=0)  # (n_features,)
        metrics['cbp/cost_trace_mean'] = jnp.mean(per_feature_cost)
        metrics['cbp/cost_trace_perfect_mean'] = jnp.sum(per_feature_cost * perfect_match_mask) / n_perfect
        metrics['cbp/cost_trace_imperfect_mean'] = jnp.sum(per_feature_cost * imperfect_match_mask) / n_imperfect

    return metrics


# TODO: Fix this function. Consider the negative feature case and stop comparing output weights.
def compute_n_best_features_pruned(
    model: MLP,
    prune_mask: Bool[Array, 'n_features'],
    task: NonlinearGEOFFTask,
):
    """
    Takes in a model, task, and a mask of which features are being pruned.
    Get the closest feature(s) in the learner for each target feature, generating a mask of shape (learner_dim, target_dim).
    If there are multiple closest features per target feature, then only the one with the smallest output weight difference
    is chosen.
    Returns the number of features in the prune mask that were the closest match to at least one target feature.
    """
    learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
    target_net_weights = task.weights[0].T # (true_hidden, in_features)
    feature_diffs, _ = compute_feature_diff_matrix(learning_net_weights, target_net_weights, include_negative_features=True) # (learner_dim, target_dim)
    
    min_diff_per_target_feature = feature_diffs.min(axis=0) # (target_dim,)
    best_match_mask = feature_diffs <= min_diff_per_target_feature[None, :] # (learner_dim, target_dim)
    

    # Narrow down the best match mask to a single 1 per target feature
    # If there are multiple 1s, choose the one with the smallest output weight difference
    
    learner_output_weights = model.layers[1].weight.squeeze(0)  # (learner_dim,)
    target_output_weights = task.weights[1].squeeze(-1)         # (target_dim,)

    # Compute absolute difference matrix between learner and target output weights
    output_diffs = jnp.abs(learner_output_weights[:, None] - target_output_weights[None, :])  # (learner_dim, target_dim)

    # Mask the output_diffs to inf where not a best match, remains valid for best matches
    masked_output_diffs = jnp.where(best_match_mask, output_diffs, jnp.inf)

    # Efficient, jittable selection: argmin over learner dimension produces exactly one best per target
    pruned_indices = jnp.argmin(masked_output_diffs, axis=0)  # (target_dim,)

    # Build a single-best-match mask: only one True per column (target feature)
    single_best_match_mask = jax.vmap(
        lambda idx, n: jax.nn.one_hot(idx, n, dtype=bool),
        in_axes=(0, None)
    )(pruned_indices, masked_output_diffs.shape[0]).T # (learner_dim, target_dim)
    
    match_and_prune_mask = jnp.logical_and(prune_mask[:, None], single_best_match_mask)
    n_best_features_pruned = match_and_prune_mask.any(axis=1).sum()
    
    return n_best_features_pruned