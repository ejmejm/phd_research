from typing import Dict, Optional
import jax.numpy as jnp
from jaxtyping import Array, Float
from omegaconf import DictConfig

from .models import MLP
from .tasks.geoff import NonlinearGEOFFTask


def compute_feature_diff_matrix(
    learning_net_weights: Float[Array, 'hidden_dim in_features'],
    target_net_weights: Float[Array, 'true_hidden in_features'],
) -> Float[Array, 'hidden_dim true_hidden']:
    """Compute the matrix of feature differences between the learning network and target network."""
    # Normalize weights
    learning_net_weights = learning_net_weights / jnp.linalg.norm(learning_net_weights, axis=1, keepdims=True)
    target_net_weights = target_net_weights / jnp.linalg.norm(target_net_weights, axis=1, keepdims=True)
    
    # (hidden_dim, 1, in_features) vs (1, true_hidden, in_features)
    learning_expanded = learning_net_weights[:, None, :]
    target_expanded = target_net_weights[None, :, :]
    
    weight_diffs = jnp.abs(learning_expanded - target_expanded)
    negative_feature_diffs = jnp.abs(learning_expanded + target_expanded)
    
    feature_diffs = weight_diffs.sum(axis=2)
    negative_feature_diffs = negative_feature_diffs.sum(axis=2)
    
    feature_diffs = jnp.minimum(feature_diffs, negative_feature_diffs) # (hidden_dim, true_hidden)
    return feature_diffs


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
    feature_diffs = compute_feature_diff_matrix(learning_net_weights, target_net_weights)
    
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
    

def compute_model_statistics(
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
    feature_diffs = compute_feature_diff_matrix(learning_net_weights, target_net_weights) # (learner_dim, target_dim)
    best_match_diffs = feature_diffs.min(axis=1) # (learner_dim,)
    
    perfect_match_mask = best_match_diffs < 1e-7
    imperfect_match_mask = ~perfect_match_mask
    
    output_weights = model.layers[1].weight.squeeze(0) # (learner_dim,)
    for prefix, mask in [('perfect_', perfect_match_mask), ('imperfect_', imperfect_match_mask)]:
        n_features = mask.sum()
        masked_weights = output_weights * mask
        metrics[f'model/{prefix}output_weight_l1'] = jnp.linalg.norm(masked_weights, ord=1) / n_features
        metrics[f'model/{prefix}output_weight_max_magnitude'] = jnp.max(jnp.abs(masked_weights))
        metrics[f'model/{prefix}output_weight_min_magnitude'] = jnp.min(
            jnp.inf * ~mask + jnp.abs(masked_weights))
    
    # for layer_idx, layer in enumerate(model.layers):
    #     weight_l1 = jnp.linalg.norm(layer.weight, ord=1) / layer.weight.size
    #     metrics[f'layer_{layer_idx}/weight_l1'] = weight_l1
    #     metrics[f'layer_{layer_idx}/max_weight_magnitude'] = jnp.max(jnp.abs(layer.weight))
    #     metrics[f'layer_{layer_idx}/min_weight_magnitude'] = jnp.min(jnp.abs(layer.weight))
        
    return metrics