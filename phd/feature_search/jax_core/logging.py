import jax.numpy as jnp
from jaxtyping import Array, Float

from .models import MLP
from .tasks.geoff import NonlinearGEOFFTask


# target_net_weights = task.weights[0].T # (true_hidden, in_features)
# learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
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


def compute_feature_match_stats(
    model: MLP,
    task: NonlinearGEOFFTask,
) -> Float[Array, 'true_hidden']:
    """Get, for each target feature, how closely the closest learning network hidden unit matches it."""
    learning_net_weights = model.layers[0].weight # (hidden_dim, in_features)
    target_net_weights = task.weights[0].T # (true_hidden, in_features)
    best_feature_match_counts = compute_best_feature_match_counts(learning_net_weights, target_net_weights)
    
    n_inputs = learning_net_weights.shape[1]
    perfect_match_count = jnp.sum(best_feature_match_counts == n_inputs)
    average_match_count = best_feature_match_counts.mean()
    
    return {
        'feature_match/perfect_match_count': perfect_match_count,
        'feature_match/average_match_count': average_match_count, # For best match, not over all features
    }
    
    