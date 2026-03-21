"""Block-sparse MLP for parallel independent tasks.

Implements K independent sub-MLPs as batched weight tensors, where each
sub-MLP processes one slice of the input (e.g., one MNIST sub-task in a
parallel MNIST setup). This is equivalent to a block-diagonal weight
structure — the oracle baseline for connectivity search.
"""
import math
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from phd.jax_core.models import ACTIVATION_MAP, lecun_uniform


class BlockSparseMLP(eqx.Module):
    """K independent sub-MLPs stored as batched weight tensors.

    Each sub-MLP maps input_dim_per_task inputs to output_dim_per_task
    outputs through hidden layers of width hidden_dim. The forward pass
    splits the input vector into K blocks, runs each through its own
    sub-MLP via batched einsum, and concatenates the outputs.
    """

    n_tasks: int = eqx.field(static=True)
    input_dim_per_task: int = eqx.field(static=True)
    output_dim_per_task: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    n_frozen_layers: int = eqx.field(static=True)
    activation_fn: Callable = eqx.field(static=True)

    # Each element: (K, out_features, in_features)
    layers: List[jnp.ndarray]

    def __init__(
        self,
        n_tasks: int,
        input_dim_per_task: int,
        output_dim_per_task: int,
        n_layers: int,
        hidden_dim: int,
        weight_init_method: str = 'lecun_uniform',
        activation: str = 'relu',
        n_frozen_layers: int = 0,
        *,
        key: PRNGKeyArray,
    ):
        self.n_tasks = n_tasks
        self.input_dim_per_task = input_dim_per_task
        self.output_dim_per_task = output_dim_per_task
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_frozen_layers = n_frozen_layers
        self.activation_fn = ACTIVATION_MAP[activation]

        keys = jax.random.split(key, n_layers)
        self.layers = []

        if n_layers == 1:
            w = self._init_weights(
                keys[0], (n_tasks, output_dim_per_task, input_dim_per_task),
                weight_init_method,
            )
            self.layers.append(w)
        else:
            # First hidden layer
            w = self._init_weights(
                keys[0], (n_tasks, hidden_dim, input_dim_per_task),
                weight_init_method,
            )
            self.layers.append(w)

            # Interior hidden layers
            for i in range(1, n_layers - 1):
                w = self._init_weights(
                    keys[i], (n_tasks, hidden_dim, hidden_dim),
                    weight_init_method,
                )
                self.layers.append(w)

            # Output layer
            w = self._init_weights(
                keys[-1], (n_tasks, output_dim_per_task, hidden_dim),
                weight_init_method,
            )
            self.layers.append(w)

    @staticmethod
    def _init_weights(key, shape, method):
        if method == 'lecun_uniform':
            return lecun_uniform(key, shape)
        elif method == 'zeros':
            return jnp.zeros(shape)
        else:
            raise ValueError(f'Unknown weight_init_method: {method}')

    def __call__(
        self,
        x: jnp.ndarray,
        set_first_element_to_one: bool = False,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Forward pass through K independent sub-MLPs.

        Args:
            x: Flattened input (K * input_dim_per_task,)

        Returns:
            output: (K * output_dim_per_task,)
            param_inputs: list of per-layer inputs, each (K, dim)
        """
        x = x.reshape(self.n_tasks, self.input_dim_per_task)  # (K, 784)
        param_inputs = []

        for layer_w in self.layers[:-1]:
            param_inputs.append(x)
            x = jnp.einsum('koh,kh->ko', layer_w, x)  # (K, hidden_dim)
            x = self.activation_fn(x)

        param_inputs.append(x)
        output = jnp.einsum('koh,kh->ko', self.layers[-1], x)

        return output.reshape(-1), param_inputs


def compute_hidden_dim_for_params(
    target_params: int,
    model_type: str,
    n_layers: int,
    n_tasks: int,
    input_dim_per_task: int = 784,
    output_dim_per_task: int = 10,
) -> int:
    """Compute hidden_dim to approximately match a target parameter count.

    Solves the quadratic equation for hidden_dim given the model architecture.

    For dense MLP (output_dim = n_tasks * output_dim_per_task):
        params = (n_tasks*input_dim_per_task)*H + (n_layers-2)*H^2 + H*(n_tasks*output_dim_per_task)

    For block_sparse:
        params = n_tasks * (input_dim_per_task*h + (n_layers-2)*h^2 + h*output_dim_per_task)
    """
    if model_type == 'block_sparse':
        per_task = target_params / n_tasks
        a = max(n_layers - 2, 0)
        b = input_dim_per_task + output_dim_per_task
        # a*h^2 + b*h = per_task
        if a == 0:
            h = per_task / b
        else:
            h = (-b + math.sqrt(b * b + 4 * a * per_task)) / (2 * a)
        return max(1, int(h))
    elif model_type in ('mlp', 'dense'):
        total_input = n_tasks * input_dim_per_task
        total_output = n_tasks * output_dim_per_task
        a = max(n_layers - 2, 0)
        b = total_input + total_output
        # a*H^2 + b*H = target_params
        if a == 0:
            h = target_params / b
        else:
            h = (-b + math.sqrt(b * b + 4 * a * target_params)) / (2 * a)
        return max(1, int(h))
    else:
        raise ValueError(f'Cannot compute hidden_dim for model_type={model_type}')
