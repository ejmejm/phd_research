"""Seed program: least mean squares on a fixed random wiring.

The two functions below are the whole interface. The evaluator calls `init` once per problem
and then `update` once per sample; see prompt.md for the problem statement and the rules.
"""

import jax
import jax.numpy as jnp


# EVOLVE-BLOCK-START
STEP_SIZE = 0.01


def init(n_inputs: int, n_outputs: int, param_budget: int, key: jax.Array) -> dict:
    """Wires every output to the same number of random inputs, one weight per connection.
    The integer wiring is free; only the float weights count against the budget."""
    n_connections = param_budget // n_outputs
    idx = jax.random.randint(key, (n_outputs, n_connections), 0, n_inputs)
    return {'weights': {'w': jnp.zeros((n_outputs, n_connections)), 'idx': idx}}


def update(state: dict, x: jax.Array, y: jax.Array):
    """Predicts, then takes one SGD step on the squared error."""
    w, idx = state['weights']['w'], state['weights']['idx']
    x_wired = x[idx]                                          # (n_outputs, n_connections)
    prediction = jnp.sum(w * x_wired, axis=1)
    w = w + STEP_SIZE * (y - prediction)[:, None] * x_wired
    return {'weights': {'w': w, 'idx': idx}}, prediction
# EVOLVE-BLOCK-END
