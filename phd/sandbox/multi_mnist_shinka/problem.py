"""The fixed benchmark a candidate algorithm is scored on. Read-only to the search.

The problem is multi-linear GEOFF (`phd.jax_core.tasks.multi_linear`): `n_tasks` independent
linear regressions laid side by side. Every output is a noisy linear function of exactly the
`n_features_per_task` inputs in its own slot and carries no information about any other slot.
The inputs and outputs are shuffled by a per-seed random permutation, so a learner sees flat
vectors of `n_inputs` and `n_outputs` and has to find the block structure from the data.

A candidate is a pair of pure functions,

    init(n_inputs, n_outputs, param_budget, key) -> state
    update(state, x, y) -> (state, prediction)

run one sample at a time: `update` predicts `y` from `x` using `state`, then learns from the
true `y`. `state` is a dict with one reserved entry. `state['weights']` holds everything the
prediction is computed from; its floating-point elements are the parameters and are capped at
`param_budget`, 90% of what the ideal solution needs, while integer arrays in it (wiring) are
free. The rest of `state` is for learning statistics. The state as a whole is capped at
`MEMORY_FACTOR` times the parameter budget: room for several numbers per weight, not for a
buffer of samples.

The score is the MSE over the last `FINAL_FRACTION` of the run divided by the MSE of the
optimal predictor (the noise variance), averaged over `N_SEEDS` independent problems. It is 1
for the optimum, 2 for always predicting the mean, and lower is better.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from phd.jax_core.tasks.multi_linear import MultiLinearTask
from phd.jax_core.utils import stack_pytrees


# One config per run, chosen by --config: `small` is for checking the loop in seconds,
# `full` is the problem the search is actually for.
SMALL = dict(n_tasks=4, n_features_per_task=10, n_outputs_per_task=5, n_steps=2_000)
FULL = dict(n_tasks=20, n_features_per_task=20, n_outputs_per_task=10, n_steps=100_000)
CONFIGS = {'small': SMALL, 'full': FULL}

NOISE_STD = 1.0
BUDGET_FRACTION = 0.9     # parameters allowed, as a fraction of the ideal solution's
MEMORY_FACTOR = 10        # total state allowed, as a multiple of the parameter budget
FINAL_FRACTION = 0.1      # the tail of the run that is scored
MEAN_PREDICTOR_LOSS = (1.0 + NOISE_STD ** 2) / NOISE_STD ** 2   # score of always predicting 0
N_SEEDS = 10
EVAL_SEEDS = list(range(N_SEEDS))                   # what the search optimizes against
HOLDOUT_SEEDS = list(range(1000, 1000 + N_SEEDS))   # never seen by the search


def param_budget(cfg: dict) -> int:
    """90% of the ideal solution's parameter count: one weight per (output, own-slot input)."""
    ideal = cfg['n_tasks'] * cfg['n_outputs_per_task'] * cfg['n_features_per_task']
    return int(BUDGET_FRACTION * ideal)


def count_params(state: dict) -> int:
    """Floating-point elements of `state['weights']`."""
    leaves = jax.tree_util.tree_leaves(state['weights'])
    return int(sum(jnp.size(a) for a in leaves if jnp.issubdtype(a.dtype, jnp.floating)))


def count_memory(state: dict) -> int:
    """Every element of every array anywhere in `state`, whatever its dtype."""
    return int(sum(jnp.size(a) for a in jax.tree_util.tree_leaves(state)))


def make_problem(cfg: dict, seed: int):
    """The teacher for one seed, plus the permutations that hide its slot layout."""
    task = MultiLinearTask(
        cfg['n_tasks'], n_features_per_task=cfg['n_features_per_task'],
        n_outputs_per_task=cfg['n_outputs_per_task'], noise_std=NOISE_STD, seed=seed,
    )
    in_key, out_key = random.split(random.fold_in(random.PRNGKey(seed), 1))
    in_perm = random.permutation(in_key, task.n_features)
    out_perm = random.permutation(out_key, task.n_outputs)
    return task, in_perm, out_perm


def run(init, update, cfg: dict, seeds: list) -> dict:
    """Scores a candidate on `len(seeds)` independent problems, all run at once under vmap.

    Returns the per-seed normalized loss over the final part of the run, and the candidate's
    parameter and total state counts. Raises if either count is over budget.
    """
    budget = param_budget(cfg)
    n_inputs = cfg['n_tasks'] * cfg['n_features_per_task']
    n_outputs = cfg['n_tasks'] * cfg['n_outputs_per_task']

    problems = jax.jit(jax.vmap(lambda seed: make_problem(cfg, seed)))(jnp.array(seeds))
    states = [init(n_inputs, n_outputs, budget, random.fold_in(random.PRNGKey(s), 2)) for s in seeds]
    if not isinstance(states[0], dict) or 'weights' not in states[0]:
        raise ValueError("state must be a dict with a 'weights' entry")
    states = stack_pytrees(states)

    n_params = count_params(jax.tree.map(lambda a: a[0], states))
    n_memory = count_memory(jax.tree.map(lambda a: a[0], states))
    if n_params > budget:
        raise ValueError(f"'weights' holds {n_params} floats but the parameter budget is {budget}")
    if n_memory > MEMORY_FACTOR * budget:
        raise ValueError(f'state holds {n_memory} numbers but the memory budget is {MEMORY_FACTOR * budget}')

    def one_problem(problem, state):
        task, in_perm, out_perm = problem

        def step(carry, _):
            task, state = carry
            task, (x, y) = task.generate_batch(1)
            x, y = x[0][in_perm], y[0][out_perm]
            state, prediction = update(state, x, y)
            assert prediction.shape == y.shape, \
                f'prediction has shape {prediction.shape}, expected {y.shape}'
            return (task, state), jnp.mean((prediction - y) ** 2)

        _, mse = jax.lax.scan(step, (task, state), length=cfg['n_steps'])
        return mse

    mse = np.asarray(jax.jit(jax.vmap(one_problem))(problems, states))   # (n_seeds, n_steps)

    n_final = int(FINAL_FRACTION * cfg['n_steps'])
    optimal_mse = problems[0].irreducible_mse
    return dict(
        normalized_loss=mse[:, -n_final:].mean(axis=1) / optimal_mse,
        n_params=n_params,
        n_memory=n_memory,
    )
