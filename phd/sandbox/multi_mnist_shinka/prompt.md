You are designing an online learning algorithm, in JAX, for a streaming regression problem.

## The problem

Each step you see one input vector x of {n_inputs} numbers, i.i.d. uniform on [-1, 1], and must predict a target vector y of {n_outputs} numbers; then the true y is revealed and the next sample arrives. Nothing is ever shown twice.

Each target is a linear function of exactly {n_features_per_task} of the {n_inputs} inputs, plus Gaussian noise of variance 1. The inputs partition into {n_tasks} hidden groups of {n_features_per_task} and the targets into {n_tasks} hidden groups of {n_outputs_per_task}; every target in a group reads exactly the inputs of one group and nothing else. So the ideal predictor has {ideal} weights, one per (target, relevant input). Which inputs and targets belong together is shuffled for every problem instance and has to be found from the data.

## The score

The mean squared error of your predictions over the last 10% of a {n_steps}-step run, divided by the mean squared error of the optimal predictor (the noise variance, 1), averaged over {n_seeds} independent problem instances. The optimum is 1, always predicting 0 gives 2, and lower is better. The score you are shown is minus this ratio, so that higher is better. Only the end of the run is scored, but the run is just {n_steps} steps long, so an algorithm that finds the structure slowly still pays for it.

## The rules

- Implement `init(n_inputs, n_outputs, param_budget, key) -> state` and `update(state, x, y) -> (state, prediction)`. `update` must produce `prediction` from `state` and `x` alone, then learn from `y`. `prediction` has the same shape as `y`.
- `state` is a dict. `state['weights']` holds everything the prediction is computed from. Its floating-point elements are your parameters and may not exceed `param_budget` = {param_budget}, which is 90% of the ideal solution's {ideal} weights, so there is no room for a dense linear map. Integer arrays inside `weights`, such as indices saying which input each weight reads, are free.
- Everything else in `state` is yours for learning statistics: step sizes, traces, relevance estimates, counters. The state as a whole, counting every element of every array in it, may not exceed {memory_budget} numbers, 10 times the parameter budget. That is room for several numbers per weight, not for a buffer of samples, and storing samples is not allowed regardless. The prediction must not read anything outside `weights`, and floats must not be packed into integer arrays.
- `update` runs under `jax.lax.scan` and `jax.vmap`. The state's structure, shapes and dtypes must stay fixed from step to step, all logic must be JAX-traceable (no Python control flow on array values, no host callbacks), and `x` and `y` are single samples, not batches.
- Only the code between the EVOLVE-BLOCK markers may change. The evaluator is fixed and not visible to you.

## What is known

Standard machinery is fair game and not interesting on its own: SGD/LMS, normalized LMS, per-weight adaptive step sizes (IDBD, Autostep, Adam-style), weight decay. The hard part is spending fewer parameters than the ideal solution well while streaming: finding which inputs matter for which targets, deciding which connections to keep and which to give up, and doing so with statistics cheap enough to fit in the state budget. Ideas with a chance include tracking cheap per-connection relevance signals and rewiring the weakest connections, sharing the discovered input group across targets that appear to belong together, and generate-and-test.
