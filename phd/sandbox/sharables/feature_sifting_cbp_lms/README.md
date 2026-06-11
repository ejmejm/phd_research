# Feature Sifting

A small, self-contained benchmark for **feature sifting** — an online linear-regression task where a learner must continually prune useless input features and replace them with fresh candidates — together with a minimal demo that solves it using continual backprop.

## Problem Description

Pruning weights to make space for new features is an important part of learning the structure of a network. If a network is to prune weights, it needs to decide which weights to prune and when. Perhaps the simplest setting to study this is a single-target linear regression problem where the network is augmented with the ability to prune features. Whenever a feature is pruned, it is always replaced with another feature. Below I propose a particular instance of this problem.

My proposed problem defines $n$ inputs, $n$ corresponding weights, and a target that is the dot product of the inputs and weights, plus a noise scalar. Each weight is randomly initialized to -1 or +1. At every step, inputs are randomly sampled from the uniform distribution $[-1, 1]$, and noise is randomly sampled from a normal distribution with mean 0 and standard deviation 0.1.

The learner has $2n$ features, and each new feature is generated via the following procedure:
1. Sample $i$ from $\{1, \dots, n\}$ for the target network feature index.
2. Sample $c$ from $[0, 1]$ for a noise coefficient.
3. Generate a feature that, for each time step has value $(1-c)\,x_i + c\,\mathcal{U}(-1,1)$, where $x_i$ is the value of the $i$-th feature in the target network.

Generating new features in this way keeps the distribution of all features fixed and bounded while varying their potential usefulness. Also because we allocate the learner more than $n$ features, we can compute the optimal possible performance from just the standard deviation of the target noise. With a standard deviation of 0.1, the optimal asymptotic loss is 0.01.

## Contents

- `feature_sifting.py` — the task as a self-contained, stateful numpy class (`FeatureSiftingTaskNP`); requires only numpy.
- `feature_sifting_jax.py` — the same task as a jittable/vmappable JAX + equinox module (`FeatureSiftingTask`); requires `jax`, `equinox`, and `jaxtyping`.
- `cbp_lms_demo.py` — a minimal demo that solves the task with LMS (online linear regression) plus continual-backprop feature recycling, using the numpy task. Run `python cbp_lms_demo.py` (needs numpy and matplotlib); it prints progress and saves a learning-curve plot. Pass `--help` to see the configurable arguments.