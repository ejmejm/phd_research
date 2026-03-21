# Parallel MNIST Baseline Experiments

## Motivation

We want to establish baselines for structure (connectivity) search by comparing a dense MLP against a block-sparse MLP (the oracle structure) on a task where sparsity is known to be optimal. The parallel MNIST task concatenates K independent MNIST sub-tasks into a single input-output problem. The ideal network for this task is block-diagonal: each output group depends only on the corresponding input group.

These baselines serve two purposes:
1. **Stationary:** Show that the right sparse structure yields better capacity efficiency — matching dense performance with fewer parameters.
2. **Non-stationary:** Show that the right sparse structure yields faster adaptation — when one sub-task's labels change, only the relevant block needs to re-learn.

The structure search algorithm will later be evaluated against these baselines. It should approach block-sparse performance without knowing the correct structure in advance.


## Task: Parallel MNIST

- K=10 independent MNIST sub-tasks
- Input: concatenated K images, dimension K*784 = 7840
- Output: K*10 = 100 logits, reshaped to (K, 10) with per-task softmax
- Cross-entropy loss averaged over tasks
- Online learning (batch_size=1)

**Non-stationary variant:** Every `permute_period` steps, one randomly chosen sub-task has its label mapping permuted (e.g., digit "3" → class "7"). The input distribution is unchanged; only the target mapping changes.


## Models

### Dense MLP
Standard fully-connected MLP with `output_dim = K*10 = 100`. Every hidden unit connects to all 7840 inputs. For n_layers=3: params = 7840*H + H^2 + 100*H.

### Block-Sparse MLP
K independent sub-MLPs, one per task. Each sub-MLP maps 784 inputs to 10 outputs through hidden layers of width h. Implemented as batched weight tensors with einsum. For n_layers=3: params = K*(784*h + h^2 + 10*h).

At the same parameter budget, both models have similar hidden dim, but the block-sparse model's hidden units connect to only 784 relevant inputs instead of 7840.


## Experiment 1: Stationary

**Goal:** Test accuracy at convergence vs parameter budget.

**Setup:**
- K=10 tasks, n_layers=3, Adam optimizer
- 200K training steps, test evaluation every 5000 steps
- 3 seeds per trial

**Sweep:**
- `model.target_params`: 2^15, 2^16, 2^17, 2^18, 2^19, 2^20
- `optimizer.learning_rate`: 2^-13, 2^-11, 2^-9, 2^-7, 2^-5
- 30 trials per model type (60 total)

**Sweep configs:**
- `conf/sweeps/parallel_mnist/dense_stationary.yaml`
- `conf/sweeps/parallel_mnist/block_sparse_stationary.yaml`

**Analysis:**
- Select best LR per (model_type, target_params) by `asymptotic_test_loss`
- Plot 1: test accuracy vs target_params (two lines: dense, block-sparse)
- Plot 2: test loss vs target_params
- Plot 3: learning curves at selected param budgets

**Expected result:** Block-sparse matches or exceeds dense at all budgets. The gap is largest at small/medium budgets where the dense network wastes capacity on irrelevant cross-task connections. At large budgets both converge.


## Experiment 2: Non-Stationary

**Goal:** Average online loss/accuracy vs label permutation frequency.

**Setup:**
- K=10 tasks, n_layers=3, Adam optimizer
- target_params = 2^19 (placeholder — pick a budget where both models converge to similar stationary performance)
- 500K training steps, no test evaluation (online metrics only)
- 3 seeds per trial

**Sweep:**
- `dataset.permute_period`: 2000, 5000, 10000, 25000, 50000
- `optimizer.learning_rate`: 2^-13, 2^-11, 2^-9, 2^-7, 2^-5
- 25 trials per model type (50 total)

**Sweep configs:**
- `conf/sweeps/parallel_mnist/dense_nonstationary.yaml`
- `conf/sweeps/parallel_mnist/block_sparse_nonstationary.yaml`

**Analysis:**
- Select best LR per (model_type, permute_period) by `average_loss`
- Plot 4: average loss vs permute_period (two lines: dense, block-sparse)
- Plot 5: average accuracy vs permute_period
- Plot 6: learning curves at a single permute_period showing adaptation spikes

**Expected result:** Block-sparse has lower average loss at all permutation periods. The gap grows as permutations become more frequent, demonstrating faster adaptation through structural credit assignment.


## Running the Experiments

```bash
# From phd/structure_search/
# Stationary
mlflow-sweeper conf/sweeps/parallel_mnist/dense_stationary.yaml
mlflow-sweeper conf/sweeps/parallel_mnist/block_sparse_stationary.yaml

# Non-stationary (after choosing target_params from stationary results)
mlflow-sweeper conf/sweeps/parallel_mnist/dense_nonstationary.yaml
mlflow-sweeper conf/sweeps/parallel_mnist/block_sparse_nonstationary.yaml
```

## Analysis

See `analysis/parallel_mnist_analysis.ipynb` for the analysis notebook. It loads experiment data from MLflow CSV exports and generates all plots listed above.
