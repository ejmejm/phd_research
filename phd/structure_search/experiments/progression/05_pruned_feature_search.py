"""Step 5 — Pruned Feature Search (DEFERRED).

Placeholder. Architecture sketch:

- Empty network at init (``init_strategy='empty'``, no units, no connections).
- Unit generation: each new unit gets 128 random inputs with LeCun-uniform
  weights (identical to ``free_generate``'s lecun-uniform path at
  ``n_conns=128``) and a single output connection with zero-init weight
  (``single_output_generate``).
- Unit pruning: fixed-rate by signed LOO utility (``loo_utility``). New units
  start with utility = median of active unit utilities
  (``median_utility_init``).
- Weight-level pruning: incoming weights pruned by the statistical-
  confidence threshold from local_pruning_progression step 6
  (``prune_statistical``). Per-connection age + resolved state must be
  tracked — a new manager class extending ``ConnectionConnectivityManager``
  is required.
- Input normalization: ``load_mnist_normalized()`` (σ_x ≈ 1 for τ_w formula).

Pending details from the user before implementation:
- Exact fixed feature prune rate and cadence.
- Whether the rate applies always, or only above a target active-unit count.
- Interaction between statistical weight pruning and dead-unit detection.
- Logging granularity for per-unit age / fan-in / utility.
"""

raise NotImplementedError(
    'Step 5 is pending detailed spec. See phd/structure_search/experiments/'
    'progression/README.md for the architecture sketch.'
)
