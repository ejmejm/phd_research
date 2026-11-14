# Pruning and Depth

**Goal.** Two changes to the method of `../step_by_step/`, each shown on one problem: LTUs prune the connections they
do not need, which frees room under a budget of connections, and an LTU moves down to just above the deepest LTU it
reads, instead of by the swap's threshold.

Each notebook asks one question, at output noise of std 0.1 and 1:

1. `1_connection_budget.ipynb` — under a fixed budget of connections, can pruning free the room to add the LTUs a
   sparse target network needs? Yes: without pruning, the LTUs' first inputs fill the budget and the method learns
   little; with pruning, the same budget holds half again as many LTUs, and the method ends about level with LTUs
   wired like the target network.
2. `2_depth_from_connections.ipynb` — with target LTUs at two depths, can each LTU simply move down to just above the
   deepest LTU it reads? Yes: the method ends level with LTUs wired like the target network. Without moves, depth-2
   LTUs that read only inputs strand their features, and it learns much more slowly.

## The method

Lines tagged with a notebook number are new here; the rest is as in `../step_by_step/`.

```
setup     a budget: a cap on the LTUs' connections from inputs and LTUs in all (biases not counted)                (1)
          each LTU has a depth, 1..D, and may read the inputs and LTUs of lower depth                              (2)
          the outputs read the inputs (the linear learner) and every LTU; an LTU's strength is its sum of
          |outgoing weights|
          at step 0, as many NEW LTUs as fit at 20 sources each, reading random inputs with random signs           (1)

every step
  LEARN   network Autostep; in the backward pass, each LTU's shadow weight (±1 on its parent) stands in for its real
          weight to the parent while that is smaller than 1 in magnitude
          G[u, j] += dL/dz_u · x_j   for every source j that LTU u may read but doesn't

every 1k steps
  GROW    each LTU u, while the budget has room: j = argmax |G[u, :]|; if |G[u, j]| > 6 · rms(G[u, :]):  u reads j
          (w 0, α 0.03); G[u, :] = 0                                                                             (1)

every 10k steps
  PRUNE   each LTU u stops reading every source j whose w[u, j] changed sign since the last PRUNE (sources read
          since then are spared; never the bias)                                                                 (1)
  MOVE    each LTU moves down to one depth above the deepest LTU it reads (depth 1 if it reads only inputs)        (2)
  SEARCH  remove the 4 weakest LTUs, then add NEW LTUs while the budget has room for their connections             (1)

NEW LTU u
  parent p = a random unit of a random layer above depth 1, the outputs (depth D + 1) included; shadow(u → p) = ±1
  u's depth is one below p's; if p is an LTU, p reads u from now on, pruned like any other connection             (2)
  u reads up to 20 sources j it may read, drawn ∝ w[p, j]²
  w[u, j] = |LeCun| · sign(shadow · w[p, j]), with step-size α = 0.03; outgoing weights 0
```

Neither change adds a threshold, and with the budget there is no fixed number of LTUs, at any depth.

`old/` holds this folder's earlier notebooks, to delete once reviewed; `old/5_changing_targets.ipynb` starts the
extension to changing targets.
