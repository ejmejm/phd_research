# Unit Search, Step by Step

**Goal.** A network that finds the units of a sparse target network (which inputs each of its units reads) by
searching over units: every so often, its weakest units are replaced by new ones (generate and test, as in continual
backprop).

**Start.** Network Autostep with LTUs and shadow weights (`../../network_autostep/`). On the deep sparse target network,
generate and test with randomly wired new units stays at the linear learner's error (notebook 6).

Each notebook asks one question, and its answer leads to the next:

1. `1_blocked_unit.ipynb` — does a new LTU learn in a network that has already learned what it can? Slowly; a high
   initial step-size on its input weights lets it learn anyway.
2. `2_growing_connections.ipynb` — what does an LTU with only some of its feature's inputs learn? Nothing; growth,
   which connects the input whose summed gradient stands out, completes it.
3. `3_connect_to_everything.ipynb` — why not connect a new LTU to every input? Wrong inputs dilute its overlap with its
   feature; reading every input, it never learns.
4. `4_parent_weights.ipynb` — how should a new LTU pick its first inputs? By its parent's weights; random ones get only
   part of the way. The parent's signs only speed it up. From here on, LTUs are added at step 0, with no linear-learner
   phase: the LTUs search makes join a trained network anyway.
5. `5_several_layers.ipynb` — how does this extend to several layers? Each LTU treats everything below it as inputs and
   everything above it as outputs; the fixed max depth needs a patch, the swap.
6. `6_sparse_target_network.ipynb` — the method on the deep sparse target network: far better than fully connected
   networks (network Autostep, tuned Adam), where generate and test with randomly wired LTUs finds nothing; LTUs wired
   like the target network still end lower. More LTUs help, with diminishing returns past 40 per depth.

## The method

```
setup     the outputs read the inputs (the linear learner) and every LTU; each LTU slot has a max depth 1..D and
          reads the inputs and lower LTUs; an LTU's strength is its sum of |outgoing weights|
          at step 0, every slot gets an LTU reading random sources with random signs (no parent weights yet)       (4)

every step
  LEARN   network Autostep; in the backward pass, each LTU's shadow weight (±1 on its parent) stands in for its real
          weight to the parent while that is smaller than 1 in magnitude                                      (prior)
          G[u, j] += dL/dz_u · x_j   for every source j that LTU u may read but doesn't                            (2)

every 1k steps
  GROW    each LTU u: j = argmax |G[u, :]|; if |G[u, j]| > 6 · rms(G[u, :]):  u reads j (w 0, α 0.03); G[u, :] = 0  (2)

every 10k steps
  SWAP    for d = D .. 2: the strongest depth-d LTU with < 10% of its input |w| on depth d−1 swaps slots with the
          weakest depth-(d−1) LTU, if that one is weaker                                                          (5)
  SEARCH  replace the weakest 1 in 8 LTUs by NEW LTUs, from the top depth down                              (prior)

NEW LTU u
  parent p = a random unit of a random higher layer, the outputs included; shadow(u → p) = ±1;
  if p is an LTU, p reads u                                                                                    (4, 5)
  u reads 20 sources j drawn ∝ w[p, j]² (fewer if p weights fewer)                                                (4)
  w[u, j] = |LeCun| · sign(shadow · w[p, j]), with step-size α = 0.03; outgoing weights 0                      (1, 4)
```

## Improvements, by component

Options to improve each component of the method, logged as they come up, to prioritize and try one at a time.

**LEARN**

**GROW**

**SWAP**
- *Depth from what a unit reads, instead of the swap* (2026-11-05). Units add and prune connections, so they are not
  connected to everything below them. A depth-2 unit that has pruned all its connections from depth-1 units, and reads
  only inputs, moves to max depth 1. Units may lower their depth, but never raise it. Needs pruning in GROW.
- *Later: no prescribed number of units per depth*, or any other fixed depth structure; something more natural.
- *Thought:* in a pipelined network, where signals move forward and backward by one unit per step, depth costs delay,
  so a network might learn to trade depth for performance.

**SEARCH**

**NEW LTU**
