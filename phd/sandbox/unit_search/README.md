# Unit search

A network with a fixed connection budget learns which hidden units exist, not just which inputs each one reads: it
prunes whole units and generates new ones in the budget they free. Connection-level learning on its own is in
`phd/sandbox/feature_sifting/networks/`.

- `parallel_mnist_label_changes.ipynb` — unit search from an empty network on 4-task parallel MNIST with label
  changes, at the ideal network's connection budget.
- `sparse_teacher.ipynb` — a deep, sparsely connected LTU target network (1,000 inputs, 3 × 20 hidden units, 100
  outputs): how fully connected networks scale with width, then unit search with units at depths 1–3 that read any
  lower depth.
- `sparse_teacher_dense_units.ipynb` — the same target, with unit search whose new units connect to everything
  before and after them, and connection pruning removes the connections that do not matter.
- `demo/` — a demo of unit search on a small random sparse, locally connected target network (33 inputs, 2 × 7 LTUs,
  7 outputs):
  - `sparse_target_demo.ipynb` — unit search against fully connected networks (SGD, Adam, batched Adam, network
    Autostep), each at its best step-size.
  - `record.py` — records one run of unit search (the notebook's code) to `data.js`, for `index.html`, a playback
    of the network as it learns (2D, layers as rows), beside the target network, with its RMSE.
- `step_by_step/` — unit search step by step, from a new unit blocked in a trained network to the deep sparse target
  network: six short notebooks, each with one question, the simplest experiment that answers it, and a simple fix. Its
  README has the method in pseudocode and a log of options to improve each part.
- `pruning_and_depth/` — two changes to that method, in two notebooks of the same kind: under a budget of
  connections, LTUs prune connections whose weight changed sign since the last pruning, which frees room for the LTUs
  a target network needs, and each LTU moves down to just above the deepest LTU it reads, instead of by the swap's
  threshold.
- `output_prioritization/` — which output each new LTU serves (its parent), when only a few of 20 outputs can be
  improved: parents drawn by error go to outputs that are noise; drawn by error times explained fraction, with no
  hyperparameters, they go where LTUs can help, and find an output that changes, but outputs that are part linear and
  part noise fool them. Parents drawn by prior success (the share of each output's LTUs that help it, with its
  explained fraction and the pooled share as two pseudo-LTUs) fix this at low noise and are never much worse than
  uniform parents. No output stops getting new LTUs, and no setting is tied to the problem. Each LTU feeds only its
  parent. Two notebooks.


## Improvements

Below I have broken the algorithm into different components and listed areas where the current algorithm needs improvement.

**Learn:** This is mostly good now. If anything, we may want to consider why a network with the ideal connectivity slows down its learning so much at some point. Ideally being given the right connectivity would allow a network to quickly reach near-optimal performance. We may eventually also deal with a vanishing gradients problem in deeper networks, though that is hypothetical as of now.

**Swap:** The current algorithm is awkward. This problem only arises because the cost of depth is not visible to the network. If information only flowed forwards and backwards one unit per step, then the lag cost of depth would naturally work its way into the structure learned. Because we don't have that we may need a better solution. I think the biggest win on this now would be to have units be able to move down in max depth but not up. The algorithm currently creates connections to every other unit in a different layer so we need to set a threshold for the maximum weight on the layer below for moving a unit down. If we instead though have a unit that can prune its connections, then we can simply move a unit down a maximum depth if it prunes all of its connections to the layer below. We should also look to get rid of the idea of a fixed number of units per layer, which will then allow us to simply move something down instead of having to swap it. So the first todo item here is actually in how connections are generated and pruned rather than directly in how the swap is done. We can revisit this once we change that. Then we can decide whether or not we still want to have a threshold, and if so, how it can be set automatically to avoid tuning.

**Grow Units:** There are two key questions that I don't think yet have satisfactory answers: which inputs and outputs to connect to at initialization. I think the answers we have now are decent for inputs—choose inputs that are useful to the master output linearly—but this is incomplete. It does not answer the question of which outputs a unit should connect to and how it should determine its master (for when shadow weights are used). A good test problem for this could be one where we intentionally create different targets with different numbers of LTUs required, or remove all of the features from one output and see if the algorithm can learn to allocate search there accordingly. 
We might also be able to do better that limiting ourselves to units that have a direct connection to the master output. One thing I imagine doing in the future is adding a notion of physical location and locality to units ([[Units With Physical Locations]]) so that semantic similarity, and hence which inputs may be relevant, can be encoded in a unit's location.

**Prune Units:** A lot of improvements need to be made here. One existing problem is that connections could be useful even if their weight values are currently small. This could be because the problem changes over time or because it is new. We don't want to prune units if we expect they will be useful in the future, but the current metric does not account for that. The other problem is the unanswered question of how to account for differences in inputs and outputs when comparing unit utilities. In most methods that do something similar, only units within the same layer are compared because they share the same set of inputs and outputs and can be compared on equal ground. Units in different layers, or in our case, units with differing connectivity may not be directly comparable.

**Grow Connections:** The current method of (RigL-style) gradient based connection adding requires computing statistics for all candidate inputs all of the time, and it requires tuning a somewhat arbitrary threshold parameter. These are both problems. I suspect the solution to the first will be to only test so many units at a time, or to prune candidate connections after they are useless for some time. To eliminate the threshold parameter, I am hoping we can do some math to get an idea of how the threshold can be dynamical determined base on the data flowing through the network.
There is also the problem of knowing which units to connect to. If you no longer test everything and have to decide a subset of things to test, then the RigL-style generation no longer works. This won't be a problem in the short term, but as we scale up to networks with millions of units, it will become a problem. This is again a place where I think the idea of [[Units With Physical Locations]] could help.

**Prune Connections:** This currently doesn't happen. It will be important to prune connections eventually though as we scale because not everything we test will be useful, and we will want to get rid of useless connections to regain space. Useless connections also can propagate gradients and assign credit where unwanted. The solution to this might look something like continual backprop's utility or my CI-based feature sifting algorithm. Though something tells me we will eventually need something better than both of these that accounts for the fact that connections can be useless for quite a while but eventually become useful again.