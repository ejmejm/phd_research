# Connection Slots

## Methods of notebooks 1 and 2

All methods have 1 depth of LTUs and one output, and a budget of `MAX_CONNECTIONS` (2048). The biases never count.
The code is in `algorithms.py`.

### Connection and unit search (`CountAllConnections`)

The budget counts every connection: the LTUs' inputs, and the output's connections to x and to the LTUs. The output
reads all of x and every LTU.

- **prune_connections** (every 10k steps, on the LTUs only): it cuts each connection whose weight sign changed since
  the previous check. New connections are exempt.
- **add_connections** (every 1k steps): each LTU keeps a gradient sum for each input it does not read. It connects the
  input with the largest sum if that sum is more than 6 times the root mean square of the sums. This happens only
  while the budget has room. A new connection starts at weight 0 with a step size of 0.03.
- **prune_units** (every 10k steps): it removes the 1/8 of the LTUs with the smallest |outgoing weight|.
- **add_units** (every 10k steps): it creates LTUs while the budget has room for 21 connections each. A new LTU reads
  20 inputs, drawn in proportion to the output's weights squared. They get random weights with a step size of 0.03.
  The output reads the new LTU from weight 0, with a shadow weight of ±1.

### No connection search (`NoConnectionSearch`, baseline)

- **prune_connections**: none.
- **add_connections**: none.
- **prune_units**: as in `CountAllConnections`.
- **add_units**: as in `CountAllConnections`.

### Static (`Static`, baseline)

- **prune_connections**: none.
- **add_connections**: none.
- **prune_units**: none.
- **add_units**: none. The network of step 0 never changes.

### Candidates in the budget (`CountCandidates`, baseline)

The budget also counts the candidates: the inputs whose gradient sums an LTU keeps but does not read. So each LTU
counts all of x, and a new LTU costs `N_INPUTS` + 1 connections.

- **prune_connections**: as in `CountAllConnections`. It frees no budget.
- **add_connections**: as in `CountAllConnections`. It uses no budget.
- **prune_units**: as in `CountAllConnections`.
- **add_units**: as in `CountAllConnections`, while the budget has room for `N_INPUTS` + 1 connections each.

### Output budget (`OutputBudget`, proposed)

Each unit owns its incoming connections. The output holds the budget and pays for its own connections and for each
LTU, at 31 connections each. Half of each freed amount goes to a connection bucket, and half goes to a unit bucket.

- **prune_connections** (every 10k steps, on the LTUs and the output): it cuts each connection whose weight sign
  changed since the previous check. New connections are exempt. An LTU that the output stops reading is removed.
- **add_connections** (every 1k steps): each LTU refills its free slots up to 30, with uniform draws from x. The
  output reads uniform draws from the sources it does not read, as many as the connection bucket holds. The part of
  the connection bucket that has nothing left to read goes to the unit bucket. A new connection starts at weight 0
  with a step size of 1e-4.
- **prune_units** (every 10k steps): it removes the 1/8 of the LTUs with the smallest |outgoing weight|.
- **add_units** (every 10k steps): it creates as many LTUs as the unit bucket holds. A new LTU reads 30 inputs,
  drawn in proportion to the output's weights squared, then uniformly where those weights are 0. 15 inputs get
  random weights with a step size of 0.03, and 15 inputs get weight 0 with a step size of 1e-4. The output reads the
  new LTU from weight 0, with a shadow weight of ±1.

### Output budget, soft candidates (`SoftCandidates`, proposed)

- **prune_connections**: as in `OutputBudget`.
- **add_connections**: as in `OutputBudget`.
- **prune_units**: as in `OutputBudget`.
- **add_units**: as in `OutputBudget`, but all 30 inputs of a new LTU get weights from U(−1e-4, 1e-4), with a step
  size of 1e-4.

### Output budget with the |w| + |h| rule (`Sifting` and `SiftingSoftCandidates`, proposed)

- **prune_connections** (every step, after the weight update, on the LTUs and the output): it cuts each connection
  whose |w| + |h| is below the bound for a dead weight of the same age and step size. This is the CI rule of
  `../../feature_sifting/networks/parallel_mnist_label_changes.ipynb`. Biases are exempt. An LTU that the output stops
  reading is removed.
- **add_connections**: as in `OutputBudget`.
- **prune_units**: as in `OutputBudget`.
- **add_units**: as in `OutputBudget` (`Sifting`) or `SoftCandidates` (`SiftingSoftCandidates`).

## Agreed environment

- Each example has 108 independent Gaussian inputs, with mean zero and variance one.
- Four target LTUs each read four randomly chosen inputs.
- Each LTU has fixed weights of +1 or -1, a zero threshold, and outputs -1 or +1.
- One output sums all four LTUs through weights of +1 or -1.
- Independent Gaussian output noise has standard deviation 0.1.
- Every 50,000 examples, all four target output weights reverse sign. All other target properties stay fixed.

## Agreed baseline sources

Use `../output_prioritization/3_progress.ipynb` as the base, with connection pruning from `../pruning_and_depth/1_connection_budget.ipynb`.
Keep the fixed LTU count and weakest-LTU replacement from output prioritization.
Every 10,000 examples, remove LTU input connections whose weights changed sign since the previous check.
Exempt new connections and biases. Removed inputs remain available for later connection growth.
Apply connection pruning before whole-LTU replacement.
This combines the two improvements without the older global connection budget.
With one output, every LTU has the same parent.

## Agreed goals

- Keep work and memory per unit bounded, with search costs included.
- Match or improve on prior methods at the same total budget for active connections and candidates.
- Test new connections with little harm to useful connections. Remove useless connections to release space.
- Adapt quickly to changes in optimal weight size and sign, with brief periods near zero allowed.

## Problems and success

1. **Repeated search.** Initial connections usually miss necessary inputs. A fixed initial set cannot solve the task, but the method that checks every input can.
   Success means eventual accuracy near that of a network with ideal connections, plus the equal-budget goal above.
2. **Changed weights.** Change the required weight sizes and signs while the ideal connections stay fixed.
   Success means fast recovery with little extra delay or error from connection removal and search.

Defer changes in ideal connections, long periods of no use, and a separate network size experiment.
Still check that no step scans every possible input for every unit.

`output_reversals.ipynb` tests the second property. `old/3_bounded_search.ipynb` tests the first.

## Initial methods

- **Separate candidates.** Candidates have no effect on predictions. Promote them after enough evidence of benefit, with zero weight and a moderate step size.
- **One connection type.** Start weights at zero with small step sizes. Let useful connections develop larger step sizes.
- **Inactive weight range.** Keep this idea for later. A weight changes internally but affects predictions only after it leaves the inactive range.

Use the same limited pool and the same input selection rule for both initial methods.
Apply the connection budget to every unit, outputs included.
Choose new inputs uniformly from allowed sources, without a scan of the parent's weights or all possible inputs.
Check duplicates only against the unit's own slots.
The longer-term aim is the same connection rules for hidden units and outputs.
New hidden units retain random initial weights. Later connections start at zero.
Choose parents with gated progress from `../output_prioritization/3_progress.ipynb`, then choose each shadow sign randomly.
Each shadow signal uses an actual output slot and counts toward that output's budget.

**Active connection removal.** Start with the rule in `../../feature_sifting/networks/parallel_mnist_label_changes.ipynb`.
Add the absolute weight to the absolute Autostep update history.
Remove the connection when this sum falls below the level expected from random variation at its age and step size.
For a promoted candidate, start its age when it becomes active.
Weight size protects settled connections. Consistent recent updates can protect a weight as it passes through zero.
This rule estimates usefulness. It does not prove that removal causes no harm.

## Current notebook

`output_reversals.ipynb` runs both proposed methods on one output that reverses every 50,000 examples. `methods.py` holds the code.

After a reversal, LTUs whose shadow is on the reversed output lose their features, so error rises a second time.
Search connects the reversed output to LTUs whose shadow is on the fixed output or that have no shadow.
These LTUs keep their features, so later reversals recover fast. With connections fixed, the losses repeat at every reversal.
The same output always reverses. If both outputs reversed, this benefit of search would probably disappear. This is not tested.

`single_output_reversals.ipynb` repeats this with one output and 12 reversals.
Every tracked LTU with a shadow reverses its feature at every reversal. Search moves the output's weight to LTUs without a shadow.
With small initial steps, the error then barely rises at later reversals.

Earlier notebooks and their notes are in `old/`.
