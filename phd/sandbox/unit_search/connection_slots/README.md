# Connection Slots

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
