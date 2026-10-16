# Incremental generalization

Goal: past learning improves future predictions, despite change and limited capacity.

## Aspirational environment

A graph of reusable computations. Some modules remain useful, some serve several
others, and some disappear and return. Weights change slowly; connections and depth
also change. Recurrence is deferred.

Use LTU hidden units, linear outputs, constant-input biases, and Gaussian noise
with standard deviation 1.

## Milestones

1. **One fixed depth-one teacher:** establish that its features can be learned.
2. **Three recurring depth-one teachers:** accumulate and retain features while
   relearning a shared readout. Inputs and outputs are shared across tasks.
3. **One fixed depth-two teacher:** establish that a two-layer representation can be
   learned from scratch.
4. **Build on existing features:** return to step 2 and introduce a new task whose
   hidden module reads features from the existing teachers. Keep the original tasks
   recurring. Does prior knowledge make learning the deeper computation easier?
5. **Update existing knowledge:** occasionally replace one teacher feature while
   preserving the others. Its change affects both direct predictions and computations
   built on it.
6. **Change composition:** change which existing features the higher module reads.
7. **Scale:** add modules and depth, one axis at a time.

Step 3 isolates learning depth before step 4 combines it with retained knowledge.
Correct-initialization retention tests are diagnostics alongside this progression.

Add one difficulty at a time. Fixed, representable targets have ideal RMSE 1.
Changing targets need achievable references; their optimal online performance is unknown.
