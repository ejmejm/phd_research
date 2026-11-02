# Blocked Unit: What We Learned

Unit search on the deep sparse teacher did no better than random sparse wiring. Notebooks 01–05 isolate why on a
one-layer version: the target is a sum of K LTUs, each reading its own 10 of 1,000 inputs.

## Why it failed, and the current fix for each

1. **A converged linear learner blocks new LTUs (01, 03).** It already has the target's linear part, so a new LTU's push
   toward its feature scales with its overlap squared, which starts near 0. With 8 target LTUs, even LTUs reading the
   right inputs barely start in 1M steps.
   **Fix:** ±1 shadow output weights in the backward pass (01), and fast learning: input step-size 3e-2 with the signs
   of the linear learner's weights (04a): 8 target LTUs in ~40k steps.
2. **Extra inputs drown the signal (02).** An LTU reading all 1,000 inputs starts with 10× more weight on distractors
   than on its 10 right inputs, and that weight random-walks, so it never gets past the linear learner.
   **Fix:** LTUs read only a few inputs; the rest are candidates that don't touch the output until connected.
3. **Sampled LTUs hold only partial features, and partial features look useless (03, 04).** With K target LTUs, a sample
   by the linear learner's weights gets ~1/K of its inputs from each. With 4 of 10 right inputs, an LTU adds ~1% of what
   a full feature does, so its output weight stays ~0 and search by output weight can't find it. The gradient on its
   missing right inputs, though, grows with how many it already has.
   **Fix:** growth (04c): connect an unread input when its summed gradient stands out (6× the RMS over unread inputs).

Not problems, or not fixes: wrong inputs only slow learning a little (04b); zero-crossing pruning cuts right inputs too
at high step-sizes (04b); letting Autostep find connections (everything connected, tiny step-sizes) works but ~3× slower
than growth, and mostly fails from sampled starts (04d).

## Where it stands (05)

Sampled LTUs + growth + fast learning + search every 10k–30k steps: RMSE 0.16 with 8 target LTUs (right wiring 0.155;
fully connected and everything without growth 0.61). Slow: ~280k steps to 0.2 (right wiring 32k).

## Not yet explored

- Candidates sampled by linear usefulness within a budget instead of every input; pruned candidates return to the pool.
- Pruning wrong connections: CI-based, CBP/SET utility; a smaller per-LTU budget with CI search.
- A growth threshold that needs no per-problem tuning; simplifying the three connection phases.
- Math or a simple model of starting with several right inputs vs growing from one.
