# Output Prioritization

**Goal.** Each new LTU serves one output, its parent. Choose the parents so that search goes to the outputs that LTUs
can improve, and not to the outputs that no LTU can improve. In the long run, an agent may pose thousands of
predictions, some noisy, some fully predictable, some not at all: the method must decide how much effort each one
deserves while it knows little about it, and never do much worse than uniform parents.

**Start.** In `../step_by_step/`, a new LTU's parent is drawn uniformly. Here each LTU feeds only its parent, as in a
sparse network: with thousands of outputs, an LTU cannot feed them all.

Each notebook asks one question, with noise of std 0.1 and 1 on the outputs that LTUs can improve:

1. `1_which_output.ipynb` — from the statistics of each output alone, can search find the outputs that LTUs can
   improve? Partly. Parents by error go to outputs that are noise. Parents by error times explained fraction keep away
   from noise as well as an oracle, but noisy linear outputs fool them: the linear learner explains half of such an
   output, and its noise looks the same as error that LTUs can remove.
2. `2_prior_success.ipynb` — can search draw parents by the share of each output's LTUs that help it? Yes. It keeps
   most of notebook 1's gain with noise outputs, does better than uniform parents with noisy linear outputs at noise
   std 0.1 and as well at noise std 1, loses nothing when all outputs are learnable, and keeps a small lead with 100
   outputs. No output ever stops getting new LTUs, the rule keeps no memory beyond the current LTUs, and it does not
   depend on the size of the network.

## The method

Lines tagged with a notebook number are new here; the rest is as in `../step_by_step/`, with each LTU feeding only its
parent.

```
every step, in sums whose weights decay by 1 - 1/T, T = 10k steps
  STATS   L_o, S_o, Q_o += (prediction_o - y_o)^2, y_o, y_o^2          for each output o                          (1)
          R_u, R2_u += r, r^2   r = the rise in u's parent's squared error if u were removed, for each LTU u      (2)
          u helps its parent if the weighted mean of r is more than 3 standard errors above 0                    (2)

every T steps, with weighted means
  SEARCH  e_o = (V_o - L_o) / V_o,  V_o = Q_o - S_o^2                   explained fraction                        (1)
          N_o = o's LTUs,  H_o = those that help it,  p = sum of H / sum of N                                     (2)
          each new LTU's parent ~ (H_o + e_o + p) / (N_o + 2)           prior success                             (2)
```

**Settings.** The rule has two numbers of its own, and both are common defaults. The help test uses 3 standard errors:
from 1 to 10, the RMSE changes by at most 0.04. The explained fraction and the pooled share p each count as one
pseudo-LTU, the usual weight of a prior. The time constant of the statistics is the search period.

## Next

The main limit is many more outputs than new LTUs: most outputs then have no LTUs, so e and p decide most parents, and
e can point the wrong way (noisy linear outputs at noise std 1).

New LTUs take their inputs from their parent's linear weights. Thus, search cannot find the inputs of a learnable
output with no linear part, even with oracle parents (a pilot run). That is the next progression.

`old/` holds the earlier notebooks (with the back-off notebook and the scale notebook), their README and the pilots,
to delete once reviewed.
