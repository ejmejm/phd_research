# Output Prioritization

**Goal.** Each new LTU serves one output, its parent. Choose the parents so that search goes to the outputs that new
features can improve, and not to the outputs that no feature can improve. In the long run, an agent may pose thousands
of predictions, some noisy, some fully predictable, some not at all: the method must decide how much effort each one
deserves while it knows little about it, and never do much worse than uniform parents.

**Start.** In `../step_by_step/`, a new LTU's parent is drawn uniformly. Here each LTU feeds only its parent, as in a
sparse network: with thousands of outputs, an LTU cannot feed them all.

**Main points.** Uniform parents do well in many cases. But often, an output whose error new inputs recently decreased
can be decreased further. The method measures this expected decrease despite noise, and it keeps half of the parents
uniform, so it does not fail badly where this heuristic is wrong.

Each notebook asks one question, on 20 outputs of which 2 can be improved by LTUs, with noise of std 0.1 and 1 on
these 2:

1. `1_which_output.ipynb` — from the level of each output's error, can search find the outputs that LTUs can improve?
   No. Parents by error go to outputs that are noise. Parents by error times explained fraction keep away from noise
   as well as an oracle, but noisy linear outputs fool them: noise looks the same as error that LTUs can remove.
2. `2_prior_success.ipynb` — can search draw parents by the share of each output's LTUs that help it now? Yes. It
   comes close to the oracle with noise outputs, does better than uniform parents with noisy linear outputs at noise
   std 0.1, and as well at noise std 1.
3. `3_progress.ipynb` — can search draw parents by progress, the expected decrease in error from one more new input?
   Only with a gate. Without it, the noise in the measured decrease gives noise outputs much of the priority. Gated at
   3 standard errors, noise outputs get about 30 times less priority, and the RMSE stays close to prior success.

## The method

```
every step, for each output o, with sums whose weights decay with time constants of 10k (fast) and 80k (slow) steps
  L_o, M_o   mean |prediction_o - y_o|, fast and slow
  n_o        o's new inputs: its new LTUs, and the 401 linear inputs and bias as new at step 0, slow (at least 1)
  s_o        the standard error of M_o - L_o: the std of |prediction_o - y_o| times the root of the summed squared
             differences between the normalized fast and slow weights
  P_o        (M_o - L_o) / M_o * L_o / n_o   if M_o - L_o > 3 s_o, else 0

every 10k steps
  each new LTU's parent ~ P_o + mean of P      (uniform if all P are 0)
```

**Settings.** The fast time constant is the search period, and the slow one the time that search takes to replace all
the LTUs (160 LTUs, 20 at each search step). The gate at 3 standard errors is a common default.

## Next

Choose the two time constants with no knowledge of the problem: with no hyperparameters, with a rule that adjusts
them, or with a rule of thumb that works on almost all problems.

New LTUs take their inputs from their parent's linear weights. Thus, search may not find the inputs of a learnable
output with no linear part.

The notebooks before this cleanup, with all the rules tried, are at commit 9b643e6. `old/` holds the earlier notebooks,
their README and the pilots, to delete once reviewed.
