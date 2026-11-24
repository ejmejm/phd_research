# Output Prioritization

**Goal.** Decide which output each new LTU serves (its parent, the output its shadow weight is on) so that search goes
where LTUs can still help: to outputs whose error is high and can be reduced, not to outputs that are noise or
already learned.

**Start.** In `../step_by_step/`, a new LTU's parent is drawn uniformly. On Asteroids RAM, drawing it in proportion to
each output's error did better, but outputs that cannot be predicted keep a high error and would draw LTUs forever.

**Test problem.** `../step_by_step/`'s notebook 4 target (8 target LTUs, each reading 10 of 400 inputs) read by the
first 10 of 100 outputs, each reading 4 target LTUs; only they can be improved by LTUs. The other 90 outputs are noise
of unit variance, linear functions of 10 inputs, or such linear functions plus noise of unit variance (noisy linear).
In notebook 5, 4 improvable outputs each read 2 target LTUs of their own instead. Learner: notebook 4's (32 LTUs,
parent's weights and signs, growth, the 4 least useful replaced every 10k steps), with usefulness the sum of squared
outgoing weights, the increase in squared error if the LTU were removed (sum of |weights| would add up the small
weights fitted to 90 noise outputs). 5 seeds; 500k steps. Measured on the improvable outputs: RMSE averaged over the
run (speed and end both count), and the share of LTUs whose parent is one of them.

Each notebook asks one question:

1. `1_outputs_that_cannot_be_improved.ipynb` — do outputs that cannot be improved take LTUs? With parents drawn by
   error, yes: with noise outputs it does far worse than uniform parents. With linear outputs, error points at the
   right outputs and nearly matches the oracle. Uniform parents are slow in both.
2. `2_parents_by_progress.ipynb` — does drawing parents by recent progress (the average decrease in each output's
   error, half the parents uniform) fix it? Partly, and no averaging time suits both kinds of other output: short
   averaging counts chance decreases of the noise outputs' error as progress; long averaging holds on to the linear
   learner's early progress on the linear outputs.
3. `3_credited_progress.ipynb` — does progress in the error the LTUs remove (removing their part of the prediction),
   which counts neither, fix it? A little better than notebook 2, but still behind parents by error with linear
   outputs, and still sensitive to averaging there: early on, the first LTUs remove some error from every output until
   the linear learner catches up, and the drop that follows counts as negative progress.
4. `4_error_times_explained.ipynb` — does error times explained fraction, the output's potential times how much of it
   has been explained so far, work without trying LTUs? With noise outputs it matches the oracle, and with linear
   outputs it nearly matches parents by error, but noisy linear outputs fool it. Drawing half the parents by credited
   progress instead (notebook 3) costs little and brings noisy linear outputs back to uniform parents' level.
5. `5_features_of_their_own.ipynb` — does it hold when each improvable output has features of its own? Yes with noise
   and linear outputs. With noisy linear outputs, the half-and-half rule is a little worse than uniform parents.
6. `6_four_times_the_ltus.ipynb` — with 4 times the LTUs, untuned? Every rule improves; the half-and-half rule nearly
   matches the oracle with noise and linear outputs, and slightly beats uniform parents with noisy linear outputs.
7. `7_a_needle_that_moves.ipynb` — when one output at a time changes, or a noise output becomes a target with
   features no LTU has learned, do new LTUs find it? Error times explained fraction does, almost as fast as an oracle
   that knows where the change is: the linear learner relearns the changed output's linear part within 6k steps, so
   its explained fraction is back by the next search step. Recent progress isn't needed for this.
8. `8_the_linear_learner_first.ipynb` — does starting the inputs' weights at a larger step-size than the LTUs'
   outgoing weights help? No: every rule is slower, the oracle included, and noisy linear outputs get worse. Averaged
   over the run, the LTUs took no credit for other outputs' linear work anyway.
9. `9_one_rule.ipynb` — is there one rule, with no hyperparameters, that weighs error by evidence from the LTUs
   instead? Error times the fraction of its error the LTUs remove handles noisy linear outputs, but never finds a
   needle, which no LTU helps yet. Error times credited progress, over one period or eight, is slow. Error times
   explained fraction remains the best single rule.
10. `10_learning_how_much_explained_says.ipynb` — can search learn, from what its new LTUs achieve, how far to trust
    the explained fraction? Error times a line fitted from the parent's explained fraction to new LTUs' outcomes does
    no worse than uniform parents on any problem here, and best on noisy linear outputs, but loses most of error times
    explained fraction's lead on the needle. On Asteroids (`../../one_off_experiments/ateroids_model/07_parent_rules.ipynb`)
    it ends about level with parents by error.
11. `11_searching_more_often.ipynb` — with search five times as often and half the new LTUs on uniform parents, does
    remembering what new LTUs achieved on each output move search past noisy linear outputs? No: it stays behind
    uniform parents there, and the random half slows the needle. Searching often helps every rule on the needle but
    hurts noisy linear outputs with shared features, the oracle included.
12. `12_settled_then_a_spike.ipynb` — with 2 learnable outputs among 20, so that search can try every output before
    anything changes, does memory of what new LTUs achieved keep search off decoys, and send it to an output that
    changes? Settled, yes, with the explained fraction standing in only until an output is tried: it beats uniform
    parents and error times explained fraction with noisy linear decoys. After a change it is slow: its memory is
    about the output as it was.

## The method

One rule, with no hyperparameters (notebook 4a):

```
every step     L_o  += (prediction_o - y_o)^2                       each output's squared error
               S_o  += y_o,  Q_o += y_o^2                            for the variance of its targets

every 10k steps (search), with means over the period:
  V_o = Q_o - S_o^2
  each new LTU's parent ~ L_o (V_o - L_o) / V_o                      error times explained fraction (0 if negative;
                                                                     uniform if all 0, as at step 0)
```

Results (uniform / error times explained fraction / oracle):

- 10 outputs sharing features, RMSE averaged over the run, other outputs noise: 0.364 / 0.230 / 0.230
- ... linear: 0.354 / 0.238 / 0.226 (parents by error: 0.235)
- ... noisy linear: 0.335 / 0.459 / 0.229
- 4 outputs with features of their own, noise: 0.419 / 0.202 / 0.204
- ... linear: 0.473 / 0.226 / 0.201
- ... noisy linear: 0.463 / 0.526 / 0.205
- 128 LTUs, shared features: noise 0.248 / 0.218 / 0.216; linear 0.258 / 0.223 / 0.214; noisy linear 0.254 / 0.286 /
  0.216
- One output at a time changes (notebook 7), the needle's RMSE over the 50k steps after its change: an output changes
  0.635 / 0.429 / 0.388; a noise output becomes a target 0.637 / 0.435 / 0.403

Drawing half the parents by credited progress instead (notebook 4b) brings noisy linear outputs to about uniform
parents' level with shared features (0.330; 0.251 with 128 LTUs), at a small cost elsewhere.

**Open problem: partly predictable outputs with large noise.** Before LTUs are tried on an output, nothing tells
whether its remaining error is noise or a feature the LTUs could learn, so error times explained fraction prefers noisy
linear outputs (error 1, half explained) to the improvable ones (error 0.37, 63% explained). Evidence from the LTUs
tells them apart (error times the fraction the LTUs remove, notebook 9b: 0.258 with shared features), but a rule
built on it never finds an output no LTU helps yet: a new needle, or an output with features of its own. One rule
would have to remember which outputs LTUs were tried on; with 96 noisy linear outputs, search tries about 2 LTUs per
output in 500k steps.

**Outputs with no linear part.** Error times explained fraction cannot see them, but neither can search itself: in a
pilot where each new needle has no linear part (a mirrored pair of thresholded target LTUs, $y \propto 1[|w \cdot x|
< c]$), even the oracle learns nothing (needle RMSE 1.02). A new LTU draws its first inputs from its parent's linear
weights, which are about 0 there.

**On Asteroids RAM** (`../../one_off_experiments/ateroids_model/`, 600 LTUs, 3 seeds, 400k steps), error times
explained fraction did a little worse than parents by error: 81% of the yardstick gap closed against 85%, worse in
every seed. Many of the bits LTUs can learn there are ones the linear learner explains little of (XOR-like counters,
the ship's heading, missile x positions), so they get little priority.

## Hypotheses

Notebooks 1–3 were planned with these hypotheses, on the RMSE averaged over the run (mean over seeds, and in at least
4 of 5 seeds, paired). Notebooks 4–6, 8 and 9 followed pilot runs (below), so their numbers were known before
they ran.

1. With noise outputs, parents by error do worse than uniform: **held** (5 of 5 seeds).
2. The oracle does better than uniform: **held** in both problems (5 of 5).
3. With linear outputs, parents by error do better than uniform: **held** (5 of 5).
4. At some averaging rate λ, progress does at least as well as the better of uniform and error in both problems:
   **failed**. With noise outputs, it does better than both at every λ (against uniform: 5 of 5 seeds at λ = 1/4 and
   1/16, 4 of 5 at λ = 1); with linear outputs, worse than error parents at every λ (best 0.261 at λ = 1 vs 0.235).
5. Progress's share of LTUs on the first 10 ends above uniform's 10%: **held**, but it tells little: uniform parents'
   share also ends at 0.31 and 0.42, because search keeps the useful LTUs on the first 10 and replaces the rest.
6. At every λ, credited progress does at least as well as the better of uniform and error in both problems:
   **failed**: with linear outputs, 0.251 / 0.274 / 0.292 at λ = 1 / 1/4 / 1/16, against 0.235 for error parents.
7. Its spread over λ is under 0.015 in both problems: **failed** with linear outputs (0.041); held with noise (0.008).
8. It does better than notebook 2's progress in both problems: **held at λ = 1 only** (0.292 vs 0.300 and 0.251 vs
   0.261, 4 of 5 seeds each).

Notebook 7 was planned with these, on the needle (the newest output reading new target LTUs):

9. The oracle finds the needles, its needle RMSE over the last 10k steps of each change's 50k below 0.3: **held**
   (0.209 for both kinds of change).
10. Error times explained fraction brings needles down faster than uniform and error parents (needle RMSE over the
    50k after each change): **held**, 5 of 5 seeds for both kinds (0.429 and 0.435, against 0.635 and 0.637 uniform,
    0.388 and 0.403 oracle). The expected weak point did not show: the changed output's error exceeded its variance
    for only about 6k steps, less than a search period.
11. So do the rules with recent progress: **held**, but barely for notebook 2's when an output changes (0.614 vs 0.635,
    4 of 5 seeds); 4b 0.450 and 0.466 (5 of 5).

Notebook 11 was planned with these, on the RMSE averaged over the run (needle: over the 50k steps after each change),
mean over seeds and in at least 4 of 5 seeds, paired:

12. Searching every 2k steps, half uniform plus remembered outcomes does better than both uniform parents and error
    times explained fraction on both noisy-linear problems: **failed**. Better than error times explained fraction
    (0.410 vs 0.488, 0.443 vs 0.520) but worse than uniform parents (0.363, 0.374).
13. On both, searching every 2k steps instead of every 10k improves it more than it improves half uniform plus error
    times explained fraction: the memory, not more search alone, is what helps: **failed**. With shared features it got
    0.060 worse (0.047 for the other); with features of their own 0.026 better (0.103).
14. With noise outputs and on the needle, it is within 0.02 of error times explained fraction, at both search periods:
    **held** with noise outputs (within 0.011), **failed** on the needle (0.499 vs 0.432, and 0.389 vs 0.361).

Notebook 12 was planned with these (drafted before a one-seed smoke run), on RMSE (mean over seeds, and in at least 4
of 5 seeds, paired): over the learnable outputs in the 100k steps before any change (settled), and the needle's over
the 100k after each change:

15. With noisy linear decoys, settled, half uniform plus remembered outcomes with the explained fraction standing in
    until tried does better than both uniform parents and error times explained fraction: **held**, 5 of 5 seeds
    against both (0.328 vs 0.477 and 0.511).
16. With noisy linear decoys, settled, half uniform plus remembered outcomes averaged with the explained fraction does
    not do better than uniform parents: **held** (0.505 vs 0.477, worse in 4 of 5 seeds).
17. After both kinds of change, with both kinds of decoy, the rule of 15 does better than uniform parents: **failed**.
    Held when a learnable output changes (0.434 vs 0.511, 0.454 vs 0.501; 4 of 5 seeds each), but when a decoy
    becomes a target, better in only 3 of 5 seeds (0.508 vs 0.545, 0.511 vs 0.577).
18. With noise decoys, the rule of 15 is within 0.02 of error times explained fraction, settled and after both kinds of
    change: **failed**. Settled, within 0.011; after a change, 0.434 vs 0.297 and 0.508 vs 0.300.

Checks of the progress values themselves (not in the notebooks): in notebook 2, with linear outputs, the first 10 hold
10% of the positive progress at 20k steps, and at 100k, 98% with λ = 1 but 20% with λ = 1/16. In notebook 3, with
linear outputs and λ = 1/16, one seed in three had no positive credited progress on the first 10 from 30k to 150k
steps: the first LTUs helped every output until the linear learner caught up, and the drop that followed counted as
negative progress for 16 periods.

## Also tried (pilots, not in the notebooks)

On all three kinds of other output with shared features, unless noted (noise / linear / noisy linear):
- Credited progress with no uniform half: 0.283 / 0.246 / 0.330; with a quarter uniform: 0.281 / 0.255 (noisy linear
  not run). Exploration by error instead of uniform: 0.365 / 0.240.
- Error times the fraction of the error the LTUs remove, $L R / (L + R)$ (notebook 9b), on an earlier version of
  the targets: 0.234 / 0.263 / 0.304; in 2 of 5 seeds with linear outputs, the first 10 held 3% of the LTUs for the
  first 80k steps.
- Help from only the LTUs added at the last search step: 0.347 / 0.252 / 0.444.
- Gain over a separate linear learner (its error minus the network's, on the same samples): error times the fraction
  of the linear learner's error removed, 0.232 / 0.256 / 0.242, but with features of their own it traps on the first
  output it solves (0.552 / 0.551 / 0.552): a solved output's error times its gain stays above the others' zero.
  Progress in that gain: 0.298 / 0.246 / 0.278 alone, 0.235 / 0.244 / 0.291 with half by error times explained
  fraction (features of their own: 0.209 / 0.241 / 0.535).
- Half by error times explained fraction, half by recent decrease in error (notebook 2's progress): 0.235 / 0.244 /
  0.395 (features of their own: 0.214 / 0.235 / 0.530).

## A learned middle ground (pilots, 2026-11-11)

Parents in proportion to error times $q_o$, an estimate of the fraction of the error a new LTU would remove. A trial
is a new LTU; its outcome is its usefulness divided by its parent's error when it was added. Replaced trials' outcomes
fade by the replaced share (1/8) per search step, so no new setting. `pilots/harness_trials.py` (run with
`pilots/nbtool.py exec`) has every variant; its targets differ from the notebooks', so uniform here is 0.336 / 0.302 /
0.329 with shared features (noise / linear / noisy linear), 0.471 / 0.433 / 0.449 with features of their own, and
0.635 / 0.637 on the needle (an output changes / a noise output becomes a target). Error times explained fraction:
0.229 / 0.238 / 0.456, 0.203 / 0.227 / 0.554, 0.429 / 0.435. Oracle on the needle: 0.388 / 0.403.

- **Per-output memory of trials**, $q_o$ = the mean outcome of the LTUs tried there, with one pseudo-trial at a prior:
  - the mean outcome of all trials: 0.404 / 0.238 / 0.417, needle 0.635 / 0.650. The 90 untried noise outputs each
    keep that prior; 4 trials per period are far too few to rule them out one by one.
  - the output's explained fraction: 0.230 / 0.240 / 0.382, 0.203 / 0.229 / 0.515, needle 0.447 / 0.454; on Asteroids
    79% (86 / 77 / 75). Blind, like error times explained fraction, to outputs with nothing explained.
- **A line fitted over all trials** from the parent's explained fraction when the LTU was added to its outcome,
  $q_o = \max(a + b\,e_o, 0)$, starting at $a = 0, b = 1$ (error times explained fraction) from two pseudo-trials at
  (0, 0) and (1, 1). With $b = 0$ it is parents by error, so the trials decide between the two rules. With outcomes
  measured on the LTU's parent only (its squared weight there, times the parent's curvature on Asteroids):
  0.229 / 0.243 / 0.314, 0.225 / 0.216 / 0.406, needle 0.585 / 0.610, never worse than uniform. On Asteroids (600
  LTUs, 3 seeds, 400k steps): 82% of the gap (85 / 85 / 77), loss 0.1042, 81.4% of bytes right, against parents by
  error's 85%, 0.1041, 81.5% and error times explained fraction's 81%, 0.1052, 81.0%. Measuring outcomes on all of
  an LTU's outputs instead credits a noise output's LTUs with help they give learnable outputs, and the fitted
  intercept stays near 0.07: 0.232 / 0.242 / 0.336, 0.225 / 0.216 / 0.414, needle 0.576 / 0.599. Adding per-output
  memory on top changes little (0.234 / 0.242 / 0.320, 0.226 / 0.216 / 0.400, needle 0.586 / 0.605; Asteroids 80%).
- Its weak point is the needle: one learnable output among 90 noise outputs with error 1 needs the line's value at
  0 explained to be almost exactly 0. The fit settles near 0.02 (slope 0.4–0.8), which still gives the noise outputs
  together about 10 times the needle's weight.

