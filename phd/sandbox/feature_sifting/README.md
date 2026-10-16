# Feature sifting

A linear learner holds a fixed number of candidate feature slots and must keep the useful ones,
pruning slots so the task (`phd/jax_core/tasks/feature_sifting.py`) redraws them.

## clean_or_noise/
Each candidate is either an exact copy of a signal or pure noise (`phd/jax_core/tasks/clean_or_noise_sifting.py`).

- `clean_or_noise.ipynb` — Autostep + G&T vs. + L1 + G&T vs. the band test, and how each handles signals with small ideal weights.

## corrupted_signal/
Each candidate is `(1 - c)·signal + c·noise` with `c ~ U[0, 1]`; the target is a ±1-weighted sum of the signals.

- `corrupted_signal.ipynb` — Autostep vs. Autostep + CI vs. Autostep + utility threshold (competitive pruning).
- `optimal_scaling.ipynb` — the best MSE any method can reach as a function of how many candidates it draws.
- `pruning_rules.ipynb` — which pruning rule finds the good features: G&T, w zero-crossings, w-or-h zero-crossings.
- `stepsize_protection.ipynb` — Autostep pruning rules protected by step-size jitter bands, plus an L1 variant.
- `adaptive_gt.ipynb` — generate-and-test whose prune period grows when it keeps pruning the same slot, vs. a fixed period.

## sign_flips/
The corrupted-signal problem where one target weight flips sign every `FLIP_PERIOD` steps.

- `sign_flips.ipynb` — the corrupted-signal methods on this nonstationary version.

## analysis/
One- or two-feature toy problems (except `l1_jitter_bands`).

- `step_size_analysis.ipynb` — weight, IDBD `h` and step-size dynamics for a single feature and for blocking by an
  established feature. `step_size_analysis/` holds the same experiments as scripts, plus `verified_flip` and the
  blocking EMA-indicator notebook.
- `weight_h_dynamics.ipynb` — weight and `h` of one weight whose target flips sign, with a Gaussian-closure model.
- `ci_estimator_suite.ipynb` — test harness for online confidence intervals on a feature's utility.
- `l1_jitter_bands.ipynb` — noise-floor bands for a dead feature's weight and `h` under LMS, L1 and Autostep.

## old/
June work that the current methods no longer build on.

- `algorithm_testing/`, `algorithm_comparison/` — Hydra harness and analysis for LMS vs. CBP + Autostep sweeps.
- `baseline_comparison.ipynb` — LMS, CBP + Adam, CBP + Autostep, Fable and hybrid on the original task.
- `utility_pruning.ipynb` — first vmapped utility-based pruning and adaptive pruning rate.
- `confidence_pruning.ipynb` — first confidence-based pruning tests.
- `cbp_lms_demo.py`, `algorithm.md` — numpy CBP + LMS demo; reservoir-index pruning spec.
