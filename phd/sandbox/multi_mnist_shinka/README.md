# LLM-driven algorithm search on multi-linear GEOFF

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) mutates the code between the
`EVOLVE-BLOCK` markers of `initial.py`, scores every candidate with `evaluate.py`, and keeps an
archive of what worked. Everything runs locally on one GPU.

## Files

| File | What it is | Who touches it |
|---|---|---|
| `problem.py` | The benchmark: problem sizes, parameter budget, the seed-parallel scoring loop | you |
| `initial.py` | Seed program: LMS on a fixed random wiring, inside the EVOLVE-BLOCK | the LLM |
| `evaluate.py` | Scores one candidate, writes `metrics.json` and `correct.json` | you |
| `prompt.md` | The problem statement handed to the LLM as the system message | you |
| `run_evo.py` | Launches or resumes a search; every knob is in this file | you |

## The task the LLM is given

Streaming regression, one sample at a time, on multi-linear GEOFF with a random permutation
hiding the slot layout. The candidate implements

```python
init(n_inputs, n_outputs, param_budget, key) -> state
update(state, x, y) -> (state, prediction)
```

and is scored by rho = 1 - (MSE - 1), averaged over every step and over 10 seeds. The one
hard constraint is the parameter budget: the total number of elements in `state`, any dtype,
must not exceed 90% of the ideal solution's weight count. That single cap also enforces the
streaming rule, since there is no room to keep samples. Over-budget, crashing, wrong-shaped
or non-finite candidates score 0 and are marked incorrect, and the exception message goes
back to the LLM as text feedback.

Two problem sizes live in `problem.py`:

| Config | Slots | Inputs | Outputs | Ideal params | Budget | Steps | Baseline rho | Eval wall time |
|---|---|---|---|---|---|---|---|---|
| `small` | 4 x (10 in, 5 out) | 40 | 20 | 200 | 180 | 10k | 0.09 | 7 s |
| `full` | 20 x (20 in, 10 out) | 400 | 200 | 4000 | 3600 | 100k | -0.01 | 8 s |

Seeds 0-9 are what the search sees; seeds 1000-1009 are held out for your own checks.

## Launch

```bash
cd phd/sandbox/multi_mnist_shinka
python run_evo.py                                          # smoke test: small, 10 generations, $2 cap
python run_evo.py --config full --generations 300 --budget 900
python run_evo.py --config full --models gpt-5-mini claude-sonnet-4-6   # add a frontier model
```

Needs `OPENAI_API_KEY` (mutations with `gpt-5-mini`, code embeddings for novelty rejection)
and a wandb login. Adding an Anthropic model needs `ANTHROPIC_API_KEY`. `shinka_models`
lists what the current keys unlock. The cost cap is a hard stop: the runner stops proposing
once it is reached and lets in-flight evaluations finish.

## Resume

Re-run the same command. Each config keeps its database at `results/<config>/programs.sqlite`;
if it exists the runner resumes from it, restores the spent budget, and treats `--generations`
as the total. To start over, delete `results/<config>/`.

## Reading the results

- **WebUI**: `shinka_visualize --db results/<config>/programs.sqlite --port 8765`, then open
  http://localhost:8765 in the Windows browser. It works while a run is going (the tree view has
  an auto-refresh toggle) and shows the genealogy tree, scores over generations, and every
  program's code, metrics and LLM feedback. The default port 8000 is taken on this machine by
  Windows' own wslrelay.exe, which answers 404 instead of forwarding, so pass a port.
- **wandb**: project `multi-linear-shinka`, one run per launch, with per-program scores and
  the best-so-far curve.
- **On disk**: `results/<config>/gen_<n>/` holds each candidate's `main.py`, `metrics.json`,
  `correct.json` and the evaluator's stdout/stderr logs. `results/<config>/best/` tracks the
  best program so far.
- **Post-hoc check on unseen seeds**:
  `python evaluate.py --program_path results/full/best/main.py --config full --holdout`.

## Scoring a candidate by hand

```bash
python evaluate.py --program_path initial.py --config small --results_dir /tmp/eval
```

## Smoke test (2026-09-09)

`python run_evo.py` on the small problem with `gpt-5-mini`: 10 generations in about 5 minutes
for $0.08. Eleven programs evaluated, eight valid, three rejected (an over-budget state, a
JAX dtype error in the scan carry, a bad attribute) with the exception message fed back.
Best score 0.0946 against the seed's 0.0906, a low-rank factorization that does not hold up
on the holdout seeds (0.082 against the seed's 0.085), which is exactly what the holdout
check is for.

## Things deliberately left simple

One mutation model, no crossover, no meta-recommendations, no prompt evolution, two islands
(the shinka default), two evaluations and two LLM calls in flight at once. Every one of these
is a single line in `run_evo.py`. The problem is stationary; set `perturb_period` in
`problem.make_problem` to make it drift.
