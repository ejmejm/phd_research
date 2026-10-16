# Global preferences

- Be concise. Lead with the answer. Leave out details that don't change a decision.
- When answering a question or discussing research and experiments, draft the answer, then reread it and cut anything that isn't important. Each point should lead naturally into the next.
- Build exactly what I asked, in the least code that does it. No extra plots, tables, markdown, config knobs, or variants. If something seems worth adding, say so in one line and let me opt in.
- When I ask "why", diagnose and explain, then stop. Don't change code or start tuning. I decide the path forward.
- Before a non-trivial build, ask about anything unclear in plain prose and wait. Otherwise just do it.
- I edit notebooks in VS Code while you work. Re-read a notebook from disk right before every edit or run. Edit by exact-match replacement and never regenerate a notebook that has been run.
- Keep smoke tests short.

# Experimental guidelines

- Use 5 seeds per configuration in parameter sweeps and 30 seeds for single-configuration experiments. Vectorize seeds with `jax.vmap`.
- Run all experiments on GPU; do not fall back to CPU.
- Sweeps measure a method's overall behavior and parameter sensitivity, rather than select a best configuration. Plot every swept configuration's learning curve together; do not follow a sweep with a separate run of its winner.
- Choose hyperparameters based on what should work or on existing results. Do not run a new sweep just to select hyperparameters.
- Plot RMSE for regression experiments, and clearly label whether each method is a baseline or a proposed method.
- Use a grid on plots. Only plot confidence intervals or uncertainty bands for experiments with 30 seeds.
- Aim for one minute per experiment, with two minutes as the maximum, including all configurations in a sweep. Simplify experiments that exceed this limit.
