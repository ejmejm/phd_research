"""Launches, or resumes, the search. Everything lives under results/<config>/.

    python run_evo.py                                       # smoke test: small problem, 3 generations, ~30 s
    python run_evo.py --config full --generations 300 --budget 900

One problem config per run: `small` only checks that the loop works, `full` is the real search.
Re-running with the same --config resumes from that run's database; --generations is the
total, not an increment. To watch a run live, in a second terminal:

    shinka_visualize --db results/<config>/programs.sqlite --port 8765   # then http://localhost:8765
"""

import argparse
from pathlib import Path

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

import problem


HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='small', choices=problem.CONFIGS)
parser.add_argument('--generations', type=int, default=3)
parser.add_argument('--budget', type=float, default=2.0, help='hard cap on API spend, in USD')
parser.add_argument('--models', nargs='+', default=['gpt-4.1-mini'], help='fast and cheap by default; pass reasoning models for a real run')
args = parser.parse_args()

cfg = problem.CONFIGS[args.config]
results_dir = HERE / 'results' / args.config
ideal = cfg['n_tasks'] * cfg['n_outputs_per_task'] * cfg['n_features_per_task']
prompt = (HERE / 'prompt.md').read_text().format(
    n_inputs=cfg['n_tasks'] * cfg['n_features_per_task'],
    n_outputs=cfg['n_tasks'] * cfg['n_outputs_per_task'],
    n_tasks=cfg['n_tasks'],
    n_features_per_task=cfg['n_features_per_task'],
    n_outputs_per_task=cfg['n_outputs_per_task'],
    ideal=ideal,
    param_budget=problem.param_budget(cfg),
    memory_budget=problem.MEMORY_FACTOR * problem.param_budget(cfg),
    n_steps=cfg['n_steps'],
    n_seeds=problem.N_SEEDS,
)

runner = ShinkaEvolveRunner(
    evo_config=EvolutionConfig(
        task_sys_msg=prompt,
        init_program_path=str(HERE / 'initial.py'),
        results_dir=str(results_dir),
        num_generations=args.generations,
        max_api_costs=args.budget,
        llm_models=args.models,
        llm_kwargs={'temperatures': [1.0], 'max_tokens': 8192, 'reasoning_efforts': ['low']},
        patch_types=['full', 'diff'],
        patch_type_probs=[0.5, 0.5],
        use_text_feedback=True,           # the evaluator's one-line summary or traceback
        enable_wandb_logging=True,
        wandb_project='multi-linear-shinka',
        wandb_dir=str(results_dir),
    ),
    job_config=LocalJobConfig(
        eval_program_path=str(HERE / 'evaluate.py'),
        extra_cmd_args={'config': args.config},
        time='00:10:00',                  # wall-clock cap per evaluation
    ),
    db_config=DatabaseConfig(),              # written to results/<config>/programs.sqlite
    max_evaluation_jobs=3,                # evaluations in flight at once, sharing the GPU
    max_proposal_jobs=3,                  # LLM calls in flight at once
    verbose=True,
)
runner.run()
