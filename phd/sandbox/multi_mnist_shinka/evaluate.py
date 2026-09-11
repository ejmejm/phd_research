"""Scores one candidate program. ShinkaEvolve runs it as

    python evaluate.py --program_path <candidate.py> --results_dir <dir> --config small|full

and reads the metrics.json and correct.json it writes to <dir>. Any failure -- a crash, too many
parameters or too much state, non-finite predictions -- is scored like the mean predictor, marked
incorrect, and hands the error back to the LLM as text feedback. Add --holdout to score on the
seeds the search never sees.
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')   # parallel evals share one GPU

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np

import problem


def load_candidate(program_path: str):
    source = Path(program_path).read_text()
    if 'callback' in source:   # the one way to smuggle data past the state cap
        raise ValueError('host callbacks are not allowed')
    spec = importlib.util.spec_from_file_location('candidate', program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str, cfg: dict, seeds: list) -> dict:
    candidate = load_candidate(program_path)
    result = problem.run(candidate.init, candidate.update, cfg, seeds)
    loss, n_params, n_memory = result['normalized_loss'], result['n_params'], result['n_memory']
    if not np.all(np.isfinite(loss)):
        raise ValueError('predictions are not finite')

    budget = problem.param_budget(cfg)
    memory_budget = problem.MEMORY_FACTOR * budget
    return {
        'combined_score': -float(loss.mean()),   # shinka maximizes, so minus the loss ratio
        'public': {
            'normalized_loss': float(loss.mean()),
            'normalized_loss_std': float(loss.std()),
            'n_params': n_params,
            'param_budget': budget,
            'n_memory': n_memory,
            'memory_budget': memory_budget,
        },
        'private': {'normalized_loss_per_seed': loss.tolist()},
        'text_feedback': f'final loss {loss.mean():.4f} times the optimal (seeds range '
                         f'{loss.min():.4f} to {loss.max():.4f}); {n_params} of {budget} '
                         f'parameters and {n_memory} of {memory_budget} state numbers used.',
    }


def failure(error: str) -> dict:
    """A broken candidate is scored like the mean predictor and marked incorrect."""
    return {'combined_score': -problem.MEAN_PREDICTOR_LOSS, 'public': {}, 'private': {},
            'text_feedback': error}


def main(program_path: str, results_dir: str, config: str, holdout: bool):
    cfg = problem.CONFIGS[config]
    seeds = problem.HOLDOUT_SEEDS if holdout else problem.EVAL_SEEDS
    try:
        metrics, correct, error = evaluate(program_path, cfg, seeds), True, ''
    except Exception as e:
        error = traceback.format_exc()
        metrics, correct = failure(''.join(traceback.format_exception_only(e))), False

    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (results / 'correct.json').write_text(json.dumps({'correct': correct, 'error': error}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--program_path', default='initial.py')
    parser.add_argument('--results_dir', default='results/manual_eval')
    parser.add_argument('--config', default='small', choices=problem.CONFIGS)
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    main(args.program_path, args.results_dir, args.config, args.holdout)
