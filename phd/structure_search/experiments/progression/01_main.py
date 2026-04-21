"""Step 1 — 20-seed main runs at the best LR per variant.

After ``01_lr_sweep.py`` completes, load the study and for each variant
run one individual MLflow run with the full 20-seed population at the
variant's best LR. These are standalone runs (not sweep trials) so their
parent in MLflow is the experiment, not a sweep parent run.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / '..' / '..' / '..' / '..'))

import math

import mlflow
import optuna

from common import (
    MLFLOW_PROJECT, build_step1_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.train import run_config


VARIANTS = ['dense', 'block_sparse', 'random_sparsity']
STUDY_NAME = f'{MLFLOW_PROJECT}/step1_lr_sweep'


def best_lr_per_variant(study: optuna.Study) -> dict:
    """Return ``{variant: best_lr}`` by min asymptotic_loss per variant."""
    best: dict = {}
    for t in study.trials:
        if t.state.name != 'COMPLETE' or t.value is None:
            continue
        v = t.params.get('variant')
        lr = t.params.get('lr')
        if v is None or lr is None:
            continue
        if v not in best or t.value < best[v][1]:
            best[v] = (lr, t.value)
    return {v: lr for v, (lr, _) in best.items()}


def main() -> None:
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri())
    mlflow.set_experiment(MLFLOW_PROJECT)

    study = optuna.load_study(
        study_name=STUDY_NAME, storage=resolve_optuna_tracking_uri())
    best = best_lr_per_variant(study)
    missing = [v for v in VARIANTS if v not in best]
    if missing:
        raise RuntimeError(
            f'No completed trials for variant(s): {missing}. Run '
            f'01_lr_sweep.py first.')

    print(f'Best LRs from {STUDY_NAME}:')
    for v in VARIANTS:
        print(f'  {v:16s}: lr=2^{math.log2(best[v]):+.1f} ({best[v]:.6f})')

    for variant in VARIANTS:
        lr = best[variant]
        run_name = f'step1_{variant}_main'
        print(f'\n=== {run_name} (lr={lr:.6f}, 20 seeds) ===')
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('phase', 'main')
            mlflow.set_tag('step', '1')
            mlflow.set_tag('variant', variant)
            cfg = build_step1_config(variant=variant, lr=lr)
            summary = run_config(cfg)
            print(
                f'[{run_name}] asymp_loss={summary["asymptotic_loss"]:.4f} '
                f'acc={summary["asymptotic_accuracy"]:.4f}')


if __name__ == '__main__':
    main()
