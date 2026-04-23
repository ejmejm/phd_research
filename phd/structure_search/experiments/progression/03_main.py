"""Step 3 — Mixed Feature Search main run at the best LR from 03_lr_sweep.

Single 20-seed individual MLflow run, not a sweep trial.
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
    MLFLOW_PROJECT, build_step_2_4_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.experiments.column_guided_search import run_config


STUDY_NAME = f'{MLFLOW_PROJECT}/step3_lr_sweep'
RUN_NAME = 'step3_mixed_feature_main'


def best_lr(study: optuna.Study) -> float:
    completed = [t for t in study.trials
                 if t.state.name == 'COMPLETE' and t.value is not None]
    if not completed:
        raise RuntimeError(
            f'No completed trials in {STUDY_NAME}. Run 03_lr_sweep.py first.')
    return min(completed, key=lambda t: t.value).params['lr']


def main() -> None:
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri())
    mlflow.set_experiment(MLFLOW_PROJECT)

    study = optuna.load_study(
        study_name=STUDY_NAME, storage=resolve_optuna_tracking_uri())
    lr = best_lr(study)
    print(f'Best LR from {STUDY_NAME}: '
          f'lr=2^{math.log2(lr):+.1f} ({lr:.6f})')

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tag('phase', 'main')
        mlflow.set_tag('step', '3')
        cfg = build_step_2_4_config(
            variant='mixed_generation', tracker_mode='unit',
            lr=lr, random_input_count=True,
        )
        summary = run_config(cfg)
        print(f'[{RUN_NAME}] asymp_loss={summary["asymptotic_loss"]:.4f} '
              f'acc={summary["asymptotic_accuracy"]:.4f}')


if __name__ == '__main__':
    main()
