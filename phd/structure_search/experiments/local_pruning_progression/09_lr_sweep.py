"""Step 9 — LR sweep across 4 variants × 4 budgets × LR grid.

Goal: sanity-check whether softmax-CE LOO and BCE utilities can drive
useful pruning at all under step 3's paradigm (dynamic prune+replace at
fixed budget, 225k steps, SPP=50). Stage-1 (budgets 50 and 500) showed
that best LR shifts by ~4× between small and large budgets for the
working variants, so all 4 budgets are swept here.

Variants (utility / training loss):
  - signed       / softmax_ce   (step 3 baseline)
  - softmax_ce   / softmax_ce   (matched softmax LOO utility)
  - bce_softmax  = bce / softmax_ce
  - bce_bce      = bce / bce    (matched BCE pair)

LR grid extended to 2.56 because stage-1 `bce_bce` saturated at the top
of the original grid (0.64). Other variants blow up well before 2.56
but the wasted trials are cheap.

Accuracy is also logged (alongside training loss) so variants with
different training losses (softmax CE vs BCE) can be compared on a
loss-independent metric.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    log_result_metrics, load_mnist,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri, run_variant,
)

load_mnist()

N_SEEDS = 5

VARIANT_SPECS = {
    'signed':       ('signed',     'softmax_ce'),
    'softmax_ce':   ('softmax_ce', 'softmax_ce'),
    'bce_softmax':  ('bce',        'softmax_ce'),
    'bce_bce':      ('bce',        'bce'),
}

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step9_lr_sweep_full",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step9_lr_full",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "method": list(VARIANT_SPECS.keys()),
        "budget": [50, 150, 500, 1500],
        "lr": [1.5625e-4, 6.25e-4, 2.5e-3, 1e-2, 4e-2, 1.6e-1, 6.4e-1, 2.56],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "alignment",
                    "separation_f1"],
        "split_by": ["budget"],
        "sensitivity": {
            "params": ["lr"],
            "hue": ["method"],
        },
    },
}


def objective(**params):
    method = str(params["method"])
    budget = int(params["budget"])
    lr = float(params["lr"])
    utility_fn, loss_fn_name = VARIANT_SPECS[method]
    r = run_variant('dynamic', lr, n_seeds=N_SEEDS, budget=budget,
                    utility_fn=utility_fn, loss_fn_name=loss_fn_name)
    log_result_metrics(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
