"""Step 3 — signed-utility variant on the same budget sweep as step 2.

Hypothesis: step 2 showed that budget pressure alone pushes task
alignment from 0.57 (budget=1500) to 0.83 (budget=150). Signed utility
should help further — or at least be competitive at large budgets —
because task-misaligned connections have contributions independent of
the per-logit error direction, so their signed utility fluctuates and
EMAs toward ~0 or negative while aligned connections accumulate
positive utility.

Runs dynamic + signed utility at every step-2 budget. The two fixed
baselines don't use utility, so we don't need to re-run them — reuse
the step 2 numbers for comparison.
"""

import os
import sys
import time

import jax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'step1', os.path.join(_HERE, '01_linear_global.py'))
step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step1)


BUDGETS = [1500, 500, 150, 50, 20]
LR = 0.16


def main():
    print('Loading MNIST...')
    mnist_images, mnist_labels = step1.load_mnist()
    print(f'  {mnist_images.shape[0]} images')

    print(f'\n{"#" * 70}')
    print(f'# STEP 3 — signed utility, budget sweep')
    print(f'{"#" * 70}')
    print(f'  budgets: {BUDGETS}')
    print(f'  utility: signed LOO, |e_k + c[k,i]| - |e_k| per logit')
    print(f'  lr={LR}  n_cycles={step1.N_CYCLES_DEFAULT}  '
          f'spp={step1.SPP}  total_steps={step1.N_CYCLES_DEFAULT * step1.SPP}')
    print(f'  seeds={step1.N_SEEDS_FINAL}')

    table = []
    for budget in BUDGETS:
        print(f'\n--- budget={budget} '
              f'(turnover={step1.N_CYCLES_DEFAULT / budget:.1f}x) ---')
        r = step1.run_variant(
            'dynamic', LR, step1.N_SEEDS_FINAL, step1.N_CYCLES_DEFAULT,
            mnist_images, mnist_labels, budget=budget, utility_fn='signed',
        )
        step1.summarize(r, f'dynamic_signed budget={budget}')
        run_name = f'step3_dynamic_signed_budget{budget}'
        step1.log_run(
            run_name, 'dynamic', LR, step1.N_SEEDS_FINAL,
            step1.N_CYCLES_DEFAULT, r, step=3, budget=budget,
            extra_params={'utility_fn': 'signed'},
        )
        table.append((budget, r))
        print(f'    logged as {run_name}')

    print(f'\n{"=" * 80}')
    print(f'SUMMARY (dynamic, signed utility, 20 seeds)')
    print(f'{"=" * 80}')
    print(f'{"budget":>8}  {"loss":>18}  '
          f'{"alignment":>16}  {"purity":>16}  {"entropy":>16}')
    print('-' * 80)
    from common import ci95
    for budget, r in table:
        fl = r['final_losses']
        al = r['alignments']
        pu = r['purities']
        en = r['entropies']
        print(f'{budget:>8}  '
              f'{fl.mean():>8.4f}+/-{ci95(fl):.4f}  '
              f'{al.mean():>6.3f}+/-{ci95(al):.3f}  '
              f'{pu.mean():>6.3f}+/-{ci95(pu):.3f}  '
              f'{en.mean():>6.3f}+/-{ci95(en):.3f}')


if __name__ == '__main__':
    main()
