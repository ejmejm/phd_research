"""Step 2 — budget sweep on the linear, global-pruning setup.

Motivation: step 1 showed that with a large-ish budget (1500), utility
prefers active pixels over task-alignment, so dynamic doesn't separate
even though it matches the within-task oracle on loss. Intuition: if the
budget is smaller, the dynamic method has no "slack" to spend on
inactive or cross-task pixels and should be forced toward task-alignment.

This script keeps everything from step 1 constant (total_steps=225000,
spp=50, lr=0.16, 20 seeds) and sweeps budget. Smaller budgets keep the
same absolute prune rate (1 connection / 50 steps), so turnover goes
up sharply — at budget=20, 4500 events is 225x turnover.
"""

import os
import sys
import time

import jax

# Reuse step 1's scaffolding (build_run_fn, aggregate_results, log_run,
# load_mnist, etc). Import from the sibling module.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'step1', os.path.join(_HERE, '01_linear_global.py'))
step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step1)


BUDGETS = [1500, 500, 150, 50, 20]
LR = 0.16  # Best from step 1 sweep — shared across variants and budgets.

VARIANTS = ['dynamic', 'fixed_random', 'fixed_intask']


def main():
    print(f'Loading MNIST...')
    mnist_images, mnist_labels = step1.load_mnist()
    print(f'  {mnist_images.shape[0]} images')

    print(f'\n{"#" * 70}')
    print(f'# STEP 2 — budget sweep')
    print(f'{"#" * 70}')
    print(f'  budgets: {BUDGETS}')
    print(f'  variants: {VARIANTS}')
    print(f'  lr={LR}  n_cycles={step1.N_CYCLES_DEFAULT}  '
          f'spp={step1.SPP}  total_steps={step1.N_CYCLES_DEFAULT * step1.SPP}')
    print(f'  seeds={step1.N_SEEDS_FINAL}')

    table = []
    for budget in BUDGETS:
        print(f'\n--- budget={budget} '
              f'(turnover={step1.N_CYCLES_DEFAULT / budget:.1f}x) ---')
        for variant in VARIANTS:
            t0 = time.time()
            r = step1.run_variant(
                variant, LR, step1.N_SEEDS_FINAL, step1.N_CYCLES_DEFAULT,
                mnist_images, mnist_labels, budget=budget,
            )
            step1.summarize(r, f'{variant} budget={budget}')
            run_name = f'step2_{variant}_budget{budget}'
            step1.log_run(
                run_name, variant, LR, step1.N_SEEDS_FINAL,
                step1.N_CYCLES_DEFAULT, r, step=2, budget=budget,
            )
            table.append((budget, variant, r))
            print(f'    logged as {run_name}')

    # Print a final summary table
    print(f'\n{"=" * 80}')
    print(f'SUMMARY')
    print(f'{"=" * 80}')
    print(f'{"budget":>8}  {"variant":<15}  '
          f'{"loss":>18}  {"purity":>16}  {"entropy":>16}')
    print('-' * 80)
    for budget, variant, r in table:
        from common import ci95
        fl = r['final_losses']
        pu = r['purities']
        en = r['entropies']
        print(f'{budget:>8}  {variant:<15}  '
              f'{fl.mean():>8.4f}+/-{ci95(fl):.4f}  '
              f'{pu.mean():>6.3f}+/-{ci95(pu):.3f}  '
              f'{en.mean():>6.3f}+/-{ci95(en):.3f}')


if __name__ == '__main__':
    main()
