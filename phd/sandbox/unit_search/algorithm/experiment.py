"""Runs algorithms on a problem over seeds, and plots their errors."""

from time import perf_counter

import jax
import matplotlib.pyplot as plt
import numpy as np


def run(methods, problem, steps, seeds=5):
    """Each method of {label: algorithm} on `seeds` seeds, vmapped. Seed s gives every method the same examples.
    Returns {label: the RMSE of each window, shaped (seed, window)}."""
    keys = jax.random.split(jax.random.PRNGKey(1), seeds)
    rmse = {}
    for label, method in methods.items():
        started = perf_counter()
        mse = jax.jit(jax.vmap(lambda key: method.run(problem, key, steps)))(keys)
        rmse[label] = np.sqrt(np.asarray(mse))
        print(f'{label}: RMSE over the last 10% {rmse[label][:, -len(mse[0]) // 10:].mean():.3f} '
              f'({perf_counter() - started:.0f}s)')
    return rmse


def plot(rmse, problem, title, window=1_000):
    """Mean RMSE over seeds against steps, with a dotted line at the noise std."""
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for (label, values), color in zip(rmse.items(), ['tab:red', 'tab:blue']):
        steps = (np.arange(values.shape[1]) + 1) * window
        ax.plot(steps / 1000, values.mean(0), color=color, lw=1, label=label)
    ax.axhline(problem.noise_std, color='black', ls=':', label='noise std: the lowest RMSE possible')
    ax.set(xlabel='step (thousands)', ylabel=f'RMSE (mean over {len(values)} seeds)', title=title)
    ax.set_ylim(bottom=0.95 * problem.noise_std)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
    plt.show()
