"""Runs algorithms on a problem over seeds, and plots their errors and where they put new LTUs."""

from time import perf_counter

import jax
import matplotlib.pyplot as plt
import numpy as np


def run(methods, problem, steps, seeds=5):
    """Each method of {label: algorithm} on `seeds` seeds, vmapped. Seed s gives every method the same examples.
    Returns {label: the RMSE of each window, shaped (seed, window)} and {label: the records of Algorithm.run, each
    shaped (seed, ...)}."""
    keys = jax.random.split(jax.random.PRNGKey(1), seeds)
    rmse, records = {}, {}
    for label, method in methods.items():
        started = perf_counter()
        mse, method_records, network = jax.jit(jax.vmap(lambda key: method.run(problem, key, steps)))(keys)
        rmse[label] = np.sqrt(np.asarray(mse))
        records[label] = {name: np.asarray(values) for name, values in method_records.items()}
        print(f'{label}: RMSE over the last 10% {rmse[label][:, -len(mse[0]) // 10:].mean():.3f} '
              f'; at the end, {(network.depth > 0).sum(1).mean():.0f} LTUs and '
              f'{records[label]["connections"][:, -1].mean():.0f} connections in the budget '
              f'({perf_counter() - started:.0f}s)')
    return rmse, records


def plot(rmse, problem, title, window=1_000):
    """Mean RMSE over seeds against steps, with a dotted line at the lowest RMSE possible: the root mean square of the
    outputs' noise stds."""
    floor = np.sqrt(np.mean(np.square(problem.noise_std)))
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for (label, values), color in zip(rmse.items(), ['tab:red', 'tab:blue']):
        steps = (np.arange(values.shape[1]) + 1) * window
        ax.plot(steps / 1000, values.mean(0), color=color, lw=1, label=label)
    ax.axhline(floor, color='black', ls=':', label='noise: the lowest RMSE possible')
    ax.set(xlabel='step (thousands)', ylabel=f'RMSE (mean over {len(values)} seeds)', title=title)
    ax.set_ylim(bottom=0.95 * floor)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
    plt.show()


def plot_new_parents(records, problem, title, unit_period=10_000, bins=5, periods_per_point=5):
    """For each method, the share of the new LTUs with an output parent that go to each bin of outputs, by their rank
    in signal-to-noise ratio (signal / noise std: infinite without noise, 0 without signal), over all seeds and over
    `periods_per_point` unit periods."""
    signal, noise_std = np.broadcast_arrays(problem.signal, problem.noise_std)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(signal == 0, 0.0, signal / noise_std)
    order = np.argsort(ratio, kind='stable')
    groups = np.array_split(order, bins)
    colors = plt.cm.viridis(np.linspace(0, 0.9, bins))
    fig, axes = plt.subplots(1, len(records), figsize=(14, 4.5), sharey=True, layout='constrained')
    for ax, (label, method_records) in zip(np.atleast_1d(axes), records.items()):
        counts = method_records['new_parents'].sum(0)   # (unit period, output), over seeds
        counts = counts.reshape(-1, periods_per_point, counts.shape[1]).sum(1)
        total = np.maximum(counts.sum(1), 1)
        steps = (np.arange(len(counts)) + 1) * periods_per_point * unit_period
        for group, color in zip(groups, colors):
            low, high = ratio[group].min(), ratio[group].max()
            ax.plot(steps / 1000, counts[:, group].sum(1) / total, color=color, marker='.',
                    label=f'SNR {low:.2g} to {high:.2g}')
        ax.axhline(1 / bins, color='black', ls=':', lw=1, label='uniform')
        ax.set(title=label, xlabel='step (thousands)')
        ax.grid(alpha=0.4)
    np.atleast_1d(axes)[0].set_ylabel('share of new LTUs with an output parent')
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title)
    plt.show()


def plot_sizes(records, title, connection_period=1_000):
    """Mean over seeds of the number of LTUs and of the connections that count against the budget, against steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for (label, method_records), color in zip(records.items(), ['tab:red', 'tab:blue']):
        for ax, name in zip(axes, ['ltus', 'connections']):
            values = method_records[name]
            steps = (np.arange(values.shape[1]) + 1) * connection_period
            ax.plot(steps / 1000, values.mean(0), color=color, lw=1, label=label)
    for ax, ylabel in zip(axes, ['LTUs', 'connections in the budget']):
        ax.set(xlabel='step (thousands)', ylabel=f'{ylabel} (mean over {len(values)} seeds)')
        ax.grid(alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    plt.show()
