# benchmark_conditionals.py
import time
from dataclasses import dataclass
from typing import Dict, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt


# -------------------------------
# Baseline config (can edit)
# -------------------------------
BASE_T = 1_000        # baseline scan length
BASE_N = 100          # baseline vector size
BASE_EVERY_N = 10     # baseline frequency for expensive op
RUNS = 10              # number of timed runs to average (set to 1 for single run)

# Sweep sets
T_VALUES = (10, 100, 1_000, 10_000) # , 100_000)
N_VALUES = (2, 8, 32, 128, 512, 2_048)
EVERY_N_VALUES = (1, 4, 16, 64, 256)

# -------------------------------
# Workload builders
# -------------------------------
def make_ops(N: int, key: jax.Array):
    """Returns (cheap_op, expensive_op) specialized to size N."""
    # Cheap op: identity
    def cheap_op(x):
        return x

    # Make a fixed weight for the expensive op (same across variants at this N)
    A = jax.random.normal(key, (N, N))

    def expensive_op(x):
        # A few chained dense+nonlinear steps to be meaningfully heavier
        y = jnp.sin(A @ x)
        y = jnp.tanh(A @ y + 0.1 * y)
        y = jnp.tanh(A @ y + 0.1 * y)
        y = jnp.tanh(A @ y + 0.1 * y)
        return y

    return cheap_op, expensive_op

def do_expensive(i: jax.Array, every_n: int):
    return (jnp.mod(i, every_n) == 0)

def build_scanners(N: int, key: jax.Array, unroll: int = 1) -> Dict[str, Callable[[jnp.ndarray, int, int], jnp.ndarray]]:
    """Builds six jitted scan runners, all sharing identical cheap/expensive ops for this N."""
    cheap_op, expensive_op = make_ops(N, key)

    def step_cond(x, i, every_n):
        pred = do_expensive(i, every_n)
        return lax.cond(pred, expensive_op, cheap_op, x)

    def step_switch(x, i, every_n):
        idx = jnp.int32(do_expensive(i, every_n))
        return lax.switch(idx, (cheap_op, expensive_op), x)

    def step_select(x, i, every_n):
        pred = do_expensive(i, every_n)
        # NOTE: both branches are computed
        return lax.select(pred, expensive_op(x), cheap_op(x))

    def step_where(x, i, every_n):
        pred = do_expensive(i, every_n)
        # NOTE: both branches are computed
        return jnp.where(pred, expensive_op(x), cheap_op(x))

    def step_always_expensive(x, i, _every_n):
        return expensive_op(x)

    def step_always_cheap(x, i, _every_n):
        return cheap_op(x)

    def make_scanner(step_fn):
        def run(x0: jnp.ndarray, every_n: int, steps: int):
            def body(carry, i):
                carry = step_fn(carry, i, every_n)
                return carry, None
            carry, _ = lax.scan(body, x0, jnp.arange(steps), unroll=unroll)
            return carry
        return jax.jit(run, static_argnums=(2,))

    return {
        "lax.cond"          : make_scanner(step_cond),
        "lax.switch"        : make_scanner(step_switch),
        "lax.select"        : make_scanner(step_select),
        "jnp.where"         : make_scanner(step_where),
        "baseline_exp_all"  : make_scanner(step_always_expensive),
        "baseline_cheap_all": make_scanner(step_always_cheap),
    }

# -------------------------------
# Benchmark helpers
# -------------------------------
@dataclass
class Stat:
    name: str
    ms_avg: float
    ms_runs: Tuple[float, ...]

def time_once(fn, x, every_n, steps) -> float:
    t0 = time.perf_counter()
    y = fn(x, every_n, steps)
    y.block_until_ready()
    return (time.perf_counter() - t0) * 1000.0

def benchmark_avg(fn, x, every_n, steps, runs: int) -> Tuple[float, Tuple[float, ...]]:
    # Warmup (compile + first run)
    _ = fn(x, every_n, steps).block_until_ready()
    # Timed runs
    times = []
    for _ in range(runs):
        times.append(time_once(fn, x, every_n, steps))
    return float(sum(times) / len(times)), tuple(times)

def run_suite_for_config(
    N: int,
    steps: int,
    every_n: int,
    runs: int,
    unroll: int = 1,
    suite_names: Sequence[str] = ("lax.cond","lax.switch","lax.select","jnp.where","baseline_exp_all","baseline_cheap_all"),
) -> Dict[str, Stat]:
    key = jax.random.PRNGKey(0)
    key_A, key_x = jax.random.split(key)
    scanners = build_scanners(N, key_A, unroll=unroll)
    x0 = jax.random.normal(key_x, (N,), dtype=jnp.float32)

    stats: Dict[str, Stat] = {}
    for name in suite_names:
        ms_avg, ms_runs = benchmark_avg(scanners[name], x0, every_n, steps, runs)
        stats[name] = Stat(name, ms_avg, ms_runs)
    return stats

# -------------------------------
# Plotting (one plot per sweep)
# -------------------------------
def plot_sweep(
    x_labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    save_path: str,
):
    # Evenly spaced categorical x-axis
    xs = list(range(len(x_labels)))
    plt.figure(figsize=(8, 5))
    for name, ys in series.items():
        # Take mean of inner quartile if enough samples
        processed_ys = []
        for y in ys:
            if isinstance(y, (list, tuple)) and len(y) >= 8:
                sorted_y = sorted(y)
                q1_idx = len(y) // 4
                q3_idx = 3 * len(y) // 4
                inner_quartile = sorted_y[q1_idx:q3_idx+1]
                processed_ys.append(sum(inner_quartile) / len(inner_quartile))
            else:
                processed_ys.append(y)
                
        plt.plot(xs, processed_ys, marker="o", label=name)  # no explicit colors/styles
    plt.xticks(xs, x_labels, rotation=0)
    plt.xlabel("Config")
    plt.ylabel(ylabel)
    plt.title(title)
    if "sweep_T" in save_path:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# -------------------------------
# Main: run sweeps + plots
# -------------------------------
def run_benchmarks_for_unroll(unroll: int):
    """Run all benchmark sweeps for a given unroll value."""
    print(f"\n{'='*60}")
    print(f"Running benchmarks with unroll={unroll}")
    print(f"Baseline: T={BASE_T}, N={BASE_N}, EVERY_N={BASE_EVERY_N}, RUNS={RUNS}")
    print(f"{'='*60}")

    # --- Sweep T ---
    print(f"\nSweep T (vary steps) - unroll={unroll}:")
    t_series: Dict[str, list] = {name: [] for name in ("lax.cond","lax.switch","lax.select","jnp.where","baseline_exp_all","baseline_cheap_all")}
    for T in T_VALUES:
        stats = run_suite_for_config(N=BASE_N, steps=T, every_n=BASE_EVERY_N, runs=RUNS, unroll=unroll)
        print(f"  T={T}: " + ", ".join(f"{k}={v.ms_avg:.2f}ms" for k,v in stats.items()))
        for name in t_series.keys():
            t_series[name].append(stats[name].ms_avg)
    plot_sweep(
        x_labels=[str(v) for v in T_VALUES],
        series=t_series,
        title=f"Runtime vs T (N={BASE_N}, EVERY_N={BASE_EVERY_N}, runs={RUNS}, unroll={unroll})",
        ylabel="Time (ms)",
        save_path=f"sweep_T_unroll{unroll}.png",
    )
    print(f"Saved plot: sweep_T_unroll{unroll}.png")

    # --- Sweep N ---
    print(f"\nSweep N (vary vector size) - unroll={unroll}:")
    n_series: Dict[str, list] = {name: [] for name in ("lax.cond","lax.switch","lax.select","jnp.where","baseline_exp_all","baseline_cheap_all")}
    for N in N_VALUES:
        stats = run_suite_for_config(N=N, steps=BASE_T, every_n=BASE_EVERY_N, runs=RUNS, unroll=unroll)
        print(f"  N={N}: " + ", ".join(f"{k}={v.ms_avg:.2f}ms" for k,v in stats.items()))
        for name in n_series.keys():
            n_series[name].append(stats[name].ms_avg)
    plot_sweep(
        x_labels=[str(v) for v in N_VALUES],
        series=n_series,
        title=f"Runtime vs N (T={BASE_T}, EVERY_N={BASE_EVERY_N}, runs={RUNS}, unroll={unroll})",
        ylabel="Time (ms)",
        save_path=f"sweep_N_unroll{unroll}.png",
    )
    print(f"Saved plot: sweep_N_unroll{unroll}.png")

    # --- Sweep EVERY_N ---
    print(f"\nSweep EVERY_N (vary expensive frequency) - unroll={unroll}:")
    e_series: Dict[str, list] = {name: [] for name in ("lax.cond","lax.switch","lax.select","jnp.where","baseline_exp_all","baseline_cheap_all")}
    for EN in EVERY_N_VALUES:
        stats = run_suite_for_config(N=BASE_N, steps=BASE_T, every_n=EN, runs=RUNS, unroll=unroll)
        print(f"  EVERY_N={EN}: " + ", ".join(f"{k}={v.ms_avg:.2f}ms" for k,v in stats.items()))
        for name in e_series.keys():
            e_series[name].append(stats[name].ms_avg)
    plot_sweep(
        x_labels=[str(v) for v in EVERY_N_VALUES],
        series=e_series,
        title=f"Runtime vs EVERY_N (T={BASE_T}, N={BASE_N}, runs={RUNS}, unroll={unroll})",
        ylabel="Time (ms)",
        save_path=f"sweep_every_n_unroll{unroll}.png",
    )
    print(f"Saved plot: sweep_every_n_unroll{unroll}.png")

def main():
    """Run benchmarks for both unroll=1 and unroll=8."""
    print("JAX Conditional Benchmark - Unroll Comparison")
    print("=" * 60)
    
    # Run benchmarks for unroll=1
    run_benchmarks_for_unroll(unroll=1)
    
    # Run benchmarks for unroll=8
    run_benchmarks_for_unroll(unroll=8)
    
    print(f"\n{'='*60}")
    print("All benchmarks completed!")
    print("Generated plots:")
    for unroll in [1, 16]:
        print(f"  - sweep_T_unroll{unroll}.png")
        print(f"  - sweep_N_unroll{unroll}.png") 
        print(f"  - sweep_every_n_unroll{unroll}.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
