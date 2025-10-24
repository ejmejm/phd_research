# spmm_benchmark_bcoo.py
import time
import argparse
import math
from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import PRNGKeyArray

import matplotlib.pyplot as plt


def make_random_bcoo(n: int, density: float, key: PRNGKeyArray, dtype=jnp.float32) -> Tuple[BCOO, int]:
    """
    Create a random NxN BCOO matrix with given density (fraction of non-zeros in [0,1]).
    Returns (A_bcoo, nnz).
    """
    assert 0.0 <= density <= 1.0
    total = n * n
    nnz = max(1, int(round(density * total)))
    
    # For large sparse matrices, use sampling with replacement and deduplicate
    # This avoids NumPy trying to allocate an array of size 'total'
    if nnz < total * 0.01:  # If less than 1% density, use efficient sparse sampling
        # Sample with replacement and deduplicate
        # Oversample slightly to account for duplicates
        oversample_factor = 1.2
        flat_idx_set = set()
        
        while len(flat_idx_set) < nnz:
            needed = nnz - len(flat_idx_set)
            sample_size = int(needed * oversample_factor)
            new_samples = np.random.randint(0, total, size=sample_size, dtype=np.int64)
            flat_idx_set.update(new_samples)
        
        flat_idx = np.array(list(flat_idx_set)[:nnz], dtype=np.int64)
    else:
        # For denser matrices, the original approach works fine
        flat_idx = np.random.choice(total, size=nnz, replace=False)
    
    rows = (flat_idx // n).astype(np.int32)
    cols = (flat_idx % n).astype(np.int32)
    indices = np.stack([rows, cols], axis=1)

    # Random nonzero values
    data = jax.random.normal(key, (nnz,), dtype=dtype)

    A = BCOO((data, jnp.asarray(indices)), shape=(n, n))
    return A, nnz


def benchmark_spmm(A: BCOO, X: jnp.ndarray, iters: int = 10) -> Tuple[float, float]:
    """
    JIT, warmup, and benchmark A @ X. Returns (avg_latency_s, gflops).
    """
    spmm = jax.jit(lambda a, x: a @ x)

    # Warmup
    _ = spmm(A, X).block_until_ready()

    # Timing
    start = time.perf_counter()
    for _ in range(iters):
        Y = spmm(A, X)
        Y.block_until_ready()
    end = time.perf_counter()

    avg_latency = (end - start) / iters  # seconds
    # FLOP model: 2 * nnz * batch (batch=1 here)
    nnz = int(A.nse)  # number of stored entries
    flops = 2.0 * nnz
    gflops = (flops / avg_latency) / 1e9
    return avg_latency, gflops


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX BCOO (COO) SpMM across size and sparsity.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 1536, 2048],
                        help="List of square matrix sizes N.")
    parser.add_argument("--densities", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                        help="List of non-zero densities (0..1).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                        help="Data type for computation.")
    parser.add_argument("--iters", type=int, default=10, help="Timing iterations per configuration.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--save", type=str, default="spmm_bcoo_heatmap.png", help="Path to save the heat map.")
    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")

    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64

    sizes: List[int] = args.sizes
    densities: List[float] = args.densities

    # Check if single size and density (single run mode)
    single_run_mode = len(sizes) == 1 and len(densities) == 1

    if single_run_mode:
        # Single run: just print results, no plots
        n = sizes[0]
        dens = densities[0]
        
        base_key = jax.random.PRNGKey(args.seed)
        keyA, keyX = jax.random.split(jax.random.fold_in(base_key, hash((n, dens))))

        # Build sparse A and dense X
        A, nnz = make_random_bcoo(n, dens, keyA, dtype=dtype)
        X = jax.random.normal(keyX, (n, 1), dtype=dtype)

        # Ensure arrays are on device before timing
        A = jax.device_put(A)
        X = jax.device_put(X)

        avg_latency_s, gflops = benchmark_spmm(A, X, iters=args.iters)
        lat_ms = avg_latency_s * 1e3

        print(f"\nSingle benchmark results:")
        print(f"  Matrix size (N):     {n}")
        print(f"  Density:             {dens:.7f}")
        print(f"  Non-zeros (nnz):     {nnz}")
        print(f"  Latency:             {lat_ms:.3f} ms")
        print(f"  Throughput:          {gflops:.3f} GFLOP/s")
    else:
        # Multi-run mode: sweep and create heatmaps
        # Results arrays: rows = densities, cols = sizes
        lat_ms = np.zeros((len(densities), len(sizes)), dtype=np.float64)
        tput_gflops = np.zeros((len(densities), len(sizes)), dtype=np.float64)

        # Fixed batch size = 1: X is (N, 1)
        base_key = jax.random.PRNGKey(args.seed)

        for j, dens in enumerate(densities):
            for i, n in enumerate(sizes):
                keyA, keyX = jax.random.split(jax.random.fold_in(base_key, hash((n, dens))))

                # Build sparse A and dense X
                A, nnz = make_random_bcoo(n, dens, keyA, dtype=dtype)
                X = jax.random.normal(keyX, (n, 1), dtype=dtype)

                # Ensure arrays are on device before timing
                A = jax.device_put(A)
                X = jax.device_put(X)

                avg_latency_s, gflops = benchmark_spmm(A, X, iters=args.iters)
                lat_ms[j, i] = avg_latency_s * 1e3
                tput_gflops[j, i] = gflops

                print(f"N={n:5d}, density={dens:8.7f}, nnz={nnz:9d} | "
                      f"lat={lat_ms[j,i]:8.3f} ms | {tput_gflops[j,i]:8.3f} GFLOP/s")

        # Plot a single 2D heat map for throughput
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(tput_gflops, aspect="auto", origin="lower")
        ax.set_title("BCOO SpMM Throughput (GFLOP/s)\nY = A @ X, batch=1")
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Density (non-zero fraction)")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=0)
        ax.set_yticks(range(len(densities)))
        ax.set_yticklabels([f"{d:.4f}" for d in densities])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("GFLOP/s")
        fig.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"Saved heat map to {args.save}")

        # Latency heatmap
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        im2 = ax2.imshow(lat_ms, aspect="auto", origin="lower")
        ax2.set_title("BCOO SpMM Latency (ms)\nY = A @ X, batch=1")
        ax2.set_xlabel("Matrix size N")
        ax2.set_ylabel("Density (non-zero fraction)")
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels([str(s) for s in sizes], rotation=0)
        ax2.set_yticks(range(len(densities)))
        ax2.set_yticklabels([f"{d:.4f}" for d in densities])
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.set_label("ms")
        fig2.tight_layout()
        plt.savefig("spmm_bcoo_latency_heatmap.png", dpi=150)

if __name__ == "__main__":
    main()
