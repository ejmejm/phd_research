"""True value function computation and MSVE evaluation.

Computes V* by discretizing the continuous grid world onto a fine grid,
building the empirical transition matrix via Monte Carlo sampling, and
solving the linear Bellman equation (I - gamma * P) V = r.

Results are cached to disk keyed by environment parameters.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional

from phd.sandbox.predictive_coding.environment import GridWorld


CACHE_DIR = Path(__file__).parent / '.v_star_cache'


def _env_cache_key(grid_world: GridWorld, resolution: int, n_samples: int, seed: int) -> str:
    """Deterministic hash of environment configuration for cache lookup."""
    config = {
        'barriers': grid_world.barriers,
        'reward_regions': [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
            for r in grid_world.reward_regions
        ],
        'sigma': grid_world.sigma,
        'gamma_td': grid_world.gamma_td,
        'bounds': list(grid_world.bounds),
        'resolution': resolution,
        'n_samples': n_samples,
        'seed': seed,
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_true_value_function(
    grid_world: GridWorld,
    resolution: int = 100,
    n_samples: int = 500,
    seed: int = 0,
    use_cache: bool = True,
) -> Tuple[np.ndarray, RegularGridInterpolator, np.ndarray, np.ndarray]:
    """Compute the true value function V* on a discretized grid.

    Uses vectorized batch stepping for speed and caches results to disk.

    Args:
        grid_world: The environment.
        resolution: Grid points per dimension.
        n_samples: Transition samples per cell for building P.
        seed: Random seed for transition sampling.
        use_cache: Whether to load from / save to disk cache.

    Returns:
        v_star_grid: (resolution, resolution) array of V* values (NaN inside barriers).
        v_star_interp: Interpolation function for off-grid queries.
        xs: (resolution,) x-coordinates of grid.
        ys: (resolution,) y-coordinates of grid.
    """
    cache_key = _env_cache_key(grid_world, resolution, n_samples, seed)
    cache_path = CACHE_DIR / f'v_star_{cache_key}.npz'

    x_min, y_min, x_max, y_max = grid_world.bounds
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)

    # Try loading from cache
    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        v_star_grid = data['v_star_grid']
        v_star_filled = np.where(np.isnan(v_star_grid), 0.0, v_star_grid)
        v_star_interp = RegularGridInterpolator(
            (xs, ys), v_star_filled, method='linear', bounds_error=False, fill_value=0.0,
        )
        return v_star_grid, v_star_interp, xs, ys

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # Build cell index mapping
    valid_mask = np.ones((resolution, resolution), dtype=bool)
    cell_to_idx = {}
    idx_to_cell = []
    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([xs[i], ys[j]])
            if grid_world.is_inside_barrier(pos):
                valid_mask[i, j] = False
            else:
                cell_to_idx[(i, j)] = idx
                idx_to_cell.append((i, j))
                idx += 1
    n_valid = idx

    # Build reward vector
    reward_vec = np.zeros(n_valid)
    for k, (i, j) in enumerate(idx_to_cell):
        reward_vec[k] = grid_world.reward(np.array([xs[i], ys[j]]))

    # Build transition matrix P via vectorized Monte Carlo
    rng = np.random.default_rng(seed)
    rows = []
    cols = []
    vals = []

    for k, (i, j) in enumerate(idx_to_cell):
        center = np.array([xs[i], ys[j]])
        # Batch sample all transitions at once
        destinations = grid_world.batch_step(center, n_samples, rng)  # (n_samples, 2)

        # Map destinations to grid cells
        di = np.clip(np.round((destinations[:, 0] - x_min) / dx).astype(int), 0, resolution - 1)
        dj = np.clip(np.round((destinations[:, 1] - y_min) / dy).astype(int), 0, resolution - 1)

        # Count transitions to each destination cell
        dest_flat = di * resolution + dj
        unique, counts = np.unique(dest_flat, return_counts=True)

        for flat_idx, count in zip(unique, counts):
            ci, cj = divmod(int(flat_idx), resolution)
            if (ci, cj) in cell_to_idx:
                rows.append(k)
                cols.append(cell_to_idx[(ci, cj)])
                vals.append(count / n_samples)

    P = sparse.csr_matrix((vals, (rows, cols)), shape=(n_valid, n_valid))

    # Solve (I - gamma * P) V = r
    I_mat = sparse.eye(n_valid, format='csr')
    A = I_mat - grid_world.gamma_td * P
    v_star_flat = spsolve(A, reward_vec)

    # Map back to grid
    v_star_grid = np.full((resolution, resolution), np.nan)
    for k, (i, j) in enumerate(idx_to_cell):
        v_star_grid[i, j] = v_star_flat[k]

    # Cache to disk
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, v_star_grid=v_star_grid)

    # Build interpolator
    v_star_filled = np.where(np.isnan(v_star_grid), 0.0, v_star_grid)
    v_star_interp = RegularGridInterpolator(
        (xs, ys), v_star_filled, method='linear', bounds_error=False, fill_value=0.0,
    )

    return v_star_grid, v_star_interp, xs, ys


def evaluate_msve_numpy(
    predictions: np.ndarray,
    grid_points: np.ndarray,
    v_star_interp: RegularGridInterpolator,
) -> float:
    """Compute MSVE given predictions and true values.

    Args:
        predictions: (N,) array of V-hat at each grid point.
        grid_points: (N, 2) array of evaluation coordinates.
        v_star_interp: Interpolation function for V*.

    Returns:
        Mean squared value error.
    """
    true_values = v_star_interp(grid_points)
    return float(np.mean((predictions - true_values) ** 2))
