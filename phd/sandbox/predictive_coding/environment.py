"""Continuous 2D grid world with barriers and reward regions.

The agent performs a random walk in [0,1]^2 with Gaussian steps, reflected at
rectangular barriers and domain boundaries. Reward regions provide r=+1 when
the agent is inside them, r=0 otherwise.

All operations are pure NumPy (CPU-side). Trajectories are pre-sampled and
shipped to JAX for training.
"""

import numpy as np
from typing import List, Tuple, Optional


# Default layout: two offset vertical barriers with a gap between them
DEFAULT_BARRIERS = [
    # (x, y, width, height) — axis-aligned rectangles
    (0.45, 0.40, 0.04, 0.55),   # Upper vertical wall
    (0.50, 0.05, 0.04, 0.45),   # Lower vertical wall (offset, gap between)
]

DEFAULT_REWARDS = [
    # {'center': (cx, cy), 'radius': r, 'value': v}
    {'center': (0.75, 0.75), 'radius': 0.08, 'value': 1.0},  # R1: upper-right
    {'center': (0.20, 0.25), 'radius': 0.08, 'value': 1.0},  # R2: lower-left
]


class GridWorld:
    """Continuous 2D grid world with barriers and circular reward regions."""

    def __init__(
        self,
        barriers: Optional[List[Tuple[float, float, float, float]]] = None,
        reward_regions: Optional[List[dict]] = None,
        sigma: float = 0.03,
        gamma_td: float = 0.95,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ):
        """
        Args:
            barriers: List of (x, y, width, height) rectangles.
            reward_regions: List of dicts with 'center', 'radius', 'value'.
            sigma: Std dev of Gaussian step noise.
            gamma_td: TD discount factor (stored for convenience).
            bounds: (x_min, y_min, x_max, y_max) domain boundaries.
        """
        self.barriers = barriers if barriers is not None else list(DEFAULT_BARRIERS)
        self.reward_regions = reward_regions if reward_regions is not None else list(DEFAULT_REWARDS)
        self.sigma = sigma
        self.gamma_td = gamma_td
        self.bounds = bounds

    def reward(self, position: np.ndarray) -> float:
        """Compute reward at a given position."""
        for region in self.reward_regions:
            cx, cy = region['center']
            r = region['radius']
            if (position[0] - cx) ** 2 + (position[1] - cy) ** 2 <= r ** 2:
                return region['value']
        return 0.0

    def is_inside_barrier(self, position: np.ndarray) -> bool:
        """Check if a position is inside any barrier."""
        x, y = position
        for bx, by, bw, bh in self.barriers:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return True
        return False

    def _clip_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """Reflect position back into domain bounds."""
        x_min, y_min, x_max, y_max = self.bounds
        x, y = position

        # Reflect off boundaries
        if x < x_min:
            x = x_min + (x_min - x)
        elif x > x_max:
            x = x_max - (x - x_max)

        if y < y_min:
            y = y_min + (y_min - y)
        elif y > y_max:
            y = y_max - (y - y_max)

        # Clamp in case of large overshoot
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        return np.array([x, y])

    def _segment_intersects_rect(
        self, p0: np.ndarray, p1: np.ndarray, rect: Tuple[float, float, float, float]
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Check if line segment p0->p1 intersects an axis-aligned rectangle.

        Returns:
            (intersects, t_min, edge): t_min is the parametric distance along
            the segment [0,1] of the first intersection, edge is 'x' or 'y'
            indicating which axis the intersected edge is aligned to.
        """
        bx, by, bw, bh = rect
        # Rectangle edges
        x_min, x_max = bx, bx + bw
        y_min, y_max = by, by + bh

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        t_enter = 0.0
        t_exit = 1.0
        enter_edge = None

        # Check x-slab
        if abs(dx) < 1e-12:
            if p0[0] < x_min or p0[0] > x_max:
                return False, None, None
        else:
            t1 = (x_min - p0[0]) / dx
            t2 = (x_max - p0[0]) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_enter:
                t_enter = t1
                enter_edge = 'x'
            if t2 < t_exit:
                t_exit = t2

        if t_enter > t_exit:
            return False, None, None

        # Check y-slab
        if abs(dy) < 1e-12:
            if p0[1] < y_min or p0[1] > y_max:
                return False, None, None
        else:
            t1 = (y_min - p0[1]) / dy
            t2 = (y_max - p0[1]) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_enter:
                t_enter = t1
                enter_edge = 'y'
            if t2 < t_exit:
                t_exit = t2

        if t_enter > t_exit:
            return False, None, None

        # Intersection must be within the segment (0, 1] — exclude t=0 (starting point)
        if t_enter > 1.0 or t_exit < 1e-9:
            return False, None, None

        t_hit = max(t_enter, 1e-9)  # Avoid reflecting at the starting point
        if t_hit > 1.0:
            return False, None, None

        return True, t_hit, enter_edge

    def _reflect_step(self, start: np.ndarray, end: np.ndarray, max_bounces: int = 10) -> np.ndarray:
        """Reflect a proposed step off barriers.

        Traces the line segment from start to end, reflecting off any barriers
        encountered. Returns the final reflected position.
        """
        current = start.copy()
        target = end.copy()

        for _ in range(max_bounces):
            # Find the closest barrier intersection
            best_t = 2.0  # > 1.0 means no intersection
            best_edge = None

            for rect in self.barriers:
                hit, t, edge = self._segment_intersects_rect(current, target, rect)
                if hit and t < best_t:
                    best_t = t
                    best_edge = edge

            if best_t > 1.0:
                # No intersection, we're done
                break

            # Move to just before the intersection point
            hit_point = current + best_t * (target - current)
            remaining = target - hit_point

            # Reflect the remaining displacement
            if best_edge == 'x':
                remaining[0] = -remaining[0]
            else:
                remaining[1] = -remaining[1]

            current = hit_point
            target = hit_point + remaining

        return target

    def batch_step(self, position: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Take n_samples steps from the same position. Returns (n_samples, 2) destinations.

        Vectorizes the common case (no barrier hit) and only loops for samples
        that cross a barrier.
        """
        noise = rng.normal(0, self.sigma, size=(n_samples, 2))
        proposed = position + noise  # (n_samples, 2)

        # Vectorized boundary reflection
        x_min, y_min, x_max, y_max = self.bounds
        for dim, (lo, hi) in enumerate([(x_min, x_max), (y_min, y_max)]):
            below = proposed[:, dim] < lo
            above = proposed[:, dim] > hi
            proposed[below, dim] = lo + (lo - proposed[below, dim])
            proposed[above, dim] = hi - (proposed[above, dim] - hi)
        proposed[:, 0] = np.clip(proposed[:, 0], x_min, x_max)
        proposed[:, 1] = np.clip(proposed[:, 1], y_min, y_max)

        if not self.barriers:
            return proposed

        # Quick vectorized pre-filter: which samples MIGHT cross a barrier?
        # A sample can only cross barrier (bx, by, bw, bh) if its x-range or y-range
        # overlaps the barrier. Use the bounding box of (start, proposed) per sample.
        needs_check = np.zeros(n_samples, dtype=bool)
        for bx, by, bw, bh in self.barriers:
            x_lo = np.minimum(position[0], proposed[:, 0])
            x_hi = np.maximum(position[0], proposed[:, 0])
            y_lo = np.minimum(position[1], proposed[:, 1])
            y_hi = np.maximum(position[1], proposed[:, 1])
            overlap = (x_hi >= bx) & (x_lo <= bx + bw) & (y_hi >= by) & (y_lo <= by + bh)
            needs_check |= overlap

        # For samples that might cross, do full reflection
        check_idx = np.where(needs_check)[0]
        for idx in check_idx:
            original_proposed = position + noise[idx]
            reflected = self._reflect_step(position, original_proposed)
            proposed[idx] = self._clip_to_bounds(reflected)

        return proposed

    def step(self, position: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        """Take one step: Gaussian displacement with reflection.

        Args:
            position: Current (x, y) position.
            rng: NumPy random generator.

        Returns:
            (next_position, reward_at_current_position)
        """
        r = self.reward(position)
        noise = rng.normal(0, self.sigma, size=2)
        proposed = position + noise

        # Reflect off barriers
        next_pos = self._reflect_step(position, proposed)

        # Reflect off domain boundaries
        next_pos = self._clip_to_bounds(next_pos)

        return next_pos, r

    def intersects_barrier(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if the line segment from start to end crosses any barrier."""
        for rect in self.barriers:
            hit, _, _ = self._segment_intersects_rect(start, end, rect)
            if hit:
                return True
        return False

    def sample_trajectory(
        self, n_steps: int, seed: int, start_position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-sample a full trajectory on CPU.

        Args:
            n_steps: Number of steps to take.
            seed: Random seed.
            start_position: Starting position. If None, sample uniformly from
                valid (non-barrier) space.

        Returns:
            positions: (n_steps + 1, 2) array of positions.
            rewards: (n_steps,) array of rewards at each position (before step).
            barrier_crossings: (n_steps,) bool array, True if the step crossed a barrier.
        """
        rng = np.random.default_rng(seed)

        if start_position is None:
            start_position = self._sample_valid_position(rng)

        positions = np.zeros((n_steps + 1, 2))
        rewards = np.zeros(n_steps)
        barrier_crossings = np.zeros(n_steps, dtype=bool)

        positions[0] = start_position

        for t in range(n_steps):
            next_pos, r = self.step(positions[t], rng)
            rewards[t] = r
            barrier_crossings[t] = self.intersects_barrier(positions[t], next_pos)
            positions[t + 1] = next_pos

        return positions, rewards, barrier_crossings

    def _sample_valid_position(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random position not inside any barrier."""
        x_min, y_min, x_max, y_max = self.bounds
        for _ in range(1000):
            pos = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
            ])
            if not self.is_inside_barrier(pos):
                return pos
        raise RuntimeError("Could not find a valid position after 1000 attempts")

    def get_eval_grid(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate evaluation grid points, excluding barrier interiors.

        Args:
            resolution: Number of grid points per dimension.

        Returns:
            grid_points: (N, 2) array of valid grid coordinates.
            valid_mask: (resolution, resolution) bool array.
        """
        x_min, y_min, x_max, y_max = self.bounds
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)

        valid_mask = np.ones((resolution, resolution), dtype=bool)
        grid_points = []

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                pos = np.array([x, y])
                if self.is_inside_barrier(pos):
                    valid_mask[i, j] = False
                else:
                    grid_points.append(pos)

        return np.array(grid_points), valid_mask
