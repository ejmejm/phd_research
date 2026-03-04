"""Data loading for continual Atari prediction experiments.

Wraps PrerecordedDataset from the continual-atari-benchmark package and provides
sequential batch sampling with background preloading for efficient training.
"""

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Tuple

import numpy as np
from omegaconf import DictConfig

from continual_atari_benchmark import PrerecordedDataset
from continual_atari_benchmark.dataset_loader import OBS_CHUNK_SIZE


# Raw Atari observation dimensions
OBS_HEIGHT = 210
OBS_WIDTH = 160
OBS_CHANNELS = 3
INPUT_DIM = OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS  # 100800


class ContinualAtariStream:
    """Sequential data stream over pre-recorded Atari trajectories.

    Yields chunks of (observations, returns) in order across multiple games.
    Observations are flattened to 1D and normalized to [0, 1].
    """

    def __init__(self, data_dir: str, game_order: list, steps_per_game: int,
                 data_seed: int = 0):
        data_path = Path(data_dir)

        # Validate that required data files exist
        for game in game_order:
            game_name = game.replace('ALE/', '').replace('-v5', '')
            npz_path = data_path / game_name / f'seed_{data_seed}.npz'
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Data file not found: {npz_path}\n"
                    f"Game '{game_name}' data has not been collected yet. "
                    f"Run the data collection script first:\n"
                    f"  cd ~/local_projects/continual_atari_benchmark && "
                    f"python dataset/collect.py games=\"['{game}']\" seeds=\"[{data_seed}]\""
                )

        self.dataset = PrerecordedDataset(
            data_dir=data_dir,
            game_order=game_order,
            steps_per_game=steps_per_game,
            seed=data_seed,
        )
        self.total_steps = len(self.dataset)
        self.current_step = 0

    def sample_batch(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the next n_steps of sequential data.

        Bulk-loads observations directly from chunk files instead of calling
        get_obs() in a loop. Handles chunk and game boundaries.

        Returns:
            observations: float32 array (n_steps, INPUT_DIM), normalized [0, 1]
            returns: float32 array (n_steps, 1)
        """
        if self.current_step + n_steps > self.total_steps:
            raise RuntimeError(
                f"Requested {n_steps} steps starting at step {self.current_step}, "
                f"but only {self.total_steps - self.current_step} steps remain. "
                f"Total dataset size: {self.total_steps} steps."
            )

        # Returns are already in memory — just slice
        returns = self.dataset._returns[
            self.current_step:self.current_step + n_steps
        ].astype(np.float32).reshape(n_steps, 1)

        # Bulk-load observations by iterating over chunk boundaries
        obs_parts = []
        remaining = n_steps
        step = self.current_step

        while remaining > 0:
            game_idx = step // self.dataset.steps_per_game
            local_step = step % self.dataset.steps_per_game
            chunk_idx = local_step // OBS_CHUNK_SIZE
            idx_in_chunk = local_step % OBS_CHUNK_SIZE

            # Load this chunk
            self.dataset._load_obs_chunk(game_idx, chunk_idx)
            chunk_obs = self.dataset._current_chunk_obs

            # How many steps can we take from this chunk?
            available_in_chunk = len(chunk_obs) - idx_in_chunk
            # Also respect game boundary
            steps_left_in_game = self.dataset.steps_per_game - local_step
            take = min(remaining, available_in_chunk, steps_left_in_game)

            obs_parts.append(chunk_obs[idx_in_chunk:idx_in_chunk + take])

            step += take
            remaining -= take

        observations = np.concatenate(obs_parts, axis=0)
        self.current_step += n_steps

        # Flatten but keep as uint8 — normalization happens on GPU.
        # Use reshape (view, no copy) rather than making a contiguous copy.
        obs_flat = observations.reshape(n_steps, -1)

        return obs_flat, returns


class BackgroundDataLoader:
    """Preloads the next batch on a background thread while training runs.

    Usage:
        loader = BackgroundDataLoader(stream, chunk_size=1000)
        loader.start_preload()  # kick off first preload

        for _ in range(num_chunks):
            obs, returns = loader.get()  # blocks until preload done
            # ... train on (obs, returns) ...
            # next preload is automatically started after get()
    """

    def __init__(self, stream: ContinualAtariStream, chunk_size: int):
        self.stream = stream
        self.chunk_size = chunk_size
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

    def _load(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.stream.sample_batch(self.chunk_size)

    def start_preload(self):
        """Begin preloading the next batch in the background."""
        self._future = self._executor.submit(self._load)

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the preloaded batch and start preloading the next one.

        Returns:
            observations: float32 array (chunk_size, INPUT_DIM)
            returns: float32 array (chunk_size, 1)
        """
        if self._future is None:
            raise RuntimeError("Call start_preload() before get()")

        result = self._future.result()

        # Start preloading the next batch if there's data remaining
        remaining = self.stream.total_steps - self.stream.current_step
        if remaining >= self.chunk_size:
            self.start_preload()
        else:
            self._future = None

        return result

    def shutdown(self):
        self._executor.shutdown(wait=False)


def load_atari_data(cfg: DictConfig) -> ContinualAtariStream:
    """Create a ContinualAtariStream from config."""
    return ContinualAtariStream(
        data_dir=cfg.dataset.data_dir,
        game_order=list(cfg.dataset.game_order),
        steps_per_game=cfg.dataset.steps_per_game,
        data_seed=cfg.dataset.data_seed,
    )
