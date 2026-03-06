"""Data loading for continual Atari prediction experiments.

Wraps PrerecordedDataset from the continual-atari-benchmark package and provides
sequential batch sampling with background preloading for efficient training.
Includes standard Atari preprocessing: grayscale, resize, and framestacking.
"""

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from omegaconf import DictConfig

from continual_atari_benchmark import PrerecordedDataset
from continual_atari_benchmark.dataset_loader import OBS_CHUNK_SIZE


# Raw Atari observation dimensions
OBS_HEIGHT = 210
OBS_WIDTH = 160
OBS_CHANNELS = 3


def compute_input_dim(cfg: DictConfig) -> int:
    """Compute flattened input dimension from preprocessing config."""
    pp = cfg.preprocessing
    h, w = pp.resize
    n_stack = pp.frame_stack
    if pp.grayscale:
        return h * w * n_stack
    else:
        return h * w * OBS_CHANNELS * n_stack


def preprocess_frames(obs: np.ndarray, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Convert raw Atari frames to grayscale and resize.

    Args:
        obs: uint8 array (n, 210, 160, 3)
        resize_hw: (height, width) target size

    Returns:
        uint8 array (n, height, width)
    """
    h, w = resize_hw
    n = len(obs)
    out = np.empty((n, h, w), dtype=np.uint8)
    for i in range(n):
        gray = cv2.cvtColor(obs[i], cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    return out


class ContinualAtariStream:
    """Sequential data stream over pre-recorded Atari trajectories.

    Yields chunks of (observations, returns) in order across multiple games.
    Applies preprocessing (grayscale, resize, framestack) on the CPU.
    """

    def __init__(self, data_dir: str, game_order: list, steps_per_game: int,
                 data_seed: int = 0, preprocessing: DictConfig = None):
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

        # Preprocessing config
        self._preprocess = preprocessing is not None
        if self._preprocess:
            self._resize_hw = tuple(preprocessing.resize)
            self._grayscale = preprocessing.grayscale
            self._n_stack = preprocessing.frame_stack
            # Frame buffer for framestacking across chunks (initialized on first call)
            self._frame_buffer = None
        else:
            self._n_stack = 1

    def _load_raw_observations(self, n_steps: int) -> np.ndarray:
        """Bulk-load raw observations from chunk files.

        Returns uint8 array (n_steps, 210, 160, 3).
        """
        obs_parts = []
        remaining = n_steps
        step = self.current_step

        while remaining > 0:
            game_idx = step // self.dataset.steps_per_game
            local_step = step % self.dataset.steps_per_game
            chunk_idx = local_step // OBS_CHUNK_SIZE
            idx_in_chunk = local_step % OBS_CHUNK_SIZE

            self.dataset._load_obs_chunk(game_idx, chunk_idx)
            chunk_obs = self.dataset._current_chunk_obs

            available_in_chunk = len(chunk_obs) - idx_in_chunk
            steps_left_in_game = self.dataset.steps_per_game - local_step
            take = min(remaining, available_in_chunk, steps_left_in_game)

            obs_parts.append(chunk_obs[idx_in_chunk:idx_in_chunk + take])

            step += take
            remaining -= take

        return np.concatenate(obs_parts, axis=0)

    def _apply_framestack(self, processed: np.ndarray) -> np.ndarray:
        """Apply framestacking over a continuous stream (no resets at any boundary).

        The entire run is treated as one long episode. The frame buffer carries
        over across chunks and game boundaries.

        Args:
            processed: uint8 array (n_steps, H, W) — preprocessed grayscale frames

        Returns:
            uint8 array (n_steps, H, W, n_stack)
        """
        n_stack = self._n_stack

        # Initialize buffer on first call with copies of the first frame
        if self._frame_buffer is None:
            self._frame_buffer = np.stack([processed[0]] * (n_stack - 1))

        # Prepend buffer and apply sliding window (zero-copy view)
        full_seq = np.concatenate([self._frame_buffer, processed], axis=0)
        stacked = np.lib.stride_tricks.sliding_window_view(
            full_seq, n_stack, axis=0)

        # Update buffer for next chunk
        self._frame_buffer = processed[-(n_stack - 1):].copy()

        return np.ascontiguousarray(stacked)

    def sample_batch(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the next n_steps of sequential data.

        Returns:
            observations: float32 array (n_steps, C, H, W), normalized to [0, 1]
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

        # Load raw observations
        raw_obs = self._load_raw_observations(n_steps)

        if self._preprocess:
            # Grayscale + resize on CPU (runs in background thread)
            processed = preprocess_frames(raw_obs, self._resize_hw)
            # Framestack: (n_steps, H, W, C)
            obs = self._apply_framestack(processed)
        else:
            obs = raw_obs

        self.current_step += n_steps

        # Channels-first and normalize: (n_steps, H, W, C) -> (n_steps, C, H, W)
        obs = obs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return obs, returns


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
            observations: float32 array (chunk_size, C, H, W)
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
    # Resolve relative data_dir against original CWD (Hydra changes CWD to output dir)
    data_dir = Path(cfg.dataset.data_dir)
    if not data_dir.is_absolute():
        import hydra
        data_dir = Path(hydra.utils.get_original_cwd()) / data_dir
    return ContinualAtariStream(
        data_dir=str(data_dir),
        game_order=list(cfg.dataset.game_order),
        steps_per_game=cfg.dataset.steps_per_game,
        data_seed=cfg.dataset.data_seed,
        preprocessing=cfg.get('preprocessing', None),
    )
