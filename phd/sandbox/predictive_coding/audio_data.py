"""Audio prediction benchmark data loading and true return computation.

Loads pre-generated audio data, computes binary FFT observations via
the audio_prediction_benchmark package, and pre-computes exact discounted
returns for MSVE evaluation.
"""

import json
import os

import numpy as np
import soundfile as sf

from audio_prediction_benchmark.preprocessing import preprocess_audio


def load_audio_data(data_dir):
    """Load and preprocess audio benchmark data.

    Loads audio.wav and rewards.npy from data_dir, computes binary FFT
    observations for all timesteps using preprocess_audio().

    Args:
        data_dir: Directory containing audio.wav, rewards.npy, metadata.json.

    Returns:
        observations: (n_steps, 2500) uint8 numpy array of binary observations.
        rewards: (n_steps,) float32 numpy array of per-step rewards.
        metadata: dict from metadata.json (or None if not found).
    """
    data_dir = os.path.expanduser(data_dir)

    # Load audio
    audio_path = os.path.join(data_dir, 'audio.wav')
    audio, sample_rate = sf.read(audio_path, dtype='float64')

    # Load rewards
    rewards_path = os.path.join(data_dir, 'rewards.npy')
    rewards = np.load(rewards_path).astype(np.float32)

    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = None

    # Pre-compute all binary observations
    print(f'Pre-computing {len(rewards)} observations (FFT + binarization)...')
    observations = preprocess_audio(audio, sample_rate)  # (n_steps, 2500) uint8

    n_nonzero = np.count_nonzero(rewards)
    print(f'Loaded: {observations.shape[0]} steps, {observations.shape[1]} obs dim, '
          f'{n_nonzero} non-zero rewards ({100 * n_nonzero / len(rewards):.2f}%)')

    return observations, rewards, metadata


def compute_true_returns(rewards, gamma):
    """Compute exact discounted returns via backward recursion.

    G[t] = r[t] + gamma * G[t+1], with G[T-1] = r[T-1].

    Args:
        rewards: (n_steps,) float32 array of per-step rewards.
        gamma: Discount factor.

    Returns:
        returns: (n_steps,) float32 array of true discounted returns.
    """
    n = len(rewards)
    # Compute in float64 for precision, return as float32
    G = np.zeros(n, dtype=np.float64)
    G[-1] = rewards[-1]
    for t in range(n - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G.astype(np.float32)
