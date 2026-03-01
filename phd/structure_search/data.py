from typing import Tuple

import numpy as np


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load and preprocess a dataset for online MLP training.

    Images are normalized to [0, 1] and flattened to 1D vectors.

    Returns:
        images: Float array (N, input_dim)
        labels: Int array (N,) with class indices
        num_classes: Number of classes
        input_dim: Flattened input dimensionality
    """
    from torchvision import datasets

    if name == 'cifar10':
        ds = datasets.CIFAR10(root='/tmp/data', train=True, download=True)
        images = np.array(ds.data, dtype=np.float32) / 255.0  # (50000, 32, 32, 3)
        images = images.reshape(images.shape[0], -1)  # (50000, 3072)
        labels = np.array(ds.targets)
        num_classes = 10
        input_dim = 3072
    else:
        raise ValueError(f'Unknown dataset: {name}')

    return images, labels, num_classes, input_dim


class DataStream:
    """CPU-side data stream for online learning.

    Randomly samples batches from the dataset with replacement.
    Each instance uses its own RNG for reproducible, independent streams.
    """

    def __init__(self, images, labels, num_classes, batch_size, seed):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.n_samples = images.shape[0]
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, n_steps):
        """Sample n_steps worth of batched data on CPU.

        Returns:
            images: numpy array (n_steps, batch_size, input_dim)
            labels: numpy array (n_steps, batch_size) with integer class indices
        """
        total = n_steps * self.batch_size
        indices = self.rng.integers(0, self.n_samples, size=total)
        images = self.images[indices].reshape(n_steps, self.batch_size, -1)
        labels = self.labels[indices].reshape(n_steps, self.batch_size)
        return images, labels
