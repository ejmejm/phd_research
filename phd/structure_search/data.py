from typing import List, Optional, Tuple

import numpy as np


def load_dataset(
    name: str, split: str = 'train',
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load and preprocess a dataset for online MLP training.

    Images are normalized to [0, 1] and flattened to 1D vectors.

    Args:
        name: Dataset name ('cifar10' or 'mnist').
        split: 'train' or 'test'.

    Returns:
        images: Float array (N, input_dim)
        labels: Int array (N,) with class indices
        num_classes: Number of classes
        input_dim: Flattened input dimensionality
    """
    from torchvision import datasets

    train = (split == 'train')

    if name == 'cifar10':
        ds = datasets.CIFAR10(root='/tmp/data', train=train, download=True)
        images = np.array(ds.data, dtype=np.float32) / 255.0
        images = images.reshape(images.shape[0], -1)
        labels = np.array(ds.targets)
        num_classes = 10
        input_dim = 3072
    elif name == 'mnist':
        ds = datasets.MNIST(root='/tmp/data', train=train, download=True)
        images = np.array(ds.data, dtype=np.float32) / 255.0  # (N, 28, 28)
        images = images.reshape(images.shape[0], -1)  # (N, 784)
        labels = np.array(ds.targets)
        num_classes = 10
        input_dim = 784
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


class ParallelMNISTStream:
    """Data stream for parallel MNIST: K independent sub-tasks concatenated.

    Each sample consists of K independently sampled MNIST images concatenated
    into a single input vector of dimension K*784, with K integer labels.

    Supports non-stationary mode: at regular intervals, the label mapping for
    one randomly chosen sub-task is permuted.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_tasks: int,
        batch_size: int,
        seed: int,
        permute_period: int = 0,
        permute_stop: int = 0,
        test_images: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
    ):
        self.images = images        # (N, 784)
        self.labels = labels        # (N,)
        self.n_tasks = n_tasks
        self.num_classes = 10
        self.batch_size = batch_size
        self.n_samples = images.shape[0]
        self.permute_period = permute_period
        self.permute_stop = permute_stop  # stop permuting after this step (0 = never stop)
        self.rng = np.random.default_rng(seed)

        self.test_images = test_images  # (N_test, 784) or None
        self.test_labels = test_labels  # (N_test,) or None

        # Per-task label permutations (identity initially)
        self.label_permutations: List[np.ndarray] = [
            np.arange(10) for _ in range(n_tasks)
        ]
        self.step_counter = 0

    def _advance_permutations(self, n_steps: int):
        """Check for and apply any permutation events in the next n_steps."""
        if self.permute_period <= 0:
            self.step_counter += n_steps
            return

        start = self.step_counter
        end = start + n_steps

        # No permutations after permute_stop
        if self.permute_stop > 0 and start >= self.permute_stop:
            self.step_counter = end
            return

        # Find the first permutation event at or after start
        if start == 0:
            first_event = self.permute_period
        else:
            first_event = ((start - 1) // self.permute_period + 1) * self.permute_period

        # Cap event range at permute_stop
        event_end = min(end, self.permute_stop) if self.permute_stop > 0 else end

        for event_step in range(first_event, event_end, self.permute_period):
            task_idx = self.rng.integers(0, self.n_tasks)
            self.label_permutations[task_idx] = self.rng.permutation(10)

        self.step_counter = end

    def sample_batch(self, n_steps: int):
        """Sample n_steps of parallel MNIST data.

        Permutation events that fall within this batch window are applied
        before sampling, so all data in this batch uses the updated
        permutation state.

        Returns:
            images: (n_steps, batch_size, K*784)
            labels: (n_steps, batch_size, K)
        """
        # Advance permutations for this batch window
        self._advance_permutations(n_steps)

        total = n_steps * self.batch_size
        task_images = []
        task_labels = []
        for k in range(self.n_tasks):
            indices = self.rng.integers(0, self.n_samples, size=total)
            imgs = self.images[indices].reshape(n_steps, self.batch_size, -1)
            lbls = self.labels[indices].reshape(n_steps, self.batch_size)
            lbls = self.label_permutations[k][lbls]
            task_images.append(imgs)
            task_labels.append(lbls)

        # (n_steps, batch_size, K*784)
        images = np.concatenate(task_images, axis=-1)
        # (n_steps, batch_size, K)
        labels = np.stack(task_labels, axis=-1)
        return images, labels

    def get_test_batch(self):
        """Return the test pool formatted for parallel MNIST evaluation.

        Produces N_test rows. For each row, every sub-task independently samples
        one test example (with replacement), matching ``sample_batch`` on the
        train pool. Per-task label permutations are applied to the sampled
        labels.

        Returns:
            images: (N_test, K*784)
            labels: (N_test, K)
        """
        assert self.test_images is not None, 'No test data provided'
        n_test = self.test_images.shape[0]

        task_images = []
        task_labels = []
        for k in range(self.n_tasks):
            indices = self.rng.integers(0, n_test, size=n_test)
            imgs = self.test_images[indices]
            lbls = self.label_permutations[k][self.test_labels[indices]]
            task_images.append(imgs)
            task_labels.append(lbls)

        images = np.concatenate(task_images, axis=-1)
        labels = np.stack(task_labels, axis=-1)
        return images, labels
