"""The multi-MNIST problem.

A multi-MNIST sample is built by drawing ``n_tasks`` MNIST examples
independently and concatenating their images and labels under a shared
ordering: the input has dimension ``784 * n_tasks`` and the target dimension
``10 * n_tasks``. Loss partitions the network output into ``n_tasks`` groups
of 10 logits and sums a softmax cross-entropy per group.

Because the sub-problems are drawn independently, the ideal connectivity is
known a priori: ``n_tasks`` independent sub-networks with no cross-task
connections. That is what makes the problem useful for isolating the effect
of connectivity on credit assignment.

In the non-stationary variant, every ``permute_period`` steps one of the
``n_tasks`` slots is chosen uniformly at random and its label mapping is
replaced by a random permutation of {0, ..., 9}.
"""

from typing import List, Optional, Tuple

import numpy as np


def load_dataset(split: str = 'train') -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load MNIST, normalized to [0, 1] and flattened.

    Returns:
        images: (N, 784) float32
        labels: (N,) int
        num_classes: 10
        input_dim: 784
    """
    from torchvision import datasets

    ds = datasets.MNIST(root='/tmp/data', train=(split == 'train'), download=True)
    images = np.array(ds.data, dtype=np.float32) / 255.0   # (N, 28, 28)
    images = images.reshape(images.shape[0], -1)           # (N, 784)
    labels = np.array(ds.targets)
    return images, labels, 10, 784


class MultiMNISTStream:
    """Online stream of multi-MNIST samples with optional label permutation.

    Each instance owns its RNG, so a vmapped multi-seed run gets independent
    but reproducible data streams.
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

        # Per-task label permutations, identity until the first event.
        self.label_permutations: List[np.ndarray] = [
            np.arange(10) for _ in range(n_tasks)
        ]
        self.step_counter = 0

    def _advance_permutations(self, n_steps: int):
        """Apply any permutation events falling in the next ``n_steps``."""
        if self.permute_period <= 0:
            self.step_counter += n_steps
            return

        start = self.step_counter
        end = start + n_steps

        if self.permute_stop > 0 and start >= self.permute_stop:
            self.step_counter = end
            return

        # First event at or after `start`.
        if start == 0:
            first_event = self.permute_period
        else:
            first_event = ((start - 1) // self.permute_period + 1) * self.permute_period

        event_end = min(end, self.permute_stop) if self.permute_stop > 0 else end

        for _ in range(first_event, event_end, self.permute_period):
            task_idx = self.rng.integers(0, self.n_tasks)
            self.label_permutations[task_idx] = self.rng.permutation(10)

        self.step_counter = end

    def sample_batch(self, n_steps: int):
        """Sample ``n_steps`` of multi-MNIST data.

        Permutation events inside this window are applied before sampling, so
        every sample in the returned block uses the updated mapping.

        Returns:
            images: (n_steps, batch_size, n_tasks * 784)
            labels: (n_steps, batch_size, n_tasks)
        """
        self._advance_permutations(n_steps)

        total = n_steps * self.batch_size
        task_images, task_labels = [], []
        for k in range(self.n_tasks):
            indices = self.rng.integers(0, self.n_samples, size=total)
            imgs = self.images[indices].reshape(n_steps, self.batch_size, -1)
            lbls = self.labels[indices].reshape(n_steps, self.batch_size)
            lbls = self.label_permutations[k][lbls]
            task_images.append(imgs)
            task_labels.append(lbls)

        images = np.concatenate(task_images, axis=-1)
        labels = np.stack(task_labels, axis=-1)
        return images, labels

    def get_test_batch(self):
        """Return the test pool formatted as multi-MNIST rows.

        Each row samples one test example per sub-task with replacement,
        matching how ``sample_batch`` draws from the train pool, and applies
        the current per-task label permutations.

        Returns:
            images: (N_test, n_tasks * 784)
            labels: (N_test, n_tasks)
        """
        assert self.test_images is not None, 'No test data provided'
        n_test = self.test_images.shape[0]

        task_images, task_labels = [], []
        for k in range(self.n_tasks):
            indices = self.rng.integers(0, n_test, size=n_test)
            task_images.append(self.test_images[indices])
            task_labels.append(self.label_permutations[k][self.test_labels[indices]])

        images = np.concatenate(task_images, axis=-1)
        labels = np.stack(task_labels, axis=-1)
        return images, labels


def standardize(images: np.ndarray, ref: Optional[np.ndarray] = None):
    """Per-pixel mean-0 var-1 standardization.

    ``ref`` supplies the statistics when given, so the test pool can be
    standardized with training-set statistics.
    """
    src = ref if ref is not None else images
    mean = src.mean(axis=0, keepdims=True)
    std = src.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (images - mean) / std
