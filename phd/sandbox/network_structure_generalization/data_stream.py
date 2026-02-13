from typing import Tuple

import numpy as np


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, int, tuple]:
    """Load and preprocess a dataset, grouping images by class.

    All data stays on CPU as numpy arrays.

    Returns:
        all_images: Float array (num_classes, max_per_class, *image_shape)
        samples_per_class: Int array (num_classes,)
        num_classes: Number of classes
        image_shape: Tuple (channels, height, width)
    """
    from torchvision import datasets

    if name == 'mnist':
        ds = datasets.MNIST(root='/tmp/data', train=True, download=True)
        images = ds.data.numpy().astype(np.float32) / 255.0
        images = images[:, None, :, :]  # (60000, 1, 28, 28)
        labels = ds.targets.numpy()
        num_classes = 10
        image_shape = (1, 28, 28)
    elif name == 'cifar100':
        ds = datasets.CIFAR100(root='/tmp/data', train=True, download=True)
        images = np.array(ds.data, dtype=np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))  # (50000, 3, 32, 32)
        labels = np.array(ds.targets)
        num_classes = 100
        image_shape = (3, 32, 32)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    # Group by class, padding to equal length
    max_per_class = max(int(np.sum(labels == c)) for c in range(num_classes))
    all_images = np.zeros((num_classes, max_per_class, *image_shape), dtype=np.float32)
    samples_per_class = np.zeros(num_classes, dtype=np.int32)

    for c in range(num_classes):
        mask = labels == c
        class_images = images[mask]
        n = class_images.shape[0]
        all_images[c, :n] = class_images
        samples_per_class[c] = n

    return all_images, samples_per_class, num_classes, image_shape


class ContinualClassStream:
    """CPU-side data stream for continual binary classification.

    Samples images from pairs of classes that switch every `switch_freq` steps.
    All data stays on CPU; use `sample_batch` to get a numpy batch that can be
    transferred to GPU for one scan cycle.
    """

    def __init__(self, all_images, samples_per_class, num_classes, image_shape,
                 switch_freq, allow_resampling, seed):
        self.all_images = all_images            # numpy, shared across streams
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.switch_freq = switch_freq
        self.allow_resampling = allow_resampling
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.current_pair = None
        self.used_classes = np.zeros(num_classes, dtype=bool)

    def _draw_new_pair(self):
        """Draw a new pair of classes to present."""
        if self.allow_resampling:
            self.current_pair = self.rng.choice(self.num_classes, 2, replace=False)
        else:
            available = ~self.used_classes
            if available.sum() < 2:
                self.used_classes[:] = False
                available = ~self.used_classes
            weights = available.astype(np.float64)
            self.current_pair = self.rng.choice(
                self.num_classes, 2, replace=False, p=weights / weights.sum())
            self.used_classes[self.current_pair[0]] = True
            self.used_classes[self.current_pair[1]] = True

    def sample_batch(self, batch_size):
        """Sample a batch of (images, labels) on CPU for one scan cycle.

        Returns:
            images: numpy array (batch_size, *image_shape)
            labels: numpy array (batch_size,) with values 0 or 1
        """
        images = np.empty((batch_size, *self.image_shape), dtype=np.float32)
        labels = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            if self.step % self.switch_freq == 0:
                self._draw_new_pair()

            which = self.rng.integers(2)
            class_idx = self.current_pair[which]
            n_samples = self.samples_per_class[class_idx]
            img_idx = self.rng.integers(n_samples)

            images[i] = self.all_images[class_idx, img_idx]
            labels[i] = which
            self.step += 1

        return images, labels
