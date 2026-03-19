from typing import Tuple

import numpy as np
from scipy.ndimage import rotate


def load_mnist() -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load MNIST, group images by class.

    Returns:
        all_images: (num_classes, max_per_class, 28, 28) float32 [0, 1]
        samples_per_class: (num_classes,) int32
        num_classes: 10
        input_dim: 784
    """
    from torchvision import datasets

    ds = datasets.MNIST(root='/tmp/data', train=True, download=True)
    images = ds.data.numpy().astype(np.float32) / 255.0  # (60000, 28, 28)
    labels = ds.targets.numpy()
    num_classes = 10

    max_per_class = max(int(np.sum(labels == c)) for c in range(num_classes))
    all_images = np.zeros((num_classes, max_per_class, 28, 28), dtype=np.float32)
    samples_per_class = np.zeros(num_classes, dtype=np.int32)

    for c in range(num_classes):
        mask = labels == c
        class_images = images[mask]
        n = class_images.shape[0]
        all_images[c, :n] = class_images
        samples_per_class[c] = n

    return all_images, samples_per_class, num_classes, 784


class RotatingMNISTStream:
    """CPU-side data stream: rotates a single MNIST digit by delta degrees per step.

    After a full 360-degree rotation, switches to a new random digit (possibly
    from a different class). The stream produces flattened 784-dim images.
    """

    def __init__(
        self,
        all_images: np.ndarray,
        samples_per_class: np.ndarray,
        num_classes: int,
        delta_degrees: float,
        seed: int,
    ):
        self.all_images = all_images          # (num_classes, max_per_class, 28, 28)
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.delta_degrees = delta_degrees
        self.steps_per_digit = int(360 / delta_degrees)
        self.rng = np.random.default_rng(seed)

        # Current state
        self.current_class = None
        self.current_img_idx = None
        self.current_base_image = None        # (28, 28)
        self.angle_step = 0                   # steps into current digit rotation

    def _draw_new_digit(self):
        """Pick a random class and a random image from that class."""
        self.current_class = self.rng.integers(self.num_classes)
        n = self.samples_per_class[self.current_class]
        self.current_img_idx = self.rng.integers(n)
        self.current_base_image = self.all_images[
            self.current_class, self.current_img_idx
        ]
        self.angle_step = 0

    def sample_batch(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample n_steps of streaming rotated MNIST data.

        Returns:
            images: (n_steps, 784) float32
            labels: (n_steps,) int32
            is_switch: (n_steps,) bool — True at digit-switch boundaries
        """
        images = np.empty((n_steps, 784), dtype=np.float32)
        labels = np.empty(n_steps, dtype=np.int32)
        is_switch = np.zeros(n_steps, dtype=bool)

        for i in range(n_steps):
            if self.current_base_image is None or self.angle_step >= self.steps_per_digit:
                self._draw_new_digit()
                is_switch[i] = True

            angle = self.angle_step * self.delta_degrees
            rotated = rotate(
                self.current_base_image, angle,
                reshape=False, mode='constant', cval=0.0,
            )
            # Clip to [0, 1] — rotation interpolation can slightly overshoot
            rotated = np.clip(rotated, 0.0, 1.0)

            images[i] = rotated.reshape(784)
            labels[i] = self.current_class
            self.angle_step += 1

        return images, labels, is_switch


class ShuffledMNISTStream:
    """CPU-side i.i.d. MNIST stream (baseline). Samples random digits each step."""

    def __init__(
        self,
        all_images: np.ndarray,
        samples_per_class: np.ndarray,
        num_classes: int,
        seed: int,
    ):
        self.all_images = all_images
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample n_steps of i.i.d. MNIST data.

        Returns:
            images: (n_steps, 784) float32
            labels: (n_steps,) int32
            is_switch: (n_steps,) bool — always False
        """
        images = np.empty((n_steps, 784), dtype=np.float32)
        labels = np.empty(n_steps, dtype=np.int32)

        for i in range(n_steps):
            c = self.rng.integers(self.num_classes)
            n = self.samples_per_class[c]
            idx = self.rng.integers(n)
            images[i] = self.all_images[c, idx].reshape(784)
            labels[i] = c

        return images, labels, np.zeros(n_steps, dtype=bool)
