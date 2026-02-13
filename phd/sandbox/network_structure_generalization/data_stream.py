from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces field values of an eqx.Module immutably."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def load_dataset(name: str) -> Tuple[jax.Array, jax.Array, int, tuple]:
    """Load and preprocess a dataset, grouping images by class.

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

    return jnp.array(all_images), jnp.array(samples_per_class), num_classes, image_shape


class ContinualClassStream(eqx.Module):
    """JAX-traceable data stream for continual binary classification.

    At each step, samples one image from one of two active classes and returns
    a binary label (0 or 1). Class pairs switch every `switch_freq` steps.
    """

    # Static (compile-time constants)
    num_classes: int = eqx.field(static=True)
    switch_freq: int = eqx.field(static=True)
    allow_resampling: bool = eqx.field(static=True)
    image_shape: tuple = eqx.field(static=True)

    # Data (constant after init)
    all_images: jax.Array
    samples_per_class: jax.Array

    # Dynamic (updated each step)
    step: jax.Array
    rng: jax.Array
    current_pair: jax.Array
    used_classes: jax.Array

    def __init__(self, all_images, samples_per_class, num_classes, image_shape,
                 switch_freq, allow_resampling, *, key):
        self.num_classes = num_classes
        self.switch_freq = switch_freq
        self.allow_resampling = allow_resampling
        self.image_shape = image_shape
        self.all_images = all_images
        self.samples_per_class = samples_per_class
        self.step = jnp.array(0)
        self.rng = key
        self.current_pair = jnp.array([0, 1])  # placeholder, overwritten on first step
        self.used_classes = jnp.zeros(num_classes, dtype=bool)

    def _draw_new_pair(self, key):
        """Draw a new pair of classes to present."""
        if self.allow_resampling:
            pair = jax.random.choice(key, self.num_classes, (2,), replace=False)
            return pair, self.used_classes
        else:
            available = ~self.used_classes
            n_available = available.sum()
            should_reset = n_available < 2
            new_used = jnp.where(should_reset,
                                 jnp.zeros(self.num_classes, dtype=bool),
                                 self.used_classes)
            available = jnp.where(should_reset,
                                  jnp.ones(self.num_classes, dtype=bool),
                                  available)
            weights = available.astype(jnp.float32)
            pair = jax.random.choice(key, self.num_classes, (2,),
                                     replace=False, p=weights / weights.sum())
            new_used = new_used.at[pair[0]].set(True)
            new_used = new_used.at[pair[1]].set(True)
            return pair, new_used

    def sample(self):
        """Sample one (image, label) pair and return updated stream.

        Returns:
            new_stream: Updated ContinualClassStream
            (image, label): Sampled image and binary label (0 or 1)
        """
        key, switch_key, class_key, img_key = jax.random.split(self.rng, 4)

        should_switch = (self.step % self.switch_freq == 0)

        # Compute candidate new pair (always computed; only used if switching)
        new_pair, new_used = self._draw_new_pair(switch_key)
        current_pair = jnp.where(should_switch, new_pair, self.current_pair)
        used_classes = jnp.where(should_switch, new_used, self.used_classes)

        # Pick one of the two active classes uniformly
        which = jax.random.randint(class_key, (), 0, 2)
        class_idx = current_pair[which]

        # Sample a random image from that class
        n_samples = self.samples_per_class[class_idx]
        img_idx = jax.random.randint(img_key, (), 0, n_samples)
        image = self.all_images[class_idx, img_idx]

        new_stream = tree_replace(
            self,
            step=self.step + 1,
            rng=key,
            current_pair=current_pair,
            used_classes=used_classes,
        )

        return new_stream, (image, which)
