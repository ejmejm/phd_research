"""ResNet with IMPALA-CNN blocks for Atari prediction.

ResNet built from IMPALA-CNN ConvSequence blocks (Espeholt et al., 2018) with
optional width scaling as used in BBF (Bigger, Better, Faster).

Architecture:
    ConvSequence(16*scale) -> ConvSequence(32*scale) -> ConvSequence(32*scale)
    -> Flatten -> Linear(256) -> ReLU -> Linear(output_dim)

Each ConvSequence:
    Conv2D -> MaxPool(3x3, stride 2) -> 2x ResBlock

Each ResBlock:
    ReLU -> Conv2D -> ReLU -> Conv2D + residual
"""

from typing import List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PRNGKeyArray


class ResBlock(eqx.Module):
    """Residual block: ReLU -> Conv -> ReLU -> Conv + skip."""
    conv1: nn.Conv2d
    conv2: nn.Conv2d

    def __init__(self, channels: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=k1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=k2)

    def __call__(self, x):
        residual = x
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + residual


class ConvSequence(eqx.Module):
    """Conv2D -> MaxPool(3x3, stride 2) -> ResBlock -> ResBlock."""
    conv: nn.Conv2d
    res_block_0: ResBlock
    res_block_1: ResBlock

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=k1)
        self.res_block_0 = ResBlock(out_channels, key=k2)
        self.res_block_1 = ResBlock(out_channels, key=k3)

    def __call__(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res_block_0(x)
        x = self.res_block_1(x)
        return x


class ResNet(eqx.Module):
    """ResNet using IMPALA-CNN ConvSequence blocks with configurable width/depth.

    Expects input shape (C, H, W) where C = frame_stack channels (e.g. 4)
    and H, W = 84, 84.

    Returns (output, param_inputs) to match the MLP interface.
    param_inputs is an empty list since per-layer input tracking is not
    meaningful for CNNs in this codebase.
    """
    conv_sequences: List[ConvSequence]
    head_linear: nn.Linear
    output_linear: nn.Linear
    output_dim: int = eqx.field(static=True)
    width_scale: float = eqx.field(static=True)
    n_conv_sequences: int = eqx.field(static=True)

    # Default IMPALA-CNN base channels; the first n_conv_sequences entries are used.
    BASE_CHANNELS: tuple = eqx.field(static=True, default=(16, 32, 32))

    def __init__(
        self,
        in_channels: int = 4,
        output_dim: int = 1,
        width_scale: float = 4.0,
        n_conv_sequences: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_channels: Number of input channels (frame_stack size).
            output_dim: Number of output predictions.
            width_scale: Multiplier for base channel widths (BBF uses 4).
                Can be fractional, e.g. 0.25, 0.5, 1, 2, 4.
            n_conv_sequences: Number of ConvSequence blocks (default 3).
                Uses the first n entries from BASE_CHANNELS, cycling if needed.
            key: PRNG key for weight initialization.
        """
        self.output_dim = output_dim
        self.width_scale = width_scale
        self.n_conv_sequences = n_conv_sequences

        # Build channel list: cycle base channels if n_conv_sequences > len(base)
        base = self.BASE_CHANNELS
        base_channels = [base[i % len(base)] for i in range(n_conv_sequences)]
        channels = [int(c * width_scale) for c in base_channels]

        keys = jax.random.split(key, len(channels) + 2)

        # Build conv sequences
        self.conv_sequences = []
        prev_ch = in_channels
        for i, ch in enumerate(channels):
            self.conv_sequences.append(
                ConvSequence(prev_ch, ch, key=keys[i]))
            prev_ch = ch

        # Compute flattened feature size after conv sequences.
        # For 84x84 input with 3 MaxPool(stride=2): 84 -> 42 -> 21 -> 11
        feat_h, feat_w = 84, 84
        for _ in channels:
            feat_h = (feat_h + 1) // 2  # ceil division for stride-2 with pad-1
            feat_w = (feat_w + 1) // 2
        flat_dim = channels[-1] * feat_h * feat_w

        self.head_linear = nn.Linear(flat_dim, 256, key=keys[-2])
        self.output_linear = nn.Linear(256, output_dim, key=keys[-1])

    def __call__(
        self,
        x,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[jnp.ndarray, List[Any]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (C, H, W).
            key: Unused, kept for interface compatibility.

        Returns:
            (output, param_inputs) matching the MLP interface.
        """
        for conv_seq in self.conv_sequences:
            x = conv_seq(x)

        x = jax.nn.relu(x)
        x = jnp.reshape(x, (-1,))  # flatten
        x = self.head_linear(x)
        x = jax.nn.relu(x)
        output = self.output_linear(x)

        return output, []
