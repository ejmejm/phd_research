import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


def n_kaiming_uniform(
    tensor: torch.Tensor,
    shape: Tuple[int, ...],
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'relu',
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    But has a customizable number of outputs.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    
    if mode == 'fan_in':
        fan = tensor.shape[-1]
    elif mode == 'fan_out':
        fan = tensor.shape[-2]
    else:
        raise ValueError(f"Invalid mode: {mode}!")
    
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    target_device = device if device is not None else tensor.device
    result = torch.rand(shape, generator=generator, device=target_device) * 2 * bound - bound
    return result