import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ACTIVATION_MAP, initialize_layer_weights, ParallelLinear


class MultipleLinear(nn.Module):
    """A linear layer that applies multiple weight matrices to different inputs using batch matrix multiplication.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        n_parallel: Number of parallel weight matrices
        bias: If True, adds a learnable bias to the output
        generator: Optional random generator for reproducibility
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.generator = generator
        
        # Create weight tensor of shape (n_parallel, out_features, in_features)
        self.weight = nn.Parameter(torch.empty(n_parallel, out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(n_parallel, out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using the same strategy as nn.Linear."""
        # Initialize each parallel weight matrix
        for i in range(self.n_parallel):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5), generator=self.generator)
        
        if self.bias is not None:
            for i in range(self.n_parallel):
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound, generator=self.generator)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_parallel, in_features) or (n_parallel, in_features)
            
        Returns:
            Output tensor of shape (batch_size, n_parallel, out_features) or (n_parallel, out_features)
        """
        # Handle single sample case
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Change the weights to the following shape for bmm:
        # x shape: (batch_size, n_parallel, in_features) -> (batch_size, n_parallel, in_features, 1)
        # weight shape: (n_parallel, out_features, in_features) -> (1, n_parallel, out_features, in_features)
        # output shape: (batch_size, n_parallel, out_features, 1)
        output = self.weight.unsqueeze(0) @ x.unsqueeze(3)
        output = output.squeeze(3) # (batch_size, n_parallel, out_features, 1) -> (batch_size, n_parallel, out_features)
        
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)
        
        # Remove batch dimension if input was single sample
        output = output.squeeze(0)
            
        return output
