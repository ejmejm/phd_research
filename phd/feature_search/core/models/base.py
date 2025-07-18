import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class LTU(nn.Module):
    def __init__(self, threshold: float = 0.0):
        """Linear threshold unit with sigmoid-like gradient."""
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applies step function but backward pass uses sigmoid gradient.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with 1s where x > 0 and 0s elsewhere
        """
        # Custom autograd function to override gradient
        class _LTUFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return (input > self.threshold).float()

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                # Sigmoid gradient is sigmoid(x) * (1 - sigmoid(x))
                grad_input = torch.sigmoid(input - self.threshold) * (1 - torch.sigmoid(input - self.threshold))
                return grad_output * grad_input

        return _LTUFunction.apply(x)


class ParallelLinear(nn.Module):
    """A linear layer that applies multiple weight matrices in parallel to the same input.
    
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
        
        # Reshape weight to (n_parallel * out_features, in_features) for efficient matmul
        self.weight = nn.Parameter(torch.empty(n_parallel * out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_parallel * out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using the same strategy as nn.Linear."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), generator=self.generator)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound, generator=self.generator)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_features) or (in_features,)
            
        Returns:
            Output tensor of shape (batch_size, n_parallel, out_features) or (n_parallel, out_features)
        """
        # Handle single sample case
        if x.dim() == 1:
            x = x.unsqueeze(0)
            expanded = True
        else:
            expanded = False
        
        # Apply parallel matrix multiplication
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
            
        # Reshape to (batch_size, n_parallel, out_features)
        batch_size = x.size(0)
        output = output.view(batch_size, self.n_parallel, self.out_features)
        
        # Remove batch dimension if input was single sample
        if expanded:
            output = output.squeeze(0)
            
        return output


ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'ltu': LTU,
}


class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        n_layers: int,
        hidden_dim: int,
        weight_init_method: str,
        activation: str = 'tanh',
        n_frozen_layers: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros', 'kaiming', or 'binary')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            n_frozen_layers: Number of frozen layers
            seed: Optional random seed for reproducibility
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_frozen_layers = n_frozen_layers
        
        # Create a generator if seed is provided
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None
        
        activation_cls = ACTIVATION_MAP[activation]
        
        # Build layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.layers.append(activation_cls())
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                self.layers.append(activation_cls())
            self.layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        
        # Freeze layers
        for i in range(n_frozen_layers):
            layer = self.layers[int(i*2)]
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        
        # Initialize weights
        initialize_layer_weights(self.layers[0], weight_init_method, self.generator)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
        param_inputs = {}
        for i in range(0, len(self.layers) - 2, 2):
            param_inputs[self.layers[i].weight] = x
            x = self.layers[i](x) # Linear layer
            x = self.layers[i + 1](x) # Activation

        param_inputs[self.layers[-1].weight] = x
        return self.layers[-1](x), param_inputs
    
    def get_first_layer_weights(self) -> torch.Tensor:
        """Returns the weights of the first layer for utility calculation."""
        return self.layers[0].weight

    
def initialize_layer_weights(
        layer: nn.Module, 
        method: str, 
        generator: Optional[torch.Generator] = None,
    ) -> None:
    """Initialize weights according to specified method."""
    if method == 'zeros':
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, generator=generator)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif method == 'binary':
        layer.weight.data = torch.randint(
            0, 2, layer.weight.shape, device=layer.weight.device, generator=generator).float() * 2 - 1
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError(f'Invalid weight initialization method: {method}')