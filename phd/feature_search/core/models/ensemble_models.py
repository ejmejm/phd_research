import math
from typing import Any, Dict, Optional, Tuple

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
        has_batch_dim = x.dim() == 3
        if not has_batch_dim:
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
        if not has_batch_dim:
            output = output.squeeze(0)
            
        return output


class EnsembleMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        n_layers: int,
        ensemble_dim: int,
        n_ensemble_members: int,
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
            ensemble_dim: Hidden units per ensemble member
            n_ensemble_members: Number of ensemble members
            weight_init_method: How to initialize the first layer weights ('zeros', 'kaiming', or 'binary')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            n_frozen_layers: Number of frozen layers
            seed: Optional random seed for reproducibility
        """
        assert n_layers == 2, "Ensemble MLPs only support a two layers!"
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim
        self.n_ensemble_members = n_ensemble_members
        self.n_frozen_layers = n_frozen_layers
        
        # Create a generator if seed is provided
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None
        
        self.activation = ACTIVATION_MAP[activation]()
        
        # Build layers
        self.hidden_dim = ensemble_dim * n_ensemble_members
        self.input_layer = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.prediction_layer = MultipleLinear(
            self.ensemble_dim, output_dim, self.n_ensemble_members,
            bias=False, generator=self.generator,
        )
        
        # Initialize weights
        initialize_layer_weights(self.input_layer, weight_init_method, self.generator)
        
        # Freeze layers as necessary
        if n_frozen_layers >= 1:
            self.input_layer.weight.requires_grad = False
        if n_frozen_layers == 2:
            self.prediction_layer.weight.requires_grad = False
        if n_frozen_layers > 2:
            raise ValueError("EnsembleMLP only supports up to 2 frozen layers!")
        
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[nn.Module, torch.Tensor], Dict[str, Any]]:
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
            target: Target tensor of shape (batch_size, out_features)
            
        Returns:
            - Prediction tensor of shape (batch_size, output_dim)
            - Dictionary of input values for each layer
            - Dictionary of auxiliary information
        """
        assert x.dim() == 2, "Input must be a 2D tensor"
        assert target is None or target.dim() == 2, "Target must be a 2D tensor"
        
        aux = {}
        
        # Input layer
        hidden_features = self.input_layer(x) # (batch_size, hidden_dim)
        hidden_features = self.activation(hidden_features)
        ensemble_input_features = hidden_features.view(
            x.shape[0], self.n_ensemble_members, self.ensemble_dim,
        ) # (batch_size, n_ensemble_members, ensemble_dim)
        
        param_inputs = {
            self.input_layer.weight: x,
            self.prediction_layer.weight: ensemble_input_features,
        }
        
        # Make predictions
        # Output shape: (batch_size, n_ensemble_members, output_dim)
        ensemble_predictions = self.prediction_layer(ensemble_input_features)
        aux['ensemble_predictions'] = ensemble_predictions
        
        prediction = ensemble_predictions.mean(dim=1) # (batch_size, output_dim)
        
        # Straight-through estimator for each ensemble member
        prediction_sum = ensemble_predictions.sum(dim=1) # (batch_size, output_dim)
        prediction = prediction.detach() + (prediction_sum - prediction_sum.detach())
        
        return prediction, param_inputs, aux
