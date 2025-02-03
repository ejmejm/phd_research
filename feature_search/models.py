import torch
import torch.nn as nn


ACTIVATION_MAP = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
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
        device: str = 'cuda'
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros' or 'kaiming')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            device: Device to put model on
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Build layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.activation = ACTIVATION_MAP[activation]
        
        # Initialize weights
        self._initialize_weights(weight_init_method)
    
    def _initialize_weights(self, method: str):
        """Initialize weights according to specified method."""
        layer = self.layers[0]
        if method == 'zeros':
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f'Invalid weight initialization method: {method}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_inputs = {}
        for layer in self.layers[:-1]:
            param_inputs[layer.weight] = x
            x = self.activation(layer(x))
        param_inputs[self.layers[-1].weight] = x
        return self.layers[-1](x), param_inputs
    
    def get_first_layer_weights(self) -> torch.Tensor:
        """Returns the weights of the first layer for utility calculation."""
        return self.layers[0].weight
