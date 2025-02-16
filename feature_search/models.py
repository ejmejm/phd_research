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
        device: str = 'cuda'
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
            device: Device to put model on
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_frozen_layers = n_frozen_layers
        self.device = device
        
        activation_cls = ACTIVATION_MAP[activation]
        
        # Build layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.layers.append(activation_cls())
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(activation_cls())
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Freeze layers
        for i in range(n_frozen_layers):
            layer = self.layers[int(i*2)]
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        
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
        elif method == 'binary':
            layer.weight.data = torch.randint(0, 2, layer.weight.shape, device=layer.weight.device) * 2 - 1
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f'Invalid weight initialization method: {method}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
