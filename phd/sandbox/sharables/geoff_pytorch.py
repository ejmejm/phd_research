import random
from typing import Optional, Tuple

import torch
from torch import nn


class LTU(nn.Module):
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).to(x.dtype)


class NonlinearGEOFFTask:
    """Non-linear version of GEOFF task with configurable depth and activation."""
    
    def __init__(
        self,
        n_features: int,
        flip_rate: float,  # Percentage of weights to flip per step
        n_layers: int = 2,
        n_stationary_layers: int = 0,
        hidden_dim: int = 64,
        weight_scale: float = 1.0,
        activation: str = 'ltu',
        sparsity: float = 0.0,
        weight_init: str = 'binary',
        input_mean_range: Tuple[float, float] = (0, 0),
        input_std_range: Tuple[float, float] = (1, 1),
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of input features
            flip_rate: Percentage of weights to flip per step (accumulates if less than 1 weight)
            n_layers: Number of layers in the target network (1 = linear)
            n_stationary_layers: Number of layers that do not flip (stationary layers start from the first layer)
            hidden_dim: Hidden dimension size for intermediate layers
            weight_scale: Scale factor for weights (weights will be ±scale)
            activation: Activation function ('ltu', 'relu', 'tanh', or 'sigmoid')
            sparsity: Percentage of weights (other than in the last layer) to set to zero
            weight_init: Weight initialization method ('binary', 'lecun_uniform', or 'kaiming_uniform')
            input_mean_range: Each feature is sampled from a different normal distribution. The means of each feature's
                distribution are sampled from this range. Defaults to standard normal distribution.
            input_std_range: Each feature is sampled from a different normal distribution. The standard deviations of each feature's
                distribution are sampled from this range. Defaults to standard normal distribution.
            seed: Random seed for reproducibility
        """
        assert weight_init in ['binary', 'lecun_uniform', 'kaiming_uniform'], f"Unsupported weight initialization: {weight_init}"
        
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_stationary_layers = n_stationary_layers
        self.hidden_dim = hidden_dim
        self.weight_scale = weight_scale
        self.flip_rate = flip_rate
        self.activation = activation
        self.sparsity = sparsity
        self.weight_init = weight_init
        self.flip_accumulators = []  # Accumulate flip probability for each layer
        
        # Create a generator for all random behaviors
        self.generator = torch.Generator()
        if seed is None:
            seed = random.randint(0, 1000000000)
        self.generator.manual_seed(seed)
            
        # Set activation function
        if activation == 'relu':
            self.activation_fn = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = torch.nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = torch.nn.Sigmoid()
        elif activation == 'ltu':
            self.activation_fn = LTU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        if tuple(input_mean_range) != (0, 0) or tuple(input_std_range) != (1, 1):
            self.standard_input = False

            # Initialize input distribution parameters
            self.input_mean = torch.zeros(n_features)
            self.input_std = torch.ones(n_features)
            
            # Sample mean and std values uniformly from specified ranges
            self.input_mean.uniform_(*input_mean_range, generator=self.generator)
            self.input_std.uniform_(*input_std_range, generator=self.generator)
        else:
            self.standard_input = True

        # Initialize network weights
        self.weights = []
        
        if n_layers == 1:
            # For linear case, single layer mapping input to output
            layer_weights = self._initialize_weights(n_features, 1)
            self.weights.append(layer_weights)
            self.flip_accumulators.append(0.0)
        else:
            # Input layer
            layer_weights = self._initialize_weights(n_features, hidden_dim)
            self._sparsify_weights(layer_weights, sparsity)
            self.weights.append(layer_weights)
            
            # Calculate number of weights that can flip in first layer
            self.flip_accumulators.append(0.0)
            
            # Hidden layers
            for i in range(n_layers - 2):
                layer_weights = self._initialize_weights(hidden_dim, hidden_dim)
                self._sparsify_weights(layer_weights, sparsity)
                self.weights.append(layer_weights)
                
                # All weights can flip in hidden layers
                self.flip_accumulators.append(0.0)
            
            # Output layer
            output_weights = self._initialize_weights(hidden_dim, 1)
            self.weights.append(output_weights)
            
            # Output layer flippable weights
            self.flip_accumulators.append(0.0)
    
    def _initialize_weights(self, in_features: int, out_features: int) -> torch.Tensor:
        """Initialize weights based on specified initialization method.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            
        Returns:
            Initialized weight tensor
        """
        if self.weight_init == 'binary':
            weights = (torch.randint(0, 2, (in_features, out_features), generator=self.generator) * 2 - 1).float()
        elif self.weight_init == 'lecun_uniform':
            weights = torch.nn.init.kaiming_uniform_(
                torch.empty(in_features, out_features),
                mode = 'fan_in',
                nonlinearity = 'linear',
            )
        else: # kaiming_uniform
            weights = torch.nn.init.kaiming_uniform_(
                torch.empty(in_features, out_features),
                mode = 'fan_in',
                nonlinearity = 'relu',
            )
        return weights * self.weight_scale
    
    def _sparsify_weights(self, weights: torch.Tensor, sparsity: float):
        """Set a percentage of weights to zero."""
        if sparsity == 0:
            return
        n_zero = int(sparsity * weights.numel())
        flat_idx = torch.randperm(weights.numel(), generator=self.generator)[:n_zero]
        weights.view(-1)[flat_idx] = 0

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the target network."""
        with torch.no_grad():
            if self.n_layers == 1:
                return x @ self.weights[0]
                
            for i in range(self.n_layers - 1):
                x = x @ self.weights[i]
                x = self.activation_fn(x)
            return x @ self.weights[-1]

    def _flip_signs(self):
        """Flip signs of weights based on accumulated probabilities."""
        layer_offset = self.n_stationary_layers
        for layer_idx, (weights, accumulator) in enumerate(
            list(zip(self.weights, self.flip_accumulators))[layer_offset:],
        ):
            n_flips = int(accumulator)
            if n_flips > 0:
                # Randomly select weights to flip
                flat_idx = torch.randperm(weights.numel(), generator=self.generator)[:n_flips]
                weights.view(-1)[flat_idx] *= -1
                
                # Update accumulator
                self.flip_accumulators[layer_offset + layer_idx] -= n_flips
    
    def get_iterator(self, batch_size: int):
        """Returns an iterator that generates batches of data."""
        while True:
            
            # Accumulate and handle weight flips
            for i in range(self.n_stationary_layers, len(self.flip_accumulators)):
                if self.n_layers == 1:
                    n_flippable = self.n_features
                elif i == 0:
                    n_flippable = self.n_features * self.hidden_dim
                elif i == len(self.weights) - 1:
                    n_flippable = self.hidden_dim
                else:
                    n_flippable = self.hidden_dim * self.hidden_dim
                self.flip_accumulators[i] += self.flip_rate * n_flippable
            
            self._flip_signs()
            
            # Generate random input features
            x = torch.randn(batch_size, self.n_features, generator=self.generator)
            if not self.standard_input:
                x = x * self.input_std.unsqueeze(0) + self.input_mean.unsqueeze(0)
            
            # Forward pass through target network
            y = self._forward(x)
            
            yield x, y


class StandardizationStats(nn.Module):
    """Holds running statistics for standardization."""
    def __init__(
        self,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.register_buffer('running_mean', torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer('step', torch.tensor(0, device=device))
        self.gamma = gamma


@torch.no_grad()
def standardize_targets(
    targets: torch.Tensor,
    stats: StandardizationStats,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, StandardizationStats]:
    """Exponentially-weighted Welford normalisation (EW-Welford).

    Normalises a 2-D tensor of shape ``(batch, 1)`` so that its running mean
    approaches zero and its running standard deviation approaches one, while
    keeping **O(1)** state and compute per call.  Statistics adapt to concept
    drift via the forgetting factor ``gamma``.

    Args:
        targets: Input tensor of shape ``(batch, 1)`` on any device / dtype.
        stats: StandardizationStats object containing running statistics.
        eps: Small constant added for numerical stability; safeguards against
            division by zero and negative variance caused by round-off.

    Returns:
        Tuple containing:
            - **z** (*torch.Tensor*): Normalised tensor with the same shape as
              ``targets``.
            - **new_stats** (*StandardizationStats*): Updated running statistics.

    Example:
        ```python
        stats = StandardizationStats(gamma=0.99, device="cuda")

        for batch in data_stream:               # batch shape: (B, 1)
            batch = batch.cuda()
            z, stats = standardize_targets(batch, stats)
            # ... use z for loss / back-prop ...
        ```
    """
    # --------------------------------------------------------------------- #
    # 1. Normalize the current batch using statistics **from the prev step**.
    # --------------------------------------------------------------------- #
    var_safe = stats.running_var.clamp_min(eps)  # ensure σ² ≥ eps
    std = torch.sqrt(var_safe)
    z = (targets - stats.running_mean) / std

    # --------------------------------------------------------------------- #
    # 2. Update running statistics with the batch mean (EW-Welford update).
    # --------------------------------------------------------------------- #
    alpha = 1.0 - stats.gamma                    # EW learning rate
    batch_mean = targets.mean()                  # scalar (dim == 1)
    delta = batch_mean - stats.running_mean
    stats.running_mean.add_(alpha * delta)       # μ_t

    delta2 = batch_mean - stats.running_mean
    stats.running_var.mul_(stats.gamma).add_(alpha * delta * delta2)

    # Numerical hygiene: clamp and squash accidental NaNs.
    stats.running_var.clamp_min_(eps)
    if torch.isnan(stats.running_var):
        stats.running_var.fill_(eps)

    stats.step.add_(1)
    return z, stats


### Example usage ###

if __name__ == '__main__':
    seed = 251103
    use_target_standardization = True
    
    # Create the GEOFF task
    task = NonlinearGEOFFTask(
        n_features = 20,
        flip_rate = 0.0,
        n_layers = 2,
        n_stationary_layers = 0, # Set to 1 if you want to freeze features to test feature search
        hidden_dim = 20,
        activation = 'ltu',
        weight_init = 'binary',
        seed = seed,
    )
    
    # The above creates a GEOFF task with all binary weights (input and output weights)
    # In my experiments, I use binary input weights but LeCun uniform output weights, which
    # is what I am changing here:
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed)
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    
    # In my experiments, I normalize the targets with Welford's algorithm.
    # Experiments should still work without this, but I do it so that the range
    # of the outputs does not change significantly if the number of hidden units
    # in the GEOFF task changes.
    target_stats = StandardizationStats(gamma=0.99)
    
    # Iterate and print the first few samples
    data_iterator = task.get_iterator(batch_size=1)
    for step in range(5):
        x, y = next(data_iterator)
        
        if use_target_standardization:
            y, target_stats = standardize_targets(y, target_stats)
        
        print(f"\nStep {step}:")
        print(f"\tInput: {x.squeeze()}")
        print(f"\tOutput: {y.squeeze()}")
            
        # Perform learning update here