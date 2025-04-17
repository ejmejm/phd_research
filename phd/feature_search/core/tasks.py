import random
from typing import Iterator, Optional, Tuple

import torch
from torch import nn
import numpy as np



class DummyTask:
    def __init__(self, feature_dim: int, n_classes: int, task_type: str = 'classification'):
        """Initialize data generator.
        
        Args:
            feature_dim: Number of input features
            n_classes: Number of classes for classification
            task_type: Type of task ('classification' or 'regression')
        """
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.task_type = task_type
        
        # Set distributions for each feature
        self.distributions = [random.choice(['uniform', 'normal']) 
                            for _ in range(feature_dim)]

    def get_iterator(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Get iterator that generates infinite batches of data."""
        while True:
            # Generate features
            features = []
            for dist in self.distributions:
                if dist == 'uniform':
                    feature = torch.rand(batch_size) * 2 - 1  # Uniform in [-1, 1]
                else:
                    feature = torch.randn(batch_size).clamp(-1, 1)  # Normal clamped to [-1, 1]
                features.append(feature)
            
            inputs = torch.stack(features, dim=1)
            
            # Generate targets
            if self.task_type == 'classification':
                targets = torch.randint(0, self.n_classes, (batch_size,))
            else:
                targets = torch.randn(batch_size)
            
            yield inputs, targets


class GEOFFTask:
    def __init__(
        self, 
        feature_dim: int = 20, 
        sign_flip_interval: int = 20,
        active_features: int = 5,
        seed: Optional[int] = None,
    ):
        """Initialize tracking task where target is sum of first k inputs with changing signs.
        
        Args:
            feature_dim: Number of input features (default 20)
            sign_flip_interval: How often to flip signs (in steps)
            active_features: Number of features that contribute to target (default 5)
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.sign_flip_interval = sign_flip_interval
        self.active_features = active_features
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Initialize signs randomly for active features (+1 or -1)
        self.signs = torch.randint(0, 2, (active_features,), generator=self.generator) * 2 - 1
        
        self.steps_since_flip = 0
        self.task_type = 'regression'
    
    def _randomize_sign(self):
        """Randomly flip the sign of one random feature."""
        # Choose random feature to flip
        feature_to_flip = torch.randint(0, self.active_features, (1,), generator=self.generator).item()
        # Flip its sign
        self.signs[feature_to_flip] *= -1
    
    def get_iterator(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Get iterator that generates infinite batches of data."""
        while True:
            # Generate all features from normal distribution
            inputs = torch.randn(batch_size, self.feature_dim, generator=self.generator)
            # Calculate target using only first k features with signs
            targets = (self.signs.unsqueeze(0) * inputs[:, :self.active_features]).sum(dim=1)
            
            # Update sign flip counter and flip if needed
            self.steps_since_flip += 1
            if self.sign_flip_interval > 0 and self.steps_since_flip >= self.sign_flip_interval:
                self._randomize_sign()
                self.steps_since_flip = 0
            
            yield inputs, targets.unsqueeze(1)


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
        activation: str = 'relu',
        sparsity: float = 0.0,
        weight_init: str = 'binary',
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of input features
            flip_rate: Percentage of weights to flip per step (accumulates if < 1 weight)
            n_layers: Number of layers in the target network (1 = linear)
            n_stationary_layers: Number of layers that do not flip
            hidden_dim: Hidden dimension size for intermediate layers
            weight_scale: Scale factor for weights (weights will be ±scale)
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            sparsity: Percentage of weights (other than the last layer) to set to zero
            weight_init: Weight initialization method ('binary' or 'kaiming_uniform')
            seed: Random seed for reproducibility
        """
        assert weight_init in ['binary', 'kaiming_uniform'], f"Unsupported weight initialization: {weight_init}"
        
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
        
        if seed is not None:
            torch.manual_seed(seed)
            
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
            

        # Initialize network weights
        self.weights = []
        
        if n_layers == 1:
            # For linear case, single layer mapping input to output
            layer_weights = self._initialize_weights(n_features, 1, is_last_layer=True)
            self.weights.append(layer_weights)
            self.flip_accumulators.append(flip_rate * n_features)
        else:
            # Input layer
            layer_weights = self._initialize_weights(n_features, hidden_dim)
            self._sparsify_weights(layer_weights, sparsity)
            self.weights.append(layer_weights)
            
            # Calculate number of weights that can flip in first layer
            n_flippable = n_features * hidden_dim
            self.flip_accumulators.append(flip_rate * n_flippable)
            
            # Hidden layers
            for i in range(n_layers - 2):
                layer_weights = self._initialize_weights(hidden_dim, hidden_dim)
                self._sparsify_weights(layer_weights, sparsity)
                self.weights.append(layer_weights)
                
                # All weights can flip in hidden layers
                n_flippable = hidden_dim * hidden_dim
                self.flip_accumulators.append(flip_rate * n_flippable)
            
            # Output layer
            output_weights = self._initialize_weights(hidden_dim, 1, is_last_layer=True)
            self.weights.append(output_weights)
            
            # Output layer flippable weights
            n_flippable = hidden_dim
            self.flip_accumulators.append(flip_rate * n_flippable)
    
    def _initialize_weights(self, in_features: int, out_features: int, is_last_layer: bool = False) -> torch.Tensor:
        """Initialize weights based on specified initialization method.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            
        Returns:
            Initialized weight tensor
        """
        if self.weight_init == 'binary':
            weights = (torch.randint(0, 2, (in_features, out_features)) * 2 - 1).float()
        else:  # kaiming_uniform
            weights = torch.nn.init.kaiming_uniform_(
                torch.empty(in_features, out_features),
                mode='fan_in',
                nonlinearity='relu'
            )
        return weights * self.weight_scale
    
    def _sparsify_weights(self, weights: torch.Tensor, sparsity: float):
        """Set a percentage of weights to zero."""
        if sparsity == 0:
            return
        n_zero = int(sparsity * weights.numel())
        flat_idx = torch.randperm(weights.numel())[:n_zero]
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
        for layer_idx, (weights, accumulator) in enumerate(
            zip(self.weights, self.flip_accumulators), start=self.n_stationary_layers,
        ):
            n_flips = int(accumulator)
            if n_flips > 0:
                # Randomly select weights to flip
                flat_idx = torch.randperm(weights.numel())[:n_flips]
                weights.view(-1)[flat_idx] *= -1
                
                # Update accumulator
                self.flip_accumulators[layer_idx] -= n_flips
    
    def get_iterator(self, batch_size: int):
        """Returns an iterator that generates batches of data."""
        while True:
            
            # Accumulate and handle weight flips
            for i in range(len(self.flip_accumulators)):
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
            x = torch.randn(batch_size, self.n_features)
            
            # Forward pass through target network
            y = self._forward(x)
            
            yield x, y
