import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import ACTIVATION_MAP, initialize_layer_weights
from phd.research_utils.weight_init import n_kaiming_uniform
from ..idbd import IDBD


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
        hidden_dim: int,
        ensemble_dim: int,
        n_ensemble_members: int,
        weight_init_method: str,
        activation: str = 'tanh',
        n_frozen_layers: int = 0,
        utility_decay: float = 0.99,
        seed: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Number of hidden units for the ensembles to pull from
            ensemble_dim: Hidden units per ensemble member
            n_ensemble_members: Number of ensemble members
            weight_init_method: How to initialize the first layer weights ('zeros', 'kaiming', or 'binary')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            n_frozen_layers: Number of frozen layers
            utility_decay: Decay rate for the utility tracking
            seed: Optional random seed for reproducibility
        """
        assert n_layers == 2, "Ensemble MLPs only support a two layers!"
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble_members = n_ensemble_members
        self.n_frozen_layers = n_frozen_layers
        self.utility_decay = utility_decay
        
        # Create a generator if seed is provided
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None
        
        self.activation = ACTIVATION_MAP[activation]()
        
        # Build layers
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=False)
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
        
        self.ensemble_utilities = torch.zeros(n_ensemble_members, dtype=torch.float32)
        self.feature_utilities = torch.zeros(hidden_dim, dtype=torch.float32)
        
        # Maps the input features to each ensemble member, shape: (n_ensemble_members, ensemble_dim)
        
        self.ensemble_input_ids = torch.stack([
            torch.randperm(hidden_dim, generator=self.generator)[:ensemble_dim]
            for _ in range(n_ensemble_members)
        ])
        
        # TODO: Because this did so well, try all of them exactly once but in random other
        #       Then try all of them exactly twice in two different random orders
        # TODO: I think there is a bug here because this doesn't work but the arange does
        #       Find and fix it
        
        # reverse = True
        
        # if reverse:
        #     # Definitely a bug because this doesn't do the same as the above
        #     self.ensemble_input_ids = torch.arange(
        #         n_ensemble_members * ensemble_dim - 1, -1, -1,
        #         dtype=torch.long,
        #     ).reshape(n_ensemble_members, ensemble_dim)
        # else:
        #     self.ensemble_input_ids = torch.arange(
        #         0, n_ensemble_members * ensemble_dim,
        #         dtype=torch.long,
        #     ).reshape(n_ensemble_members, ensemble_dim)
    
        # self.ensemble_input_ids = torch.cat([torch.randperm(
        #     40 * ensemble_dim, dtype=torch.long, generator=self.generator,
        # ).reshape(40, ensemble_dim) for _ in range(5)], dim=0)
        

        self.update_step = 0

        # Trace the target to allow for normalizing the utilities if necessary
        self.register_buffer('target_trace', torch.zeros(1, dtype=torch.float32))
    
    
    def _get_ensemble_input_features(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Get the input features for each ensemble member.
        
        Args:
            hidden_features: The hidden features from the input layer,
                Shape: (batch_size, hidden_dim)
                
        Returns:
            The input features for each ensemble member,
                Shape: (batch_size, n_ensemble_members * ensemble_dim)
        """
        return hidden_features[:, self.ensemble_input_ids.ravel()]
    
    def _convert_hidden_idxs_to_ensemble_idxs(self, hidden_idxs: List[int]) -> Tuple[List[int], List[int]]:
        """Convert hidden dim indices to the indices of features used in the ensemble inputs.
        
        Args:
            hidden_idxs: List of hidden dim indices
        
        Returns:
            The indices of the ensemble inputs that use the hidden features,
                Shape: (n_ensemble_members, ensemble_dim)
        """
        # Convert these hidden dim indices to their indices in the ensemble inputs to know which weights to reinitialize
        hidden_features_mask = torch.zeros(self.hidden_dim, dtype=torch.bool, device=self.ensemble_input_ids.device)
        hidden_features_mask[hidden_idxs] = True
        prune_ensemble_inputs_mask = hidden_features_mask[self.ensemble_input_ids.ravel()]
        prune_ensemble_inputs_idxs = prune_ensemble_inputs_mask.nonzero(as_tuple=False).squeeze(1)
        ensemble_idxs, feature_idxs = np.unravel_index(
            prune_ensemble_inputs_idxs, (self.n_ensemble_members, self.ensemble_dim))
        return ensemble_idxs, feature_idxs
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        update_state: bool = False,
    ) -> Tuple[torch.Tensor, Dict[nn.Module, torch.Tensor], Dict[str, Any]]:
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
            target: Target tensor of shape (batch_size, out_features)
            update_state: Whether to update the state of the model (for utility tracking)
            
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
        
        # Get the input features for each ensemble member
        ensemble_input_features = self._get_ensemble_input_features(hidden_features)
        ensemble_input_features = ensemble_input_features.view(
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
        
        # Calculate losses if applicable
        if target is not None:
            loss = torch.mean((target - prediction) ** 2)
            aux['loss'] = loss
            
            # Update utility
            if update_state:
                with torch.no_grad():
                    # Measure how much each ensemble is reducing the loss
                    # Shape: (batch_size, n_ensemble_members, output_dim)
                    prediction_errors = target.unsqueeze(1) - ensemble_predictions
                    
                    # TODO: Given the signed utility didn't work well for features here,
                    #       maybe just try the negative of the loss per ensemble for ensemble utilities?
                    ensemble_utilities = torch.abs(target.unsqueeze(1)) - torch.abs(prediction_errors)
                    ensemble_utilities = ensemble_utilities.sum(dim=2).mean(0) # Shape: (n_ensemble_members,)
                    
                    # Need to update this if there is more than one prediction
                    assert self.prediction_layer.weight.shape[1] == 1, \
                        "Only one prediction is supported for now!"
                    
                    # Measure how much each feature is reducing the loss within its ensemble
                    # Features shape: (batch, n_ensemble_members, ensemble_dim)
                    # Weights shape: (n_ensemble_members, out_dim, ensemble_dim) -> (1, n_ensemble_members, ensemble_dim)
                    feature_contribs = ensemble_input_features * self.prediction_layer.weight.squeeze(1).unsqueeze(0)
                    
                    # Calculate how much much each feature individually contributed to decreasing the loss
                    # (e.g. if the feature was 0, how much worse would the error be?)
                    # Shape: (batch_size, n_ensemble_members, ensemble_dim)
                    
                    # Signed utility version, did not work well in testing:
                    # feature_utilities = (
                    #     torch.abs(prediction_errors + feature_contribs) - \
                    #     torch.abs(prediction_errors)
                    # )
                    
                    # CBP utility version:
                    # TODO: Back compute utilities to their original features
                    # feature_utilities = torch.abs(feature_contribs).mean(dim=0).ravel()
                    
                    
                    
                    
                    # Shape: (batch_size, n_ensemble_members, ensemble_dim) -> (n_ensemble_members, ensemble_dim)
                    ensemble_input_utilities = torch.abs(feature_contribs).mean(dim=0)
                    feature_utilities = torch.zeros(
                        (self.n_ensemble_members, self.hidden_dim),
                        dtype=torch.float32, device=feature_contribs.device,
                    )
                    mask = torch.zeros_like(feature_utilities, dtype=torch.bool)
                    for i in range(self.n_ensemble_members):
                        feature_utilities[i, self.ensemble_input_ids[i]] = ensemble_input_utilities[i]
                        mask[i, self.ensemble_input_ids[i]] = True
                    
                    # TODO: Also try max here or median???
                    
                    # Option 1:
                    feature_utilities = feature_utilities.sum(dim=0) / mask.sum(dim=0)
                    feature_utilities = torch.nan_to_num(feature_utilities, nan=0.0)
                    
                    # Option 2:
                    # feature_utilities = feature_utilities.max(dim=0).values
                    
                    
                    
                    # feature_utilities[self.ensemble_input_ids.ravel()] = ensemble_input_utilities
                    
                    
                    self.ensemble_utilities = (
                        self.utility_decay * self.ensemble_utilities +
                        (1 - self.utility_decay) * ensemble_utilities
                    )
                    self.feature_utilities = (
                        self.utility_decay * self.feature_utilities +
                        (1 - self.utility_decay) * feature_utilities
                    )
                    self.target_trace = (
                        self.utility_decay * self.target_trace +
                        (1 - self.utility_decay) * torch.abs(target).mean()
                    )
                
                self.update_step += 1
        
        return prediction, param_inputs, aux


def _reinit_input_weights(model: EnsembleMLP, idxs: List[int], init_type: str):
    """Reinitialize the weights of the input layer after pruning.
    
    Args:
        model: The model to prune
        prune_idxs: List of hidden units to prune
        init_type: The type of initialization to use {kaiming_uniform, binary}
    """
    if init_type == 'kaiming_uniform':
        weight_data = model.input_layer.weight.data
        model.input_layer.weight.data[idxs] = n_kaiming_uniform(
            weight_data, weight_data[idxs].shape, a=math.sqrt(5), generator=model.generator)
    elif init_type == 'binary':
        weight_data = model.input_layer.weight.data
        model.input_layer.weight.data[idxs] = torch.randint(
            0, 2,
            weight_data[idxs].shape,
            device = model.input_layer.weight.device,
            generator = model.generator
        ).float() * 2 - 1
    else:
        raise ValueError(f'Invalid weight initialization: {init_type}!')


def _reinit_output_weights(model: EnsembleMLP, idxs: Tuple[List[int], List[int]], init_type: str):
    """Reinitialize the weights of the output layer after pruning.
    
    Args:
        model: The model to prune
        idxs: Tuple of (List of ensemble indices, List of feature indices)
        init_type: The type of initialization to use {zeros, kaiming_uniform}
    """
    # This is how linear layers are initialized in PyTorch
    weight_data = model.prediction_layer.weight.data
    ensemble_idxs, feature_idxs = idxs
    
    if init_type == 'zeros':
        model.prediction_layer.weight.data[ensemble_idxs, :, feature_idxs] = torch.zeros_like(
            weight_data[ensemble_idxs, :, feature_idxs])
    elif init_type == 'kaiming_uniform':
        model.prediction_layer.weight.data[ensemble_idxs, :, feature_idxs] = n_kaiming_uniform(
            weight_data,
            weight_data[ensemble_idxs, :, feature_idxs].shape,
            a = math.sqrt(5),
            generator = model.generator,
        )
    else:
        raise ValueError(f'Invalid weight initialization: {init_type}!')


def _reset_input_optim_state(model: EnsembleMLP, optimizer: optim.Optimizer, idxs: List[int]):
    """Reset the optimizer state of the input layer after pruning."""
    optim_state = optimizer.state[model.input_layer.weight]
        
    for key, value in optim_state.items():
        if value.shape == model.input_layer.weight.shape:
            if value is not None:
                optim_state[key][idxs, :] = 0
        else:
            warnings.warn(
                f"Cannot reset optimizer state for {key} of layer '{model.input_layer}' because shapes do not match, "
                f"parameter: {model.input_layer.weight.shape}, state value: {value.shape}"
            )
            
    if isinstance(optimizer, IDBD) and 'beta' in optim_state:
        optim_state['beta'][idxs, :] = math.log(optimizer.init_lr)


def _reset_output_optim_state(model: EnsembleMLP, optimizer: optim.Optimizer, idxs: Tuple[List[int], List[int]]):
    """Reset the optimizer state of the output layer after pruning.
    
    Args:
        model: The model to prune
        optimizer: The optimizer that holds the optimization state for the given model
        idxs: Tuple of (List of ensemble indices, List of feature indices)
    """
    optim_state = optimizer.state[model.prediction_layer.weight]
    
    ensemble_idxs, feature_idxs = idxs
    
    for key, value in optim_state.items():
        if value.shape == model.prediction_layer.weight.shape:
            if value is not None:
                optim_state[key][ensemble_idxs, :, feature_idxs] = 0
        else:
            warnings.warn(
                f"Cannot reset optimizer state for {key} of layer '{model.prediction_layer}' because shapes do not match, "
                f"parameter: {model.prediction_layer.weight.shape}, state value: {value.shape}"
            )
    
    if isinstance(optimizer, IDBD) and 'beta' in optim_state:
        optim_state['beta'][ensemble_idxs, :, feature_idxs] = math.log(optimizer.init_lr)


def prune_features(
    model: EnsembleMLP,
    optimizer: optim.Optimizer,
    prune_idxs: List[int],
    input_init_type: str = 'binary',
    output_init_type: str = 'zeros',
):
    """Prune features from the model.
    
    Args:
        model: The model to prune
        optimizer: The optimizer that holds the optimization state for the given model
        prune_idxs: List of hidden units to prune
        input_init_type: The type of initialization to use for the input weights {binary, kaiming_uniform}
        output_init_type: The type of initialization to use for the output weights {zeros, kaiming_uniform}
    """
    if prune_idxs is None or len(prune_idxs) == 0:
        return
    
    ensemble_prune_idxs = model._convert_hidden_idxs_to_ensemble_idxs(prune_idxs)
    
    _reinit_input_weights(model, prune_idxs, input_init_type)
    _reinit_output_weights(model, ensemble_prune_idxs, output_init_type)
    
    _reset_input_optim_state(model, optimizer, prune_idxs)
    _reset_output_optim_state(model, optimizer, ensemble_prune_idxs)
    
    median_utility = model.feature_utilities.median()
    model.feature_utilities[prune_idxs] = median_utility


def prune_ensembles(
    model: EnsembleMLP,
    optimizer: optim.Optimizer,
    prune_idxs: List[int],
    output_init_type: str = 'zeros',
):
    """Prune ensembles from the model.
    
    Args:
        model: The model to prune
        optimizer: The optimizer that holds the optimization state for the given model
        prune_idxs: List of ensembles to prune
        output_init_type: The type of initialization to use for the output weights {zeros, kaiming_uniform}
    """
    if prune_idxs is None or len(prune_idxs) == 0:
        return

    # Get the indices of the output weights that need to be reset
    device = model.ensemble_input_ids.device
    ensemble_prune_idxs = torch.tensor(prune_idxs, dtype=torch.long, device=device)
    ensemble_prune_idxs = ensemble_prune_idxs.repeat_interleave(model.ensemble_dim)
    
    feature_prune_idxs = torch.arange(model.ensemble_dim, dtype=torch.long, device=device)
    feature_prune_idxs = feature_prune_idxs.repeat(len(prune_idxs))
    
    full_reset_ids = (ensemble_prune_idxs, feature_prune_idxs)
    
    # Reset the output weights and optimizer states for the ensembles being pruned
    _reinit_output_weights(model, full_reset_ids, output_init_type)
    _reset_output_optim_state(model, optimizer, full_reset_ids)
    
    # Randomize input ids for the ensembles that are being pruned
    for ensemble_idx in prune_idxs:
        model.ensemble_input_ids[ensemble_idx] = torch.randperm(
            model.hidden_dim, generator=model.generator)[:model.ensemble_dim]
    
    # Reset utilities for the ensembles that are being pruned
    median_utility = model.ensemble_utilities.median()
    model.ensemble_utilities[prune_idxs] = median_utility