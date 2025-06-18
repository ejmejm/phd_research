import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
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
        prediction_mode: str = 'mean',
        feature_utility_mode: str = 'mean',
        ensemble_utility_mode: str = 'objective_improvement',
        ensemble_feature_selection_method: str = 'random',
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
            prediction_mode: Mode for making predictions {mean, mean_top_half, median, median_top_half, max}
            feature_utility_mode: Mode for calculating feature utilities {mean, median, best}
            ensemble_utility_mode: Mode for calculating ensemble utilities {objective_improvement, negative_loss}
            ensemble_feature_selection_method: Method for selecting features for the ensembles {random, inverse_frequency}
            seed: Optional random seed for reproducibility
        """
        assert n_layers == 2, "Ensemble MLPs only support a two layers!"
        assert ensemble_feature_selection_method in ['random', 'inverse_frequency'], \
            f"Invalid ensemble feature selection method: {ensemble_feature_selection_method}!"
        assert prediction_mode in ['mean', 'mean_top_half', 'median', 'median_top_half', 'best_ensemble'], \
            f"Invalid prediction mode: {prediction_mode}!"
        assert feature_utility_mode in ['mean', 'median', 'best'], \
            f"Invalid feature utility mode: {feature_utility_mode}!"
        assert ensemble_utility_mode in ['objective_improvement', 'negative_loss'], \
            f"Invalid ensemble utility mode: {ensemble_utility_mode}!"
        assert ensemble_dim <= hidden_dim, \
            f"Ensemble dimension must be less than or equal to hidden dimension!"
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble_members = n_ensemble_members
        self.n_frozen_layers = n_frozen_layers
        self.utility_decay = utility_decay
        self.prediction_mode = prediction_mode
        self.feature_utility_mode = feature_utility_mode
        self.ensemble_utility_mode = ensemble_utility_mode
        self.ensemble_feature_selection_method = ensemble_feature_selection_method
        
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
        
        self.register_buffer(
            'ensemble_utilities', torch.zeros(n_ensemble_members, dtype=torch.float32))
        self.register_buffer(
            'feature_utilities', torch.zeros(hidden_dim, dtype=torch.float32))
        
        # Maps the input features to each ensemble member, shape: (n_ensemble_members, ensemble_dim)
        
        self.ensemble_input_ids = torch.full(
            (n_ensemble_members, ensemble_dim), fill_value=-1, dtype=torch.long)

        for i in range(n_ensemble_members):
            self._reinit_ensemble_input_ids(i)
        
        # Gives a 1:1 hidden dim to ensemble input mapping
        # self.ensemble_input_ids = torch.arange(
        #     0, n_ensemble_members * ensemble_dim,
        #     dtype=torch.long,
        # ).reshape(n_ensemble_members, ensemble_dim)
        
        self.update_step = 0

        # Trace the target to allow for normalizing the utilities if necessary
        self.register_buffer('target_trace', torch.zeros(1, dtype=torch.float32))
        
    def get_device(self):
        return self.ensemble_utilities.device
    
    def _reinit_ensemble_input_ids(self, ensemble_idx: int):
        """Reinitialize the input ids for a given ensemble.
        
        Args:
            ensemble_idx: The index of the ensemble to reinitialize
        """
        if self.ensemble_feature_selection_method == 'random':
            self.ensemble_input_ids[ensemble_idx] = torch.randperm(
                self.hidden_dim, generator=self.generator)[:self.ensemble_dim].to(self.get_device())
            
        elif self.ensemble_feature_selection_method == 'inverse_frequency':
            feature_frequencies = torch.zeros(self.hidden_dim)
            flat_ensemble_input_ids = self.ensemble_input_ids.ravel()
            for hidden_idx in flat_ensemble_input_ids:
                if hidden_idx >= 0:
                    feature_frequencies[hidden_idx] += 1
                    
            feature_probs = softmax(-feature_frequencies, dim=0)
            self.ensemble_input_ids[ensemble_idx] = torch.multinomial(
                feature_probs, self.ensemble_dim, generator=self.generator).to(self.get_device())
        
        else:
            raise ValueError(f"Invalid ensemble feature selection method: {self.ensemble_feature_selection_method}!")
    
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
    
    def _merge_predictions(self, ensemble_predictions: torch.Tensor) -> torch.Tensor:
        """Make a prediction based on the individual ensemble predictions.
        
        Args:
            ensemble_predictions: The predictions from the ensemble,
                Shape: (batch_size, n_ensemble_members, output_dim)
        """
        if self.prediction_mode == 'mean':
            return ensemble_predictions.mean(dim=1)
        
        # Sort predictions based off of the ensemble utilities before taking the mean
        elif self.prediction_mode == 'mean_top_half':
            best_ensemble_idxs = torch.argsort(
                self.ensemble_utilities, descending=True)[:self.n_ensemble_members // 2]
            return ensemble_predictions[:, best_ensemble_idxs, :].mean(dim=1)
        
        elif self.prediction_mode == 'median':
            return ensemble_predictions.median(dim=1)
        
        # Sort predictions based off of the ensemble utilities before taking the median
        elif self.prediction_mode == 'median_top_half':
            best_ensemble_idxs = torch.argsort(
                self.ensemble_utilities, descending=True)[:self.n_ensemble_members // 2]
            return ensemble_predictions[:, best_ensemble_idxs, :].median(dim=1)
        
        # Take the prediction of the ensemble with the highest utility
        elif self.prediction_mode == 'best_ensemble':
            best_ensemble_idx = torch.argsort(self.ensemble_utilities, descending=True)[0]
            return ensemble_predictions[:, best_ensemble_idx, :]
    
        else:
            raise ValueError(f"Invalid prediction mode: {self.prediction_mode}!")
    
    def _calculate_ensemble_utilities(self, ensemble_errors: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the utilities for each ensemble.
        
        Args:
            ensemble_errors: The errors for each ensemble,
                Shape: (batch_size, n_ensemble_members, output_dim)
            target: The target tensor,
                Shape: (batch_size, output_dim)
        
        Returns:
            The utilities for each ensemble,
                Shape: (n_ensemble_members,)
        """
        if self.ensemble_utility_mode == 'objective_improvement':
        # TODO: Given the signed utility didn't work well for features here,
        #       maybe just try the negative of the loss per ensemble for ensemble utilities?
            ensemble_utilities = torch.abs(target.unsqueeze(1)) - torch.abs(ensemble_errors)
            ensemble_utilities = ensemble_utilities.sum(dim=2).mean(0) # Shape: (n_ensemble_members,)
            
        elif self.ensemble_utility_mode == 'negative_loss':
            ensemble_utilities = -torch.abs(ensemble_errors)
            ensemble_utilities = ensemble_utilities.sum(dim=2).mean(0) # Shape: (n_ensemble_members,)
            
        else:
            raise ValueError(f"Invalid ensemble utility mode: {self.ensemble_utility_mode}!")
        
        return ensemble_utilities

    def _calculate_feature_utilities(self, feature_contribs: torch.Tensor) -> torch.Tensor:
        """Calculate the utilities for each feature.
        
        Args:
            feature_contribs: The contributions of each feature to the predictions,
                Shape: (batch_size, n_ensemble_members, ensemble_dim)
        
        Returns:
            The utilities for each feature,
                Shape: (hidden_dim,)
        """
        # TODO: This didn't work well before, but I have since found a major bug that could
        #       have been the cause. Retry this.
        # Signed utility version, did not work well in testing:
        # feature_utilities = (
        #     torch.abs(prediction_errors + feature_contribs) - \
        #     torch.abs(prediction_errors)
        # )
        
        # Shape: (batch_size, n_ensemble_members, ensemble_dim) -> (n_ensemble_members, ensemble_dim)
        ensemble_input_utilities = torch.abs(feature_contribs).mean(dim=0)
        
        # Note that this could result in a very sparse tensor and use a lot of resources as we scale
        # I may need a more efficient way to do this in the future
        
        # More efficient replacement for the loop
        ensemble_indices = torch.arange(self.n_ensemble_members, device=feature_contribs.device)
        ensemble_indices = ensemble_indices.unsqueeze(1).expand(-1, self.ensemble_dim)
        feature_utilities = torch.full(
            (self.n_ensemble_members, self.hidden_dim),
            fill_value=torch.nan,
            dtype=torch.float32, device=feature_contribs.device,
        )
        feature_utilities[ensemble_indices, self.ensemble_input_ids] = ensemble_input_utilities
        
        if self.feature_utility_mode == 'mean':
            feature_utilities = feature_utilities.nanmean(dim=0)
            feature_utilities = torch.nan_to_num(feature_utilities, nan=0.0)
            
        elif self.feature_utility_mode == 'median':
            feature_utilities = feature_utilities.nanquantile(0.5, dim=0)
            feature_utilities = torch.nan_to_num(feature_utilities, nan=0.0)
            
        elif self.feature_utility_mode == 'best':
            feature_utilities = torch.nan_to_num(feature_utilities, nan=0.0)
            feature_utilities = feature_utilities.max(dim=0).values
            
        else:
            raise ValueError(f"Invalid feature utility mode: {self.feature_utility_mode}!")
        
        return feature_utilities
    
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
        
        prediction = self._merge_predictions(ensemble_predictions) # (batch_size, output_dim)
        
        # Calculate losses if applicable
        if target is not None:
            loss = torch.mean((target - prediction) ** 2)
            ensemble_errors = target.unsqueeze(1) - ensemble_predictions # Shape: (batch_size, n_ensemble_members, output_dim)
            aux['loss'] = loss
            aux['ensemble_losses'] = (ensemble_errors ** 2).mean(dim=2).mean(dim=0) # Shape: (n_ensemble_members,)
            
            # Update utility
            if update_state:
                with torch.no_grad():
                    # Measure utility as a proxy for how much this ensemble is reducing the loss
                    ensemble_utilities = self._calculate_ensemble_utilities(ensemble_errors, target)
                    
                    # Need to update this if there is more than one prediction
                    assert self.prediction_layer.weight.shape[1] == 1, \
                        "Only one prediction is supported for now!"
                    
                    # Measure utility as a proxy estimate for how much each feature is reducing the loss within its ensemble
                    # Features shape: (batch, n_ensemble_members, ensemble_dim)
                    # Weights shape: (n_ensemble_members, out_dim, ensemble_dim) -> (1, n_ensemble_members, ensemble_dim)
                    feature_contribs = ensemble_input_features * self.prediction_layer.weight.squeeze(1).unsqueeze(0)
                    feature_utilities = self._calculate_feature_utilities(feature_contribs) # Output shape: (hidden_dim,)
                    
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
            weight_data, weight_data[idxs].shape, a=math.sqrt(5), generator=model.generator,
        ).to(model.get_device())
    elif init_type == 'binary':
        weight_data = model.input_layer.weight.data
        model.input_layer.weight.data[idxs] = torch.randint(
            0, 2,
            weight_data[idxs].shape,
            generator = model.generator
        ).to(model.get_device()).float() * 2 - 1
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
        ).to(model.get_device())
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
            model.hidden_dim, generator=model.generator)[:model.ensemble_dim].to(model.get_device())
    
    # Reset utilities for the ensembles that are being pruned
    median_utility = model.ensemble_utilities.median()
    model.ensemble_utilities[prune_idxs] = median_utility