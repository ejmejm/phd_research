from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
import torch.optim as optim

from adam import Adam
from idbd import IDBD, RMSPropIDBD
from models import MLP


EPSILON = 1e-8


@dataclass
class FeatureInfo:
    """Stores information about a feature including its utility and distribution parameters."""
    is_real: bool
    utility: float
    distribution_params: Dict[str, Any]
    last_update: int
    creation_step: int


class InputRecycler:
    def __init__(
        self,
        n_features: int,
        n_real_features: int,
        distractor_chance: float,
        recycle_rate: float,
        utility_decay: float,
        use_cbp_utility: bool,
        feature_protection_steps: int,
        sample_with_replacement: bool = False,
        std_normal_distractors_only: bool = False,
        n_start_real_features: int = -1,
        device: str = 'cuda',
    ):
        """
        Args:
            n_features: Total number of features model receives
            n_real_features: Number of real features available
            distractor_chance: Chance of selecting distractor vs real feature
            recycle_rate: How many features to recycle per step (can be fractional)
            utility_decay: Decay rate for feature utility
            use_cbp_utility: Whether to use CBP utility or random selection
            feature_protection_steps: Number of steps to protect new features
            n_start_real_features: When not -1, forces the the recycler to start with exactly this many real features
            device: Device to put tensors on
        """
        self.n_features = n_features
        self.n_real_features = n_real_features
        self.distractor_chance = distractor_chance
        self.recycle_rate = recycle_rate
        self.utility_decay = utility_decay
        self.use_cbp_utility = use_cbp_utility
        self.feature_protection_steps = feature_protection_steps
        self.sample_with_replacement = sample_with_replacement
        self.std_normal_distractors_only = std_normal_distractors_only
        self.n_start_real_features = n_start_real_features
        self.device = device
        
        self.recycle_accumulator = 0.0
        self.features = {}
        self._initialize_features()
        self.total_recycled = 0  # Add counter for total recycled features
    
    def _initialize_features(self):
        """Initialize the initial pool of features."""
        
        if self.n_start_real_features > 0:
            n_real = min(self.n_start_real_features, self.n_features)
            for i in range(n_real):
                self._add_new_feature(i, 0, force_real=True)
                
            n_remaining = max(0, self.n_features - n_real)
            for i in range(n_remaining):
                self._add_new_feature(n_real + i, 0, force_distractor=True)

        else:
            for i in range(self.n_features):
                self._add_new_feature(i, 0)
    
    def _add_new_feature(self, idx: int, step: int, force_real: bool = False, force_distractor: bool = False):
        """Add a new feature (real or distractor) at the given index."""
        if force_real:
            is_real = True
        elif force_distractor:
            is_real = False
        else:
            is_real = random.random() > self.distractor_chance
        
        # Get list of currently used feature indices
        used_indices = set([
            f.distribution_params['feature_idx'] 
            for f in self.features.values() 
            if f.is_real
        ]) if not self.sample_with_replacement else set()
            
        if is_real and len(used_indices) < self.n_real_features:
            # Get available indices
            available_indices = [
                i for i in range(self.n_real_features) 
                if i not in used_indices
            ]
            
            dist_params = {
                'type': 'real',
                'feature_idx': random.choice(available_indices)
            }
        else:
            is_real = False
            
            # 50% chance of uniform vs normal distribution
            if self.std_normal_distractors_only:
                dist_params = {
                    'type': 'distractor',
                    'distribution': 'normal',
                    'mean': 0.0,
                    'std': 1.0,
                }
            elif random.random() < 0.5:
                dist_params = {
                    'type': 'distractor',
                    'distribution': 'uniform',
                    'low': random.uniform(-1, 0),
                    'high': random.uniform(0, 1)
                }
            else:
                dist_params = {
                    'type': 'distractor',
                    'distribution': 'normal',
                    'mean': random.uniform(-0.5, 0.5),
                    'std': random.uniform(0.2, 0.5)
                }
        
        self.features[idx] = FeatureInfo(
            is_real=is_real,
            utility=0.0,
            distribution_params=dist_params,
            last_update=step,
            creation_step=step,
        )
    
    def _generate_feature_values(self, batch_size: int, real_features: torch.Tensor) -> torch.Tensor:
        """Generate feature values for the current feature set."""
        values = torch.zeros(batch_size, self.n_features, device=real_features.device)
        
        # Initialize arrays for distractor indices
        uniform_indices = []
        normal_indices = []
        uniform_lows = []
        uniform_highs = []
        normal_means = []
        normal_stds = []
        real_feature_sources = []
        real_feature_targets = []
        
        # Single loop to handle all features
        for i in range(self.n_features):
            if self.features[i].is_real:
                feature_idx = self.features[i].distribution_params['feature_idx']
                real_feature_sources.append(feature_idx)
                real_feature_targets.append(i)
            else:
                params = self.features[i].distribution_params
                if params['distribution'] == 'uniform':
                    uniform_indices.append(i)
                    uniform_lows.append(params['low'])
                    uniform_highs.append(params['high'])
                else:  # normal
                    normal_indices.append(i)
                    normal_means.append(params['mean'])
                    normal_stds.append(params['std'])
        values[:, real_feature_targets] = real_features[:, real_feature_sources]
        

        # Handle uniform distractors in batch
        if uniform_indices:
            lows = torch.tensor(uniform_lows)
            highs = torch.tensor(uniform_highs)
            
            uniform_values = torch.rand(batch_size, len(uniform_indices))
            uniform_values = uniform_values * (highs - lows) + lows
            values[:, uniform_indices] = uniform_values
            
        # Handle normal distractors in batch
        if normal_indices:
            means = torch.tensor(normal_means)
            stds = torch.tensor(normal_stds)
            
            eps = torch.randn(batch_size, len(normal_indices))
            normal_values = (means + eps * stds).clamp(-1, 1)
            values[:, normal_indices] = normal_values
            
        values = values.to(self.device)
        
        return values
    
    def _update_utilities(
        self, 
        feature_values: torch.Tensor, 
        first_layer_weights: torch.Tensor,
        step: int
    ):
        """Update utility values for all features."""
        if not self.use_cbp_utility:
            return
            
        weight_norms = torch.norm(first_layer_weights, p=1, dim=0)
        feature_impacts = torch.abs(feature_values) * weight_norms
        feature_impacts = feature_impacts.mean(dim=0).detach().cpu().numpy()
        old_utilities = np.array([self.features[i].utility for i in range(self.n_features)])
        new_utilities = self.utility_decay * old_utilities + (1 - self.utility_decay) * feature_impacts
        
        # Update running averages
        for i in range(self.n_features):
            self.features[i].utility = new_utilities[i]
            self.features[i].last_update = step
    
    def get_features_to_recycle(self, current_step: int) -> list:
        """Determine which features should be recycled this step."""
        self.recycle_accumulator += self.recycle_rate
        n_recycle = int(self.recycle_accumulator)
        self.recycle_accumulator -= n_recycle
        
        if n_recycle == 0:
            return []
        
        # Filter out protected features
        eligible_features = [
            i for i, f in self.features.items() 
            if current_step - f.creation_step >= self.feature_protection_steps
        ]
        
        if not eligible_features:
            return []
        
        if self.use_cbp_utility:
            utilities = {i: self.features[i].utility for i in eligible_features}
            return sorted(utilities.keys(), key=lambda x: utilities[x])[:n_recycle]
        else:
            return random.sample(eligible_features, min(n_recycle, len(eligible_features)))

    def get_statistics(self, current_step: int, model: MLP, optimizer: optim.Optimizer) -> dict:
        """Calculate statistics about current features."""
        real_features = [f for f in self.features.values() if f.is_real]
        distractor_features = [f for f in self.features.values() if not f.is_real]
        
        # Get indices of real and distractor features
        real_indices = [i for i, f in self.features.items() if f.is_real]
        distractor_indices = [i for i, f in self.features.items() if not f.is_real]
        
        # Get first layer weights and calculate l1 norms
        first_layer_weights = model.get_first_layer_weights()
        weight_norms = torch.norm(first_layer_weights, p=1, dim=0) / first_layer_weights.shape[0]
        
        stats = {
            'avg_lifespan_real': np.mean([current_step - f.creation_step for f in real_features]) if real_features else 0,
            'avg_lifespan_distractor': np.mean([current_step - f.creation_step for f in distractor_features]) if distractor_features else 0,
            'num_real_features': len(real_features),
            'num_distractor_features': len(distractor_features),
            'total_recycled_features': self.total_recycled,
            'mean_weight_norm_real': weight_norms[real_indices].mean().item() if real_indices else 0,
            'mean_weight_norm_distractor': weight_norms[distractor_indices].mean().item() if distractor_indices else 0
        }
        
        if self.use_cbp_utility:
            stats['mean_utility_real'] = np.mean([f.utility for f in real_features]) if real_features else 0
            stats['mean_utility_distractor'] = np.mean([f.utility for f in distractor_features]) if distractor_features else 0
        
        # Log step sizes for the inputs, but only if the first layer is not frozen
        first_layer_weights = model.layers[0].weight
        if len(optimizer.state[first_layer_weights]) > 0:
            
            if isinstance(optimizer, (IDBD, RMSPropIDBD)):
                idbd_beta = optimizer.state[first_layer_weights]['beta']
                learning_rates = torch.exp(idbd_beta).mean(dim=0)
                stats['mean_learning_rate_real'] = learning_rates[real_indices].mean().item() if real_indices else 0
                stats['mean_learning_rate_distractor'] = learning_rates[distractor_indices].mean().item() if distractor_indices else 0
                
            elif isinstance(optimizer, Adam):
                state = optimizer.state[first_layer_weights]
                
                # Get Adam parameters
                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = optimizer.defaults['betas']
                lr = optimizer.defaults['lr']
                eps = optimizer.defaults['eps']
                
                # Calculate bias corrections
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Calculate step size
                step_size = lr / bias_correction1
                
                # Calculate denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                
                # Calculate effective learning rates
                effective_lrs = (step_size / denom).mean(dim=0)
                
                stats['mean_learning_rate_real'] = effective_lrs[real_indices].mean().item() if real_indices else 0
                stats['mean_learning_rate_distractor'] = effective_lrs[distractor_indices].mean().item() if distractor_indices else 0
        
        return stats
    
    def step(
        self, 
        batch_size: int,
        real_features: torch.Tensor,
        first_layer_weights: torch.Tensor,
        step_num: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Perform one step of feature recycling and return feature values.
        
        Args:
            batch_size: Size of current batch
            real_features: Real feature values for this batch
            first_layer_weights: Weights from first layer of model
            step_num: Current training step
        
        Returns:
            Tensor of feature values to use for this step
            List of indices of features that were recycled
        """
        # Generate current feature values
        feature_values = self._generate_feature_values(batch_size, real_features)
        
        # Update utilities
        self._update_utilities(feature_values, first_layer_weights, step_num)
        
        # Update total recycled counter
        recycled_features = self.get_features_to_recycle(step_num)
        self.total_recycled += len(recycled_features)
        
        for idx in recycled_features:
            self._add_new_feature(idx, step_num)
        
        return feature_values, recycled_features


def reset_input_weights(idxs: Union[int, Sequence[int]], model: MLP, optimizer: optim.Optimizer, cfg: DictConfig):
    """Reset the weights and associated optimizer state for a feature."""
    if isinstance(idxs, Sequence) and len(idxs) == 0:
        return
    
    first_layer = model.layers[0]
    
    # Reset weights
    if cfg.model.weight_init_method == 'zeros':
        with torch.no_grad():
            first_layer.weight[:, idxs] = 0
    elif cfg.model.weight_init_method == 'kaiming_uniform':
        fan = first_layer.weight.shape[1] # fan_in
        gain = 1
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            first_layer.weight[:, idxs] = first_layer.weight[:, idxs].uniform_(-bound, bound)
    else:
        raise ValueError(f'Invalid weight initialization method: {cfg.model.weight_init_method}')

    # Reset optimizer states
    if isinstance(optimizer, Adam):
        # Reset Adam state for the specific feature
        state = optimizer.state[first_layer.weight]
        if len(state) > 0: # State is only populated after the first call to step
            state['step'][:, idxs] = 0
            state['exp_avg'][:, idxs] = 0
            state['exp_avg_sq'][:, idxs] = 0
            if 'max_exp_avg_sq' in state:  # For AMSGrad
                state['max_exp_avg_sq'][:, idxs] = 0
    elif isinstance(optimizer, IDBD):
        state = optimizer.state[first_layer.weight]
        state['beta'][:, idxs] = math.log(cfg.train.learning_rate)
        state['h'][:, idxs] = 0
    else:
        raise ValueError(f'Invalid optimizer type: {type(optimizer)}')


def n_kaiming_uniform(tensor, shape, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    But has a customizable number of outputs.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    result = torch.rand(shape) * 2 * bound - bound
    result = result.to(tensor.device)
    return result


class CBPTracker():
    """Perform CBP for recycling features."""
    
    # TODO: Reimplement weight initialization between model, cbp, and input recycler
    # so that they share the same initialization methods
    def __init__(
        self,
        optimizer = None,
        replace_rate = 1e-4,
        decay_rate = 0.99,
        maturity_threshold = 100,
        incoming_weight_init = 'kaiming_uniform', # {'kaiming_uniform', 'binary'}
        outgoing_weight_init = 'zeros', # {'zeros', 'kaiming_uniform'}
    ):
        assert optimizer is None or isinstance(optimizer, (Adam, IDBD, torch.optim.SGD))
        
        # Dictionary mapping feature output layer to previous and next layers
        self._tracked_layers = {}
        self._feature_stats = {}
        self._replace_accumulator = defaultdict(float) 

        self.optimizer = optimizer
        
        self.replace_rate = replace_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.incoming_weight_init = incoming_weight_init
        self.outgoing_weight_init = outgoing_weight_init

    def track(self, previous, current, next):
        """Track a list of layers used for CBP calculations."""
        if not isinstance(previous, nn.Linear) or not isinstance(next, nn.Linear):
            raise NotImplementedError('CBP is only implemented for linear layers.')

        self._tracked_layers[current] = (previous, next) # Previous layer is currently unused, but could be for other utility metrics
        self._feature_stats[current] = defaultdict(lambda: torch.zeros(1, requires_grad=False, device=next.weight.device))
        current.register_forward_hook(self._get_hook(current))

    def track_sequential(self, sequential: Sequence[nn.Module]):
        """
        Track a sequential model for CBP calculations.
        Must be an alternating sequence of linear layers and activations.
        """
        for i in range(0, len(sequential) - 2, 2):
            self.track(sequential[i], sequential[i+1], sequential[i+2])

    def track_optimizer(self, optimizer):
        """Track an optimizer for CBP calculations."""
        if self.optimizer is not None:
            warnings.warn("Replacing previously tracked optimizer.")
        self.optimizer = optimizer

    def _get_input_weight_sums(self, layer):
        """Return the sum of the absolute values of the weights for each outputted feature."""
        return torch.sum(torch.abs(layer.weight), dim=1)

    def _get_output_weight_sums(self, layer):
        """Return the sum of the absolute values of the weights for each inputted feature."""
        return torch.sum(torch.abs(layer.weight), dim=0)

    def _reinit_input_weights(self, layer, idxs):
        """Reinitialize the weights that output features at the given indices."""
        # This is how linear layers are initialized in PyTorch
        if self.incoming_weight_init == 'kaiming_uniform':
            weight_data = layer.weight.data
            layer.weight.data[idxs] = n_kaiming_uniform(
                weight_data, weight_data[idxs].shape, a=math.sqrt(5))

            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                layer.bias.data[idxs] = torch.rand(len(idxs), device=layer.bias.device) * 2 * bound - bound
                
        elif self.incoming_weight_init == 'binary':
            weight_data = layer.weight.data
            layer.weight.data[idxs] = torch.randint(
                0, 2, weight_data[idxs].shape, device=layer.weight.device).float() * 2 - 1
            if layer.bias is not None:
                layer.bias.data[idxs] = torch.zeros_like(layer.bias.data[idxs])

        else:
            raise ValueError(f'Invalid weight initialization: {self.incoming_weight_init}')

    def _reset_input_optim_state(self, layer, idxs):
        """
        Reset the optimizer state for the weights that output features at the given indices.
        Currently works for SGD and Adam (without step reset) optimizers.
        """
        optim_state = self.optimizer.state[layer.weight]
        
        for key, value in optim_state.items():
            if value.shape == layer.weight.shape:
                if value is not None:
                    optim_state[key][idxs, :] = 0
            else:
                warnings.warn(
                    f"Cannot reset optimizer state for {key} of layer '{layer}' because shapes do not match, "
                    f"parameter: {layer.weight.shape}, state value: {value.shape}"
                )
                
        if isinstance(self.optimizer, IDBD) and 'beta' in optim_state:
            optim_state['beta'][idxs, :] = math.log(self.optimizer.init_lr)
                
        if layer.bias is not None:
            optim_state = self.optimizer.state[layer.bias]
            
            for key, value in optim_state.items():
                if value.shape == layer.bias.shape:
                    if value is not None:
                        optim_state[key][idxs] = 0
                else:
                    warnings.warn(
                        f"Cannot reset optimizer state for {key} of layer '{layer}' because shapes do not match, "
                        f"parameter: {layer.bias.shape}, state value: {value.shape}"
                    )
            
            if isinstance(self.optimizer, IDBD) and 'beta' in optim_state:
                optim_state['beta'][idxs] = math.log(self.optimizer.init_lr)

    def _reinit_output_weights(self, layer, idxs):
        """Reinitialize the weights that take in features at the given indices."""
        # This is how linear layers are initialized in PyTorch
        weight_data = layer.weight.data
        
        if self.outgoing_weight_init == 'zeros':
            layer.weight.data[:, idxs] = torch.zeros_like(weight_data[:, idxs])
        elif self.outgoing_weight_init == 'kaiming_uniform':
            layer.weight.data[:, idxs] = n_kaiming_uniform(
                weight_data, weight_data[:, idxs].shape, a=math.sqrt(5))
        else:
            raise ValueError(f'Invalid weight initialization: {self.outgoing_weight_init}')
    
    def _reset_output_optim_state(self, layer, idxs):
        """
        Reset the optimizer state for the weights that take in features at the given indices.
        Currently works for SGD and Adam optimizers.
        """
        optim_state = self.optimizer.state[layer.weight]
        
        for key, value in optim_state.items():
            if value.shape == layer.weight.shape:
                if value is not None:
                    optim_state[key][:, idxs] = 0
            else:
                warnings.warn(
                    f"Cannot reset optimizer state for {key} of layer '{layer}' because shapes do not match, "
                    f"parameter: {layer.weight.shape}, state value: {value.shape}"
                )
        
        # TODO: Consider making different variables for initial step-size of reset outgoing and incoming weights
        if isinstance(self.optimizer, IDBD) and 'beta' in optim_state:
            optim_state['beta'][:, idxs] = math.log(self.optimizer.init_lr)

    def _get_hook(self, layer):
        """Return a hook function for a given layer."""
        def track_cbp_stats(module, input, output):
            if not module.training:
                return

            input = input[0]
            
            with torch.no_grad():
                output_magnitude = torch.abs(output)
                
                # Mean over batch dimension if present
                if len(output_magnitude.shape) == 2:
                    output_magnitude = output_magnitude.mean(dim=0)
                
                self._feature_stats[module]['age'] = torch.ones_like(output_magnitude) + self._feature_stats[module]['age']

                output_layer = self._tracked_layers[layer][1]
                utility = output_magnitude * self._get_output_weight_sums(output_layer)
                
                self._feature_stats[module]['utility'] = (1 - self.decay_rate) * utility \
                    + self.decay_rate * self._feature_stats[module]['utility']

        return track_cbp_stats

    def _reset_feature_stats(self, layer, idxs):
        """Resets the feature stats for the given layer and indices."""
        for key in self._feature_stats[layer]:
            self._feature_stats[layer][key][idxs] = 0

    def _prune_layer(self, layer):
        ages = self._feature_stats[layer]['age']

        # Get number of features to reset
        n_features = ages.numel()
        self._replace_accumulator[layer] += self.replace_rate * n_features

        # If there are not enough features to reset, return
        if self._replace_accumulator[layer] < 1:
            return

        # Get eligible features
        eligible_idxs = ages > self.maturity_threshold
        eligible_idxs = torch.nonzero(eligible_idxs).squeeze()
        n_reset = int(self._replace_accumulator[layer])
        
        # If there are no eligible features, return
        if eligible_idxs.numel() == 0 or n_reset == 0:
            return

        # Get features to reset based on lowest utility
        self._replace_accumulator[layer] -= n_reset
        reset_idxs = torch.argsort(
            self._feature_stats[layer]['utility'][eligible_idxs])[:n_reset]
        reset_idxs = eligible_idxs[reset_idxs]
        
        # Reset feature stats
        self._reset_feature_stats(layer, reset_idxs)

        # Reset features
        self._reinit_input_weights(self._tracked_layers[layer][0], reset_idxs)
        self._reinit_output_weights(self._tracked_layers[layer][1], reset_idxs)

        # Reset optimizer state
        if self.optimizer is not None:
            self._reset_input_optim_state(self._tracked_layers[layer][0], reset_idxs)
            self._reset_output_optim_state(self._tracked_layers[layer][1], reset_idxs)

        return reset_idxs

    def prune_features(self):
        """Prune features based on the CBP score."""
        reset_idxs = {}
        for layer in self._tracked_layers.keys():
            layer_idxs = self._prune_layer(layer)
            if layer_idxs is not None:
                reset_idxs[layer] = layer_idxs
        return reset_idxs