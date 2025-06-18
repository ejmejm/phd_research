"""
Rupam's problem setup, but with distractors and other additions.

This script is a more complete version of the `rupam_experiment.py` script, which adds the following features:
- Distractors in the input
- Target noise
- Separate optimizers for the intermediate and output layers
"""

from dataclasses import dataclass
import logging
from typing import Iterator, Tuple, Callable, List, Optional

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from phd.feature_search.core.experiment_helpers import (
    get_model_statistics,
    prepare_task,
    prepare_optimizer,
    seed_from_string,
    set_seed,
    standardize_targets,
    StandardizationStats,
)
from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.models import LTU, EnsembleMLP, MultipleLinear
from phd.feature_search.core.models.ensemble_models import prune_features as prune_ensemble_features
from phd.feature_search.core.models.ensemble_models import prune_ensembles
from phd.feature_search.core.feature_recycling import InputRecycler, n_kaiming_uniform
from phd.feature_search.core.feature_recycling import CBPTracker as OriginalCBPTracker
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.research_utils.logging import *


logger = logging.getLogger(__name__)


class CBPTracker(OriginalCBPTracker):
    def _reinit_output_weights(self, layer: MultipleLinear, idxs: List[int]):
        """Reinitialize the weights that take in features at the given indices."""
        # This is how linear layers are initialized in PyTorch
        weight_data = layer.weight.data
        
        # Convert 1D indices to 2D indices
        ensemble_dim = layer.weight.shape[2]
        ensemble_idxs = [idx // ensemble_dim for idx in idxs]
        feature_idxs = [idx % ensemble_dim for idx in idxs]
        
        if self.outgoing_weight_init == 'zeros':
            layer.weight.data[ensemble_idxs, :, feature_idxs] = torch.zeros_like(weight_data[ensemble_idxs, :, feature_idxs])
        elif self.outgoing_weight_init == 'kaiming_uniform':
            layer.weight.data[ensemble_idxs, :, feature_idxs] = n_kaiming_uniform(
                weight_data,
                weight_data[ensemble_idxs, :, feature_idxs].shape,
                a=math.sqrt(5),
                generator=self.rng,
            )
        else:
            raise ValueError(f'Invalid weight initialization: {self.outgoing_weight_init}')
    
    def _reset_output_optim_state(self, layer: nn.Module, idxs: List[int]):
        """
        Reset the optimizer state for the weights that take in features at the given indices.
        Currently works for SGD and Adam optimizers.
        """
        optim_state = self.optimizer.state[layer.weight]
        
        # Convert 1D indices to 2D indices
        ensemble_dim = layer.weight.shape[2]
        ensemble_idxs = [idx // ensemble_dim for idx in idxs]
        feature_idxs = [idx % ensemble_dim for idx in idxs]
        
        for key, value in optim_state.items():
            if value.shape == layer.weight.shape:
                if value is not None:
                    optim_state[key][ensemble_idxs, :, feature_idxs] = 0
            else:
                warnings.warn(
                    f"Cannot reset optimizer state for {key} of layer '{layer}' because shapes do not match, "
                    f"parameter: {layer.weight.shape}, state value: {value.shape}"
                )
        
        # TODO: Consider making different variables for initial step-size of reset outgoing and incoming weights
        if isinstance(self.optimizer, IDBD) and 'beta' in optim_state:
            optim_state['beta'][ensemble_idxs, :, feature_idxs] = math.log(self.optimizer.init_lr)


class DistractorTracker():
    def __init__(
            self,
            model: EnsembleMLP,
            distractor_chance: float,
            mean_range: Tuple[float, float],
            std_range: Tuple[float, float],
            seed: Optional[int] = None,
        ):
        self.model = model
        self.distractor_chance = distractor_chance
        self.mean_range = mean_range
        self.std_range = std_range
        self.n_features = model.input_layer.out_features
        self.device = next(model.parameters()).device

        # Initialize random number generator with random seed if none provided
        self.torch_rng = torch.Generator(device=self.device)
        if seed is not None:
            self.torch_rng.manual_seed(seed)
        else:
            self.torch_rng.seed()

        # Initialize tensors with seeded random values
        self.distractor_mask = torch.zeros(self.n_features, dtype=torch.bool, device=self.device)
        self.distractor_means = torch.zeros(self.n_features, device=self.device)
        self.distractor_stds = torch.ones(self.n_features, device=self.device)
        
        # Use torch_rng for uniform sampling
        self.distractor_means.uniform_(*mean_range, generator=self.torch_rng)
        self.distractor_stds.uniform_(*std_range, generator=self.torch_rng)

        self.distractor_values = None  # Will be initialized on first use
    
    @torch.no_grad()
    def process_new_features(self, new_feature_idxs: List[int]):
        if len(new_feature_idxs) == 0:
            return

        if not all(0 <= idx < self.n_features for idx in new_feature_idxs):
            raise ValueError(f"Feature indices must be between 0 and {self.n_features-1}")

        # Use torch_rng for random sampling
        new_distractor_mask = torch.rand(
            len(new_feature_idxs), 
            device=self.device, 
            generator=self.torch_rng
        ) < self.distractor_chance
        
        new_means = torch.zeros(len(new_feature_idxs), device=self.device)
        new_stds = torch.ones(len(new_feature_idxs), device=self.device)
        
        # Use torch_rng for uniform sampling
        new_means.uniform_(*self.mean_range, generator=self.torch_rng)
        new_stds.uniform_(*self.std_range, generator=self.torch_rng)
        
        self.distractor_mask[new_feature_idxs] = new_distractor_mask
        self.distractor_means[new_feature_idxs] = new_means
        self.distractor_stds[new_feature_idxs] = new_stds
        
        # Zero out weights and biases for distractor features
        distractor_idxs = [idx for idx in new_feature_idxs if self.distractor_mask[idx]]
        if len(distractor_idxs) > 0:
            self.model.input_layer.weight[distractor_idxs] = 0.0
    
    def replace_features(self, x: torch.Tensor) -> torch.Tensor:
        """Replace distractor feature values with random values.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Tensor of same shape with distractor features replaced with random values
        """
        assert x.shape[1] == self.n_features, \
            f"Input tensor has {x.shape[1]} features but expected {self.n_features}!"
            
        # Generate random values for entire batch
        batch_size = x.shape[0]
        
        # Initialize or resize the pre-allocated tensor if needed
        if self.distractor_values is None or self.distractor_values.shape[0] != batch_size:
            self.distractor_values = torch.empty(
                (batch_size, self.n_features), 
                device=self.device
            )
        
        # Generate random values in-place
        self.distractor_values.normal_(generator=self.torch_rng)
        self.distractor_values.mul_(self.distractor_stds.unsqueeze(0))
        self.distractor_values.add_(self.distractor_means.unsqueeze(0))
        
        # TODO: Fix this taking so long on gpu
        # x = x * ~self.distractor_mask + self.distractor_values * self.distractor_mask
        x[:, self.distractor_mask] = self.distractor_values[:, self.distractor_mask]
        return x


# This is used to overwrite the model's forward pass so that distractor features
# can be replaced with random values each step during the forward pass
def model_distractor_forward_pass(
        self: EnsembleMLP,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        update_state: bool = False,
        distractor_callback: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[nn.Module, torch.Tensor], Dict[str, Any]]:
    """Forward pass with a callback that can replace hidden unit outputs with distractor values.
    
    Args:
        x: Input tensor of shape (batch_size, in_features)
        target: Target tensor of shape (batch_size, out_features)
        update_state: Whether to update the state of the model (for utility tracking)
        distractor_callback: Callable that takes a tensor and returns a tensor of the same shape
            which should be a mix of the same values and distractor values.

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
    if distractor_callback is not None:
        hidden_features = distractor_callback(hidden_features)
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
                
                # Estimate how much each feature is reducing the loss within its ensemble as a proxy for utility
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



def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    global model, cbp_tracker
    # Initialize model and optimizer
    model = EnsembleMLP(
        input_dim = cfg.task.n_features,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim,
        ensemble_dim = cfg.model.ensemble_dim,
        n_ensemble_members = cfg.model.n_ensemble_members,
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        utility_decay = cfg.feature_recycling.utility_decay,
        prediction_mode = cfg.model.prediction_mode,
        feature_utility_mode = cfg.model.feature_utility_mode,
        ensemble_utility_mode = cfg.model.ensemble_utility_mode,
        ensemble_feature_selection_method = cfg.model.ensemble_feature_selection_method,
        seed = seed_from_string(base_seed, 'model'),
    )
    model.to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    
    # Determine if we need separate optimizers for the intermediate and output layers
    repr_optimizer_name = cfg.get('representation_optimizer', {}).get('name')
    assert repr_optimizer_name != 'idbd', "IDBD is not supported for the representation optimizer!"
    repr_module = model.input_layer
    prediction_module = model.prediction_layer
    n_repr_trainable_layers = len([p for p in repr_module.parameters() if p.requires_grad])
    
    if repr_optimizer_name is not None and n_repr_trainable_layers > 0:
        # Use separate optimizers for the intermediate and output layers
        repr_optimizer = prepare_optimizer(repr_module, repr_optimizer_name, cfg.representation_optimizer)
        optimizer = prepare_optimizer(prediction_module, cfg.optimizer.name, cfg.optimizer)
    else:
        # Only use one optimizer
        repr_optimizer = None
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
    
    # Initialize feature recycler
    recycler = InputRecycler(
        n_features = cfg.task.n_features,
        n_real_features = cfg.task.n_real_features,
        distractor_chance = cfg.input_recycling.distractor_chance,
        recycle_rate = cfg.input_recycling.recycle_rate,
        utility_decay = cfg.input_recycling.utility_decay,
        use_cbp_utility = cfg.input_recycling.use_cbp_utility,
        feature_protection_steps = cfg.input_recycling.feature_protection_steps,
        n_start_real_features = cfg.input_recycling.get('n_start_real_features', -1),
        device = 'cpu',
        seed = seed_from_string(base_seed, 'recycler'),
    )
    
    # Initialize CBP tracker
    # if cfg.feature_recycling.use_cbp_utility:
    #     cbp_tracker = CBPTracker(
    #         optimizer = optimizer,
    #         replace_rate = cfg.feature_recycling.recycle_rate,
    #         decay_rate = cfg.feature_recycling.utility_decay,
    #         maturity_threshold = cfg.feature_recycling.feature_protection_steps,
    #         seed = seed_from_string(base_seed, 'cbp_tracker'),
    #     )
    #     cbp_tracker.track(model.input_layer, model.activation, model.prediction_layer)
    # else:
    #     cbp_tracker = None
    cbp_tracker = None
        
    return task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker


def prepare_ltu_geoff_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker = \
        prepare_components(cfg)

    assert isinstance(task, NonlinearGEOFFTask)
    
    assert cfg.model.weight_init_method == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.weight_init == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.model.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"

    if cbp_tracker is not None:
        cbp_tracker.incoming_weight_init = 'binary'
    
    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed_from_string(base_seed, 'task_init_generator'))
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    torch.nn.init.zeros_(model.prediction_layer.weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.0 # 0.1 * cfg.task.n_features
    if isinstance(model.activation, LTU):
        model.activation.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    torch.manual_seed(seed_from_string(base_seed, 'experiment_setup'))

    return task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker


@dataclass
class PruningState:
    feature_prune_accumulator: float = 0.0
    ensemble_prune_accumulator: float = 0.0


def prune_model(
    model: EnsembleMLP,
    optimizer: Optimizer,
    distractor_tracker: DistractorTracker,
    pruning_state: PruningState,
    cfg: DictConfig,
):
    """Prune the model, handle distractor replacement, and return pruning metrics.
    
    Args:
        model: The model to prune
        optimizer: The optimizer that holds the optimization state for the given model
        distractor_tracker: The tracker that handles the distractor features
        pruning_state: The state of the pruning process
        cfg: The configuration
        
    Returns:
        A dictionary containing the pruning results
    """
    log_pruning_stats = cfg.train.get('log_pruning_stats', False)
    
    # Compute how many features and ensembles to prune
    n_features = model.input_layer.out_features
    n_ensembles = model.n_ensemble_members
    
    pruning_state.feature_prune_accumulator += n_features * cfg.feature_recycling.recycle_rate
    pruning_state.ensemble_prune_accumulator += n_ensembles * cfg.feature_recycling.ensemble_recycle_rate
    
    n_features_to_prune = int(pruning_state.feature_prune_accumulator)
    n_ensembles_to_prune = int(pruning_state.ensemble_prune_accumulator)
    
    pruning_state.feature_prune_accumulator -= n_features_to_prune
    pruning_state.ensemble_prune_accumulator -= n_ensembles_to_prune
    
    # Determine the indices to prune
    feature_utilities = model.feature_utilities.cpu().numpy()
    feature_idxs_to_prune = np.argsort(feature_utilities, stable=True)[:n_features_to_prune]
    
    ensemble_utilities = model.ensemble_utilities.cpu().numpy()
    ensemble_idxs_to_prune = np.argsort(ensemble_utilities, stable=True)[:n_ensembles_to_prune]
    
    # Prune the features and ensembles
    prune_ensemble_features(
        model, optimizer, feature_idxs_to_prune,
        input_init_type = cfg.model.weight_init_method,
        output_init_type = 'zeros',
    )
    
    prune_ensembles(
        model, optimizer, ensemble_idxs_to_prune,
        output_init_type = 'zeros',
    )
     
    # Replace features with distractors where necessary
    if len(feature_idxs_to_prune) > 0:
        distractor_tracker.process_new_features(feature_idxs_to_prune)
    
    return {
        'n_features_pruned': n_features_to_prune,
        'n_ensembles_pruned': n_ensembles_to_prune,
        'feature_idxs_pruned': feature_idxs_to_prune,
        'ensemble_idxs_pruned': ensemble_idxs_to_prune,
    }


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: EnsembleMLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        repr_optimizer: Optional[Optimizer],
        cbp_tracker: CBPTracker,
        distractor_tracker: DistractorTracker,
    ):
    # Distractor setup
    n_hidden_units = model.input_layer.out_features
    distractor_tracker.process_new_features(list(range(n_hidden_units)))

    # Training loop
    step = 0
    prev_pruned_idxs = set()
    prune_layer = model.activation
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Flags
    log_utility_stats = cfg.train.get('log_utility_stats', False)
    log_pruning_stats = cfg.train.get('log_pruning_stats', False)
    log_model_stats = cfg.train.get('log_model_stats', False)

    # Initialize accumulators
    cumulant_stats = StandardizationStats(gamma=0.99)
    pruning_state = PruningState()
    cumulative_loss = np.float128(0.0)
    loss_accum = 0.0
    ensemble_loss_accum = 0.0
    mean_pred_loss_accum = 0.0
    pruned_accum = 0
    pruned_newest_feature_accum = 0
    n_steps_since_log = 0
    total_features_pruned = 0
    total_ensembles_pruned = 0
    prune_thresholds = []
    target_buffer = []

    while step < cfg.train.total_steps:

        ### Data Processing ###

        # Generate batch of data
        inputs, targets = next(task_iterator)

        # Add noise to targets
        if cfg.task.noise_std > 0:
            targets += torch.randn_like(targets) * cfg.task.noise_std
        
        with torch.no_grad():
            standardized_targets, cumulant_stats = standardize_targets(targets, cumulant_stats)
        
        if cfg.train.standardize_cumulants:
            targets = standardized_targets
        target_buffer.extend(targets.view(-1).tolist())
        
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)


        ### Pruning ###

        if log_pruning_stats:
            pre_prune_feature_utilities = model.feature_utilities.ravel().cpu().clone().numpy()
            pre_prune_ensemble_utilities = model.ensemble_utilities.cpu().clone().numpy()
        
        # Prune the model if necessary
        prune_results = prune_model(
            model, optimizer, distractor_tracker, pruning_state, cfg)
        
        # Update pruning metrics
        total_features_pruned += prune_results['n_features_pruned']
        total_ensembles_pruned += prune_results['n_ensembles_pruned']
        
        if prune_results['n_features_pruned'] > 0:
            feature_idxs_pruned = prune_results['feature_idxs_pruned'].tolist()
            
            # Log pruning statistics
            pruned_accum += len(feature_idxs_pruned)
            n_new_pruned_features = len(set(feature_idxs_pruned).intersection(prev_pruned_idxs))
            pruned_newest_feature_accum += n_new_pruned_features
            prev_pruned_idxs = set(feature_idxs_pruned)
            
            if log_pruning_stats:
                prune_thresholds.append(pre_prune_feature_utilities[feature_idxs_pruned].max())
        
        # if cbp_tracker is not None:
        #     if log_pruning_stats:
        #         pre_prune_utilities = cbp_tracker.get_statistics(prune_layer)['utility']

        #     pruned_idxs = cbp_tracker.prune_features()
        #     n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
        #     total_features_pruned += n_pruned

        #     if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
        #         new_feature_idxs = pruned_idxs[prune_layer].tolist()

        #         # Turn some features into distractors
        #         distractor_tracker.process_new_features(new_feature_idxs)

        #         # Log pruning statistics
        #         pruned_accum += len(new_feature_idxs)
        #         n_new_pruned_features = len(set(new_feature_idxs).intersection(prev_pruned_idxs))
        #         pruned_newest_feature_accum += n_new_pruned_features
        #         prev_pruned_idxs = set(new_feature_idxs)
                
        #         if log_pruning_stats:
        #             prune_thresholds.append(pre_prune_utilities[new_feature_idxs].max())
        
        
        ### Forward Pass ###
        
        outputs, param_inputs, aux = model(
            features, targets, update_state=True,
            distractor_callback=distractor_tracker.replace_features,
        )
        loss = aux['loss']
        ensemble_loss_sum = aux['ensemble_losses'].sum()
        
        with torch.no_grad():
            if cfg.train.standardize_cumulants:
                baseline_pred = torch.zeros_like(targets)
            else:
                baseline_pred = cumulant_stats.running_mean.cpu().view(1, 1)
            mean_pred_loss = criterion(baseline_pred, targets)


        ### Backward Pass ###

        # Backward pass
        optimizer.zero_grad()
        if repr_optimizer is not None:
            repr_optimizer.zero_grad()
        
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            retain_graph = optimizer.version == 'squared_grads'
            ensemble_loss_sum.backward(retain_graph=retain_graph)
            optimizer.step(outputs, param_inputs)
        else:
            ensemble_loss_sum.backward()
            optimizer.step()
            
        if repr_optimizer is not None:
            repr_optimizer.step()
            
        
        ### Metrics ###
        
        # Accumulate metrics
        loss_accum += loss.item()
        ensemble_loss_accum += ensemble_loss_sum.item()
        cumulative_loss += loss.item()
        mean_pred_loss_accum += mean_pred_loss.item()
        n_steps_since_log += 1
        
        
        ### Logging ###
        
        if step % cfg.train.log_freq == 0:
            n_distractors = distractor_tracker.distractor_mask.sum().item()
            n_real_features = distractor_tracker.distractor_mask.numel() - n_distractors
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_accum / n_steps_since_log,
                'ensemble_loss': ensemble_loss_accum / n_steps_since_log,
                'cumulative_loss': float(cumulative_loss),
                'mean_prediction_loss': mean_pred_loss_accum / n_steps_since_log,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'n_distractors': n_distractors,
                'n_real_features': n_real_features,
            }

            if log_pruning_stats:
                if pruned_accum > 0:
                    metrics['fraction_pruned_were_new'] = pruned_newest_feature_accum / pruned_accum
                    pruned_newest_feature_accum = 0
                    pruned_accum = 0
                metrics['units_pruned'] = total_pruned
                if len(prune_thresholds) > 0:
                    metrics['prune_threshold'] = np.mean(prune_thresholds)
                prune_thresholds.clear()
            
            if log_utility_stats:
                all_utilities = cbp_tracker.get_statistics(prune_layer)['utility']
                distractor_mask = distractor_tracker.distractor_mask
                real_utilities = all_utilities[~distractor_mask]
                distractor_utilities = all_utilities[distractor_mask]
                
                cumulative_utility = all_utilities.sum().item()
                metrics['cumulative_utility'] = cumulative_utility
                
                if len(real_utilities) > 0:
                    metrics['real_utility_median'] = real_utilities.median().item()
                    metrics['real_utility_25th'] = real_utilities.quantile(0.25).item()
                    metrics['real_utility_75th'] = real_utilities.quantile(0.75).item()
                
                if len(distractor_utilities) > 0:
                    metrics['distractor_utility_median'] = distractor_utilities.median().item()
                    metrics['distractor_utility_25th'] = distractor_utilities.quantile(0.25).item() 
                    metrics['distractor_utility_75th'] = distractor_utilities.quantile(0.75).item()

            # Add model statistics separately for real and distractor features
            if log_model_stats:
                real_feature_masks = [
                    torch.ones(model.layers[0].weight.shape[1], dtype=torch.bool, device=model.layers[0].weight.device),
                    ~distractor_tracker.distractor_mask,
                ]
                metrics.update(get_model_statistics(
                    model, features, param_inputs, real_feature_masks, metric_prefix='real_'))
                
                distractor_feature_masks = [
                    real_feature_masks[0],
                    distractor_tracker.distractor_mask,
                ]
                metrics.update(get_model_statistics(
                    model, features, param_inputs, distractor_feature_masks, metric_prefix='distractor_'))

            log_metrics(metrics, cfg, step=step)
            
            pbar.set_postfix(loss=metrics['loss'])
            pbar.update(cfg.train.log_freq)
            
            # Reset accumulators
            loss_accum = 0.0
            mean_pred_loss_accum = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1

    pbar.close()


@hydra.main(config_path='../conf', config_name='full_feature_search')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"

    cfg = init_experiment(cfg.project, cfg)

    task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker = \
        prepare_ltu_geoff_experiment(cfg)
    model.forward = model_distractor_forward_pass.__get__(model)
    
    distractor_tracker = DistractorTracker(
        model,
        cfg.task.distractor_chance,
        tuple(cfg.task.distractor_mean_range),
        tuple(cfg.task.distractor_std_range),
        seed = seed_from_string(cfg.seed, 'distractor_tracker'),
    )
    
    run_experiment(
        cfg, task, task_iterator, model, criterion, optimizer,
        repr_optimizer, cbp_tracker, distractor_tracker,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
