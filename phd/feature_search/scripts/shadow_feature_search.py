"""
Rupam's problem setup, but with distractors and other additions.

This script is a more complete version of the `rupam_experiment.py` script, which adds the following features:
- Distractors in the input
- Target noise
- Separate optimizers for the intermediate and output layers
"""

import logging
from typing import Iterator, Tuple, Callable, List, Optional

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
from phd.feature_search.core.models import LTU, DynamicShadowMLP
from phd.feature_search.core.feature_recycling import InputRecycler, CBPTracker
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.research_utils.logging import *


logger = logging.getLogger(__name__)


class DistractorTracker():
    def __init__(
            self,
            model: DynamicShadowMLP,
            distractor_chance: float,
            mean_range: Tuple[float, float],
            std_range: Tuple[float, float],
            seed: Optional[int] = None,
        ):
        self.model = model
        self.distractor_chance = distractor_chance
        self.mean_range = mean_range
        self.std_range = std_range
        self.n_features = model.active_output_layer.in_features
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
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        update_state: bool = False,
        distractor_callback: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[nn.Module, torch.Tensor], Dict[str, Any]]:
    """Forward pass with a callback that can replace hidden unit outputs with distractor values.
    
    Args:
        x: Input tensor
        target: Target value to predict
        update_state: Whether to update the model's state (promotions, demotions, and utilities)
        distractor_callback: Callable that takes a tensor and returns a tensor of the same shape
            which should be a mix of the same values and distractor values.

    Returns:
        tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
            - Output tensor
            - Dictionary of parameter inputs
    """
    assert len(x.shape) == 1, "DynamicMLP does not support batching!"
        
    aux = {}
    if update_state:
        aux['promoted_features'] = self._handle_promotions()
        aux['demoted_features'] = self._handle_demotions()

    hidden_features = self.input_layer(x)
    
    if distractor_callback is not None:
        hidden_features = distractor_callback(hidden_features)
    
    active_hidden_features = hidden_features * self.active_feature_mask
    inactive_hidden_features = hidden_features * ~self.active_feature_mask

    # Need to update this if there is more than one prediction
    active_feature_contribs = self.active_output_layer.weight.squeeze(0) * active_hidden_features
    target_pred = torch.sum(active_feature_contribs)
    
    residual_preds = self.inactive_output_layer.weight.squeeze(0) * inactive_hidden_features

    param_inputs = {
        self.active_output_layer.weight: active_hidden_features,
        self.inactive_output_layer.weight: inactive_hidden_features,
    }

    # Calculate losses if applicable
    if target is not None:
        target_error = target - target_pred
        target_loss = target_error ** 2
        residual_errors = target_loss.detach() - residual_preds
        residual_losses = residual_errors ** 2
        cumulative_loss = target_loss + residual_losses.sum()

        assert len(target.shape) == 0, "Target must be a scalar"
        assert len(active_feature_contribs.shape) == 1, "Active feature contributions must be a 1D tensor"
        assert len(residual_losses.shape) == 1, "Residual losses must be a 1D tensor"
        assert len(residual_preds.shape) == 1, "Residual predictions must be a 1D tensor"
        assert len(target_error.shape) == 0, "Target error must be a scalar"
        assert len(target_loss.shape) == 0, "Target loss must be a scalar"

        aux = {
            'loss': cumulative_loss,
            'target_loss': target_loss,
            'residual_losses': residual_losses,
            'target_pred': target_pred,
            'residual_preds': residual_preds,
        }
        
        # Update utility
        if update_state:
            with torch.no_grad():
                active_utilities = torch.abs(target_error + active_feature_contribs) - torch.abs(target_error)
                residual_utilities = torch.abs(target_error) - torch.abs(target_error - residual_preds)
                
                self.active_feature_utilities = (
                    self.utility_decay * self.active_feature_utilities +
                    (1 - self.utility_decay) * active_utilities
                )
                self.inactive_feature_utilities = (
                    self.utility_decay * self.inactive_feature_utilities +
                    (1 - self.utility_decay) * residual_utilities
                )
                
                self.target_error_trace = (
                    self.utility_decay * self.target_error_trace +
                    (1 - self.utility_decay) * torch.abs(target_error)
                )
                self.target_trace = (
                    self.utility_decay * self.target_trace +
                    (1 - self.utility_decay) * torch.abs(target)
                )
            self.update_step += 1
    
    return target_pred, param_inputs, aux


def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    model = DynamicShadowMLP(
        input_dim = cfg.task.n_features,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim,
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        utility_decay = cfg.model.get('utility_decay', 0.99),
        max_promotions_per_step = cfg.model.get('max_promotions_per_step', 0.5),
        max_demotions_per_step = cfg.model.get('max_demotions_per_step', 0.5),
        promotion_threshold = cfg.model.get('promotion_threshold', 0.01),
        demotion_threshold = cfg.model.get('demotion_threshold', 0.0),
        keep_promotion_weights = cfg.model.get('keep_promotion_weights', False),
        n_initial_real_features = cfg.model.get('n_initial_real_features', cfg.model.hidden_dim),
        seed = seed_from_string(base_seed, 'model'),
    )
    model.to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    
    # Determine if we need separate optimizers for the intermediate and output layers
    repr_optimizer_name = cfg.get('representation_optimizer', {}).get('name')
    assert repr_optimizer_name != 'idbd', "IDBD is not supported for the representation optimizer!"
    repr_module = model.input_layer
    n_repr_trainable_layers = len([p for p in repr_module.parameters() if p.requires_grad])
    
    assert n_repr_trainable_layers == 0 or repr_optimizer_name is not None, \
        "Representation optimizer must be specified if there are trainable representationlayers!"
    
    if repr_optimizer_name is not None and n_repr_trainable_layers > 0:
        # Use separate optimizers for the intermediate and output layers
        repr_optimizer = prepare_optimizer(repr_module, repr_optimizer_name, cfg.representation_optimizer)
        active_optimizer = prepare_optimizer(model.active_output_layer, cfg.optimizer.name, cfg.optimizer)
        inactive_optimizer = prepare_optimizer(model.inactive_output_layer, cfg.optimizer.name, cfg.optimizer)
    else:
        # Only use one optimizer
        repr_optimizer = None
        active_optimizer = prepare_optimizer(model.active_output_layer, cfg.optimizer.name, cfg.optimizer)
        inactive_optimizer = prepare_optimizer(model.inactive_output_layer, cfg.optimizer.name, cfg.optimizer)
        
    return task, task_iterator, model, criterion, active_optimizer, inactive_optimizer, repr_optimizer


def prepare_ltu_geoff_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task, task_iterator, model, criterion, active_optimizer, inactive_optimizer, repr_optimizer = \
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
    assert cfg.optimizer.name == 'idbd', "IDBD is the only supported optimizer!"
    
    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed_from_string(base_seed, 'task_init_generator'))
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    
    # Change LTU threshold for target and predictors
    model.activation.threshold = 0.0 # Redundant, but keeps same format as other scripts
    task.activation_fn.threshold = 0.0

    torch.manual_seed(seed_from_string(base_seed, 'experiment_setup'))

    return task, task_iterator, model, criterion, active_optimizer, inactive_optimizer, repr_optimizer


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: DynamicShadowMLP,
        criterion: nn.Module,
        active_optimizer: Optimizer,
        inactive_optimizer: Optimizer,
        repr_optimizer: Optional[Optimizer],
        distractor_tracker: DistractorTracker,
    ):
    # Distractor setup
    n_hidden_units = model.active_output_layer.in_features
    distractor_tracker.process_new_features(list(range(n_hidden_units)))

    # Training loop
    step = 0
    prev_pruned_idxs = set()
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    # Initialize accumulators
    cumulant_stats = StandardizationStats(gamma=0.99)
    cumulative_loss = np.float128(0.0)
    loss_accum = 0.0
    mean_pred_loss_accum = 0.0
    pruned_accum = 0
    pruned_newest_feature_accum = 0
    n_steps_since_log = 0
    total_pruned = 0
    target_buffer = []

    while step < cfg.train.total_steps:

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

        # # Reset weights and optimizer states for recycled features
        # if cbp_tracker is not None:
        #     pruned_idxs = cbp_tracker.prune_features()
        #     n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
        #     total_pruned += n_pruned

        #     if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
        #         new_feature_idxs = pruned_idxs[prune_layer].tolist()

        #         # Turn some features into distractors
        #         distractor_tracker.process_new_features(new_feature_idxs)

        #         # Log pruning statistics
        #         pruned_accum += len(new_feature_idxs)
        #         for new_feature_idx in new_feature_idxs:
        #             pruned_newest_feature_accum += int(new_feature_idx in prev_pruned_idxs)
        #         prev_pruned_idxs = set(new_feature_idxs)
        
        
        # Forward pass
        predictions, param_inputs, aux = model(
            features,
            target = targets,
            update_state = True,
            distractor_callback = distractor_tracker.replace_features,
        )
        predictions = predictions.squeeze()
        loss = aux['target_loss']
        
        active_optimizer.zero_grad()
        inactive_optimizer.zero_grad()
        if repr_optimizer is not None:
            repr_optimizer.zero_grad()
        
        retain_graph = active_optimizer.version == 'squared_grads'
        aux['loss'].backward(retain_graph=retain_graph) # Graph used in IDBD step function
        
        promoted_idxs = aux.get('promoted_features', [])
        demoted_idxs = aux.get('demoted_features', [])

        active_weights = model.active_output_layer.weight
        inactive_weights = model.inactive_output_layer.weight

        init_step_size = active_optimizer.init_lr / (model.active_feature_mask.sum() + 1) ** 2
        init_beta = torch.log(init_step_size)
                    
        # Update optimizer step sizes if there are promoted features
        if len(promoted_idxs) > 0:
            if model.keep_promotion_weights:
                active_optimizer.state[active_weights]['beta'][:, promoted_idxs] = \
                    inactive_optimizer.state[inactive_weights]['beta'][:, promoted_idxs]
            else:
                active_optimizer.state[active_weights]['beta'][:, promoted_idxs] = init_beta
        
        # Update optimizer step sizes if there are demoted features
        if len(demoted_idxs) > 0:
            if model.keep_promotion_weights:
                inactive_optimizer.state[inactive_weights]['beta'][:, demoted_idxs] = \
                    active_optimizer.state[active_weights]['beta'][:, demoted_idxs]
            else:
                inactive_optimizer.state[inactive_weights]['beta'][:, demoted_idxs] = torch.minimum(
                    init_beta,
                    active_optimizer.state[active_weights]['beta'][:, demoted_idxs],
                )
    
        # Perform parameter updates
        active_optimizer.step(param_inputs, predictions)
        # TODO: Add in the weights_independent flag
        inactive_optimizer.step(param_inputs, aux['shadow_preds'], weights_independent=True)
        
        if repr_optimizer is not None:
            repr_optimizer.step()
        
        # Compute loss for a baseline prediction
        with torch.no_grad():
            if cfg.train.standardize_cumulants:
                baseline_pred = torch.zeros_like(targets)
            else:
                baseline_pred = cumulant_stats.running_mean.cpu().view(1, 1)
            mean_pred_loss = criterion(baseline_pred, targets)

        
        # TODO: FINISHED REVISIONS UP TO HERE
        
        
        # Accumulate metrics
        # TODO: Add metrics based on shadow features
        loss_accum += loss.item()
        cumulative_loss += loss.item()
        mean_pred_loss_accum += mean_pred_loss.item()
        n_steps_since_log += 1
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            n_distractors = distractor_tracker.distractor_mask.sum().item()
            n_real_features = distractor_tracker.distractor_mask.numel() - n_distractors
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_accum / n_steps_since_log,
                'cumulative_loss': float(cumulative_loss),
                'mean_prediction_loss': mean_pred_loss_accum / n_steps_since_log,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
                'n_distractors': n_distractors,
                'n_real_features': n_real_features,
            }

            if pruned_accum > 0:
                metrics['fraction_pruned_were_new'] = pruned_newest_feature_accum / pruned_accum
                pruned_newest_feature_accum = 0
                pruned_accum = 0

            # Add model statistics separately for real and distractor features
            if cfg.model.get('log_model_stats', False):
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

    task, task_iterator, model, criterion, active_optimizer, inactive_optimizer, repr_optimizer = \
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
        cfg, task, task_iterator, model, criterion, active_optimizer, inactive_optimizer,
        repr_optimizer, distractor_tracker,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
