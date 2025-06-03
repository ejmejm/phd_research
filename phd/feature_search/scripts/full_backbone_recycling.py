"""
Rupam's problem setup, but with distractors
"""

import logging
from typing import Iterator, Tuple, Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.experiment_helpers import *
from phd.feature_search.scripts.feature_maturity_experiment import *
from phd.research_utils.logging import *


logger = logging.getLogger(__name__)


class DistractorTracker():
    def __init__(
            self,
            model: MLP,
            distractor_chance: float,
            mean_range: Tuple[float, float],
            std_range: Tuple[float, float],
            seed: Optional[int] = None,
        ):
        self.model = model
        self.distractor_chance = distractor_chance
        self.mean_range = mean_range
        self.std_range = std_range
        self.n_features = model.layers[-1].in_features
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
            self.model.layers[0].weight[distractor_idxs] = 0
            if self.model.layers[0].bias is not None:
                self.model.layers[0].bias[distractor_idxs] = 0
    
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
        
        # TODO: Fix this taking so long
        # x = x * ~self.distractor_mask + self.distractor_values * self.distractor_mask
        x[:, self.distractor_mask] = self.distractor_values[:, self.distractor_mask]
        return x


# This is used to overwrite the model's forward pass so that distractor features
# can be replaced with random values each step during the forward pass
def model_distractor_forward_pass(
        self,
        x: torch.Tensor,
        distractor_callback: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
    """Forward pass with a callback that can replace hidden unit outputs with distractor values.
    
    Args:
        x: Input tensor
        distractor_callback: Callable that takes a tensor and returns a tensor of the same shape
            which should be a mix of the same values and distractor values.

    Returns:
        tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
            - Output tensor
            - Dictionary of parameter inputs
    """
    param_inputs = {}
    for i in range(0, len(self.layers) - 2, 2):
        param_inputs[self.layers[i].weight] = x
        x = self.layers[i](x) # Linear layer
        x = self.layers[i + 1](x) # Activation
    
    if distractor_callback is not None:
        x = distractor_callback(x)

    param_inputs[self.layers[-1].weight] = x
    return self.layers[-1](x), param_inputs


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        cbp_tracker: CBPTracker,
        distractor_tracker: DistractorTracker,
    ):
    # Distractor setup
    n_hidden_units = model.layers[-1].in_features
    distractor_tracker.process_new_features(list(range(n_hidden_units)))

    # Training loop
    step = 0
    prev_pruned_idxs = set()
    prune_layer = model.layers[-2]
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
        
        with torch.no_grad():
            standardized_targets, cumulant_stats = standardize_targets(targets, cumulant_stats)
        
        if cfg.train.standardize_cumulants:
            targets = standardized_targets
        target_buffer.extend(targets.view(-1).tolist())
        
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)

        # Add noise to targets
        if cfg.task.noise_std > 0:
            targets += torch.randn_like(targets) * cfg.task.noise_std

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
            total_pruned += n_pruned

            if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
                new_feature_idxs = pruned_idxs[prune_layer].tolist()

                # Turn some features into distractors
                distractor_tracker.process_new_features(new_feature_idxs)

                # Log pruning statistics
                pruned_accum += len(new_feature_idxs)
                for new_feature_idx in new_feature_idxs:
                    pruned_newest_feature_accum += int(new_feature_idx in prev_pruned_idxs)
                prev_pruned_idxs = set(new_feature_idxs)
        
        # Forward pass
        outputs, param_inputs = model(
            features, distractor_tracker.replace_features)
        loss = criterion(outputs, targets)
        
        with torch.no_grad():
            if cfg.train.standardize_cumulants:
                baseline_pred = torch.zeros_like(targets)
            else:
                baseline_pred = cumulant_stats.running_mean.cpu().view(1, 1)
            mean_pred_loss = criterion(baseline_pred, targets)

        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            optimizer.step(loss, outputs, param_inputs)
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate metrics
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

            log_metrics(metrics, cfg)
            
            pbar.set_postfix(loss=metrics['loss'])
            pbar.update(cfg.train.log_freq)
            
            # Reset accumulators
            loss_accum = 0.0
            mean_pred_loss_accum = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1

    pbar.close()
    wandb.finish()


@hydra.main(config_path='../conf', config_name='full_backbone_recycling')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"

    init_experiment(cfg.project, cfg)

    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)
    model.forward = model_distractor_forward_pass.__get__(model)
    
    distractor_tracker = DistractorTracker(
        model,
        cfg.task.distractor_chance,
        tuple(cfg.task.distractor_mean_range),
        tuple(cfg.task.distractor_std_range),
        seed = seed_from_string(cfg.seed, 'distractor_tracker'),
    )
    
    run_experiment(
        cfg, task, task_iterator, model, criterion,
        optimizer, cbp_tracker, distractor_tracker,
    )


if __name__ == '__main__':
    main()
