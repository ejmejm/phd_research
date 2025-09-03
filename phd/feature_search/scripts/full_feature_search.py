"""
Rupam's problem setup, but with distractors and other additions.

Adds the following features:
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
from phd.feature_search.core.models import LTU, MLP
from phd.feature_search.core.feature_recycling import CBPTracker, InputRecycler, SignedCBPTracker
from phd.feature_search.core.tasks import NonlinearGEOFFTask
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
            self.model.layers[0].weight[distractor_idxs] = 0.0
            if self.model.layers[0].bias is not None:
                self.model.layers[0].bias[distractor_idxs] = 0.0
    
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
        distractor_callback: Callable[[torch.Tensor], torch.Tensor] = None,
        use_bias: bool = True,
    ) -> tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
    """Forward pass with a callback that can replace hidden unit outputs with distractor values.
    
    Args:
        x: Input tensor
        distractor_callback: Callable that takes a tensor and returns a tensor of the same shape
            which should be a mix of the same values and distractor values.
        use_bias: Whether to use a bias in the hidden layer.

    Returns:
        tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
            - Output tensor
            - Dictionary of parameter inputs
    """
    param_inputs = {}
    for i in range(0, len(self.layers) - 2, 2):
        param_inputs[self.layers[i].weight] = x
        x = self.layers[i](x) # Linear layer
        
        if i == 0 and distractor_callback is not None:
            x = distractor_callback(x)
        
        x = self.layers[i + 1](x) # Activation
        
        if use_bias:
            x[..., 0] = 1.0

    param_inputs[self.layers[-1].weight] = x
    return self.layers[-1](x), param_inputs


def prepare_components(cfg: DictConfig):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    use_bias = cfg.model.get('use_bias', True)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim = cfg.task.n_features,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim + int(use_bias),
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        seed = seed_from_string(base_seed, 'model'),
    )
    model.to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    
    # Determine if we need separate optimizers for the intermediate and output layers
    repr_optimizer_name = cfg.get('representation_optimizer', {}).get('name')
    assert repr_optimizer_name != 'idbd', "IDBD is not supported for the representation optimizer!"
    repr_module = model.layers[:-1]
    n_repr_trainable_layers = len([p for p in repr_module.parameters() if p.requires_grad])
    
    if repr_optimizer_name is not None and n_repr_trainable_layers > 0:
        # Use separate optimizers for the intermediate and output layers
        repr_optimizer = prepare_optimizer(repr_module, repr_optimizer_name, cfg.representation_optimizer)
        optimizer = prepare_optimizer(model.layers[-1], cfg.optimizer.name, cfg.optimizer)
        logger.info(f"Using separate optimizers for the intermediate and output layers: {repr_optimizer_name} and {cfg.optimizer.name}")
    else:
        # Only use one optimizer
        repr_optimizer = None
        optimizer = prepare_optimizer(model, cfg.optimizer.name, cfg.optimizer)
        logger.info(f"Using single optimizer: {cfg.optimizer.name}")
    
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
    if cfg.feature_recycling.use_cbp_utility:
        cbp_cls = SignedCBPTracker if cfg.feature_recycling.use_signed_utility else CBPTracker
        cbp_tracker = cbp_cls(
            optimizer = optimizer,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate = cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.feature_recycling.feature_protection_steps,
            initial_step_size_method = cfg.feature_recycling.initial_step_size_method,
            seed = seed_from_string(base_seed, 'cbp_tracker'),
        )
        cbp_tracker.track_sequential(model.layers)
    else:
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
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.0 # 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    torch.manual_seed(seed_from_string(base_seed, 'experiment_setup'))

    return task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        repr_optimizer: Optional[Optimizer],
        cbp_tracker: CBPTracker,
        distractor_tracker: DistractorTracker,
    ):
    use_bias = cfg.model.get('use_bias', True)
    
    # Distractor setup
    n_hidden_units = model.layers[-1].in_features
    first_feature_idx = 1 if use_bias else 0 # First feature is bias if enabled
    distractor_tracker.process_new_features(list(range(first_feature_idx, n_hidden_units)))

    # Training loop
    step = 0
    prev_pruned_idxs = set()
    prune_layer = model.layers[-2]
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Flags
    log_utility_stats = cfg.train.get('log_utility_stats', False)
    log_pruning_stats = cfg.train.get('log_pruning_stats', False)
    log_model_stats = cfg.train.get('log_model_stats', False)
    log_optimizer_stats = cfg.train.get('log_optimizer_stats', False)

    # Initialize accumulators
    cumulant_stats = StandardizationStats(gamma=0.99)
    cumulative_loss = np.float128(0.0)
    loss_accum = 0.0
    mean_pred_loss_accum = 0.0
    effective_lr_accum = 0.0
    pruned_accum = 0
    pruned_newest_feature_accum = 0
    n_steps_since_log = 0
    total_pruned = 0
    prune_thresholds = []
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

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            if log_pruning_stats:
                pre_prune_utilities = cbp_tracker.get_statistics(prune_layer)['utility']

            if isinstance(cbp_tracker, SignedCBPTracker):
                pruned_idxs = cbp_tracker.prune_features(targets)
            else:
                pruned_idxs = cbp_tracker.prune_features()
            n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
            total_pruned += n_pruned

            if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
                new_feature_idxs = pruned_idxs[prune_layer].tolist()
                distractor_process_idxs = new_feature_idxs
    
                # Don't turn bias into a distractor
                if use_bias:
                    distractor_process_idxs = [idx for idx in distractor_process_idxs if idx != 0]

                # Turn some features into distractors
                distractor_tracker.process_new_features(distractor_process_idxs)

                # Log pruning statistics
                pruned_accum += len(new_feature_idxs)
                n_new_pruned_features = len(set(new_feature_idxs).intersection(prev_pruned_idxs))
                pruned_newest_feature_accum += n_new_pruned_features
                prev_pruned_idxs = set(new_feature_idxs)
                
                if log_pruning_stats:
                    prune_thresholds.append(pre_prune_utilities[new_feature_idxs].max().item())
        
        # Forward pass
        outputs, param_inputs = model(
            features, distractor_tracker.replace_features, use_bias)
        loss = criterion(outputs, targets)
        
        with torch.no_grad():
            if cfg.train.standardize_cumulants:
                baseline_pred = torch.zeros_like(targets)
            else:
                baseline_pred = cumulant_stats.running_mean.cpu().view(1, 1)
            mean_pred_loss = criterion(baseline_pred, targets)

        # Backward pass
        optimizer.zero_grad()
        if repr_optimizer is not None:
            repr_optimizer.zero_grad()
        
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            retain_graph = optimizer.version == 'squared_grads'
            loss.backward(retain_graph=retain_graph)
            stats = optimizer.step(outputs, param_inputs)
            effective_lr_accum += list(stats.values())[0]['effective_step_size'].mean().item()
        else:
            loss.backward()
            optimizer.step()
            
        if repr_optimizer is not None:
            repr_optimizer.step()
        
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
            
            if log_optimizer_stats and isinstance(optimizer, IDBD):
                states = list(optimizer.state.values())
                assert len(states) == 1, "There should not be more than one optimizer state!"
                state = states[0]
                step_sizes = torch.exp(state['beta'])
                metrics['mean_step_size'] = step_sizes.mean().item()
                metrics['median_step_size'] = step_sizes.median().item()
                metrics['effective_lr'] = effective_lr_accum / n_steps_since_log
            effective_lr_accum = 0.0

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
