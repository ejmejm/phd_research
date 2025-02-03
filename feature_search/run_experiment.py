from dataclasses import dataclass
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from adam import Adam
from idbd import IDBD, RMSPropIDBD
from models import MLP
from tasks import DummyTask, GEOFFTask, NonlinearGEOFFTask


omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))


@dataclass
class FeatureInfo:
    """Stores information about a feature including its utility and distribution parameters."""
    is_real: bool
    utility: float
    distribution_params: Dict[str, Any]
    last_update: int
    creation_step: int


class FeatureRecycler:
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
            
        if isinstance(optimizer, (IDBD, RMSPropIDBD)):
            first_layer = model.layers[0]
            idbd_beta = optimizer.state[first_layer.weight]['beta']
            learning_rates = torch.exp(idbd_beta).mean(dim=0)
            stats['mean_learning_rate_real'] = learning_rates[real_indices].mean().item() if real_indices else 0
            stats['mean_learning_rate_distractor'] = learning_rates[distractor_indices].mean().item() if distractor_indices else 0
            
        elif isinstance(optimizer, Adam):
            first_layer = model.layers[0]
            state = optimizer.state[first_layer.weight]
            
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


def reset_feature_weights(idxs: Union[int, Sequence[int]], model: MLP, optimizer: optim.Optimizer, cfg: DictConfig):
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


def prepare_task(cfg: DictConfig):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'dummy':
        return DummyTask(cfg.task.n_features, cfg.model.output_dim, cfg.task.type)
    elif cfg.task.name.lower() == 'static_linear_geoff':
        # Non-stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, -1, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'linear_geoff':
        # Stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, 20, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features=cfg.task.n_real_features,
            flip_rate=cfg.task.flip_rate,
            n_layers=cfg.task.n_layers,
            hidden_dim=cfg.task.hidden_dim if cfg.task.n_layers > 1 else 0,
            weight_scale=cfg.task.weight_scale,
            activation=cfg.task.activation,
            sparsity=cfg.task.sparsity,
            weight_init=cfg.task.weight_init,
        )


def prepare_optimizer(model: nn.Module, cfg: DictConfig):
    """Prepare the optimizer based on configuration."""
    if cfg.train.optimizer == 'adam':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'rmsprop':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            betas=(0, 0.999),
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'sgd_momentum':
        return optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
        )
    elif cfg.train.optimizer == 'idbd':
        return IDBD(
            model.parameters(),
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            version=cfg.idbd.version,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'rmsprop_idbd':
        return RMSPropIDBD(
            model.parameters(),
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            trace_diagonal_approx=cfg.idbd.diagonal_approx,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f'Invalid optimizer type: {cfg.train.optimizer}')


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_statistics(model: MLP, features: torch.Tensor, param_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics about the model's weights, biases, and layer inputs.
    
    Args:
        model: The MLP model to analyze
        features: Input features to compute layer activations
        param_inputs: Dictionary mapping weight parameters to their inputs
        
    Returns:
        Dictionary containing various model statistics
    """
    stats = {}
    
    # Compute statistics for each layer
    for i, layer in enumerate(model.layers):
        # Weight norms
        weight_l1 = torch.norm(layer.weight, p=1).item() / layer.weight.numel()
        stats[f'layer_{i}/weight_l1'] = weight_l1
        
        # Bias norms (if exists)
        if layer.bias is not None:
            bias_l1 = torch.norm(layer.bias, p=1).item() / layer.bias.numel()
            stats[f'layer_{i}/bias_l1'] = bias_l1
        
        # Input norms
        if i == 0:
            input_l1 = torch.norm(features, p=1, dim=1).mean().item() / features.shape[1]
        else:
            layer_inputs = param_inputs[layer.weight]
            input_l1 = torch.norm(layer_inputs, p=1, dim=-1).mean().item() / layer_inputs.shape[-1]
        stats[f'layer_{i}/input_l1'] = input_l1
    
    return stats


def standardize_targets(
    targets: torch.Tensor,
    cumulant_mean: float,
    cumulant_square_mean: float,
    cumulant_gamma: float,
    step: int,
) -> torch.Tensor:
    """Standardize targets using a running mean and variance."""
    cumulant_mean = cumulant_gamma * cumulant_mean + (1 - cumulant_gamma) * targets.mean()
    cumulant_square_mean = cumulant_gamma * cumulant_square_mean + (1 - cumulant_gamma) * targets.square().mean()
    bias_correction = 1 / (1 - cumulant_gamma ** (step + 1))
    curr_mean = cumulant_mean * bias_correction
    curr_square_mean = cumulant_square_mean * bias_correction
    std_dev = (curr_square_mean - curr_mean.square()).sqrt()
    std_dev = 1 if std_dev == 0 else std_dev
    targets = (targets - curr_mean) / std_dev
    return targets, cumulant_mean, cumulant_square_mean



@hydra.main(config_path='conf', config_name='defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    set_seed(cfg.seed)
    
    if not cfg.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Initialize wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.project, config=wandb_config, allow_val_change=True)
    
    task = prepare_task(cfg)
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim=cfg.task.n_features,
        output_dim=cfg.model.output_dim,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        weight_init_method=cfg.model.weight_init_method,
        activation=cfg.model.activation,
        device=cfg.device
    ).to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    optimizer = prepare_optimizer(model, cfg)
    
    # Initialize feature recycler
    recycler = FeatureRecycler(
        n_features=cfg.task.n_features,
        n_real_features=cfg.task.n_real_features,
        distractor_chance=cfg.feature_recycling.distractor_chance,
        recycle_rate=cfg.feature_recycling.recycle_rate,
        utility_decay=cfg.feature_recycling.utility_decay,
        use_cbp_utility=cfg.feature_recycling.use_cbp_utility,
        feature_protection_steps=cfg.feature_recycling.feature_protection_steps,
        n_start_real_features=cfg.feature_recycling.get('n_start_real_features', -1),
        device=cfg.device,
    )
    
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    cumulant_mean = 0.0
    cumulant_square_mean = 0.0
    cumulant_gamma = 0.999
    target_buffer = []
    
    while step < cfg.train.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        target_buffer.extend(targets.view(-1).tolist())
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        # Get recycled features
        features, recycled_features = recycler.step(
            batch_size=inputs.size(0),
            real_features=inputs,
            first_layer_weights=model.get_first_layer_weights(),
            step_num=step
        )
        
        if cfg.train.standardize_cumulants:
            with torch.no_grad():
                targets, cumulant_mean, cumulant_square_mean = standardize_targets(
                    targets, cumulant_mean, cumulant_square_mean, cumulant_gamma, step)

        # Reset weights and optimizer states for recycled features
        reset_feature_weights(recycled_features, model, optimizer, cfg)
        
        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        if isinstance(optimizer, RMSPropIDBD):
            loss.backward(create_graph=True)
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            optimizer.step(param_inputs)
        elif isinstance(optimizer, IDBD):
            # Mean over batch dimension
            param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
            optimizer.step(loss, outputs, param_inputs)
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate metrics
        loss_acc += loss.item()
        cumulative_loss += loss.item()
        n_steps_since_log += 1
        
        # Calculate and accumulate accuracy for classification
        if isinstance(criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            accuracy_acc += accuracy
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_acc / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'accuracy': accuracy_acc / n_steps_since_log if isinstance(criterion, nn.CrossEntropyLoss) else None,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
            }
            # Add recycler statistics
            metrics.update(recycler.get_statistics(step, model, optimizer))
            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'], accuracy=metrics['accuracy'])
            
            # Reset accumulators
            loss_acc = 0.0
            accuracy_acc = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1
        pbar.update(1)
    
    pbar.close()
    wandb.finish()

if __name__ == '__main__':
    main()
