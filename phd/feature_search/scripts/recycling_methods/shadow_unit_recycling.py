import logging
import math
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig

from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.models import MLP, ACTIVATION_MAP
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.experiment_helpers import *
from phd.feature_search.scripts.feature_maturity_experiment import *


logger = logging.getLogger(__name__)


class ShadowUnitsMLP(MLP):
    def __init__(
        self, 
        input_dim: int,
        n_shadow_units: int,
        output_dim: int,
        n_layers: int,
        hidden_dim: int,
        weight_init_method: str,
        activation: str = 'tanh',
        n_frozen_layers: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Number of input features
            n_shadow_units: Number of shadow units
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros', 'kaiming', or 'binary')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            n_frozen_layers: Number of frozen layers
        """
        super().__init__(
            input_dim = input_dim,
            output_dim = output_dim,
            n_layers = n_layers,
            hidden_dim = hidden_dim,
            weight_init_method = weight_init_method,
            activation = activation,
            n_frozen_layers = n_frozen_layers,
            seed = seed,
        )
        assert n_layers == 2, "Shadow units MLP must have exactly 2 layers!"

        self.n_shadow_units = n_shadow_units
        activation_cls = ACTIVATION_MAP[activation]
        
        # Build layers
        self.shadow_layers = nn.ModuleList()
        self.shadow_layers.append(nn.Linear(input_dim, n_shadow_units, bias=False))
        self.shadow_layers.append(activation_cls())
        for _ in range(n_layers - 2):
            self.shadow_layers.append(nn.Linear(n_shadow_units, n_shadow_units, bias=False))
            self.shadow_layers.append(activation_cls())
        self.shadow_layers.append(nn.Linear(n_shadow_units, output_dim, bias=False))
        
        # Freeze layers
        for i in range(n_frozen_layers):
            layer = self.shadow_layers[int(i*2)]
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        
        # Initialize shadow weights
        self._initialize_weights(self.shadow_layers[0], weight_init_method)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[nn.Module, torch.Tensor]]:
        param_inputs = {}

        param_inputs[self.layers[0].weight] = x
        features = self.layers[0](x)
        features = self.layers[1](features)
        param_inputs[self.layers[2].weight] = features
        
        param_inputs[self.shadow_layers[0].weight] = x
        shadow_features = self.shadow_layers[0](x)
        shadow_features = self.shadow_layers[1](shadow_features)
        param_inputs[self.shadow_layers[2].weight] = shadow_features

        real_value = self.layers[2](features)
        shadow_value = self.shadow_layers[2](shadow_features)

        value = real_value + shadow_value - shadow_value.detach()

        return value, param_inputs


class ShadowCBPTracker(CBPTracker):
    """Perform CBP for recycling features."""
    
    # TODO: Reimplement weight initialization between model, cbp, and input recycler
    # so that they share the same initialization methods
    def __init__(
        self,
        *args,
        utility_type: str = 'cbp', # {'cbp', 'age_normalized', 'influence'}
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert utility_type in ('cbp', 'age_normalized', 'influence'), \
            "Invalid shadow utility type! Must be one of: {cbp, age_normalized, influence}."
        self.utility_type = utility_type

    def _get_hook(self, layer: nn.Module):
        """Return a hook function for a given layer."""
        def track_cbp_stats(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
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
                raw_utility = output_magnitude * self._get_output_weight_sums(output_layer)
                
                if self.utility_type == 'age_normalized':
                    age_normalized_utility = raw_utility / (self._feature_stats[module]['age'] + 1)
                    self._feature_stats[module]['utility'] = (1 - self.decay_rate) * age_normalized_utility \
                        + self.decay_rate * self._feature_stats[module]['utility']
                else:
                    self._feature_stats[module]['utility'] = (1 - self.decay_rate) * raw_utility \
                        + self.decay_rate * self._feature_stats[module]['utility']

        return track_cbp_stats
    


# TODO: Consider changing the calculation for shadow weights so that the loss includes the contribution of just that single shadow unit


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        cbp_tracker: CBPTracker,
        shadow_cbp_tracker: CBPTracker,
    ):

    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_accum = 0.0
    pruned_accum = 0
    pruned_newest_feature_accum = 0
    n_steps_since_log = 0
    total_pruned = 0
    total_shadow_pruned = 0
    target_buffer = []

    while step < cfg.train.total_steps:

        # Generate batch of data
        inputs, targets = next(task_iterator)
        target_buffer.extend(targets.view(-1).tolist())
        features, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        # Forward pass
        outputs, param_inputs = model(features)
        loss = criterion(outputs, targets)

        # Reset weights and optimizer states for recycled features
        if cbp_tracker is not None:
            pruned_idxs = cbp_tracker.prune_features()
            n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
            total_pruned += n_pruned

            if pruned_idxs:
                pruned_idxs = pruned_idxs[model.layers[-2]]
                assert len(pruned_idxs) > 0, "No features were pruned!"
                assert len(pruned_idxs) <= model.shadow_layers[-1].in_features, \
                    "Not enough shadow units to replace all pruned features!"
                
                # TODO: When the shadow utility type is set to 'influence', consider also copying over outgoing weights
                # Replace each of the pruned features with the highest utility shadow features

                shadow_feature_utilities = list(shadow_cbp_tracker._feature_stats.values())[0]['utility']
                shadow_feature_rankings = torch.argsort(shadow_feature_utilities, descending=True)

                for i, real_idx in enumerate(pruned_idxs):
                    with torch.no_grad():
                        shadow_feature_weights = model.shadow_layers[0].weight[shadow_feature_rankings[i], :]
                        model.layers[0].weight[real_idx, :] = shadow_feature_weights
                
                used_shadow_unit_idxs = shadow_feature_rankings[:len(pruned_idxs)]
                shadow_feature_layer = model.shadow_layers[1] # Activation layer
                shadow_cbp_tracker._prune_layer(shadow_feature_layer, used_shadow_unit_idxs)
        
        if shadow_cbp_tracker is not None:
            pruned_idxs = shadow_cbp_tracker.prune_features()
            n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
            total_shadow_pruned += n_pruned

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
        n_steps_since_log += 1
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            feature_utilities = list(cbp_tracker._feature_stats.values())[0]['utility']
            shadow_feature_utilities = list(shadow_cbp_tracker._feature_stats.values())[0]['utility']
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_accum / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                'units_pruned': total_pruned,
                'pruned_shadow_units': total_shadow_pruned,
                'utility_mean': feature_utilities.mean(),
                'utility_std': feature_utilities.std(),
                'shadow_utility_mean': shadow_feature_utilities.mean(),
                'shadow_utility_std': shadow_feature_utilities.std(),
            }

            if pruned_accum > 0:
                metrics['fraction_pruned_were_new'] = pruned_newest_feature_accum / pruned_accum
                pruned_newest_feature_accum = 0
                pruned_accum = 0

            # Add model statistics
            metrics.update(get_model_statistics(model, features, param_inputs))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'])
            pbar.update(cfg.train.log_freq)
            
            # Reset accumulators
            loss_accum = 0.0
            n_steps_since_log = 0
            target_buffer = []

        step += 1

    pbar.close()
    wandb.finish()


def prepare_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)

    full_hidden_dim = cfg.model.hidden_dim
    n_shadow_units = int(cfg.model.fraction_shadow_units * full_hidden_dim)
    n_real_units = full_hidden_dim - n_shadow_units
    model = ShadowUnitsMLP(
        input_dim = cfg.task.n_features,
        n_shadow_units = n_shadow_units,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = n_real_units,
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        seed = seed_from_string(base_seed, 'model'),
    )

    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_components(cfg, model=model)

    assert isinstance(task, NonlinearGEOFFTask)
    
    assert cfg.model.weight_init_method == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.weight_init == 'binary', \
        "Binary weight initialization is required for reproducing Mahmood and Sutton (2013)"
    assert cfg.model.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"
    assert cfg.task.activation == 'ltu', \
        "LTU activations are required for reproducing Mahmood and Sutton (2013)"

    # Additionaly initialize the CBP tracker for the shadow units
    if cfg.shadow_feature_recycling.use_cbp_utility:
        shadow_cbp_tracker = ShadowCBPTracker(
            optimizer = optimizer,
            replace_rate = cfg.shadow_feature_recycling.recycle_rate,
            decay_rate = cfg.shadow_feature_recycling.utility_decay,
            maturity_threshold = cfg.shadow_feature_recycling.feature_protection_steps,
            seed = seed_from_string(cfg.seed, 'shadow_cbp_tracker'),
            utility_type = cfg.shadow_feature_recycling.utility_type,
        )
        shadow_cbp_tracker.track_sequential(model.shadow_layers)
    
    if cbp_tracker is not None:
        cbp_tracker.incoming_weight_init = 'binary'
    if shadow_cbp_tracker is not None:
        shadow_cbp_tracker.incoming_weight_init = 'binary'

    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed_from_string(cfg.seed, 'task_init_generator'))
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    torch.manual_seed(seed_from_string(cfg.seed, 'experiment_setup'))

    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker, shadow_cbp_tracker


@hydra.main(config_path='../../conf', config_name='rupam_task_shadow_weights')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker, shadow_cbp_tracker = \
        prepare_experiment(cfg)

    run_experiment(cfg, task, task_iterator, model, criterion, optimizer, cbp_tracker, shadow_cbp_tracker)


if __name__ == '__main__':
    main()
