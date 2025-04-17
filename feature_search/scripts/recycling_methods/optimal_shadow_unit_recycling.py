"""
This script is used to run experiments for better understanding how long it takes for new
features to mature and converge.
More specifically, the goal of this script is to produce metrics that explain how long it
takes for a feature to become "safe", and how long it takes for the associated weight to converge
as a function of the optimal utility of the feature and the number of other existing features.
"""


from collections import defaultdict, OrderedDict
import copy
import logging
import os
import sys
from typing import Iterator, List, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig, open_dict

from feature_recycling import reset_input_weights
from idbd import IDBD, RMSPropIDBD
from models import ACTIVATION_MAP, LTU, MLP
from tasks import NonlinearGEOFFTask
from experiment_helpers import *
from scripts.feature_maturity_experiment import *


CONVERGENCE_N_SAMPLES = 200_000


logger = logging.getLogger(__name__)


def compute_optimal_stats(
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the optimal weights and utilities given the task and model structure."""

    device = next(model.parameters()).device

    inputs, targets = next(task.get_iterator(batch_size=CONVERGENCE_N_SAMPLES))
    inputs = inputs.to(device)
    targets = targets.to(device)

    loss = np.inf

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_features = model.layers[-1].in_features
    init_lr = 1.0 / np.sqrt(n_features)
    optimizer = torch.optim.SGD(trainable_params, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.5,
        patience = 10,
        verbose = False,
    )
    step = 0
    min_lr = init_lr * 0.001
    
    while True:
        # Compute optimal weights
        outputs, param_inputs = model(inputs)
        loss = criterion(outputs, targets)

        scheduler.step(loss.item())
        
        # Stop if learning rate gets too small
        if optimizer.param_groups[0]['lr'] <= min_lr:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    logger.info(f'Optimal model converged in {step} steps with a loss of {loss.item():.5f}')

    with torch.no_grad():
        weight_layer = model.layers[-1].weight
        shadow_weight_layer = model.shadow_layers[-1].weight
        optimal_weights = weight_layer.data
        shadow_optimal_weights = shadow_weight_layer.data
        feature_weight_sums = weight_layer.abs().sum(dim=0)
        shadow_feature_weight_sums = shadow_weight_layer.abs().sum(dim=0)

        layer_inputs = param_inputs[weight_layer]
        shadow_layer_inputs = param_inputs[shadow_weight_layer]
        input_magnitudes = layer_inputs.abs().mean(dim=0)
        shadow_input_magnitudes = shadow_layer_inputs.abs().mean(dim=0)

        optimal_utilities = input_magnitudes * feature_weight_sums
        shadow_optimal_utilities = shadow_input_magnitudes * shadow_feature_weight_sums

    return optimal_weights, optimal_utilities, shadow_optimal_weights, shadow_optimal_utilities, loss.item()


def reset_model(model: MLP):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data.zero_()
        model._initialize_weights(model.shadow_layers[0], 'binary')


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        model: MLP,
        criterion: nn.Module,
        cbp_tracker: CBPTracker,
    ):
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    feature_layer = model.layers[-2]
    feature_layer._forward_hooks = OrderedDict()
    
    # Initialize accumulators
    total_pruned = 0

    while step < cfg.train.total_steps:
        
        # Compute utilities with optimal weights
        reset_model(model) # Set all trainable params to zero
        optimal_weights, optimal_utilities, optimal_shadow_utilities, optimal_shadow_utilities, loss = \
            compute_optimal_stats(task, model, criterion)
        optimal_utilities = optimal_utilities.cpu().numpy()
        optimal_shadow_utilities = optimal_shadow_utilities.cpu().numpy()

        # Prune the lowest utility feature, and set weights based on the highest utility shadow unit
        prune_feature_idx = np.argmin(optimal_utilities)
        cbp_tracker._prune_layer(feature_layer, [prune_feature_idx])
        best_shadow_unit_idx = np.argmax(optimal_shadow_utilities)
        with torch.no_grad():
            model.layers[0].weight[prune_feature_idx, :] = model.shadow_layers[0].weight[best_shadow_unit_idx, :]
        total_pruned += 1
    
        # Log the results
        wandb.log({
            'pruned_utility': optimal_utilities[prune_feature_idx],
            'total_pruned': total_pruned,
            'loss': loss,
            'min_utility': optimal_utilities.min(),
            'max_utility': optimal_utilities.max(),
            'mean_utility': optimal_utilities.mean(),
        })
    
    pbar.close()
    wandb.finish()


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
        device: str = 'cuda'
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
            device: Device to put model on
        """
        super().__init__(
            input_dim = input_dim,
            output_dim = output_dim,
            n_layers = n_layers,
            hidden_dim = hidden_dim,
            weight_init_method = weight_init_method,
            activation = activation,
            n_frozen_layers = n_frozen_layers,
            device = device
        )
        assert n_layers == 2, "Shadow units MLP must have exactly 2 layers!"

        self.n_shadow_units = n_shadow_units
        activation_cls = ACTIVATION_MAP[activation]
        
        # Build layers
        self.shadow_layers = nn.ModuleList()
        self.shadow_layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.shadow_layers.append(activation_cls())
        for _ in range(n_layers - 2):
            self.shadow_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.shadow_layers.append(activation_cls())
        self.shadow_layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        
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


def custom_prepare_components(cfg: DictConfig):
    model = ShadowUnitsMLP(
        input_dim = cfg.task.n_features,
        n_shadow_units = cfg.n_shadow_units,
        output_dim = cfg.model.output_dim,
        n_layers = cfg.model.n_layers,
        hidden_dim = cfg.model.hidden_dim,
        weight_init_method = cfg.model.weight_init_method,
        activation = cfg.model.activation,
        n_frozen_layers = cfg.model.n_frozen_layers,
        device = cfg.device,
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

    if cbp_tracker is not None:
        cbp_tracker.incoming_weight_init = 'binary'
    
    # Init target output weights to kaiming uniform and predictor output weights to zero
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
    )
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker


def prepare_experiment(cfg: DictConfig):
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        custom_prepare_components(cfg)

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
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
    )
    torch.nn.init.zeros_(model.layers[-1].weight)
    
    # Change LTU threshold for target and predictors
    ltu_threshold = 0.1 * cfg.task.n_features
    for layer in model.layers:
        if isinstance(layer, LTU):
            layer.threshold = ltu_threshold
    task.activation_fn.threshold = ltu_threshold

    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker


@hydra.main(config_path='../../conf', config_name='feature_maturity_defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    if not cfg.get('n_shadow_units'):
        with open_dict(cfg):
            cfg.n_shadow_units = 100
            cfg.train.optimizer = 'adam' # TODO: Don't forget about this workaround (because IDBD doesn't support shadow units)

    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
        prepare_experiment(cfg)
    cbp_tracker._utility_reset_mode = 'zero'
    model.to(cfg.get('optimal_weight_device', cfg.device))

    run_experiment(cfg, task, model, criterion, cbp_tracker)


if __name__ == '__main__':
    main()
