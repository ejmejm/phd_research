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

from phd.feature_search.core.idbd import IDBD as OriginalIDBD
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


# TODO: Update ShadowCBPTracker so that that both real and shadow unit input weights are initialized
#       as if there are |real weights| inputs or |real weights| + 1 inputs when using influence utility
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


# TODO: Update IDBD so that that both real and shadow unit weights are updated as if there
#       are |real weights| inputs or |real weights| + 1 inputs when using influence utility
class IDBD(OriginalIDBD):
    def __init__(
        self,
        *args,
        utility_type: str = 'cbp',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.utility_type = utility_type
        assert self.version in ('squared_grads', 'squared_inputs'), \
            f"Invalid version: {self.version}. Must be one of: squared_grads, squared_inputs."

    @torch.no_grad()
    def step(
        self,
        predictions: torch.Tensor,
        param_inputs: Dict[torch.nn.parameter.Parameter, torch.Tensor],
    ) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            predictions: Predictions tensor of shape (batch_size, n_classes).
                Only needed for `squared_grads` and `hvp` versions of IDBD.
            param_inputs: Dictionary mapping linear layer weight parameters to their inputs
                Only needed for `squared_inputs` version of IDBD.
            retain_graph: Whether to retain the graph of the computation
        """
        if self.version == 'squared_grads':
            all_params = [p for group in self.param_groups for p in group['params']]
            with torch.enable_grad():
                prediction_sum = torch.sum(predictions)
            prediction_grads = torch.autograd.grad(
                outputs = prediction_sum,
                inputs = all_params,
                retain_graph = False,
            )
            prediction_grads = {p: g for p, g in zip(all_params, prediction_grads)}

        param_updates = []
        for group in self.param_groups:
            meta_lr = group['meta_lr']
            tau = group['tau']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if p in param_inputs:
                    assert len(param_inputs[p].shape) == 1, "Inputs must be 1D tensors"
                    inputs = param_inputs[p].unsqueeze(0)
                elif len(grad.shape) == 1:
                    # This branch is currently not used because I disabled support for bias parameters
                    inputs = torch.ones_like(grad)
                else:
                    raise ValueError(f"Parameter {p} not found in activations dictionary.")
                
                # Get state variables
                state = self.state[p]
                beta = state['beta']
                h = state['h']
                if self.autostep:
                    v = state['v']
                
                
                ### Different versions of IDBD change how h is decayed ###
                
                if self.version == 'squared_inputs':
                    h_decay_term = inputs.pow(2)
                elif self.version == 'squared_grads':
                    h_decay_term = prediction_grads[p].pow(2)
                else:
                    raise ValueError(f"Invalid IDBD version: {self.version}")
                
                
                ### Update state variables ###
                
                # Calculate and update step-size (learning rate / alpha)
                if self.autostep:
                    alpha = torch.exp(state['beta'])
                    v = torch.max(
                        torch.abs(grad * h),
                        v + 1.0 / tau * alpha * h_decay_term * (torch.abs(grad * h) - v),
                    )
                    new_alpha = alpha * torch.exp(meta_lr * grad * h / v)
                    alpha = torch.where(
                        v != 0,
                        new_alpha,
                        alpha,
                    )
                    
                    # Normalize the step-size
                    effective_step_size = torch.clamp(torch.sum(alpha * h_decay_term, dim=-1), min=1.0)
                    alpha = alpha / effective_step_size.unsqueeze(1)
                    state['beta'] = torch.log(alpha)
                else:
                    beta.add_(meta_lr * grad * h)
                    state['beta'] = beta
                    alpha = torch.exp(beta)
                
                # Queue paramter update
                weight_decay_term = self.weight_decay * p.data if self.weight_decay != 0 else 0
                param_update = -alpha * (grad + weight_decay_term)
                param_updates.append((p, param_update))
                
                # Update h (gradient trace)
                state['h'] = h * (1 - alpha * h_decay_term).clamp(min=0) + alpha * grad
                
        for p, param_update in param_updates:
            p.add_(param_update)


# TODO: Consider changing the calculation for shadow weights so that the loss includes the contribution of just that single shadow unit
# TODO: Check to make sure the shadow weights are not changing how squared_grads version of IDBD works.
#       Can check this by making sure squared_grads and squared_inputs give the exact same results.


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
        
        # Forward pass
        outputs, param_inputs = model(features)
        param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        retain_graph = isinstance(optimizer, IDBD) and optimizer.version == 'squared_grads'
        loss.backward(retain_graph=retain_graph)

        # Modify shadow unit grads if using influence utility
        if shadow_cbp_tracker is not None and shadow_cbp_tracker.utility_type == 'influence':
            # TODO: This will need an update if I want to add in backprop at some point
            with torch.no_grad():
                shadow_out_weights = model.shadow_layers[2].weight
                influence_grads = 2 * param_inputs[shadow_out_weights].unsqueeze(0) * shadow_out_weights
                shadow_out_weights.grad += influence_grads


        # Update weights
        if isinstance(optimizer, IDBD):
            # Mean over batch dimension
            optimizer.step(outputs, param_inputs)
        else:
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


def prepare_components(cfg: DictConfig, model: Optional[nn.Module] = None):
    """Prepare the components based on configuration."""
    base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32)
    
    if not cfg.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Initialize wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.project, config=wandb_config, allow_val_change=True)
    task = prepare_task(cfg, seed=seed_from_string(base_seed, 'task'))
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    if model is None:
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
    model.to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification'
                else nn.MSELoss())
    if cfg.train.optimizer == 'idbd':
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = IDBD(
            trainable_params,
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            version=cfg.idbd.version,
            weight_decay=cfg.train.weight_decay,
            autostep=cfg.idbd.autostep,
            utility_type=cfg.shadow_feature_recycling.utility_type,
        )
    else:
        optimizer = prepare_optimizer(model, cfg)
    
    # Initialize feature recycler
    recycler = InputRecycler(
        n_features=cfg.task.n_features,
        n_real_features=cfg.task.n_real_features,
        distractor_chance=cfg.input_recycling.distractor_chance,
        recycle_rate=cfg.input_recycling.recycle_rate,
        utility_decay=cfg.input_recycling.utility_decay,
        use_cbp_utility=cfg.input_recycling.use_cbp_utility,
        feature_protection_steps=cfg.input_recycling.feature_protection_steps,
        n_start_real_features=cfg.input_recycling.get('n_start_real_features', -1),
        device=cfg.device,
        seed=seed_from_string(base_seed, 'recycler'),
    )
    
    # Initialize CBP tracker
    if cfg.feature_recycling.use_cbp_utility:
        cbp_tracker = CBPTracker(
            optimizer = optimizer,
            replace_rate = cfg.feature_recycling.recycle_rate,
            decay_rate = cfg.feature_recycling.utility_decay,
            maturity_threshold = cfg.feature_recycling.feature_protection_steps,
            seed=seed_from_string(base_seed, 'cbp_tracker'),
        )
        cbp_tracker.track_sequential(model.layers)
    else:
        cbp_tracker = None
        
    return task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker

def prepare_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    task, task_iterator, model, criterion, optimizer, recycler, cbp_tracker = \
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
