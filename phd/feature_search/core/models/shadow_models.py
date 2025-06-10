import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Iterator, List, Tuple

from .base import ACTIVATION_MAP


EPSILON = 1e-8
MIN_UTILITY = -10.0


class DynamicShadowMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        n_layers: int,
        hidden_dim: int = 128,
        weight_init_method: str = 'zeros',
        activation: str = 'relu',
        n_frozen_layers: int = 0,
        utility_decay: float = 0.99,
        max_promotions_per_step: float = 0.5,
        max_demotions_per_step: float = 0.5,
        promotion_threshold: float = 0.01,
        demotion_threshold: float = 0.0,
        keep_promotion_weights: bool = False,
        n_initial_real_features: int = 0,
        seed: Optional[int] = None,
    ):
        """Extension of the SparseInputMLP that is able to generate new features and prune old ones.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros' or 'kaiming')
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            n_frozen_layers: Number of frozen layers
            utility_decay: Rate at which feature utilities are decayed.
            max_promotions_per_step: The maximum number of shadow features that can be promoted to
                real features per step. If fractional, then the value is accumulated until it reaches
                one, at which point a feature may be promoted.
            max_demotions_per_step: The maximum number of real features that can be demoted to
                shadow features per step. If fractional, then the value is accumulated until it reaches
                one, at which point a feature may be demoted.
            promotion_threshold: The fraction of the residual error a shadow feature must reduce to
                be promoted.
            demotion_threshold: The fraction of the total error (i.e. target) a real feature must
                reduce to avoid being demoted.
            keep_promotion_weights: Whether to maintain the the weights of a shadow feature when it is
                promoted to a real feature.
            n_initial_real_features: The number of real features to initialize with.
            seed: Optional random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_frozen_layers = n_frozen_layers
        self.weight_init_method = weight_init_method
        self.utility_decay = utility_decay
        self.max_promotions_per_step = max_promotions_per_step
        self.max_demotions_per_step = max_demotions_per_step
        self.keep_promotion_weights = keep_promotion_weights
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.seed = seed

        self.promotion_accumulator = 0.0
        self.demotion_accumulator = 0.0

        assert self.weight_init_method == 'zeros', "DynamicShadowMLP only supports weight initialization to zero"
        assert n_layers == 2, "DynamicShadowMLP must have exactly 2 layers!"

        # Create a generator if seed is provided
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None(seed) if seed is not None else None
        
        self.activation = ACTIVATION_MAP[activation]
        
        # Build layers
        # Really the combined real and shadow features together should add up to a total count of 
        # input_dim, but instead I will make both input_dim and use masking to get the desired behavior.
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.active_output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.inactive_output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

        # Initialize weights
        self._zero_init_layer(self.active_output_layer)
        self._zero_init_layer(self.inactive_output_layer)
        
        # Freeze layers as necessary
        if n_frozen_layers >= 1:
            self.input_layer.weight.requires_grad = False
        if n_frozen_layers == 2:
            self.active_output_layer.weight.requires_grad = False
            self.inactive_output_layer.weight.requires_grad = False
        if n_frozen_layers > 2:
            raise ValueError("DynamicShadowMLP only supports up to 2 frozen layers!")
        
        # Set up feature mask and utilities
        active_feature_mask = torch.zeros(hidden_dim, dtype=torch.bool)
        active_feature_mask[:n_initial_real_features] = True
        self.register_buffer('active_feature_mask', active_feature_mask)
        
        self.active_feature_utilities = torch.zeros(hidden_dim, dtype=torch.float32)
        self.inactive_feature_utilities = torch.zeros(hidden_dim, dtype=torch.float32)

        self.update_step = 0

        # Both are traces of the absolute values of the corresponding terms
        self.register_buffer('target_error_trace', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('target_trace', torch.zeros(1, dtype=torch.float32))

    def _zero_init_layer(self, layer: nn.Linear):
        nn.init.zeros_(layer.weight)
    
    # TODO: Handle setting step size on promotions (and demotions and maybe adjusting other step sizes?)
    @torch.no_grad()
    def _handle_promotions(self) -> List[int]:
        """Handle promotions of shadow features to real features.
        
        Returns:
            A list of the indices of the promoted features.
        """
        available_promotions = int(self.promotion_accumulator)
        self.promotion_accumulator += self.max_promotions_per_step - available_promotions

        if available_promotions == 0:
            return []
        
        bias_corrected_error_trace = self.target_error_trace / (1 - self.utility_decay ** self.update_step)
        abs_promotion_threshold = self.promotion_threshold * bias_corrected_error_trace
        n_promotions_possible = (
            (self.inactive_feature_utilities > abs_promotion_threshold)
            * ~self.active_feature_mask
        ).sum()
        n_promotions = min(available_promotions, n_promotions_possible)

        if n_promotions == 0:
            return []

        # Get indices of top n_promotions inactive feature utilities
        _, top_indices = torch.sort(self.inactive_feature_utilities, descending=True)
        promotion_indices = top_indices[:n_promotions]

        # Update the feature mask
        self.active_feature_mask[promotion_indices] = True

        # Update utilities
        self.active_feature_utilities[promotion_indices] = self.inactive_feature_utilities[promotion_indices]
        self.inactive_feature_utilities[promotion_indices] = 0.0

        # Update outgoing weights of both real and shadow features
        active_feature_weights = self.active_output_layer.weight
        inactive_feature_weights = self.inactive_output_layer.weight
        if self.keep_promotion_weights:
            active_feature_weights[:, promotion_indices] = inactive_feature_weights[:, promotion_indices]
        else:
            active_feature_weights[:, promotion_indices] = 0.0
        inactive_feature_weights[:, promotion_indices] = 0.0

        return promotion_indices

    @torch.no_grad()
    def _handle_demotions(self) -> List[int]:
        """Handle demotions of real features to shadow features.
        
        Returns:
            A list of the indices of the promoted features.
        """
        available_demotions = int(self.demotion_accumulator)
        self.demotion_accumulator += self.max_demotions_per_step - available_demotions

        if available_demotions == 0:
            return []
        
        bias_corrected_error_trace = self.target_error_trace / (1 - self.utility_decay ** self.update_step)
        abs_demotion_threshold = self.demotion_threshold * bias_corrected_error_trace
        n_demotions_possible = (
            (self.active_feature_utilities <= abs_demotion_threshold)
            * self.active_feature_mask
        ).sum()
        n_demotions = min(available_demotions, n_demotions_possible)

        if n_demotions == 0:
            return []

        # Get indices of top n_demotions real utilities
        _, bottom_indices = torch.sort(self.active_feature_utilities, descending=False)
        demotion_indices = bottom_indices[:n_demotions]

        # Update the feature mask
        self.active_feature_mask[demotion_indices] = False

        # Update utilities
        self.inactive_feature_utilities[demotion_indices] = self.active_feature_utilities[demotion_indices]
        self.active_feature_utilities[demotion_indices] = 0.0

        # Update outgoing weights of both real and shadow features
        active_feature_weights = self.active_output_layer.weight
        inactive_feature_weights = self.inactive_output_layer.weight
        if self.keep_promotion_weights:
            inactive_feature_weights[:, demotion_indices] = active_feature_weights[:, demotion_indices]
        else:
            inactive_feature_weights[:, demotion_indices] = 0.0
        active_feature_weights[:, demotion_indices] = 0.0

        return demotion_indices

    def active_parameters(self) -> Iterator[nn.Parameter]:
        return self.active_output_layer.parameters()
        
    def inactive_parameters(self) -> Iterator[nn.Parameter]:
        return self.inactive_output_layer.parameters()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        update_state: bool = False,
    ) -> Tuple[torch.Tensor, Dict[nn.Module, torch.Tensor], Dict[str, Any]]:
        assert len(x.shape) == 1, "DynamicMLP does not support batching!"
        
        aux = {}
        if update_state:
            aux['promoted_features'] = self._handle_promotions()
            aux['demoted_features'] = self._handle_demotions()

        hidden_features = self.input_layer(x)
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