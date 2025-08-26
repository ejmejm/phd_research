import math
import logging
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterator, Optional


EPSILON = 1e-8


logger = logging.getLogger(__name__)


class IDBD(Optimizer):
    """Incremental Delta-Bar-Delta optimizer.
    
    This is an implementation of the IDBD algorithm adapted for deep neural networks.
    Instead of working with input features directly, it uses gradients with respect
    to parameters and maintains separate learning rates for each parameter.
    
    Args:
        params: Iterable of parameters to optimize
        meta_lr: Meta learning rate (default: 0.01)
        init_lr: Initial learning rate (default: 0.01)
        weight_decay: Weight decay (default: 0.0)
        version: Version of IDBD to use (default: squared_inputs)
        autostep: Whether to use autostep (default: False)
        tau: Tau parameter for autostep (default: 1e4)
    """
    
    def __init__(
        self, 
        params: Iterator[torch.Tensor],
        meta_lr: float = 0.01,
        init_lr: float = 0.01,
        weight_decay: float = 0.0,
        version: str = 'squared_grads', # {squared_inputs, squared_grads}
        autostep: bool = False,
        tau: float = 1e4,
    ):
        param_list = list(params)
        defaults = dict(meta_lr=meta_lr, tau=tau)
        super().__init__(param_list, defaults)
        self.weight_decay = weight_decay
        self.init_lr = init_lr
        self.version = version
        self.autostep = autostep
        self.tau = tau
        
        assert self.version in ['squared_inputs', 'squared_grads'], \
            f"Invalid version: {self.version}. Must be one of: squared_inputs, squared_grads."
        
        if autostep:
            # Check that parameters match a linear layer structure
            weights = [p for p in param_list if len(p.shape) in (2, 3)]
            biases = [p for p in param_list if len(p.shape) == 1]
            
            assert len(biases) == 0, "AutoStep optimizer does not support biases!"
            assert len(weights) > 0, "No valid weight parameters found!"

            if len(weights) > 1:
                logger.warning(
                    "Found multiple sets of weights, but AutoStep does not support non-linear  "
                    "layer structures. If the weights provided to AutoStep are stacked and not "
                    "independent, then this will probably cause a silent bug."
                )

        # Initialize beta and h for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['beta'] = torch.full_like(p.data, math.log(init_lr))
                state['h'] = torch.zeros_like(p.data)
                if autostep:
                    state['v'] = torch.zeros_like(p.data)
    

    @torch.no_grad()
    def step(
        self,
        predictions: Optional[torch.Tensor] = None,
        param_inputs: Optional[Dict[torch.nn.parameter.Parameter, torch.Tensor]] = None,
        retain_graph: bool = False,
        features_independent: bool = False,
    ) -> Dict[str, Any]:
        """Performs a single optimization step.
        
        Args:
            predictions: Predictions tensor of shape (batch_size, n_classes).
                Only needed for `squared_grads` and `hvp` versions of IDBD.
            param_inputs: Dictionary mapping linear layer weight parameters to their inputs
                Only needed for `squared_inputs` version of IDBD.
            retain_graph: Whether to retain the graph of the computation
            features_independent: Whether to treat each unit as an independent predictor
            
        Returns:
            Dictionary of statistics for each parameter group.
        """
        if self.version == 'squared_inputs':
            assert param_inputs is not None, "Parameter inputs are required for squared_inputs version of IDBD"
        
        elif self.version == 'squared_grads':
            all_params = [p for group in self.param_groups for p in group['params']]
            with torch.enable_grad():
                prediction_sum = torch.sum(predictions)
            prediction_grads = torch.autograd.grad(
                outputs = prediction_sum,
                inputs = all_params,
                retain_graph = retain_graph,
            )
            prediction_grads = {p: g for p, g in zip(all_params, prediction_grads)}

        param_updates = []
        stats = {}
        for group in self.param_groups:
            meta_lr = group['meta_lr']
            tau = group['tau']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                group_stats = {}
                stats[p] = group_stats
            
                grad = p.grad
                
                if self.version == 'squared_inputs':
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
                    if features_independent:
                        raw_effective_step_size = alpha * h_decay_term
                        effective_step_size = torch.clamp(raw_effective_step_size, min=1.0)
                    else:
                        raw_effective_step_size = torch.sum(alpha * h_decay_term, dim=-1)
                        effective_step_size = torch.clamp(raw_effective_step_size, min=1.0)
                        effective_step_size = effective_step_size.unsqueeze(-1)
                    
                    group_stats['effective_step_size'] = raw_effective_step_size.squeeze()
                    
                    alpha = alpha / effective_step_size
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
            
        return stats


if __name__ == '__main__':
    print("Testing IDBD optimizer...")
    
    # Test 1
    print("\nTest 1: Linear Regression w/ Overshooting (IDBD decreases learning rate)")
    torch.manual_seed(42)
    X = torch.tensor([[1.0, 2.0]])
    true_weights = torch.tensor([[1.5, -0.5]])
                              # [0.5, 1.0]])
    y = X @ true_weights.t()

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.data.copy_(torch.tensor([[1.0, -1.0]]))
                                           # [0.5, 1.0]]))
    optimizer = IDBD(model.parameters(), meta_lr=0.0001, init_lr=0.5, autostep=True)
    
    for _ in range(10):
        y_pred = model(X[0])
        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y[0])
        print('loss:', loss.item(), 'step-size:', torch.exp(optimizer.state[model.weight]['beta']))
        
        param_inputs = {model.weight: X[0]}
        optimizer.zero_grad()
        optimizer.step(loss, y_pred, param_inputs)

    # Test 2
    print("\nTest 2: Linear Regression w/ Undershooting (IDBD increases learning rate)")
    with torch.no_grad():
        model.weight.data.copy_(torch.tensor([[1.0, -1.0]]))
                                           # [0.5, 1.0]]))
    optimizer = IDBD(model.parameters(), meta_lr=0.1, init_lr=0.001, autostep=True)
    
    for _ in range(10):
        y_pred = model(X[0])
        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y[0])
        print('loss:', loss.item(), 'step-size:', torch.exp(optimizer.state[model.weight]['beta']))
        
        param_inputs = {model.weight: X[0]}
        optimizer.zero_grad()
        optimizer.step(loss, y_pred, param_inputs)
    
    
    # # Test with 2-layer network
    # print("\nTest: 2-layer network")
    # torch.manual_seed(42)
    
    # # Create random input
    # X = torch.randn(10, 4)
    # y = torch.randn(10, 2)

    # # Create 2-layer network
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(4, 8),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(8, 2)
    # )
    
    # optimizer = IDBD(model.parameters(), meta_lr=0.01, init_lr=0.01)
    
    # # Single training step
    
    # for _ in range(100):
    #     y_pred = model(X)
    #     loss = torch.nn.functional.mse_loss(y_pred, y)
    #     print('Loss:', loss.item())
    #     loss.backward(create_graph=True)
    #     optimizer.step()