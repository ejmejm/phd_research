from typing import Optional, Tuple

import equinox as eqx
from jaxtyping import PyTree
import optax

from phd.feature_search.jax_core.utils import tree_replace


class EqxOptimizer(eqx.Module):
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    filter_spec: Optional[PyTree] = eqx.field(default=None, static=True)
    state: PyTree

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        model: eqx.Module,
        filter_spec: Optional[PyTree] = None,
    ):
        self.optimizer = optimizer
        self.filter_spec = filter_spec
        
        if filter_spec is not None:
            trainable_params = eqx.filter(model, filter_spec)
        else:
            trainable_params = model
        
        self.state = self.optimizer.init(trainable_params)

    def with_update(self, grads, params) -> Tuple[PyTree, 'EqxOptimizer']:
        """Update the optimizer state and return a new optimizer."""
        if self.filter_spec is not None:
            grads = eqx.filter(grads, self.filter_spec)
            
        updates, new_state = self.optimizer.update(grads, self.state, params)
        return updates, tree_replace(self, state=new_state)