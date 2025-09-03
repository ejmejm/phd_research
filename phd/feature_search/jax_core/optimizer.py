import equinox as eqx
from jaxtyping import PyTree
import optax


class EqxOptimizer(eqx.Module):
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    optimizer_state: PyTree

    def __init__(self, optimizer: optax.GradientTransformation, optimizer_state: PyTree):
        self.optimizer = optimizer
        self.optimizer_state = optimizer_state

    def with_state(self, optimizer_state: PyTree) -> 'EqxOptimizer':
        return EqxOptimizer(self.optimizer, optimizer_state)

    def with_update(self, updates, params) -> PyTree:
        return EqxOptimizer(
            self.optimizer,
            self.optimizer.update(updates, self.optimizer_state, params),
        )