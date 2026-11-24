"""The algorithms of the connection_slots progression, from the algorithm of step 2 of the meta progression."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from phd.sandbox.unit_search.algorithm.algorithms import Network, PruneAndMove, UnitSearch


class CountAllConnections(PruneAndMove):
    """PruneAndMove with 1 depth, whose budget counts every connection except the biases: the LTUs' connections from x,
    and the outputs' connections from x and from the LTUs. The outputs read x, the bias and every LTU, so a new LTU
    takes initial_fan_in + n_outputs connections.

    init         the `width` LTUs. A ValueError at construction if their connections and the outputs' do not fit in
                 the budget.
    prune_units  as in PruneAndMove. The outputs stop reading the removed LTUs.
    add_units    NEW LTUs in the free slots, while the budget has room for initial_fan_in + n_outputs connections
                 each. The outputs read the new LTUs.
    """
    depths: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        if self.depths != 1:
            raise ValueError('CountAllConnections has 1 depth')
        needed = self.n_outputs * self.n_inputs + self.width * (self.initial_fan_in + self.n_outputs)
        if needed > self.budget:
            raise ValueError(f'the initial network needs {needed} connections, more than the budget of {self.budget}')

    def n_connections(self, network: Network) -> Int[Array, '']:
        return network.ltus.connections[:, :-1].sum() + network.outputs.connections[:, :-1].sum()

    def with_output_connections(self, network: Network) -> Network:
        """The outputs read x, the bias and the LTUs, and no empty slot."""
        reads = jnp.concatenate([jnp.ones(self.n_inputs, bool), network.depth > 0, jnp.ones(1, bool)])
        connections = jnp.broadcast_to(reads, network.outputs.connections.shape)
        return network._replace(outputs=network.outputs._replace(connections=connections))

    def init(self, key: PRNGKeyArray) -> Network:
        """The first `width` slots hold new LTUs. __check_init__ ensures that they fit in the budget."""
        filled = jnp.arange(self.n_ltus) < self.width
        no_weights = jnp.zeros((self.n_ltus, self.n_layer_inputs))
        network = self.new_network(filled.astype(int), no_weights, no_weights != 0,
                                   jnp.zeros((self.n_outputs, self.n_layer_inputs), bool))
        network = UnitSearch.add_units(self, network, key, filled, at_random=True)
        return self.with_output_connections(network)

    def prune_units(self, network: Network) -> tuple[Network, Bool[Array, 'ltus']]:
        network, free = super().prune_units(network)
        return self.with_output_connections(network), free

    def add_units(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus'],
                  at_random: bool = False) -> Network:
        room = self.budget - self.n_connections(network)
        new = slots & (jnp.cumsum(slots) <= room // (self.initial_fan_in + self.n_outputs))
        return self.with_output_connections(UnitSearch.add_units(self, network, key, new, at_random))
