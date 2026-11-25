"""The algorithms of the connection_slots progression, from the algorithm of step 2 of the meta progression."""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from phd.sandbox.unit_search.algorithm.algorithms import Network, PruneAndMove, UnitSearch, select


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
        needed = self.n_outputs * self.n_inputs + self.width * self.new_ltu_cost
        if needed > self.budget:
            raise ValueError(f'the initial network needs {needed} connections, more than the budget of {self.budget}')

    @property
    def new_ltu_cost(self) -> int:
        """The connections that a new LTU adds to n_connections."""
        return self.initial_fan_in + self.n_outputs

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
        new = slots & (jnp.cumsum(slots) <= room // self.new_ltu_cost)
        return self.with_output_connections(UnitSearch.add_units(self, network, key, new, at_random))


class CountCandidates(CountAllConnections):
    """CountAllConnections whose budget also counts the candidates: the layer inputs that an LTU may read but does
    not read, whose gradient sums add_connections tests. Each LTU then counts every layer input it may read, so a new
    LTU costs n_inputs + n_outputs, whatever it reads."""

    @property
    def new_ltu_cost(self) -> int:
        return self.n_inputs + self.n_outputs

    def n_connections(self, network: Network) -> Int[Array, '']:
        return (self.may_read(network.depth)[:, :-1].sum()   # an LTU's connections and candidates
                + network.outputs.connections[:, :-1].sum())


class NoConnectionSearch(CountAllConnections):
    """CountAllConnections with no add_connections and no prune_connections: an LTU keeps the connections it has at
    creation."""

    def add_connections(self, network: Network) -> Network:
        return network

    def prune_connections(self, network: Network) -> Network:
        return network


class Static(NoConnectionSearch):
    """NoConnectionSearch with also no prune_units and no add_units: the network of step 0 never changes."""

    def prune_units(self, network: Network) -> tuple[Network, Bool[Array, 'ltus']]:
        return network, jnp.zeros(self.n_ltus, bool)

    def add_units(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus'],
                  at_random: bool = False) -> Network:
        return network


# --------------------------------------------------------------------------------------------------------------------
# The output's budget
# --------------------------------------------------------------------------------------------------------------------

class Slots(NamedTuple):
    """What OutputBudget keeps for its structure changes."""
    output_signs: Float[Array, 'outputs layer_inputs']   # the output's weight signs at the last connection pruning
    unit_bucket: Float[Array, '']                        # the free connections kept for new LTUs
    key: PRNGKeyArray                                    # for the uniform draws of add_connections
    traces: Any = None                                   # what a subclass keeps for its connection pruning


SlotNetwork = NamedTuple('SlotNetwork', [*Network.__annotations__.items(), ('slots', Slots)])


class OutputBudget(PruneAndMove):
    """The output holds the whole budget. Each unit owns its incoming connections, and the biases do not count. Each
    LTU has initial_fan_in connections from x, and the output's connection to it, so a new LTU costs
    initial_fan_in + 1 connections. An LTU exists while a unit reads it.

    Two buckets hold the free connections: one for the output's new connections, and one for new LTUs. Half of the
    connections that the output frees, by pruning its connections or LTUs, go to each bucket. At step 0, half of the
    budget goes to each bucket.

    prune_connections  every prune_period steps: the sign rule of PruneAndMove on the LTUs and on the output. The LTUs
                       that no unit reads any more are removed.
    add_connections    each LTU refills its free slots with uniform draws from the x it does not read, from weight 0
                       at Autostep's initial step size. The output reads layer inputs it does not read, drawn
                       uniformly, as many as the connection bucket holds, the same way. The part of the connection
                       bucket that has no layer input left to read goes to the unit bucket.
    prune_units        as in PruneAndMove.
    add_units          NEW LTUs in the free slots, as many as the unit bucket holds. A new LTU reads initial_fan_in of
                       x, drawn in proportion to its parent's weights squared, then uniformly among the x with a parent
                       weight of 0. The n_random_weights inputs drawn first get LeCun uniform weights with the signs of
                       the parent weights, at the new-weight step size. The others start at weight 0 at Autostep's
                       initial step size. The output reads the new LTU from weight 0.
    """
    depths: int = eqx.field(static=True, default=1)
    initial_fan_in: int = eqx.field(static=True, default=30)     # each LTU's connections from x
    n_random_weights: int = eqx.field(static=True, default=15)   # of a new LTU's connections, those with random weights

    def __check_init__(self):
        if self.depths != 1 or self.n_outputs != 1:
            raise ValueError('OutputBudget has 1 depth and 1 output')
        if not self.n_random_weights <= self.initial_fan_in <= self.n_inputs:
            raise ValueError('OutputBudget needs n_random_weights <= initial_fan_in <= n_inputs')

    def n_connections(self, network: Network) -> Int[Array, '']:
        return network.ltus.connections[:, :-1].sum() + network.outputs.connections[:, :-1].sum()

    def owned(self, network: Network) -> Int[Array, '']:
        """The connections that the output pays for: its own, and initial_fan_in for each LTU."""
        return network.outputs.connections[:, :-1].sum() + self.initial_fan_in * (network.depth > 0).sum()

    def split_freed(self, before: Network, after: Network) -> Network:
        """Half of the connections freed between `before` and `after` go to the unit bucket."""
        slots = after.slots
        return after._replace(slots=slots._replace(
            unit_bucket=slots.unit_bucket + (self.owned(before) - self.owned(after)) / 2))

    def draw(self, key: PRNGKeyArray, open_: Bool[Array, 'units layer_inputs'],
             counts: Int[Array, 'units']) -> Bool[Array, 'units layer_inputs']:
        """For each unit, `counts` of its `open_` layer inputs, drawn uniformly without replacement."""
        scores = jnp.where(open_, jax.random.gumbel(key, open_.shape), -jnp.inf)
        return (jnp.argsort(jnp.argsort(-scores)) < counts[:, None]) & open_

    def new_network(self, depth, ltu_weights, ltu_connections, output_connections):
        network = super().new_network(depth, ltu_weights, ltu_connections, output_connections)
        slots = Slots(jnp.zeros_like(network.outputs.weights), jnp.zeros(()), jax.random.PRNGKey(0))
        return SlotNetwork(*network, slots)

    # --- init ---

    def init(self, key: PRNGKeyArray):
        """No LTUs, and the output reads only the bias. Half of the budget goes to each bucket, then add_connections
        and add_units spend them."""
        connections_key, units_key = jax.random.split(key)
        no_weights = jnp.zeros((self.n_ltus, self.n_layer_inputs))
        bias_only = jnp.zeros((self.n_outputs, self.n_layer_inputs), bool).at[:, -1].set(True)
        network = self.new_network(jnp.zeros(self.n_ltus, int), no_weights, no_weights != 0, bias_only)
        network = network._replace(slots=network.slots._replace(unit_bucket=jnp.array(self.budget / 2),
                                                                key=connections_key))
        network = self.add_connections(network)
        return self.add_units(network, units_key, network.depth == 0)

    # --- prune_connections ---

    def prune_connections(self, network):
        """The sign rule on the LTUs (PruneAndMove) and on the output, then the removal of the LTUs that no unit
        reads."""
        pruned = super().prune_connections(network)
        signs = pruned.slots.output_signs
        weight_signs = jnp.sign(pruned.outputs.weights)
        output_cut = pruned.outputs.connections & (signs != 0) & (weight_signs != signs)
        pruned = self.cut(pruned, jnp.zeros_like(pruned.ltus.connections), output_cut)
        output_signs = (weight_signs * pruned.outputs.connections).at[:, -1].set(0.0)
        pruned = pruned._replace(slots=pruned.slots._replace(output_signs=output_signs))
        return select(network.step % self.prune_period == 0, pruned, network)

    def cut(self, network, ltu_cut: Bool[Array, 'ltus layer_inputs'], output_cut: Bool[Array, 'outputs layer_inputs']):
        """The LTUs and the output stop reading the cut layer inputs. Then the LTUs that no unit reads are removed,
        and half of the connections that the output frees go to the unit bucket."""
        cut = network._replace(
            ltus=self.reset(network.ltus, ltu_cut, 0.0)._replace(connections=network.ltus.connections & ~ltu_cut),
            outputs=self.reset(network.outputs, output_cut, 0.0)._replace(
                connections=network.outputs.connections & ~output_cut))
        read = (cut.outputs.connections[:, self.ltu_activation_idxs].any(0)
                | cut.ltus.connections[:, self.ltu_activation_idxs].any(0))
        return self.split_freed(network, self.empty(cut, (cut.depth > 0) & ~read))

    # --- add_connections ---

    def add_connections(self, network):
        """The LTUs refill their slots, then the output spends its connection bucket."""
        ltu_key, output_key, key = jax.random.split(network.slots.key, 3)
        ltus, outputs = network.ltus, network.outputs
        open_ = (self.may_read(network.depth) & ~ltus.connections).at[:, -1].set(False)
        missing = jnp.where(network.depth > 0, self.initial_fan_in - ltus.connections[:, :-1].sum(1), 0)
        new = self.draw(ltu_key, open_, missing)
        ltus = self.reset(ltus, new, 0.0)._replace(connections=ltus.connections | new)

        exists = jnp.concatenate([jnp.ones(self.n_inputs, bool), network.depth > 0, jnp.zeros(1, bool)])
        open_ = exists[None] & ~outputs.connections
        bucket = self.budget - self.owned(network) - network.slots.unit_bucket
        n_open = open_.sum(1)
        new = self.draw(output_key, open_, jnp.minimum(jnp.floor(bucket), n_open))
        outputs = self.reset(outputs, new, 0.0)._replace(connections=outputs.connections | new)
        unit_bucket = network.slots.unit_bucket + jnp.maximum(bucket - n_open, 0).sum()
        return network._replace(ltus=ltus, outputs=outputs,
                                slots=network.slots._replace(unit_bucket=unit_bucket, key=key))

    # --- prune_units ---

    def prune_units(self, network):
        pruned, free = super().prune_units(network)
        return self.split_freed(network, pruned), free

    def empty(self, network, removed: Bool[Array, 'ltus']):
        """As in PruneAndMove, and the output stops reading the removed LTUs."""
        network = super().empty(network, removed)
        columns = jnp.zeros(self.n_layer_inputs, bool).at[self.ltu_activation_idxs].set(removed)
        return network._replace(
            outputs=network.outputs._replace(connections=network.outputs.connections & ~columns),
            slots=network.slots._replace(output_signs=jnp.where(columns, 0.0, network.slots.output_signs)))

    # --- add_units ---

    def add_units(self, network, key: PRNGKeyArray, slots: Bool[Array, 'ltus'], at_random: bool = False):
        cost = self.initial_fan_in + 1
        new = slots & (jnp.cumsum(slots) <= jnp.floor(network.slots.unit_bucket / cost))
        network = network._replace(slots=network.slots._replace(
            unit_bucket=network.slots.unit_bucket - cost * new.sum()))
        return UnitSearch.add_units(self, network, key, new, at_random)

    def wire(self, network, key: PRNGKeyArray, removed: Bool[Array, 'ltus'], new: Bool[Array, 'ltus'],
             at_random: bool):
        """The `new` LTUs read the bias and initial_fan_in of x, and their parent output reads them."""
        draw_key, weight_key = jax.random.split(key)
        parent_weights = self.parent_weights(network)
        gumbel = jax.random.gumbel(draw_key, parent_weights.shape)
        scores = jnp.where(parent_weights != 0, jnp.log(parent_weights ** 2) + gumbel, gumbel - 1e4)
        scores = jnp.where(self.may_read(network.depth).at[:, -1].set(False), scores, -jnp.inf)
        rank = jnp.argsort(jnp.argsort(-scores))
        drawn = rank < self.initial_fan_in
        weights, fast = self.initial_weights(weight_key, parent_weights, rank)

        ltus = self.reset(network.ltus, new[:, None], weights * drawn)
        step_sizes = jnp.where(new[:, None] & fast, jnp.log(self.new_weight_step_size), ltus.optimizer_state.beta)
        ltus = ltus._replace(connections=jnp.where(new[:, None], drawn.at[:, -1].set(True), ltus.connections),
                             optimizer_state=ltus.optimizer_state._replace(beta=step_sizes))
        columns = jnp.zeros(self.n_layer_inputs, bool).at[self.ltu_activation_idxs].set(new)
        links = jnp.zeros_like(network.outputs.connections).at[:, self.ltu_activation_idxs].set(
            network.parent[None] - self.n_ltus == jnp.arange(self.n_outputs)[:, None]) & columns
        outputs = self.reset(network.outputs, links, 0.0)
        return network._replace(ltus=ltus, outputs=outputs._replace(connections=outputs.connections | links),
                                norm=network.norm.restart(new),
                                gradient_sums=jnp.where(new[:, None], 0.0, network.gradient_sums))

    def initial_weights(self, key: PRNGKeyArray, parent_weights, rank: Int[Array, 'ltus layer_inputs']):
        """A new LTU's weights on the inputs it draws, and where they start at the new-weight step size (the others
        start at Autostep's initial step size): LeCun uniform weights with the signs of the parent weights on the
        n_random_weights inputs drawn first, and 0 on the others."""
        fast = rank < self.n_random_weights
        lecun = jax.random.uniform(key, rank.shape, minval=-1.0, maxval=1.0) * jnp.sqrt(3.0 / self.n_random_weights)
        weights = jnp.where(parent_weights != 0, jnp.abs(lecun) * jnp.sign(parent_weights), lecun) * fast
        return weights, fast


class SoftCandidates(OutputBudget):
    """OutputBudget in which a new LTU starts every input as a soft candidate: weights from
    U(-initial_step_size, initial_step_size), at Autostep's initial step size."""

    def initial_weights(self, key: PRNGKeyArray, parent_weights, rank: Int[Array, 'ltus layer_inputs']):
        weights = jax.random.uniform(key, rank.shape, minval=-self.initial_step_size, maxval=self.initial_step_size)
        return weights, jnp.zeros(rank.shape, bool)


class SiftingTraces(NamedTuple):
    """What Sifting keeps, for the LTUs then the output."""
    log_immaturity: Float[Array, 'units layer_inputs']   # log(1 - maturity) of each connection, 0 at its creation
    delta_squared: Float[Array, 'units']                 # running mean of dL/d(pre-activation)^2, before bias correction


class Sifting(OutputBudget):
    """OutputBudget whose prune_connections is the CI rule of ../../feature_sifting/networks/
    parallel_mnist_label_changes.ipynb in place of the sign rule. Every step, after the weight update, each connection
    of the LTUs and the output (not the biases) is cut if |w| + |h| is below the bound that a dead weight's stays under
    68% of the time at the same age: sqrt(alpha E[delta^2] / 2 * m) (1.61 + 0.39 (1 - m)^0.75). delta is the unit's
    dL/d(pre-activation), E[delta^2] its running mean (bias-corrected), and m the connection's maturity before this
    step. m = 1 - prod_k (1 - alpha_k c_k)^2, with c_k = 1 - (1 - running_average_rate)^k. The cuts then act as in
    OutputBudget: an LTU that the output stops reading is removed, and the freed connections go to the buckets. A new
    LTU starts with E[delta^2] = 0."""

    def new_network(self, depth, ltu_weights, ltu_connections, output_connections):
        network = super().new_network(depth, ltu_weights, ltu_connections, output_connections)
        n_units = self.n_ltus + self.n_outputs
        traces = SiftingTraces(jnp.zeros((n_units, self.n_layer_inputs)), jnp.zeros(n_units))
        return network._replace(slots=network.slots._replace(traces=traces))

    def prune_connections(self, network):
        return network

    def update_measures(self, network, pre_activation_gradients, layer_inputs):
        network = super().update_measures(network, pre_activation_gradients, layer_inputs)
        traces = network.slots.traces
        rate = self.running_average_rate
        delta_squared = traces.delta_squared + rate * (pre_activation_gradients ** 2 - traces.delta_squared)
        warmed = -jnp.expm1(network.step * jnp.log1p(-rate))   # 1 - (1 - rate)^step
        layers = (network.ltus, network.outputs)
        weights, connections = (jnp.concatenate([getattr(layer, name) for layer in layers])
                                for name in ('weights', 'connections'))
        h, step_sizes = (jnp.concatenate([getattr(layer.optimizer_state, name) for layer in layers])
                         for name in ('h', 'beta'))
        step_sizes = jnp.exp(step_sizes)
        log_immaturity = traces.log_immaturity
        bound = (jnp.sqrt(step_sizes * (delta_squared / warmed)[:, None] / 2 * -jnp.expm1(log_immaturity))
                 * (1.61 + 0.39 * jnp.exp(0.75 * log_immaturity)))
        cut = (connections & (jnp.abs(weights) + jnp.abs(h) < bound)).at[:, -1].set(False)
        network = self.cut(network, cut[:self.n_ltus], cut[self.n_ltus:])
        connections = jnp.concatenate([network.ltus.connections, network.outputs.connections])
        log_immaturity = jnp.where(connections, log_immaturity + 2 * jnp.log1p(-jnp.minimum(step_sizes * warmed, 1.0)),
                                   0.0)
        return network._replace(slots=network.slots._replace(traces=SiftingTraces(log_immaturity, delta_squared)))

    def empty(self, network, removed: Bool[Array, 'ltus']):
        """As in OutputBudget, and the removed LTUs' connections in and out lose their maturity."""
        network = super().empty(network, removed)
        columns = jnp.zeros(self.n_layer_inputs, bool).at[self.ltu_activation_idxs].set(removed)
        rows = jnp.append(removed, jnp.zeros(self.n_outputs, bool))
        traces = network.slots.traces
        traces = traces._replace(log_immaturity=jnp.where(rows[:, None] | columns[None], 0.0, traces.log_immaturity),
                                 delta_squared=jnp.where(rows, 0.0, traces.delta_squared))
        return network._replace(slots=network.slots._replace(traces=traces))


class SiftingSoftCandidates(Sifting, SoftCandidates):
    """SoftCandidates with the connection pruning of Sifting."""
