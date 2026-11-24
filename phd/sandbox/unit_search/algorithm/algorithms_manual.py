"""BACKUP: the version with a hand-written backward pass. The algorithm at each step of the meta progression, one class per step.

Each algorithm is an equinox module, and its static fields are its hyperparameters. `Algorithm.run` calls its parts:

    init(key)                              the network at step 0
    every step
        learn_step(network, x, y)          network Autostep, then update_measures for the structure changes
    every connection_period steps          change_connections
        prune_connections(network)
        add_connections(network)
    every unit_period steps                change_units
        move_units(network)
        prune_units(network)               returns the LTU slots to fill
        add_units(network, key, slots)

A part that an algorithm does not use leaves the network unchanged. A part that runs less often than its period does
nothing at the other calls. Each later class inherits from the class before it and overrides only the parts that
change.

The network. Every unit reads from the sources [inputs, LTUs, 1], where the last source is the bias. `Network.ltus`
holds the LTUs' incoming weights and `Network.outputs` holds the outputs'. Each LTU has a depth, and reads only
inputs, the bias and LTUs of lower depth.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from phd.jax_core.optimizers.idbd import IDBDState, optax_idbd


def ltu(pre_activations: Array) -> Array:
    """A linear threshold unit: 1 where the pre-activation is above 0, else 0."""
    return (pre_activations > 0).astype(jnp.float32)


def ltu_slope(pre_activations: Array) -> Array:
    """The slope that the backward pass uses for an LTU: the slope of the sigmoid."""
    sigmoid = jax.nn.sigmoid(pre_activations)
    return sigmoid * (1 - sigmoid)


def select(condition: Bool[Array, ''], if_true, if_false):
    """if_true where the condition holds, else if_false, for every array of two networks."""
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), if_true, if_false)


class Layer(NamedTuple):
    """A group of units, and their weights from the sources [inputs, LTUs, 1]."""
    weights: Float[Array, 'units sources']
    connections: Bool[Array, 'units sources']     # the sources that each unit reads
    shadow_weights: Float[Array, 'units sources'] # +1 or -1 on each LTU whose parent is this unit, else 0
    optimizer_state: IDBDState                    # Autostep's step sizes and traces, shaped like the weights


class Network(NamedTuple):
    ltus: Layer
    outputs: Layer
    depth: Int[Array, 'ltus']                             # 1 for the lowest LTUs
    firing_rate: Float[Array, 'ltus']                     # running average of each LTU's output
    error_mean: Float[Array, 'ltus']                      # running average of the error arriving at each LTU
    error_activation_mean: Float[Array, 'ltus']           # running average of that error times the LTU's activation
    gradient_sums: Float[Array, 'ltus sources_but_bias']  # sum of dL/d(pre-activation) * source since the last growth
    usefulness: Float[Array, 'ltus']                      # sum of |outgoing weights|


class Algorithm(eqx.Module):
    """Network Autostep on a network of LTUs, with a main loop that calls the structure changes."""
    n_inputs: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    depths: int = eqx.field(static=True, default=3)
    width: int = eqx.field(static=True, default=40)                     # LTU slots per depth
    initial_step_size: float = eqx.field(static=True, default=1e-4)     # Autostep's initial step size
    meta_step_size: float = eqx.field(static=True, default=0.02)        # Autostep's meta step size
    running_average_rate: float = eqx.field(static=True, default=1e-3)  # of the firing rates and error averages
    variance_floor: float = eqx.field(static=True, default=0.01)        # least variance that standardization divides by
    window: int = eqx.field(static=True, default=1_000)                 # steps per recorded error
    connection_period: int = eqx.field(static=True, default=1_000)      # steps between connection changes
    unit_period: int = eqx.field(static=True, default=10_000)           # steps between unit changes

    @property
    def n_ltus(self) -> int:
        return self.depths * self.width

    @property
    def n_sources(self) -> int:
        return self.n_inputs + self.n_ltus + 1

    @property
    def ltu_sources(self) -> slice:
        """The LTU columns of the sources."""
        return slice(self.n_inputs, self.n_inputs + self.n_ltus)

    @property
    def optimizer(self):
        return optax_idbd(meta_lr=self.meta_step_size, init_lr=self.initial_step_size, autostep=True, tau=1e4,
                          version='squared_inputs')

    # --- the main loop ---

    def run(self, problem, key: PRNGKeyArray, steps: int) -> Float[Array, 'windows']:
        """`steps` online steps on examples of `problem`. Returns the mean squared error of each window.
        Three nested scans, so that each structure change runs only at the end of its period. The window divides
        connection_period, connection_period divides unit_period, and unit_period divides `steps`."""
        structure_key, key = jax.random.split(key)
        network = self.init(structure_key)

        def learn_connection_period(network, key):
            network, squared_errors = jax.lax.scan(
                lambda network, key: self.learn_step(network, *problem.sample(key)), network,
                jax.random.split(key, self.connection_period))
            return self.change_connections(network), squared_errors.reshape(-1, self.window).mean(1)

        def learn_unit_period(network, inputs):
            index, keys = inputs
            network, mean_squared_errors = jax.lax.scan(learn_connection_period, network, keys)
            return self.change_units(network, jax.random.fold_in(structure_key, index)), mean_squared_errors

        n_unit_periods = steps // self.unit_period
        keys = jax.random.split(key, (n_unit_periods, self.unit_period // self.connection_period))
        mean_squared_errors = jax.lax.scan(learn_unit_period, network, (jnp.arange(n_unit_periods), keys))[1]
        return mean_squared_errors.reshape(-1)

    def change_connections(self, network: Network) -> Network:
        network = self.prune_connections(network)
        return self.add_connections(network)

    def change_units(self, network: Network, key: PRNGKeyArray) -> Network:
        network = self.move_units(network)
        network, slots = self.prune_units(network)
        return self.add_units(network, key, slots)

    # --- the parts that change the structure: none here ---

    def update_measures(self, network: Network, ltu_gradients, sources) -> Network:
        return network

    def prune_connections(self, network: Network) -> Network:
        return network

    def add_connections(self, network: Network) -> Network:
        return network

    def move_units(self, network: Network) -> Network:
        return network

    def prune_units(self, network: Network):
        return network, None

    def add_units(self, network: Network, key: PRNGKeyArray, slots) -> Network:
        return network

    # --- learn_step: network Autostep ---

    def learn_step(self, network: Network, x: Float[Array, 'inputs'],
                   y: Float[Array, 'outputs']) -> tuple[Network, Float[Array, '']]:
        """One online update on example (x, y). Returns the network and the squared error averaged over the outputs,
        before the update."""
        pre_activations, activations, firing_rate, sources = self.forward(network, x)
        predictions = (network.outputs.weights * network.outputs.connections) @ sources
        errors = predictions - y
        arriving_errors, ltu_gradients = self.backward(network, pre_activations, activations, firing_rate, errors)

        network = network._replace(
            ltus=self.update_weights(network.ltus, ltu_gradients, sources),
            outputs=self.update_weights(network.outputs, errors, sources),
            firing_rate=firing_rate,
            error_mean=self.running_average(network.error_mean, arriving_errors),
            error_activation_mean=self.running_average(network.error_activation_mean, arriving_errors * activations))
        return self.update_measures(network, ltu_gradients, sources), jnp.mean(errors ** 2)

    def forward(self, network: Network, x: Float[Array, 'inputs']):
        """The LTUs depth by depth. Returns their pre-activations, standardized activations and new firing rates, and
        the sources [x, activations, 1]."""
        weights = network.ltus.weights * network.ltus.connections
        pre_activations = jnp.zeros(self.n_ltus)
        activations = jnp.zeros(self.n_ltus)
        firing_rate = network.firing_rate
        for depth in range(1, self.depths + 1):
            at_depth = network.depth == depth
            depth_pre_activations = weights @ jnp.concatenate([x, activations, jnp.ones(1)])
            bits = ltu(depth_pre_activations)
            depth_firing_rate = self.running_average(network.firing_rate, bits)

            pre_activations = jnp.where(at_depth, depth_pre_activations, pre_activations)
            firing_rate = jnp.where(at_depth, depth_firing_rate, firing_rate)
            activations = jnp.where(at_depth, self.standardize(bits, depth_firing_rate), activations)
        return pre_activations, activations, firing_rate, jnp.concatenate([x, activations, jnp.ones(1)])

    def backward(self, network: Network, pre_activations, activations, firing_rate, errors):
        """From the top depth down: the error arriving at each LTU from the units that read it, then the gradient of
        the loss with respect to its pre-activation."""
        to_outputs = self.backward_weights(network.outputs)[:, self.ltu_sources]
        to_ltus = self.backward_weights(network.ltus)[:, self.ltu_sources]
        arriving_errors = jnp.zeros(self.n_ltus)
        ltu_gradients = jnp.zeros(self.n_ltus)
        for depth in range(self.depths, 0, -1):
            at_depth = network.depth == depth
            depth_errors = errors @ to_outputs + ltu_gradients @ to_ltus
            depth_gradients = (self.standardize_backward(network, depth_errors, activations, firing_rate)
                               * ltu_slope(pre_activations))

            arriving_errors = jnp.where(at_depth, depth_errors, arriving_errors)
            ltu_gradients = jnp.where(at_depth, depth_gradients, ltu_gradients)
        return arriving_errors, ltu_gradients

    def backward_weights(self, layer: Layer) -> Float[Array, 'units sources']:
        """The weights that the backward pass uses: each shadow weight while the real weight is smaller than 1 in
        size, and the real weights elsewhere."""
        weights = layer.weights * layer.connections
        return jnp.where((layer.shadow_weights != 0) & (jnp.abs(weights) < 1.0), layer.shadow_weights, weights)

    def update_weights(self, layer: Layer, unit_gradients, sources) -> Layer:
        """Autostep on a layer's weights, from the gradient of the loss with respect to each unit's pre-activation."""
        gradients = jnp.outer(unit_gradients, sources) * layer.connections
        inputs = sources * layer.connections
        updates, optimizer_state = self.optimizer.update((gradients, None, inputs), layer.optimizer_state,
                                                         layer.weights)
        return layer._replace(weights=layer.weights + updates, optimizer_state=optimizer_state)

    # --- standardization of the LTUs, as batch norm with running averages ---

    def running_average(self, average, value):
        return average + self.running_average_rate * (value - average)

    def standard_deviation(self, firing_rate):
        """An LTU's standard deviation from its firing rate, at least the square root of the variance floor."""
        return jnp.sqrt(jnp.maximum(firing_rate * (1 - firing_rate), self.variance_floor))

    def standardize(self, bits, firing_rate):
        return (bits - firing_rate) / self.standard_deviation(firing_rate)

    def standardize_backward(self, network: Network, errors, activations, firing_rate):
        """Batch norm's backward pass with running averages: the error, less its running average and less the
        activation times the running average of the error times the activation, over the standard deviation. While
        the variance is at its floor, the standard deviation does not depend on the LTU, so the second term drops."""
        above_floor = firing_rate * (1 - firing_rate) > self.variance_floor
        centered = (errors - network.error_mean
                    - jnp.where(above_floor, activations * network.error_activation_mean, 0.0))
        return centered / self.standard_deviation(firing_rate)

    # --- construction ---

    def new_network(self, depth, ltu_weights, ltu_connections, output_connections) -> Network:
        """A network with these LTU weights, zero output weights and no shadow weights."""
        output_weights = jnp.zeros((self.n_outputs, self.n_sources))
        units = jnp.zeros(self.n_ltus)
        return Network(
            ltus=Layer(ltu_weights, ltu_connections, jnp.zeros_like(ltu_weights), self.optimizer.init(ltu_weights)),
            outputs=Layer(output_weights, output_connections, jnp.zeros_like(output_weights),
                          self.optimizer.init(output_weights)),
            depth=depth, firing_rate=units + 0.5, error_mean=units, error_activation_mean=units,
            gradient_sums=jnp.zeros((self.n_ltus, self.n_sources - 1)), usefulness=units)


# --------------------------------------------------------------------------------------------------------------------
# Baseline: a fully connected network
# --------------------------------------------------------------------------------------------------------------------

class FullyConnected(Algorithm):
    """Network Autostep on `depths` layers of `width` LTUs. Each layer reads the one below (the lowest reads the
    inputs), and the outputs read the top layer. LTU weights start in U(-initial_step_size, initial_step_size); output
    weights and biases start at 0. The structure never changes."""
    width: int = eqx.field(static=True, default=160)
    meta_step_size: float = eqx.field(static=True, default=0.01)

    def init(self, key: PRNGKeyArray) -> Network:
        depth = jnp.arange(self.n_ltus) // self.width + 1
        source_depth = jnp.concatenate([jnp.zeros(self.n_inputs, int), depth, jnp.array([-1])])  # bias: -1
        is_bias = source_depth == -1
        ltu_connections = (source_depth[None] == depth[:, None] - 1) | is_bias[None]
        output_connections = jnp.broadcast_to((source_depth == self.depths) | is_bias, (self.n_outputs, self.n_sources))
        ltu_weights = jax.random.uniform(key, ltu_connections.shape, minval=-self.initial_step_size,
                                         maxval=self.initial_step_size) * ltu_connections * ~is_bias
        return self.new_network(depth, ltu_weights, ltu_connections, output_connections)


# --------------------------------------------------------------------------------------------------------------------
# Step 1: unit search, from step_by_step/
# --------------------------------------------------------------------------------------------------------------------

class UnitSearch(Algorithm):
    """Step 1: the algorithm of step_by_step/.

    `depths` depths of `width` LTU slots, every slot always holding an LTU. The outputs read every source.

    update_measures  each LTU's gradient sums over its sources, and its usefulness: the sum of |outgoing weights|.
    add_connections  GROW: an LTU connects the unread source whose gradient sum is largest, if that sum is more than
                     growth_threshold times the root mean square of the sums over its unread sources.
    move_units       SWAP: from the top depth down, the most useful LTU of depth d with less than swap_threshold of
                     its input |weights| on depth d - 1 takes the slot of the least useful LTU of depth d - 1, if
                     that one is less useful. That LTU goes to the freed slot, and prune_units removes it.
    prune_units      SEARCH: the least useful replaced_fraction of the LTUs.
    add_units        NEW LTUs, from the top depth down. A new LTU's parent is a unit of a random layer above its
                     depth, the outputs included, and its shadow weight on the parent is +1 or -1. It reads
                     initial_fan_in sources drawn in proportion to its parent's weights squared, with LeCun uniform
                     weights with the signs of its shadow weight times its parent's weights. A parent LTU reads its
                     new LTU from weight 0.
    """
    initial_fan_in: int = eqx.field(static=True, default=20)              # sources that a new LTU first reads
    new_weight_step_size: float = eqx.field(static=True, default=3e-2)    # of a new LTU's weights and new connections
    growth_threshold: float = eqx.field(static=True, default=6.0)         # GROW, in root mean squares
    swap_threshold: float = eqx.field(static=True, default=0.1)           # SWAP
    replaced_fraction: float = eqx.field(static=True, default=1 / 8)      # SEARCH

    @property
    def n_replaced(self) -> int:
        return round(self.replaced_fraction * self.n_ltus)

    def may_read(self, depth: Int[Array, 'ltus']) -> Bool[Array, 'ltus sources']:
        """Every input, the bias, and the LTUs of lower depth."""
        lower = (depth[None] > 0) & (depth[None] < depth[:, None])
        return jnp.concatenate([jnp.ones((self.n_ltus, self.n_inputs), bool), lower,
                                jnp.ones((self.n_ltus, 1), bool)], 1)

    # --- init ---

    def init(self, key: PRNGKeyArray) -> Network:
        """Every slot holds a new LTU. The outputs have no weights yet, so the LTUs draw their sources uniformly."""
        depth = jnp.arange(self.n_ltus) // self.width + 1
        no_weights = jnp.zeros((self.n_ltus, self.n_sources))
        network = self.new_network(depth, no_weights, no_weights != 0,
                                   jnp.ones((self.n_outputs, self.n_sources), bool))
        return self.add_units(network, key, jnp.ones(self.n_ltus, bool), at_random=True)

    # --- update_measures ---

    def update_measures(self, network: Network, ltu_gradients, sources) -> Network:
        to_outputs = jnp.abs(network.outputs.weights[:, self.ltu_sources]).sum(0)
        to_ltus = jnp.abs(network.ltus.weights * network.ltus.connections)[:, self.ltu_sources].sum(0)
        return network._replace(gradient_sums=network.gradient_sums + jnp.outer(ltu_gradients, sources[:-1]),
                                usefulness=to_outputs + to_ltus)

    # --- add_connections ---

    def add_connections(self, network: Network) -> Network:
        """GROW."""
        unread = self.may_read(network.depth)[:, :-1] & ~network.ltus.connections[:, :-1]
        sums = jnp.where(unread, jnp.abs(network.gradient_sums), 0.0)
        root_mean_square = jnp.sqrt(jnp.sum(sums ** 2, 1) / unread.sum(1))
        grows = jnp.max(sums, 1) > self.growth_threshold * root_mean_square
        new = (jnp.arange(self.n_sources) == jnp.argmax(sums, 1)[:, None]) & grows[:, None]
        network = self.connect(network, new)
        return network._replace(gradient_sums=jnp.where(grows[:, None], 0.0, network.gradient_sums))

    def connect(self, network: Network, new: Bool[Array, 'ltus sources']) -> Network:
        """The LTUs read the `new` sources, from weight 0 at the new-weight step size."""
        ltus = network.ltus
        step_sizes = jnp.where(new, jnp.log(self.new_weight_step_size), ltus.optimizer_state.beta)
        return network._replace(ltus=ltus._replace(
            weights=jnp.where(new, 0.0, ltus.weights), connections=ltus.connections | new,
            optimizer_state=ltus.optimizer_state._replace(beta=step_sizes)))

    # --- move_units ---

    def move_units(self, network: Network) -> Network:
        """SWAP."""
        for depth in range(self.depths, 1, -1):
            input_weights = jnp.abs(network.ltus.weights[:, :-1]) * network.ltus.connections[:, :-1]
            from_below = jnp.sum(jnp.where(network.depth == depth - 1, input_weights[:, self.ltu_sources], 0.0), 1)
            flat = (network.depth == depth) & (from_below < self.swap_threshold * input_weights.sum(1))
            mover = jnp.argmax(jnp.where(flat, network.usefulness, -jnp.inf))
            weak = jnp.argmin(jnp.where(network.depth == depth - 1, network.usefulness, jnp.inf))
            swaps = flat.any() & (network.usefulness[mover] > network.usefulness[weak])
            swapped = self.swap(network, weak, mover)
            swapped = swapped._replace(usefulness=swapped.usefulness.at[mover].set(-1.0))   # prune_units removes it
            network = select(swaps, swapped, network)
        return network

    def swap(self, network: Network, i, j) -> Network:
        """Slots i and j exchange their LTUs, with their weights, statistics and gradient sums. The shadow weights
        of a parent stay in its slot. The LTU that arrives at slot i, the lower depth, loses its connections from
        LTUs no longer below it."""
        order = jnp.arange(self.n_ltus).at[i].set(j).at[j].set(i)
        columns = jnp.concatenate([jnp.arange(self.n_inputs), self.n_inputs + order, jnp.array([self.n_sources - 1])])
        rows_and_columns = lambda a: a[order][:, columns]
        ltus, outputs = network.ltus, network.outputs
        ltus = Layer(rows_and_columns(ltus.weights), rows_and_columns(ltus.connections),
                     ltus.shadow_weights[:, columns], jax.tree.map(
                         lambda a: rows_and_columns(a) if jnp.ndim(a) == 2 else a, ltus.optimizer_state))
        outputs = Layer(outputs.weights[:, columns], outputs.connections[:, columns],
                        outputs.shadow_weights[:, columns],
                        jax.tree.map(lambda a: a[:, columns] if jnp.ndim(a) == 2 else a, outputs.optimizer_state))
        network = network._replace(
            ltus=ltus, outputs=outputs, firing_rate=network.firing_rate[order], error_mean=network.error_mean[order],
            error_activation_mean=network.error_activation_mean[order],
            gradient_sums=network.gradient_sums[order][:, columns[:-1]],
            usefulness=network.usefulness[order])
        cut = jnp.zeros((self.n_ltus, self.n_sources), bool).at[i].set(~self.may_read(network.depth)[i])
        return network._replace(ltus=self.reset(network.ltus, cut, 0.0)._replace(
            connections=network.ltus.connections & ~cut))

    # --- prune_units ---

    def prune_units(self, network: Network):
        """SEARCH: returns the least useful LTUs, which add_units replaces."""
        weakest = jnp.argsort(network.usefulness)[:self.n_replaced]
        return network, jnp.zeros(self.n_ltus, bool).at[weakest].set(True)

    # --- add_units ---

    def add_units(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus'],
                  at_random: bool = False) -> Network:
        """NEW LTUs in `slots`: first their parents, then their connections, from the top depth down, so that an LTU
        whose parent is also new draws on the parent's new weights."""
        keys = jax.random.split(key, self.depths + 1)
        network = self.draw_parents(network, keys[0], slots)
        for depth, depth_key in zip(range(self.depths, 0, -1), keys[1:]):
            new = slots & (network.depth == depth)
            network = self.wire(network, depth_key, slots, new, at_random)
            network = self.align(network, new)
        return network

    def draw_parents(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus']) -> Network:
        """Each new LTU's parent: a layer drawn uniformly among those above the LTU's depth (the outputs are the top
        layer), then a unit of that layer drawn uniformly. The shadow weight is +1 in even slots and -1 in odd
        slots, and it sits in the parent's weights, on the new LTU."""
        layer_key, unit_key = jax.random.split(key)
        layer = jax.random.randint(layer_key, (self.n_ltus,), network.depth + 1, self.depths + 2)
        on_output = layer > self.depths
        index = jax.random.randint(unit_key, (self.n_ltus,), 0, jnp.where(on_output, self.n_outputs, self.width))
        sign = jnp.where(jnp.arange(self.n_ltus) % 2 == 0, 1.0, -1.0)
        output_parent = jnp.where(on_output, index, -1)
        ltu_parent = jnp.where(on_output, -1, (layer - 1) * self.width + index)
        on_outputs = (jnp.arange(self.n_outputs)[:, None] == output_parent) * sign
        on_ltus = (jnp.arange(self.n_ltus)[:, None] == ltu_parent) * sign
        set_shadow = lambda layer, shadow: layer._replace(shadow_weights=layer.shadow_weights.at[:, self.ltu_sources].set(
            jnp.where(slots[None], shadow, layer.shadow_weights[:, self.ltu_sources])))
        return network._replace(ltus=set_shadow(network.ltus, on_ltus), outputs=set_shadow(network.outputs, on_outputs))

    def parent_weights(self, network: Network) -> Float[Array, 'ltus sources']:
        """Each LTU's shadow weight times its parent's weights on the sources that the LTU may read (an output's
        weights on the inputs are the linear learner's); 0 on the bias."""
        from_outputs = network.outputs.shadow_weights[:, self.ltu_sources].T @ network.outputs.weights
        from_ltus = (network.ltus.shadow_weights[:, self.ltu_sources].T
                     @ (network.ltus.weights * network.ltus.connections))
        return (from_outputs + from_ltus).at[:, -1].set(0.0) * self.may_read(network.depth)

    def wire(self, network: Network, key: PRNGKeyArray, removed: Bool[Array, 'ltus'], new: Bool[Array, 'ltus'],
             at_random: bool) -> Network:
        """The `new` LTUs read initial_fan_in sources (at most as many as their parent weights), drawn without
        replacement in proportion to their parent's weights squared (uniformly among the sources they may read if
        `at_random`), and the bias, with LeCun uniform weights. Their output weights are 0, and their statistics,
        Autostep state and gradient sums are fresh. Every LTU loses its connections from the `removed` LTUs. Every
        parent LTU reads its child LTUs; a link that it does not have yet starts from weight 0 at the new-weight
        step size."""
        draw_key, weight_key = jax.random.split(key)
        cut = jnp.zeros((self.n_ltus, self.n_sources), bool).at[:, self.ltu_sources].set(removed[None])
        chances = jnp.where(at_random, self.may_read(network.depth), self.parent_weights(network))
        chances = jnp.where(cut, 0.0, chances)[:, :-1]
        fan_in = jnp.minimum(self.initial_fan_in, (chances != 0).sum(1, keepdims=True))
        scores = jnp.log(chances ** 2) + jax.random.gumbel(draw_key, chances.shape)   # Gumbel top-k
        drawn = jnp.append(jnp.argsort(jnp.argsort(-scores)) < fan_in, jnp.ones((self.n_ltus, 1), bool), axis=1)
        weights = (jax.random.uniform(weight_key, drawn.shape, minval=-1.0, maxval=1.0)
                   * jnp.sqrt(3.0 / jnp.maximum(fan_in, 1)) * drawn).at[:, -1].set(0.0)

        changed = new[:, None] | cut
        connections = jnp.where(changed, drawn, network.ltus.connections)
        links_to_children = network.ltus.shadow_weights != 0
        missing_links = links_to_children & ~connections
        ltus = self.reset(network.ltus, changed, weights)
        step_sizes = jnp.where(missing_links, jnp.log(self.new_weight_step_size), ltus.optimizer_state.beta)
        ltus = ltus._replace(connections=connections | missing_links,
                             optimizer_state=ltus.optimizer_state._replace(beta=step_sizes))
        new_columns = jnp.zeros(self.n_sources, bool).at[self.ltu_sources].set(new)
        outputs = self.reset(network.outputs, jnp.broadcast_to(new_columns, network.outputs.weights.shape), 0.0)
        return network._replace(
            ltus=ltus, outputs=outputs, firing_rate=jnp.where(new, 0.5, network.firing_rate),
            error_mean=jnp.where(new, 0.0, network.error_mean),
            error_activation_mean=jnp.where(new, 0.0, network.error_activation_mean),
            gradient_sums=jnp.where(changed[:, :-1], 0.0, network.gradient_sums))

    def align(self, network: Network, new: Bool[Array, 'ltus']) -> Network:
        """The `new` LTUs' weights keep their sizes but take the signs of their shadow weight times their parent's
        weights (where those are not 0), and every weight but the bias gets the new-weight step size."""
        ltus = network.ltus
        sources = ltus.connections.at[:, -1].set(False) & new[:, None]
        parent = self.parent_weights(network)
        weights = jnp.where(sources & (parent != 0), jnp.abs(ltus.weights) * jnp.sign(parent), ltus.weights)
        step_sizes = jnp.where(sources, jnp.log(self.new_weight_step_size), ltus.optimizer_state.beta)
        return network._replace(ltus=ltus._replace(
            weights=weights, optimizer_state=ltus.optimizer_state._replace(beta=step_sizes)))

    def reset(self, layer: Layer, where: Bool[Array, 'units sources'], weights) -> Layer:
        """The weights at `where` become `weights`, with Autostep's initial step size and no traces."""
        state = layer.optimizer_state
        return layer._replace(weights=jnp.where(where, weights, layer.weights), optimizer_state=state._replace(
            beta=jnp.where(where, state.init_beta, state.beta), h=jnp.where(where, 0.0, state.h),
            v=jnp.where(where, 0.0, state.v)))
