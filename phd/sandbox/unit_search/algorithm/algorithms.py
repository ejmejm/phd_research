"""The algorithm at each step of the meta progression, one class per step.

Each algorithm is an equinox module, and its static fields are its hyperparameters. `Algorithm.run` calls its parts:

    init(key)                              the network at step 0
    every step
        learn_step(network, x, y)          network Autostep, then update_measures for the structure changes
    every connection_period steps          change_connections
        prune_connections(network)
        add_connections(network)
    every unit_period steps                change_units
        move_units(network)
        prune_units(network)               returns the LTUs to replace
        add_units(network, key, slots)

A part that an algorithm does not use leaves the network unchanged. A part that runs less often than its period does
nothing at the other calls. Each later class inherits from the class before it and overrides only the parts that
change.

The network. Every unit reads from the layer inputs [x, standardized LTU activations, 1], where the last one is the
bias. `Network.ltus` holds the LTUs' incoming weights and `Network.outputs` holds the outputs'. Each LTU has a depth,
and reads only x, the bias and LTUs of lower depth. An LTU can have a parent, a unit that reads it, and a shadow weight
of +1 or -1 on that parent: the backward pass uses the shadow weight in place of the real weight while the real weight
is smaller than 1 in size.

The learn step uses autograd, with the layers of layers.py, which have their own backward passes.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from phd.jax_core.optimizers.idbd import IDBDState, optax_idbd
from phd.sandbox.unit_search.algorithm.layers import Standardize, linear, ltu


def select(condition: Bool[Array, ''], if_true, if_false):
    """if_true where the condition holds, else if_false, for every array of two networks."""
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), if_true, if_false)


class Layer(NamedTuple):
    """A group of units, and their weights from the layer inputs."""
    weights: Float[Array, 'units layer_inputs']
    connections: Bool[Array, 'units layer_inputs']   # the layer inputs that each unit reads
    optimizer_state: IDBDState                       # Autostep's step sizes and traces, shaped like the weights


class Network(NamedTuple):
    ltus: Layer
    outputs: Layer
    norm: Standardize                                   # the LTUs' standardization and its running averages
    depth: Int[Array, 'ltus']                           # 1 for the lowest LTUs
    parent: Int[Array, 'ltus']                          # an index into [LTUs, outputs]
    shadow: Float[Array, 'ltus']                        # the shadow weight on the parent: +1, -1, or 0 for no parent
    gradient_sums: Float[Array, 'ltus layer_inputs']    # sum of dL/d(pre-activation) * layer input since last growth
    utility: Float[Array, 'ltus']                       # sum of |outgoing weights|


class Algorithm(eqx.Module):
    """Network Autostep on a network of LTUs, with a main loop that calls the structure changes."""
    n_inputs: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    depths: int = eqx.field(static=True, default=3)
    width: int = eqx.field(static=True, default=40)                     # LTUs per depth
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
    def n_layer_inputs(self) -> int:
        return self.n_inputs + self.n_ltus + 1

    @property
    def ltu_activation_idxs(self) -> slice:
        """The indices of the LTUs' activations in the layer inputs."""
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

    def update_measures(self, network: Network, pre_activation_gradients, layer_inputs) -> Network:
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
        def loss_function(parameters):
            ltu_weights, output_weights, norm = parameters
            predictions, (layer_inputs, activations) = self.predict(network, ltu_weights, output_weights, norm, x)
            return 0.5 * jnp.sum((predictions - y) ** 2), (layer_inputs, activations)

        parameters = (network.ltus.weights, network.outputs.weights, network.norm)
        (loss, (layer_inputs, activations)), gradients = eqx.filter_value_and_grad(loss_function, has_aux=True)(
            parameters)
        ltu_gradients, output_gradients, norm_gradients = gradients

        network = network._replace(
            ltus=self.update_weights(network.ltus, ltu_gradients, layer_inputs),
            outputs=self.update_weights(network.outputs, output_gradients, layer_inputs),
            norm=network.norm.update_statistics(activations, norm_gradients))
        pre_activation_gradients = ltu_gradients[:, -1]   # the bias's gradient is dL/d(pre-activation)
        return self.update_measures(network, pre_activation_gradients, layer_inputs), 2 * loss / self.n_outputs

    def predict(self, network: Network, ltu_weights, output_weights, norm: Standardize, x: Float[Array, 'inputs']):
        """The LTUs depth by depth, then the outputs. Returns the predictions, the layer inputs and the LTUs'
        activations."""
        ltu_weights = ltu_weights * network.ltus.connections
        output_weights = output_weights * network.outputs.connections
        shadow_weights = self.shadow_weights(network)
        ltu_backward_weights = self.backward_weights(ltu_weights, shadow_weights[:self.n_ltus])
        output_backward_weights = self.backward_weights(output_weights, shadow_weights[self.n_ltus:])

        activations = jnp.zeros(self.n_ltus)
        standardized_activations = jnp.zeros(self.n_ltus)
        for depth in range(1, self.depths + 1):
            at_depth = network.depth == depth
            layer_inputs = jnp.concatenate([x, standardized_activations, jnp.ones(1)])
            depth_activations = ltu(linear(ltu_weights, ltu_backward_weights, layer_inputs))
            activations = jnp.where(at_depth, depth_activations, activations)
            standardized_activations = jnp.where(at_depth, norm(depth_activations, at_depth),
                                                 standardized_activations)

        layer_inputs = jnp.concatenate([x, standardized_activations, jnp.ones(1)])
        predictions = linear(output_weights, output_backward_weights, layer_inputs)
        return predictions, (layer_inputs, jax.lax.stop_gradient(activations))

    def shadow_weights(self, network: Network) -> Float[Array, 'ltus_then_outputs layer_inputs']:
        """The shadow weights in the weights of the LTUs, then of the outputs: each LTU's shadow weight sits in its
        parent's row, on the LTU's column."""
        units = jnp.arange(self.n_ltus + self.n_outputs)[:, None]
        on_ltus = jnp.where(units == network.parent, network.shadow, 0.0)
        return jnp.pad(on_ltus, ((0, 0), (self.n_inputs, 1)))

    def backward_weights(self, weights, shadow_weights) -> Float[Array, 'units layer_inputs']:
        """The weights that the backward pass uses: each shadow weight while the real weight is smaller than 1 in
        size, and the real weights elsewhere."""
        uses_shadow = (shadow_weights != 0) & (jnp.abs(weights) < 1.0)
        return jax.lax.stop_gradient(jnp.where(uses_shadow, shadow_weights, weights))

    def update_weights(self, layer: Layer, gradients, layer_inputs) -> Layer:
        """Autostep on a layer's weights. Each layer is its own linear learner, with the layer inputs that it reads as
        features."""
        features = layer_inputs * layer.connections
        updates, optimizer_state = self.optimizer.update((gradients, None, features), layer.optimizer_state,
                                                         layer.weights)
        return layer._replace(weights=layer.weights + updates, optimizer_state=optimizer_state)

    # --- construction ---

    def new_network(self, depth, ltu_weights, ltu_connections, output_connections) -> Network:
        """A network with these LTU weights, zero output weights and no parents."""
        output_weights = jnp.zeros((self.n_outputs, self.n_layer_inputs))
        return Network(
            ltus=Layer(ltu_weights, ltu_connections, self.optimizer.init(ltu_weights)),
            outputs=Layer(output_weights, output_connections, self.optimizer.init(output_weights)),
            norm=Standardize(self.n_ltus, self.running_average_rate, self.variance_floor), depth=depth,
            parent=jnp.zeros(self.n_ltus, int), shadow=jnp.zeros(self.n_ltus),
            gradient_sums=jnp.zeros_like(ltu_weights), utility=jnp.zeros(self.n_ltus))


# --------------------------------------------------------------------------------------------------------------------
# Baseline: a fully connected network
# --------------------------------------------------------------------------------------------------------------------

class FullyConnected(Algorithm):
    """Network Autostep on `depths` layers of `width` LTUs. Each layer reads the one below (the lowest reads x), and
    the outputs read the top layer. LTU weights start in U(-initial_step_size, initial_step_size); output weights and
    biases start at 0. The structure never changes."""
    width: int = eqx.field(static=True, default=160)
    meta_step_size: float = eqx.field(static=True, default=0.01)

    def init(self, key: PRNGKeyArray) -> Network:
        depth = jnp.arange(self.n_ltus) // self.width + 1
        input_depth = jnp.concatenate([jnp.zeros(self.n_inputs, int), depth, jnp.array([-1])])  # bias: -1
        is_bias = input_depth == -1
        ltu_connections = (input_depth[None] == depth[:, None] - 1) | is_bias[None]
        output_connections = jnp.broadcast_to((input_depth == self.depths) | is_bias,
                                              (self.n_outputs, self.n_layer_inputs))
        ltu_weights = jax.random.uniform(key, ltu_connections.shape, minval=-self.initial_step_size,
                                         maxval=self.initial_step_size) * ltu_connections * ~is_bias
        return self.new_network(depth, ltu_weights, ltu_connections, output_connections)


# --------------------------------------------------------------------------------------------------------------------
# Step 1: unit search, from step_by_step/
# --------------------------------------------------------------------------------------------------------------------

class UnitSearch(Algorithm):
    """Step 1: the algorithm of step_by_step/.

    `depths` depths of `width` LTUs. Every LTU has a parent, and its shadow weight is +1 for even indices and -1 for
    odd indices. The outputs read every layer input.

    update_measures  each LTU's gradient sums over its layer inputs, and its utility: the sum of |outgoing weights|.
    add_connections  GROW: an LTU connects the unread layer input with the largest gradient sum, if that sum is more
                     than growth_threshold times the root mean square of the sums over its unread layer inputs.
    move_units       SWAP: from the top depth down, the depth-d LTU of highest utility with less than swap_threshold
                     of its input |weights| on depth d - 1 changes depth with the depth-(d - 1) LTU of lowest
                     utility, if that one has a lower utility. prune_units removes the LTU that moves up.
    prune_units      SEARCH: the replaced_fraction of the LTUs with the lowest utility.
    add_units        NEW LTUs, from the top depth down. A new LTU's parent is a unit of a random layer above its
                     depth, the outputs included. It reads initial_fan_in layer inputs drawn in proportion to its
                     parent's weights squared, with LeCun uniform weights with the signs of its shadow weight times
                     its parent's weights. A parent LTU reads its new LTU from weight 0.
    """
    initial_fan_in: int = eqx.field(static=True, default=20)              # layer inputs that a new LTU first reads
    new_weight_step_size: float = eqx.field(static=True, default=3e-2)    # of a new LTU's weights and new connections
    growth_threshold: float = eqx.field(static=True, default=6.0)         # GROW, in root mean squares
    swap_threshold: float = eqx.field(static=True, default=0.1)           # SWAP
    replaced_fraction: float = eqx.field(static=True, default=1 / 8)      # SEARCH

    def may_read(self, depth: Int[Array, 'ltus']) -> Bool[Array, 'ltus layer_inputs']:
        """x, the bias, and the LTUs of lower depth."""
        input_depth = jnp.concatenate([jnp.zeros(self.n_inputs, int), depth, jnp.zeros(1, int)])
        return input_depth[None] < depth[:, None]

    # --- init ---

    def init(self, key: PRNGKeyArray) -> Network:
        """Every LTU is new. The outputs have no weights yet, so the LTUs draw their layer inputs uniformly."""
        depth = jnp.arange(self.n_ltus) // self.width + 1
        no_weights = jnp.zeros((self.n_ltus, self.n_layer_inputs))
        network = self.new_network(depth, no_weights, no_weights != 0,
                                   jnp.ones((self.n_outputs, self.n_layer_inputs), bool))
        network = network._replace(shadow=jnp.where(jnp.arange(self.n_ltus) % 2 == 0, 1.0, -1.0))
        return self.add_units(network, key, jnp.ones(self.n_ltus, bool), at_random=True)

    # --- update_measures ---

    def update_measures(self, network: Network, pre_activation_gradients, layer_inputs) -> Network:
        to_outputs = jnp.abs(network.outputs.weights[:, self.ltu_activation_idxs]).sum(0)
        to_ltus = jnp.abs(network.ltus.weights * network.ltus.connections)[:, self.ltu_activation_idxs].sum(0)
        return network._replace(
            gradient_sums=network.gradient_sums + jnp.outer(pre_activation_gradients, layer_inputs),
            utility=to_outputs + to_ltus)

    # --- add_connections ---

    def add_connections(self, network: Network) -> Network:
        """GROW."""
        unread = self.may_read(network.depth) & ~network.ltus.connections
        sums = jnp.where(unread, jnp.abs(network.gradient_sums), 0.0)
        root_mean_square = jnp.sqrt(jnp.sum(sums ** 2, 1) / unread.sum(1))
        grows = jnp.max(sums, 1) > self.growth_threshold * root_mean_square
        new = (jnp.arange(self.n_layer_inputs) == jnp.argmax(sums, 1)[:, None]) & grows[:, None]
        network = self.connect(network, new)
        return network._replace(gradient_sums=jnp.where(grows[:, None], 0.0, network.gradient_sums))

    def connect(self, network: Network, new: Bool[Array, 'ltus layer_inputs']) -> Network:
        """The LTUs read the `new` layer inputs, from weight 0 at the new-weight step size."""
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
            from_below = jnp.sum(jnp.where(network.depth == depth - 1,
                                           input_weights[:, self.ltu_activation_idxs], 0.0), 1)
            flat = (network.depth == depth) & (from_below < self.swap_threshold * input_weights.sum(1))
            mover = jnp.argmax(jnp.where(flat, network.utility, -jnp.inf))
            weak = jnp.argmin(jnp.where(network.depth == depth - 1, network.utility, jnp.inf))
            swaps = flat.any() & (network.utility[mover] > network.utility[weak])
            network = select(swaps, self.swap(network, mover, weak), network)
        return network

    def swap(self, network: Network, mover, weak) -> Network:
        """The mover and the weak LTU change depths and children, so that every parent stays above its children. The
        mover loses its connections from LTUs no longer below it. The weak LTU gets utility -1, so that prune_units
        removes it."""
        depth = network.depth.at[mover].set(network.depth[weak]).at[weak].set(network.depth[mover])
        parent = jnp.where(network.parent == mover, weak, jnp.where(network.parent == weak, mover, network.parent))
        cut = (jnp.arange(self.n_ltus) == mover)[:, None] & ~self.may_read(depth)
        ltus = self.reset(network.ltus, cut, 0.0)._replace(connections=network.ltus.connections & ~cut)
        return network._replace(ltus=ltus, depth=depth, parent=parent, utility=network.utility.at[weak].set(-1.0))

    # --- prune_units ---

    def prune_units(self, network: Network):
        """SEARCH: returns the LTUs with the lowest utility, which add_units replaces."""
        weakest = jnp.argsort(network.utility)[:round(self.replaced_fraction * self.n_ltus)]
        return network, jnp.zeros(self.n_ltus, bool).at[weakest].set(True)

    # --- add_units ---

    def add_units(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus'],
                  at_random: bool = False) -> Network:
        """NEW LTUs in `slots`: first their parents, then their connections, from the top depth down, so that an LTU
        whose parent is also new draws on the parent's new weights."""
        keys = jax.random.split(key, self.depths + 1)
        network = self.draw_parents(network, keys[0], slots)
        for depth, depth_key in zip(range(self.depths, 0, -1), keys[1:]):
            network = self.wire(network, depth_key, slots, slots & (network.depth == depth), at_random)
        return network

    def draw_parents(self, network: Network, key: PRNGKeyArray, slots: Bool[Array, 'ltus']) -> Network:
        """Each new LTU's parent: a layer drawn uniformly among those above the LTU's depth (the outputs are the top
        layer), then a unit of that layer drawn uniformly."""
        layer_key, unit_key = jax.random.split(key)
        layer = jax.random.randint(layer_key, (self.n_ltus, 1), network.depth[:, None] + 1, self.depths + 2)
        unit_layer = jnp.concatenate([network.depth, jnp.full(self.n_outputs, self.depths + 1)])
        scores = jnp.where(unit_layer == layer, jax.random.gumbel(unit_key, (self.n_ltus, len(unit_layer))), -jnp.inf)
        return network._replace(parent=jnp.where(slots, jnp.argmax(scores, 1), network.parent))

    def parent_weights(self, network: Network) -> Float[Array, 'ltus layer_inputs']:
        """Each LTU's shadow weight times its parent's weights on the layer inputs that the LTU may read (an output's
        weights on x are the linear learner's); 0 on the bias."""
        unit_weights = jnp.concatenate([network.ltus.weights * network.ltus.connections, network.outputs.weights])
        weights = network.shadow[:, None] * unit_weights[network.parent]
        return weights.at[:, -1].set(0.0) * self.may_read(network.depth)

    def wire(self, network: Network, key: PRNGKeyArray, removed: Bool[Array, 'ltus'], new: Bool[Array, 'ltus'],
             at_random: bool) -> Network:
        """The `new` LTUs read the bias and initial_fan_in layer inputs (at most as many as their parent weights),
        drawn without replacement in proportion to their parent weights squared (uniformly among those they may read
        if `at_random`). Their weights are LeCun uniform in size, with the signs of their parent weights where those
        are not 0, at the new-weight step size. Their output weights, statistics and gradient sums start fresh.
        Every LTU loses its connections from the `removed` LTUs, and every parent LTU reads its children."""
        draw_key, weight_key = jax.random.split(key)
        parent_weights = self.parent_weights(network)
        removed_inputs = jnp.zeros(self.n_layer_inputs, bool).at[self.ltu_activation_idxs].set(removed)
        chances = jnp.where(at_random, self.may_read(network.depth), parent_weights)
        chances = jnp.where(removed_inputs, 0.0, chances)[:, :-1]   # the bias is not drawn
        fan_in = jnp.minimum(self.initial_fan_in, (chances != 0).sum(1, keepdims=True))
        scores = jnp.log(chances ** 2) + jax.random.gumbel(draw_key, chances.shape)   # Gumbel top-k
        drawn = jnp.append(jnp.argsort(jnp.argsort(-scores)) < fan_in, jnp.zeros((self.n_ltus, 1), bool), axis=1)
        lecun = (jax.random.uniform(weight_key, drawn.shape, minval=-1.0, maxval=1.0)
                 * jnp.sqrt(3.0 / jnp.maximum(fan_in, 1)))
        weights = jnp.where(parent_weights != 0, jnp.abs(lecun) * jnp.sign(parent_weights), lecun) * drawn

        changed = new[:, None] | removed_inputs
        ltus = self.reset(network.ltus, changed, weights)
        step_sizes = jnp.where(new[:, None] & drawn, jnp.log(self.new_weight_step_size), ltus.optimizer_state.beta)
        connections = jnp.where(new[:, None], drawn.at[:, -1].set(True), ltus.connections & ~removed_inputs)
        ltus = ltus._replace(connections=connections, optimizer_state=ltus.optimizer_state._replace(beta=step_sizes))
        new_inputs = jnp.zeros(self.n_layer_inputs, bool).at[self.ltu_activation_idxs].set(new)
        network = network._replace(ltus=ltus, outputs=self.reset(network.outputs, new_inputs, 0.0),
                                   norm=network.norm.restart(new),
                                   gradient_sums=jnp.where(changed, 0.0, network.gradient_sums))
        children = self.shadow_weights(network)[:self.n_ltus] != 0
        return self.connect(network, children & ~network.ltus.connections)

    def reset(self, layer: Layer, where: Bool[Array, 'units layer_inputs'], weights) -> Layer:
        """The weights at `where` become `weights`, with Autostep's initial step size and no traces."""
        state = layer.optimizer_state
        return layer._replace(weights=jnp.where(where, weights, layer.weights), optimizer_state=state._replace(
            beta=jnp.where(where, state.init_beta, state.beta), h=jnp.where(where, 0.0, state.h),
            v=jnp.where(where, 0.0, state.v)))
