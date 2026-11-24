"""The problems of the meta progression. Each has `n_inputs`, `n_outputs`, `noise_std` and `sample(key) -> (x, y)`."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray


class SparseTargetNetwork:
    """A fixed deep sparse network of LTUs, from step_by_step/6_sparse_target_network.ipynb.

    1000 standard normal inputs and `depths` hidden depths of `width` target LTUs. Each target LTU reads 10 distinct
    units of lower depths: a lower depth chosen uniformly (the inputs are depth 0), then a unit of that depth. Each of
    the 100 outputs reads `output_fan_in` target LTUs the same way. All weights are N(0, 1). Each target LTU's
    threshold makes it fire 30% of the time. Each output is scaled to zero mean and unit variance, multiplied by
    `signal`, and gets noise of std `noise_std`. `signal` and `noise_std` are numbers or one value per output.
    """
    n_inputs, n_outputs = 1000, 100
    fan_in, firing_rate = 10, 0.3

    def __init__(self, noise_std: ArrayLike = 1.0, signal: ArrayLike = 1.0, depths: int = 3, width: int = 20,
                 output_fan_in: int = 10, seed: int = 123):
        self.noise_std, self.signal = noise_std, signal
        self.depths, self.width = depths, width
        key = jax.random.PRNGKey(seed)
        n_hidden = self.depths * self.width
        self.depth = jnp.repeat(jnp.arange(1, self.depths + 1), self.width)          # each target LTU's depth
        source_depth = jnp.concatenate([jnp.zeros(self.n_inputs, int), self.depth])
        group_size = jnp.array([self.n_inputs] + [self.width] * self.depths)

        def pick(key, allowed, fan_in=self.fan_in):
            # fan_in distinct sources: a uniformly chosen allowed depth, then a uniformly chosen unit of it
            # (Gumbel top-k samples without replacement in proportion to these probabilities).
            p = jnp.where(allowed[source_depth], 1.0 / group_size[source_depth], 0.0)
            scores = jnp.where(p > 0, jnp.log(p) + jax.random.gumbel(key, p.shape), -jnp.inf)
            return jnp.zeros(p.shape, bool).at[jax.lax.top_k(scores, fan_in)[1]].set(True)

        keys = jax.random.split(key, 4)
        lower = jnp.arange(self.depths + 1)[None, :] < self.depth[:, None]
        hidden_reads = jax.vmap(pick)(jax.random.split(keys[0], n_hidden), lower)
        output_allowed = jnp.arange(self.depths + 1) > 0
        output_reads = jax.vmap(lambda k: pick(k, output_allowed, output_fan_in))(
            jax.random.split(keys[1], self.n_outputs))[:, self.n_inputs:]
        self.hidden_w = jax.random.normal(keys[2], hidden_reads.shape) * hidden_reads   # over [inputs, target LTUs]
        self.output_w = jax.random.normal(keys[3], output_reads.shape) * output_reads   # over target LTUs

        # Thresholds, depth by depth, from a batch of inputs; then the outputs' means and stds.
        x = jax.random.normal(jax.random.fold_in(key, 1), (20_000, self.n_inputs))
        bits = jnp.zeros((len(x), n_hidden))
        self.threshold = jnp.zeros(n_hidden)
        for d in range(1, self.depths + 1):
            z = jnp.concatenate([x, bits], axis=1) @ self.hidden_w.T
            self.threshold = jnp.where(self.depth == d, jnp.quantile(z, 1 - self.firing_rate, axis=0), self.threshold)
            bits = jnp.where(self.depth == d, (z > self.threshold).astype(jnp.float32), bits)
        raw = bits @ self.output_w.T
        self.output_mean, self.output_std = raw.mean(0), raw.std(0)

    def outputs(self, x: Float[Array, 'inputs']) -> Float[Array, 'outputs']:
        """The target network's outputs for input x, without noise."""
        bits = jnp.zeros(self.depths * self.width)
        for d in range(1, self.depths + 1):
            z = self.hidden_w @ jnp.concatenate([x, bits])
            bits = jnp.where(self.depth == d, (z > self.threshold).astype(jnp.float32), bits)
        return self.signal * (self.output_w @ bits - self.output_mean) / self.output_std

    def sample(self, key: PRNGKeyArray) -> tuple[Float[Array, 'inputs'], Float[Array, 'outputs']]:
        """One example: standard normal inputs, and the outputs plus noise."""
        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, (self.n_inputs,))
        return x, self.outputs(x) + self.noise_std * jax.random.normal(noise_key, (self.n_outputs,))


def signal_noise_grid(levels: int = 10) -> tuple[Float[Array, 'outputs'], Float[Array, 'outputs']]:
    """The signal multipliers and noise stds of levels ** 2 outputs: every pair of levels evenly spaced from 0 to 1."""
    values = jnp.linspace(0.0, 1.0, levels)
    signal, noise_std = jnp.meshgrid(values, values, indexing='ij')
    return signal.reshape(-1), noise_std.reshape(-1)
