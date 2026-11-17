"""Network Autostep: a minimal standalone example.

Network Autostep learns online, one sample per update. It has three parts:
1. Autostep (Mahmood et al., 2012) on every weight. Each layer counts as its own linear
   learner, with the layer's inputs as its features.
2. Small initialization. Hidden weights start in U(-a0, a0), where a0 is Autostep's
   initial step-size. Output weights and biases start at 0.
3. Standardization of each hidden LTU with running statistics, with batch norm's
   backward pass.

The example learns a random target network of LTUs.
"""
import equinox as eqx
import jax
import jax.numpy as jnp


# Problem
N_INPUTS = 20
N_TARGET_LTUS = 10
N_OUTPUTS = 1
NOISE_STD = 0.1             # noise on the targets, so the lowest possible RMSE

# Learner
HIDDEN_SIZES = (64, 64)     # LTUs in each hidden layer
INIT_STEP_SIZE = 2 ** -4    # a0
META_STEP_SIZE = 0.02       # Autostep's mu
TAU = 1e4                   # Autostep's tau
STAT_RATE = 1e-3            # rate of the running statistics for standardization
VAR_FLOOR = 0.01            # smallest variance that standardization divides by

# Run
N_STEPS = 200_000
LOG_EVERY = 20_000
SEED = 0


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class Target(eqx.Module):
    """A random network of LTUs: y = output_weights @ ltu(hidden_weights @ x) + noise."""
    hidden_weights: jax.Array
    output_weights: jax.Array

    def __init__(self, key):
        hidden_key, output_key = jax.random.split(key)
        self.hidden_weights = jax.random.normal(hidden_key, (N_TARGET_LTUS, N_INPUTS))
        self.output_weights = jax.random.normal(output_key, (N_OUTPUTS, N_TARGET_LTUS))

    def sample(self, key):
        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, (N_INPUTS,))
        hidden = (self.hidden_weights @ x > 0).astype(jnp.float32)
        noise = NOISE_STD * jax.random.normal(noise_key, (N_OUTPUTS,))
        return x, self.output_weights @ hidden + noise


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@jax.custom_vjp
def ltu(z):
    """1 where z > 0, else 0. The backward pass uses the sigmoid's derivative."""
    return (z > 0).astype(jnp.float32)


def ltu_fwd(z):
    return ltu(z), z


def ltu_bwd(z, out_grad):
    sigmoid = jax.nn.sigmoid(z)
    return (out_grad * sigmoid * (1 - sigmoid),)


ltu.defvjp(ltu_fwd, ltu_bwd)


def moving_average(mean, value):
    return mean + STAT_RATE * (value - mean)


@jax.custom_vjp
def standardize(x, mean, shift_grad_mean, scale_grad_mean):
    """(x - mean) / std, where std = sqrt(mean (1 - mean)) is the std of a bit."""
    return standardize_fwd(x, mean, shift_grad_mean, scale_grad_mean)[0]


def standardize_fwd(x, mean, shift_grad_mean, scale_grad_mean):
    variance = mean * (1 - mean)
    std = jnp.sqrt(jnp.maximum(variance, VAR_FLOOR))
    out = (x - mean) / std
    return out, (out, std, variance > VAR_FLOOR, shift_grad_mean, scale_grad_mean)


def standardize_bwd(saved, out_grad):
    # Batch norm's backward pass: dL/dx = (dL/dout - mean(dL/dshift) - out * mean(dL/dscale)) / std.
    # While the variance is at its floor, std does not depend on x, so the scale term drops out.
    # The running statistics get no gradient.
    out, std, unfloored, shift_grad_mean, scale_grad_mean = saved
    x_grad = out_grad - shift_grad_mean - jnp.where(unfloored, out * scale_grad_mean, 0.0)
    return x_grad / std, None, None, None


standardize.defvjp(standardize_fwd, standardize_bwd)


class Standardize(eqx.Module):
    """Standardizes LTU outputs with running statistics, like batch norm with running means.

    Batch norm standardizes with the mean and std of the batch, and autograd also
    differentiates through them. Here the mean and std come from running means over time,
    which autograd sees as constants. So `standardize` has batch norm's backward pass, with
    running means over time in place of means over the batch.

    That backward pass needs the mean gradients of batch norm's shift and scale parameters,
    in out = scale * standardize(x) + shift. Here shift and scale stay at 0 and 1, so they
    do not change the output. We keep them only for their gradients.
    """
    mean: jax.Array             # running mean of x. For an LTU, its firing rate.
    shift_grad_mean: jax.Array  # running mean of dL/dshift = dL/dout
    scale_grad_mean: jax.Array  # running mean of dL/dscale = dL/dout * out
    shift: jax.Array            # always 0
    scale: jax.Array            # always 1

    def __init__(self, n):
        self.mean = jnp.full(n, 0.5)
        self.shift_grad_mean = jnp.zeros(n)
        self.scale_grad_mean = jnp.zeros(n)
        self.shift = jnp.zeros(n)
        self.scale = jnp.ones(n)

    def __call__(self, x):
        mean = moving_average(self.mean, x)     # includes this sample
        return self.scale * standardize(x, mean, self.shift_grad_mean, self.scale_grad_mean) + self.shift

    def update_statistics(self, x, grads):
        """Returns this layer with its running statistics updated, from its input x and its gradients."""
        return eqx.tree_at(
            lambda norm: (norm.mean, norm.shift_grad_mean, norm.scale_grad_mean),
            self,
            (
                moving_average(self.mean, x),
                moving_average(self.shift_grad_mean, grads.shift),
                moving_average(self.scale_grad_mean, grads.scale),
            ),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Network(eqx.Module):
    """An MLP. Each hidden layer is linear, then LTU, then Standardize."""
    weights: list   # one (n_out, n_in + 1) matrix per layer. The last column is the bias.
    norms: list     # one Standardize per hidden layer

    def __init__(self, key):
        sizes = (N_INPUTS, *HIDDEN_SIZES)
        layer_keys = jax.random.split(key, len(HIDDEN_SIZES))
        self.weights = []
        for layer_key, n_in, n_out in zip(layer_keys, sizes[:-1], sizes[1:]):
            shape = (n_out, n_in + 1)
            weights = jax.random.uniform(layer_key, shape, minval=-INIT_STEP_SIZE, maxval=INIT_STEP_SIZE)
            self.weights.append(weights.at[:, -1].set(0.0))  # biases at 0
        self.weights.append(jnp.zeros((N_OUTPUTS, sizes[-1] + 1)))
        self.norms = [Standardize(n) for n in HIDDEN_SIZES]

    def __call__(self, x):
        """Returns the prediction, the inputs to each layer, and each hidden layer's LTU outputs."""
        layer_inputs, activations = [], []
        for weights, norm in zip(self.weights[:-1], self.norms):
            layer_inputs.append(jnp.append(x, 1.0))
            activations.append(ltu(weights @ layer_inputs[-1]))
            x = norm(activations[-1])
        layer_inputs.append(jnp.append(x, 1.0))
        return self.weights[-1] @ layer_inputs[-1], (layer_inputs, activations)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def autostep_update(grad, x, alpha, h, v):
    """Autostep for one linear learner, with weights (n_out, n_in) and features x (n_in,).

    Returns the weight update and the new alpha, h and v. `grad` is the loss gradient, so h
    has the opposite sign to the paper's.
    """
    x_squared = x ** 2      # broadcast over the rows
    hg = h * grad
    v = jnp.maximum(jnp.abs(hg), v + alpha * x_squared / TAU * (jnp.abs(hg) - v))
    alpha = jnp.where(v > 0, alpha * jnp.exp(META_STEP_SIZE * hg / v), alpha)
    # Scale down each unit's step-sizes so that one update cannot overshoot its target
    alpha = alpha / jnp.maximum(jnp.sum(alpha * x_squared, axis=1, keepdims=True), 1.0)
    h = h * jnp.maximum(1 - alpha * x_squared, 0.0) + alpha * grad
    return -alpha * grad, alpha, h, v


class Autostep(eqx.Module):
    """Autostep on a list of weight matrices, one per layer.

    Autostep is derived for a linear learner. Here each layer counts as its own linear
    learner, so `update` needs each layer's inputs as well as the gradients.
    """
    alpha: list     # step-sizes, one array per weight matrix, with the same shape
    h: list         # traces
    v: list         # normalizers

    def __init__(self, weights, init_step_size):
        self.alpha = [jnp.full_like(w, init_step_size) for w in weights]
        self.h = [jnp.zeros_like(w) for w in weights]
        self.v = [jnp.zeros_like(w) for w in weights]

    def update(self, grads, layer_inputs):
        """Returns the weight updates and the new optimizer."""
        layers = zip(grads, layer_inputs, self.alpha, self.h, self.v)
        results = [autostep_update(*layer) for layer in layers]
        updates, alpha, h, v = (list(result) for result in zip(*results))
        return updates, eqx.tree_at(lambda opt: (opt.alpha, opt.h, opt.v), self, (alpha, h, v))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: Network
    autostep: Autostep
    target: Target
    key: jax.Array


def setup(seed):
    target_key, model_key, key = jax.random.split(jax.random.PRNGKey(seed), 3)
    model = Network(model_key)
    autostep = Autostep(model.weights, INIT_STEP_SIZE)
    return TrainState(model=model, autostep=autostep, target=Target(target_key), key=key)


def learn_step(state, _):
    """One update from one new sample. Returns the new state and the squared error before the update."""
    key, sample_key = jax.random.split(state.key)
    x, y = state.target.sample(sample_key)

    def loss_fn(model):
        prediction, info = model(x)
        return 0.5 * jnp.sum((prediction - y) ** 2), info

    (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)
    layer_inputs, activations = info

    # Autostep updates the weights
    updates, autostep = state.autostep.update(grads.weights, layer_inputs)
    weights = eqx.apply_updates(state.model.weights, updates)

    # Each Standardize layer updates its running statistics
    norms = []
    for norm, norm_input, norm_grads in zip(state.model.norms, activations, grads.norms):
        norms.append(norm.update_statistics(norm_input, norm_grads))

    model = eqx.tree_at(lambda model: (model.weights, model.norms), state.model, (weights, norms))
    new_state = TrainState(model=model, autostep=autostep, target=state.target, key=key)
    return new_state, 2 * loss / N_OUTPUTS


def main():
    state = setup(SEED)
    learn_many_steps = jax.jit(lambda state: jax.lax.scan(learn_step, state, length=LOG_EVERY))
    for step in range(LOG_EVERY, N_STEPS + 1, LOG_EVERY):
        state, squared_errors = learn_many_steps(state)
        print(f'step {step:>7,}: RMSE {jnp.sqrt(squared_errors.mean()):.3f} (noise floor {NOISE_STD})')


if __name__ == '__main__':
    main()
