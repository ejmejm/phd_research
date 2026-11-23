# The frozen linear model: logistic regression of every next-RAM bit on the standardized inputs + bias, Adam (batch 256,
# 40 epochs, lr 1e-3) on steps 0-400k. Saves offline_linear.pkl.
import os, pickle
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
jax.config.update('jax_platforms', 'cuda')
import jax.numpy as jnp, numpy as np, optax
jax.device_get(jnp.zeros(1))
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
data = np.load(D + 'data_mode0_diff0.npz'); valid = data['valid']
ram, action, next_ram = data['ram'][:-1][valid], data['action'][valid], data['ram'][1:][valid]
N_TRAIN, BATCH, EPOCHS, LR = 400_000, 256, 40, 1e-3
mean = np.concatenate([np.unpackbits(ram[:N_TRAIN], axis=1).mean(0), np.eye(14)[action[:N_TRAIN]].mean(0)])
scale = np.sqrt(np.maximum(mean * (1 - mean), 0.01))
def feats(r, a):
    b = jnp.concatenate([jnp.unpackbits(r, axis=-1).astype(jnp.float32), jax.nn.one_hot(a, 14)], -1)
    return jnp.concatenate([(b - mean) / scale, jnp.ones((*r.shape[:-1], 1))], -1)
def loss(w, r, a, n): return optax.sigmoid_binary_cross_entropy(feats(r, a) @ w.T, jnp.unpackbits(n, axis=-1).astype(jnp.float32))
opt = optax.adam(LR)
R, A, N = jnp.asarray(ram), jnp.asarray(action), jnp.asarray(next_ram)
@jax.jit
def epoch(w, s, key):
    idx = jax.random.permutation(key, N_TRAIN)[: N_TRAIN // BATCH * BATCH].reshape(-1, BATCH)
    def step(c, i):
        w, s = c
        g = jax.grad(lambda w: loss(w, R[i], A[i], N[i]).mean())(w)
        u, s = opt.update(g, s, w)
        return (optax.apply_updates(w, u), s), None
    return jax.lax.scan(step, (w, s), idx)[0]
@jax.jit
def test(w):
    return jax.lax.map(lambda i: loss(w, R[i], A[i], N[i]).mean(0), N_TRAIN + jnp.arange(100_000).reshape(-1, 1000)).mean(0)
w = jnp.zeros((1024, 1039)); s = opt.init(w)
for e in range(EPOCHS):
    w, s = epoch(w, s, jax.random.PRNGKey(e))
print('test loss per bit', float(test(w).mean()))
pickle.dump(dict(w=np.asarray(w), mean=mean), open('offline_linear.pkl', 'wb'))
