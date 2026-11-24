# How much do the dynamics change between modes? The frozen linear model (fit on mode 0, steps 0-400k): loss per bit on
# held-out mode-0 steps (400k-500k) vs other modes' data, and the bytes whose loss rises most.
import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import pickle, sys, numpy as np, jax, jax.numpy as jnp, optax
np.asarray(jnp.zeros(1))
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
fit = pickle.load(open('offline_linear.pkl', 'rb'))
W, mean = jnp.asarray(fit['w']), jnp.asarray(fit['mean'], jnp.float32)
@jax.jit
def lin(r, a, n):
    b = jnp.concatenate([jnp.unpackbits(r, axis=-1), jax.nn.one_hot(a, 14)], -1)
    x = jnp.concatenate([(b - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), 0.01)), jnp.ones((*b.shape[:-1], 1))], -1)
    return optax.sigmoid_binary_cross_entropy(x @ W.T, jnp.unpackbits(n, axis=-1).astype(jnp.float32)).mean(0)
def per_bit(name, lo, hi):
    d = np.load(D + name); v = d['valid']
    r, a, n = d['ram'][:-1][v][lo:hi], d['action'][v][lo:hi], d['ram'][1:][v][lo:hi]
    return np.mean([np.asarray(lin(r[i:i + 4000], a[i:i + 4000], n[i:i + 4000])) for i in range(0, len(r) - 3999, 4000)], 0)
base = per_bit('data_mode0_diff0.npz', 400_000, 500_000)
print(f'mode 0 held out: {base.mean():.4f} per bit')
for name in sys.argv[1:]:
    l = per_bit(name, 0, 100_000)
    byte = (l - base).reshape(128, 8).sum(1)
    top = np.argsort(-byte)[:10]
    print(f'{name}: {l.mean():.4f} per bit (+{(l - base).sum():.1f} nats/step); bytes rising most: '
          + ', '.join(f'{b} (+{byte[b]:.2f})' for b in top))
