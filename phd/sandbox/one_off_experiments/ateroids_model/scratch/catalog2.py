# Wider yardstick: per next-RAM bit, held-out loss of a count model on (own byte, action) and on (own byte, one other
# byte, action), counts from steps 0-100k, loss on 100k-200k; unseen keys back off to the smaller model (smoothing 1).
import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import pickle, numpy as np, jax, jax.numpy as jnp
np.asarray(jnp.zeros(1))
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
data = np.load(D + 'data_mode0_diff0.npz'); valid = data['valid']
N = 200_000; H = N // 2
ram = jnp.asarray(data['ram'][:-1][valid][:N].astype(np.int32)); act = jnp.asarray(data['action'][valid][:N].astype(np.int32))
nb = jnp.asarray(np.unpackbits(data['ram'][1:][valid][:N], axis=1).astype(np.float32))

def counts(key, bits, query):
    # per query key: train count and per-bit ones count (bits: (H, 8))
    order = jnp.argsort(key[:H]); k = key[:H][order]; c = jnp.concatenate([jnp.zeros((1, bits.shape[1])), jnp.cumsum(bits[:H][order], 0)])
    lo, hi = jnp.searchsorted(k, query, 'left'), jnp.searchsorted(k, query, 'right')
    return (hi - lo)[:, None], c[hi] - c[lo]

def nll(p, y):
    p = jnp.clip(p, 1e-6, 1 - 1e-6); return -(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)).mean(0)

@jax.jit
def own_model(a, ram, act, nb):
    bits = jax.lax.dynamic_slice_in_dim(nb, a * 8, 8, 1)
    prior = bits[:H].mean(0)
    key = ram[:, a] * 14 + act
    n, n1 = counts(key, bits, key[H:])
    p1 = (n1 + prior) / (n + 1)
    return p1, nll(p1, bits[H:])

@jax.jit
def pair_model(a, b, p1, ram, act, nb):
    bits = jax.lax.dynamic_slice_in_dim(nb, a * 8, 8, 1)
    key = ram[:, a] * 256 * 14 + ram[:, b] * 14 + act
    n, n1 = counts(key, bits, key[H:])
    return nll((n1 + p1) / (n + 1), bits[H:])

best1 = np.zeros(1024); best2 = np.zeros(1024); arg2 = np.zeros(1024, int)
for a in range(128):
    p1, l1 = own_model(a, ram, act, nb)
    best1[a * 8:a * 8 + 8] = np.asarray(l1)
    l2 = np.stack([np.asarray(pair_model(a, b, p1, ram, act, nb)) if b != a else np.asarray(l1) for b in range(128)])
    best2[a * 8:a * 8 + 8] = l2.min(0); arg2[a * 8:a * 8 + 8] = l2.argmin(0)
c = pickle.load(open('catalog.pkl', 'rb'))
pickle.dump(dict(best1=best1, best2=best2, arg2=arg2), open('catalog2.pkl', 'wb'))
lin = c['linear']
for name, b in [('own byte + action', best1), ('own + one other byte + action', best2)]:
    m = lin - b > 0.1
    print(f'{name}: {m.sum()} bits beat linear by > 0.1 nats, gap {(lin - b)[m].sum():.1f} nats/step')
