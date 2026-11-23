# Known-solvable bits: best loss from own byte + action (conditional entropy) vs the frozen linear model, steps 0-200k.
import pickle, numpy as np, jax, jax.numpy as jnp, optax
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
data = np.load(D + 'data_mode0_diff0.npz'); valid = data['valid']
N = 200_000
ram, action, next_ram = data['ram'][:-1][valid][:N], data['action'][valid][:N], data['ram'][1:][valid][:N]
nb = np.unpackbits(next_ram, axis=1)
def cond_H(target, key):
    joint = key * 2 + target
    _, cj = np.unique(joint, return_counts=True); _, ck = np.unique(key, return_counts=True)
    return -(cj / N * np.log(cj / N)).sum() + (ck / N * np.log(ck / N)).sum()
best = np.array([cond_H(nb[:, i], ram[:, i // 8].astype(np.int64) * 14 + action) for i in range(1024)])
fit = pickle.load(open('offline_linear.pkl', 'rb'))
W, mean = jnp.asarray(fit['w']), jnp.asarray(fit['mean'], jnp.float32)
@jax.jit
def lin(r, a, n):
    b = jnp.concatenate([jnp.unpackbits(r, axis=-1), jax.nn.one_hot(a, 14)], -1)
    x = jnp.concatenate([(b - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), 0.01)), jnp.ones((*b.shape[:-1], 1))], -1)
    return optax.sigmoid_binary_cross_entropy(x @ W.T, jnp.unpackbits(n, axis=-1).astype(jnp.float32)).mean(0)
linear = np.mean([np.asarray(lin(ram[i:i + 10000], action[i:i + 10000], next_ram[i:i + 10000])) for i in range(0, N, 10000)], 0)
pickle.dump(dict(best=best, linear=linear), open('catalog.pkl', 'wb'))
m = linear - best > 0.1
print(f'catalog: {m.sum()} bits with gap > 0.1, gap sum {(linear - best)[m].sum():.1f} nats/step')
