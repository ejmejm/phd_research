# Per full run, over its last 10% of steps: loss per bit, varying bytes exactly right, and the share of the known-solvable
# gap closed (bits whose own byte + action beat the best linear model by > 0.1 nats).
import pickle, sys, numpy as np, jax, jax.numpy as jnp, optax
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
c = pickle.load(open('catalog.pkl', 'rb')); best = c['best']
data = np.load(D + 'data_mode0_diff0.npz'); valid = data['valid']
fit = pickle.load(open('offline_linear.pkl', 'rb'))
W, mean = jnp.asarray(fit['w']), jnp.asarray(fit['mean'], jnp.float32)
@jax.jit
def lin(r, a, n):
    b = jnp.concatenate([jnp.unpackbits(r, axis=-1), jax.nn.one_hot(a, 14)], -1)
    x = jnp.concatenate([(b - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), 0.01)), jnp.ones((*b.shape[:-1], 1))], -1)
    return optax.sigmoid_binary_cross_entropy(x @ W.T, jnp.unpackbits(n, axis=-1).astype(jnp.float32)).mean(0)
lo, hi = int(sys.argv[1]), int(sys.argv[2])      # the steps to evaluate on (the runs' last 10%)
ram, action, next_ram = data['ram'][:-1][valid][lo:hi], data['action'][valid][lo:hi], data['ram'][1:][valid][lo:hi]
lin_last = np.mean([np.asarray(lin(ram[i:i + 4000], action[i:i + 4000], next_ram[i:i + 4000])) for i in range(0, hi - lo, 4000)], 0)
m = (c['linear'] - best) > 0.1
target = (lin_last[m] - best[m]).sum()
best2 = pickle.load(open('catalog2.pkl', 'rb'))['best2']; m2 = (c['linear'] - best2) > 0.1   # wider yardstick
target2 = (lin_last[m2] - best2[m2]).sum()
for name in sys.argv[3:]:
    s = pickle.load(open(name + '.pkl', 'rb'))
    last = slice(-s.shape[1] // 10, None)
    run = s[..., :1024][:, last].mean(1)
    cap = (lin_last[m] - run[:, m]).sum(1) / target
    cap2 = (lin_last[m2] - run[:, m2]).sum(1) / target2
    print(f'  {name:32s} loss {run.mean():.4f} | bytes right {s[:, last, -1].mean():.3f} | gap closed {cap.mean():.0%} '
          f'(seeds {", ".join(f"{v:.0%}" for v in cap)}) | two-byte gap closed {cap2.mean():.0%}')
