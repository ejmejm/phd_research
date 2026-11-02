"""Records one run of our method on sparse_target_demo.ipynb's target network for the viewer (index.html): a snapshot of
the network every SNAP steps, written to data.js. The method's code is run from the notebook's cells."""
import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
for cell in json.loads((HERE / 'sparse_target_demo.ipynb').read_text())['cells']:
    if cell['id'] in ('cell-1', 'cell-2', 'cell-4', 'cell-6', 'cell-8'):
        exec(''.join(line for line in cell['source'] if not line.startswith('%')))
np.asarray(jnp.zeros(1))   # a tiny device-to-host copy first avoids a pinned-memory OOM on WSL

SNAP = 100                  # steps between snapshots
STEP_SIZE = 2.0 ** -11      # ours' best step-size in the notebook
SEED = 0
N_MATCH = 2000              # inputs on which each learned LTU is matched to the target LTU it correlates with most


def target_bits(x):
    s = jnp.zeros((len(x), 2 * N_PER_DEPTH))
    for rows in (slice(0, N_PER_DEPTH), slice(N_PER_DEPTH, None)):
        s = s.at[:, rows].set((jnp.concatenate([x, s], 1) @ target.w[rows].T > target.threshold[rows]).astype(float))
    return s


def hidden_bits(net, x):
    a, bits, ones = jnp.zeros((len(x), N_LTUS)), jnp.zeros((len(x), N_LTUS)), jnp.ones((len(x), 1))
    for rows in DEPTHS:
        b = (jnp.concatenate([x, a, ones], 1) @ (net.w['hidden'][rows] * net.reads[rows]).T > 0).astype(float)
        mean = net.stats.hidden_mean[rows]
        a = a.at[:, rows].set((b - mean) / jnp.sqrt(jnp.maximum(mean * (1 - mean), VAR_FLOOR)))
        bits = bits.at[:, rows].set(b)
    return bits


def normalized(b):
    b = b - b.mean(0)
    return b / jnp.maximum(jnp.linalg.norm(b, axis=0), 1e-6)


def swap_slots(net):
    # search's swap: whether the strongest flat depth-2 LTU moves to the weakest depth-1 LTU's slot, and the two slots.
    outgoing = usefulness(net)
    links = jnp.abs(net.w['hidden'][:, :-1]) * net.reads[:, :-1]
    flat = (jnp.arange(N_LTUS) >= HALF) & (links[:, N_INPUTS:].sum(1) < FLAT_SHARE * links.sum(1))
    mover = jnp.argmax(jnp.where(flat, outgoing, -jnp.inf))
    weak = jnp.argmin(jnp.where(jnp.arange(N_LTUS) < HALF, outgoing, jnp.inf))
    return flat.any() & (outgoing[mover] > outgoing[weak]), weak, mover


keys = lambda k: jax.random.split(jax.random.PRNGKey(k), N_SEEDS)[SEED]
target = make_target(keys(2))
x_match = jax.random.normal(jax.random.PRNGKey(3), (N_MATCH, N_INPUTS))
target_match = normalized(target_bits(x_match))


def window(net, inputs):
    # train's window for ours, also returning which slots got a new LTU and which got the LTU of another slot (-1: none).
    index, key = inputs
    net, mse = jax.lax.scan(lambda s, k: learn(s, *sample(target, k)), net, jax.random.split(key, WINDOW))
    net = grow(net)
    due = (index + 1) * WINDOW % PERIOD == 0
    searched = search(net, jax.random.fold_in(key, 1))
    moved, weak, mover = swap_slots(net)
    moved_from = jnp.full(N_LTUS, -1).at[weak].set(jnp.where(due & moved, mover, -1))
    changed = jnp.any(searched.w['output'][:, N_INPUTS:-1] != net.w['output'][:, N_INPUTS:-1], 0)
    net = jax.tree.map(lambda a, b: jnp.where(due, a, b), searched, net)
    return net, (mse.mean(), due & changed & (moved_from < 0), moved_from)


def frame(net):
    corr = jnp.abs(normalized(hidden_bits(net, x_match)).T @ target_match)
    return dict(hidden=net.w['hidden'][:, :-1] * net.reads[:, :-1], reads=net.reads[:, :-1],
                output=net.w['output'][:, :-1], beta_hidden=net.opt.beta['hidden'][:, :-1],
                beta_output=net.opt.beta['output'][:, :-1], match=jnp.argmax(corr, 1), corr=jnp.max(corr, 1))


def chunk(net, inputs):
    net, (mse, new, moved_from) = jax.lax.scan(window, net, inputs)
    return net, (frame(net), jnp.sqrt(mse.mean()), new.any(0), moved_from.max(0))


@jax.jit
def record():
    init_key, key = keys(0), keys(1)
    n, per = NUM_STEPS // WINDOW, SNAP // WINDOW
    net = start(init_net(init_key, STEP_SIZE, False), jax.random.fold_in(init_key, 1), jnp.ones(N_LTUS, bool))
    _, (frames, rmse, new, moved_from) = jax.lax.scan(
        chunk, net, (jnp.arange(n).reshape(-1, per), jax.random.split(key, n).reshape(-1, per, 2)))
    first = frame(net)
    frames = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), first, frames)
    return frames, rmse, new, moved_from


frames, rmse, new, moved_from = jax.tree.map(np.asarray, record())
print(f'final RMSE {rmse[-len(rmse) // 10:].mean():.3f}, {new.sum()} new LTUs, {(moved_from >= 0).sum()} moved')

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
quantize = lambda a: (np.round(a / np.abs(a).max() * 127).astype(np.int8), float(np.abs(a).max() / 127))
LOG_RANGE = (np.log(1e-6), 0.0)   # log step-sizes stored as uint8 over this range
log_step = lambda a: np.round(np.clip((a - LOG_RANGE[0]) / (LOG_RANGE[1] - LOG_RANGE[0]), 0, 1) * 255).astype(np.uint8)
data = dict(n_inputs=N_INPUTS, n_ltus=N_LTUS, n_outputs=N_OUTPUTS, n_target=2 * N_PER_DEPTH, snap=SNAP,
            position=POSITION.tolist(),
            n_frames=len(frames['hidden']), log_range=LOG_RANGE, noise=NOISE_STD,
            rmse=np.round(rmse, 4).tolist(), new=b64(new.astype(np.uint8)), moved_from=b64(moved_from.astype(np.int8)),
            reads=b64(np.packbits(frames['reads'])), match=b64(frames['match'].astype(np.uint8)),
            corr=b64(np.round(frames['corr'] * 255).astype(np.uint8)),
            beta_hidden=b64(log_step(frames['beta_hidden'])), beta_output=b64(log_step(frames['beta_output'])),
            target_hidden=b64(np.asarray(target.w).astype(np.int8)), target_output=b64(np.asarray(target.out).astype(np.int8)))
for name in ('hidden', 'output'):
    data[name], data[name + '_scale'] = quantize(frames[name])
    data[name] = b64(data[name])
(HERE / 'data.js').write_text('window.DATA = ' + json.dumps(data) + ';\n')
print(f"wrote data.js ({(HERE / 'data.js').stat().st_size / 1e6:.1f} MB)")
