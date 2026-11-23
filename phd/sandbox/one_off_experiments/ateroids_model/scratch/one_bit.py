# 04's method (frozen linear model + 60 searched LTUs on one output) on another bit: env BYTE, VALUE, N_SEEDS, N_STEPS.
import os, json, numpy as np
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
E = os.environ
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
cells = [''.join(c['source']) for c in json.load(open(D + '04_heading_bit.ipynb'))['cells'] if c['cell_type'] == 'code']
def rep(s, old, new):
    assert s.count(old) == 1, old[:90]
    return s.replace(old, new)
code = 'import numpy as np, jax.numpy as jnp\nnp.asarray(jnp.zeros(1))\n' + cells[0].replace('%matplotlib inline', '') + '\n'
c1 = rep(cells[1], "DATA_FILE = 'data_mode0_diff0.npz'", f"DATA_FILE = '{D}data_mode0_diff0.npz'")
c1 = rep(c1, "N_STEPS = 400_000", f"N_STEPS = {int(E.get('N_STEPS', 200_000))}")
c1 = rep(c1, "TARGET_BYTE, TARGET_VALUE = 60, 1", f"TARGET_BYTE, TARGET_VALUE = {E['BYTE']}, {E['VALUE']}")
c1 = rep(c1, "N_SEEDS = 30", f"N_SEEDS = {int(E.get('N_SEEDS', 5))}")
g = {}
exec(code + c1 + '\n' + cells[2] + '\n' + cells[3] + '\n' + cells[4], g)
s = g['scores'][g['SEARCH']][..., 0]
last = slice(-s.shape[1] // 10, None)
print(f"byte {E['BYTE']} value {E['VALUE']}: frozen linear {np.mean(g['linear_scores'][0][last]):.3f}; search per seed, last 10%: "
      + ' '.join(f'{v:.3f}' for v in s[:, last].mean(1)))
