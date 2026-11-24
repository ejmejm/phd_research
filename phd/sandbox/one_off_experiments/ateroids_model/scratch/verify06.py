# Runs notebook 06's code at 600 LTUs (3 seeds, no baseline) and compares it with run.py's l3_lossfix.pkl.
import json, os, pickle, numpy as np
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
os.chdir(D)
cells = [''.join(c['source']) for c in json.load(open('06_scaling_unit_search.ipynb'))['cells'] if c['cell_type'] == 'code']
g = {}
exec(cells[0].replace('%matplotlib inline', '') + '\nimport numpy as np\nnp.asarray(jnp.zeros(1))\n' + cells[1] + '\n' + cells[2] + '\n' + cells[3], g)
g['set_width'](600)
s = g['run']([600])[600]
ref = pickle.load(open(D + 'scratch/l3_lossfix.pkl', 'rb'))
print('shapes', s.shape, ref.shape, '| identical:', np.array_equal(s, ref), '| max abs diff', np.abs(s - ref).max())
last = slice(-10, None)
print('notebook last 10%:', s[:, last, :-1].mean((1, 2)), s[:, last, -1].mean(1))
print('run.py   last 10%:', ref[:, last, :-1].mean((1, 2)), ref[:, last, -1].mean(1))
