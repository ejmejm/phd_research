"""Build a notebook from a cell file, or exec its code cells with overrides (smoke test / timing).

Cell file format: cells separated by lines '# %%' (code) or '# %% [markdown]' (markdown, lines prefixed '# ').
usage: python nbtool.py build CELLS.py OUT.ipynb
       python nbtool.py exec CELLS.py 'NUM_STEPS=20_000' 'N_SEEDS=1'
       python nbtool.py extract NOTEBOOK.ipynb CELLS.py
"""
import json
import sys

import nbformat


def parse(path):
    cells, kind, lines = [], None, []
    for line in open(path).read().split('\n'):
        if line.startswith('# %%'):
            if kind is not None:
                cells.append((kind, '\n'.join(lines).strip('\n')))
            kind, lines = ('md' if '[markdown]' in line else 'code'), []
        else:
            lines.append(line[2:] if kind == 'md' and line.startswith('# ') else ('' if kind == 'md' and line == '#' else line))
    cells.append((kind, '\n'.join(lines).strip('\n')))
    return cells


if sys.argv[1] == 'build':
    nb = nbformat.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'jax', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.10.14'}}
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == 'md' else nbformat.v4.new_code_cell(s)
                for k, s in parse(sys.argv[2])]
    nbformat.write(nb, sys.argv[3])
elif sys.argv[1] == 'extract':
    out = []
    for c in json.load(open(sys.argv[2]))['cells']:
        src = ''.join(c['source'])
        if c['cell_type'] == 'code':
            out.append('# %%\n' + src)
        else:
            out.append('# %% [markdown]\n' + '\n'.join('# ' + l if l else '#' for l in src.split('\n')))
    open(sys.argv[3], 'w').write('\n'.join(out) + '\n')
else:
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    overrides = [a for a in sys.argv[3:] if not a.startswith('--')]
    code = [s for k, s in parse(sys.argv[2]) if k == 'code']
    g = {}
    for i, s in enumerate(code):
        s = '\n'.join(l for l in s.split('\n') if not l.startswith('%'))
        exec(compile(s, f'cell{i}', 'exec'), g)
        if i == 0:
            exec('np.asarray(jnp.zeros(1))', g)   # WSL pinned-memory workaround
        if i == 1:   # the config cell
            for o in overrides:
                exec(o, g)
