# Mode-switch runs (400k steps, mode 0 then mode 1 from SWITCH): mean loss per bit before and after the switch, over
# all bits but byte 0 (it holds the mode) and over the bytes whose dynamics change (asteroid x and sizes).
import pickle, sys, numpy as np
SWITCH, LOG = int(sys.argv[1]), 4000
CHANGED = [17, 21, 22, 23, 30, 31, 32, 33, 41]
bits = lambda bytes_: np.concatenate([np.arange(b * 8, b * 8 + 8) for b in bytes_])
rest, changed = bits(range(1, 128)), bits(CHANGED)
s0 = SWITCH // LOG
spans = {'pre (last 20k)': slice(s0 - 5, s0), 'post 0-4k': slice(s0, s0 + 1), 'post 4-20k': slice(s0 + 1, s0 + 5),
         'post 20-100k': slice(s0 + 5, s0 + 25), 'post last 20k': slice(-5, None)}
print(f'{"":14s}' + ''.join(f'{k:>16s}' for k in spans))
for name in sys.argv[2:]:
    s = pickle.load(open(name + '.pkl', 'rb')).mean(0)
    for label, cols in [('all but byte 0', rest), ('changed bytes', changed)]:
        print(f'{name:14s}' + ''.join(f'{s[sp][:, cols].mean():16.4f}' for sp in spans.values()) + f'   {label}')
    print(f'{name:14s}' + ''.join(f'{s[sp][:, -1].mean():16.3f}' for sp in spans.values()) + '   varying bytes right')
