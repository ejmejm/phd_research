# Adaptation after a switch: each learner switched from mode 0 at 200k vs the same learner trained only on the new
# setting, on the same transitions (steps 200k-400k of the new setting). Mean loss per bit, all bits but byte 0.
import pickle, sys, numpy as np
s0 = 50   # the switch, in 4,000-step points
spans = {'0-4k': slice(s0, s0 + 1), '4-20k': slice(s0 + 1, s0 + 5), '20-100k': slice(s0 + 5, s0 + 25),
         'last 20k': slice(-5, None)}
cols = np.arange(8, 1024)
for target in sys.argv[1:]:
    print(f'{target}: steps since switch {"":4s}' + ''.join(f'{k:>10s}' for k in spans))
    for method in ['new', 'old', 'lin']:
        sw, fr = (pickle.load(open(f'{kind}_{target}_{method}.pkl', 'rb')).mean(0)[:, cols].mean(1) for kind in ('sw', 'fr'))
        print(f'  {method}: switched {"":11s}' + ''.join(f'{sw[sp].mean():10.4f}' for sp in spans.values()))
        print(f'  {method}: trained on it only{"":2s}' + ''.join(f'{fr[sp].mean():10.4f}' for sp in spans.values()))
