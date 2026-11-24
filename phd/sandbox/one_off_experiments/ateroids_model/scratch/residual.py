# Per catalog bit: frozen linear, best (own byte + action), and each run's loss over the last 10% (steps 180k-200k).
import pickle, sys, numpy as np
exec(open('capture.py').read().split("for name in sys.argv[3:]")[0].replace("int(sys.argv[1]), int(sys.argv[2])", "180_000, 200_000"))
names = sys.argv[1:]
runs = [pickle.load(open(n + '.pkl', 'rb'))[..., :1024][:, -10:].mean((0, 1)) for n in names]
idx = np.where(m)[0]
gap = lin_last - best
order = idx[np.argsort(-(runs[-1][idx] - best[idx]))]
print('byte.bit  linear  best  ' + '  '.join(n.split('_')[1] for n in names) + '   left(nats)')
for i in order:
    print(f'{i // 8:4d}.{7 - i % 8}  {lin_last[i]:.3f}  {best[i]:.3f}  ' + '  '.join(f'{r[i]:.3f}' for r in runs) + f'   {runs[-1][i] - best[i]:.3f}')
left = np.array([runs[-1][i] - best[i] for i in idx])
print('total gap', gap[idx].sum(), 'left at last', left.sum(), 'negative (beat best):', (left < 0).sum())
