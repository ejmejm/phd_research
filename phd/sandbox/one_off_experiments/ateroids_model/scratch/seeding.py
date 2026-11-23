# For each catalog bit: final LTUs whose parent it is, and how many of them read its own byte's other bits (the inputs the
# best predictor uses), from a DUMP run's _final.pkl.
import pickle, sys, numpy as np
c = pickle.load(open('catalog.pkl', 'rb')); cat = np.where(c['linear'] - c['best'] > 0.1)[0]
for name in sys.argv[1:]:
    f = pickle.load(open(name + '_final.pkl', 'rb')); f = {k: v[0] for k, v in f.items()}   # seed 0
    s = pickle.load(open(name + '.pkl', 'rb'))[0, -10:, :1024].mean(0)
    print(name)
    for i in cat:
        kids = np.where(f['active'] & (f['parent'] == i))[0]
        own = [b for b in range(i // 8 * 8, i // 8 * 8 + 8) if b != i]
        n_own = f['reads'][kids][:, own].sum(1) if len(kids) else np.array([])
        print(f'  {i // 8:3d}.{7 - i % 8}  loss {s[i]:.3f} (lin {c["linear"][i]:.3f} best {c["best"][i]:.3f})  LTUs {len(kids):2d}  '
              f'own-byte inputs read: {" ".join(map(str, n_own.astype(int)))}')
