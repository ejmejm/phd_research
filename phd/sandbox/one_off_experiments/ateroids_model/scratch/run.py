# 03's method (one hidden layer, full RAM, linear part learned online) with step-sizes capped (0.1; 1e-3 on the outputs'
# weights on the LTUs) and an optional maturity: LTUs younger than MATURITY steps are not replaced by search.
# Env: LABEL, WIDTH, N_SEEDS, N_STEPS, MATURITY, PERIOD. Saves LABEL.pkl, shaped (seed, point, 1025): the loss of each
# next-RAM bit and the fraction of varying bytes exactly right, averaged over LOG_EVERY steps.
import os, json, pickle
import numpy as np
from time import perf_counter
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
E = os.environ
D = '/home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/'
cells = [''.join(c['source']) for c in json.load(open(D + '03_unit_search.ipynb'))['cells'] if c['cell_type'] == 'code']
def rep(s, old, new):
    assert s.count(old) == 1, old[:90]
    return s.replace(old, new)
imports = 'import pickle, numpy as np, jax.numpy as jnp\nnp.asarray(jnp.zeros(1))\n' + cells[0].replace('%matplotlib inline', '')
data = cells[1]
data = rep(data, "DATA_FILE = 'data_mode0_diff0.npz'", f"DATA_FILE = '{D}{E.get('DATA_FILE', 'data_mode0_diff0.npz')}'")
data = rep(data, "N_STEPS = 100_000", f"N_STEPS = {int(E.get('N_STEPS', 200_000))}")
if E.get('SWITCH'):   # mode switch: the first SWITCH steps from DATA_FILE, the rest from SWITCH_FILE (from its step
    # SWITCH if ALIGN, so a run on SWITCH_FILE alone sees the same transitions after the switch; else from its start)
    data = rep(data, "DATA = tuple(jnp.asarray(v) for v in (ram, action, next_ram))",
               f"d2 = np.load('{D}' + '{E['SWITCH_FILE']}'); v2 = d2['valid']; S = {int(E['SWITCH'])}; O = {int(E['SWITCH']) if E.get('ALIGN') else 0}\n"
               "ram, action, next_ram = (np.concatenate([ram[:S], d2['ram'][:-1][v2][O:O + N_STEPS - S]]),\n"
               "                         np.concatenate([action[:S], d2['action'][v2][O:O + N_STEPS - S]]),\n"
               "                         np.concatenate([next_ram[:S], d2['ram'][1:][v2][O:O + N_STEPS - S]]))\n"
               "DATA = tuple(jnp.asarray(v) for v in (ram, action, next_ram))")
data = rep(data, "N_DEPTHS = 3 ", "N_DEPTHS = 1 ")
data = rep(data, "LOG_EVERY = N_STEPS // 100", "LOG_EVERY = max(N_STEPS // 100, 1_000)")
data = rep(data, "WIDTH = 60 ", f"WIDTH = {int(E.get('WIDTH', 600))} ")
data = rep(data, "PERIOD = 10_000", f"PERIOD = {int(E.get('PERIOD', 10_000))}")
data = rep(data, "N_SEEDS = 5", f"N_SEEDS = {int(E.get('N_SEEDS', 3))}\nSTEP_CAP, LTU_STEP_CAP = 0.1, 1e-3\nMATURITY = {int(E.get('MATURITY', 0))}")
learner = cells[2]
PARENTS = E.get('PARENTS', 'random')
FOCUS = float(E.get('FOCUS', 0.5))
data += ("\ndef mixed(score, net):\n"
         "    # 1 - FOCUS of the new LTUs go to outputs drawn uniformly (EXPLORE=uniform) or in proportion to their loss this\n"
         "    # search period (EXPLORE=loss); FOCUS in proportion to the positive part of score (as exploring if none is)\n"
         + ("    explore = jnp.full(N_OUTPUTS, 1 / N_OUTPUTS)\n" if E.get('EXPLORE', 'uniform') == 'uniform' else
            "    explore = net.loss_sum / jnp.maximum(net.loss_sum.sum(), 1e-30)\n") +
         "    score = jnp.maximum(score, 0.0)\n"
         "    share = jnp.where(score.sum() > 0, score / jnp.maximum(score.sum(), 1e-30), explore)\n"
         f"    return jnp.log({1 - FOCUS} * explore + {FOCUS} * share)\n")
data += ("\ndef explained_logits(net):\n"
         "    # parents in proportion to each output's loss L times its explained fraction (H - L) / H, H the entropy of its\n"
         "    # base rate, over this search period (0 where negative; uniform if all 0)\n"
         "    L, p = net.loss_sum / PERIOD, net.y_sum / PERIOD\n"
         "    H = -jax.scipy.special.xlogy(p, p) - jax.scipy.special.xlogy(1 - p, 1 - p)\n"
         "    w = L * jnp.maximum(H - L, 0.0) / jnp.maximum(H, 1e-30)\n"
         "    return jnp.log(jnp.where(w.sum() > 0, w / jnp.maximum(w.sum(), 1e-30), 1 / N_OUTPUTS))\n")
data += f"\nPARENTS = '{PARENTS}'\n"
data += ("\ndef trial_use(net):\n"
         "    # each LTU's usefulness to its parent alone (fit_line), or to all its outputs\n"
         "    parent = jnp.argmax(jnp.abs(net.shadow), 0)\n"
         "    own = net.w['output'][parent, N_INPUTS + jnp.arange(N_LTUS)] ** 2 * net.mask[parent, N_INPUTS + jnp.arange(N_LTUS)] * net.curv_sum[parent] / PERIOD\n"
         "    return own if PARENTS == 'fit_line' else usefulness(net)\n")
data += ("\ndef trials_logits(net, fit):\n"
         "    # parents in proportion to each output's loss L times the estimated fraction of it a new LTU removes: the mean\n"
         "    # outcome (usefulness / parent's loss when wired) of the LTUs tried with it as parent, with one pseudo-trial\n"
         "    # worth its explained fraction (fit=False) or a line fitted from explained fraction to outcome over all trials\n"
         "    L, p = net.loss_sum / PERIOD, net.y_sum / PERIOD\n"
         "    H = -jax.scipy.special.xlogy(p, p) - jax.scipy.special.xlogy(1 - p, 1 - p)\n"
         "    fraction = jnp.maximum(H - L, 0.0) / jnp.maximum(H, 1e-30)\n"
         "    parent = jnp.argmax(jnp.abs(net.shadow), 0)\n"
         "    trial = ~jnp.isnan(net.birth)\n"
         "    outcome = jnp.where(trial, trial_use(net) / jnp.maximum(jnp.nan_to_num(net.birth), 1e-30), 0.0)\n"
         "    alive_sum = jnp.zeros(N_OUTPUTS).at[parent].add(outcome)\n"
         "    alive_count = jnp.zeros(N_OUTPUTS).at[parent].add(trial * 1.0)\n"
         "    e = jnp.where(trial, net.birth_fraction, 0.0)\n"
         "    sums = net.fit_sums + jnp.array([2.0, 1.0, 1.0, 1.0, 1.0]) + jnp.stack([trial.sum(), e.sum(), outcome.sum(), (e * e).sum(), (e * outcome).sum()])\n"
         "    slope = (sums[4] - sums[1] * sums[2] / sums[0]) / jnp.maximum(sums[3] - sums[1] ** 2 / sums[0], 1e-30)\n"
         "    prior = jnp.maximum((sums[2] - slope * sums[1]) / sums[0] + slope * fraction, 0.0) if fit else fraction\n"
         "    w = L * prior if PARENTS == 'fit_line' else L * (net.dead_sum + alive_sum + prior) / (net.dead_count + alive_count + 1)\n"
         "    return jnp.log(jnp.where(w.sum() > 0, w / jnp.maximum(w.sum(), 1e-30), 1 / N_OUTPUTS))\n")
data += ("\nCATALOG = jnp.asarray((lambda c: c['linear'] - c['best'] > 0.1)(pickle.load(open('" + D + "scratch/catalog.pkl', 'rb'))))\n"
         "def parent_logits(net):\n"
         "    return " + {'random': 'jnp.zeros(N_OUTPUTS)', 'catalog': 'jnp.where(CATALOG, 0.0, -jnp.inf)',
                       'loss': 'jnp.where(net.loss_sum.sum() > 0, jnp.log(net.loss_sum / net.loss_sum.sum()), 0.0)',
                       'catalog_late': 'jnp.where(net.loss_sum.sum() > 0, jnp.where(CATALOG, 0.0, -jnp.inf), 0.0)',
                       'benefit': 'mixed(net.ltu_gain / PERIOD, net)',
                       'harmonic': 'mixed((lambda l, b: l * b / jnp.maximum(l + b, 1e-30))(net.loss_sum / PERIOD, jnp.maximum(net.ltu_gain / PERIOD, 0.0)), net)', 'progress': 'mixed(net.ltu_gain / PERIOD - net.prev_gain, net)', 'explained': 'explained_logits(net)', 'trials': 'trials_logits(net, False)', 'trials_fit': 'trials_logits(net, True)', 'fit_line': 'trials_logits(net, True)'}[PARENTS] + "\n")
learner = rep(learner, "    index = jax.random.randint(unit_key, (N_LTUS,), 0, jnp.where(on_output, N_OUTPUTS, PER_DEPTH))\n",
              "    index = jax.random.categorical(unit_key, parent_logits(net), shape=(N_LTUS,))   # one hidden layer: always an output\n")
learner = rep(learner, "    input_mean: jax.Array       # (N_INPUTS,): running mean of each input bit\n",
              "    input_mean: jax.Array       # (N_INPUTS,): running mean of each input bit\n"
              "    age: jax.Array              # (N_LTUS,): steps since each slot's unit was wired\n")
learner = rep(learner, "jnp.full(N_INPUTS, 0.5))", "jnp.full(N_INPUTS, 0.5), jnp.zeros(N_LTUS, jnp.int32))")
learner = rep(learner, "                        candidates=jnp.where(changed[:, :-1], 0.0, net.candidates))",
              "                        candidates=jnp.where(changed[:, :-1], 0.0, net.candidates), age=jnp.where(rows, 0, net.age))")
learner = rep(learner, "    updates, opt = OPTIMIZER.update((grads, None, sources), net.opt, net.w)\n",
              "    updates, opt = OPTIMIZER.update((grads, None, sources), net.opt, net.w)\n"
              "    beta = jax.tree.map(lambda b: jnp.minimum(b, np.log(STEP_CAP)), opt.beta)\n"
              "    opt = opt._replace(beta={**beta, 'output': beta['output'].at[:, N_INPUTS:-1].min(np.log(LTU_STEP_CAP))})\n")
learner = rep(learner, "jnp.stack([loss / N_OUTPUTS, accuracy])", "jnp.append(optax.sigmoid_binary_cross_entropy(logits, y), accuracy)")
OUT_GROW = bool(int(E.get('OUT_GROW', 0)))
if OUT_GROW:   # sparse LTU -> output connections: a new LTU feeds its parent output only; outputs grow LTU connections
    learner = rep(learner, "    age: jax.Array              # (N_LTUS,): steps since each slot's unit was wired\n",
                  "    age: jax.Array              # (N_LTUS,): steps since each slot's unit was wired\n"
                  "    out_candidates: jax.Array   # (N_OUTPUTS, N_LTUS): sum of dL/dlogit * LTU since each output's last new LTU\n")
    learner = rep(learner, "jnp.zeros(N_LTUS, jnp.int32))", "jnp.zeros(N_LTUS, jnp.int32), jnp.zeros((N_OUTPUTS, N_LTUS)))")
    learner = rep(learner, "input_mean=input_mean), \\\n",
                  "input_mean=input_mean, out_candidates=net.out_candidates + jnp.where(\n"
                  "            net.mask[:, N_INPUTS:-1], 0.0, (jax.nn.sigmoid(logits) - y)[:, None] * source[None, N_INPUTS:-1])), \\\n")
    learner = rep(learner, "age=jnp.where(rows, 0, net.age))",
                  "age=jnp.where(rows, 0, net.age),\n"
                  "                        mask=net.mask.at[:, N_INPUTS:-1].set(jnp.where(rows[None], net.shadow != 0, net.mask[:, N_INPUTS:-1])),\n"
                  "                        out_candidates=jnp.where(rows[None], 0.0, net.out_candidates))")
    learner = rep(learner, "def learn(net, ram, action, next_ram):\n",
                  "def grow_outputs(net):\n"
                  "    # Each output whose largest gradient sum, over the active LTUs it does not read, is more than GROW_Z times\n"
                  "    # their root mean square starts reading that LTU, at weight 0 with Autostep's initial step-size, and restarts its sums.\n"
                  "    unread = ~net.mask[:, N_INPUTS:-1] & active(net)[None]\n"
                  "    sums = jnp.where(unread, jnp.abs(net.out_candidates), 0.0)\n"
                  "    rms = jnp.sqrt(jnp.sum(sums ** 2, 1) / jnp.maximum(unread.sum(1), 1))\n"
                  "    grows = (jnp.max(sums, 1) > GROW_Z * rms) & unread.any(1)\n"
                  "    new = (jnp.arange(N_LTUS) == jnp.argmax(sums, 1)[:, None]) & grows[:, None]\n"
                  "    cols = jnp.zeros_like(net.mask).at[:, N_INPUTS:-1].set(new)\n"
                  "    zero = lambda t: {**t, 'output': jnp.where(cols, 0.0, t['output'])}\n"
                  "    beta = {**net.opt.beta, 'output': jnp.where(cols, net.opt.init_beta, net.opt.beta['output'])}\n"
                  "    return net._replace(w=zero(net.w), mask=net.mask | cols, opt=net.opt._replace(beta=beta, h=zero(net.opt.h), v=zero(net.opt.v)),\n"
                  "                        out_candidates=jnp.where(grows[:, None], 0.0, net.out_candidates))\n\n\n"
                  "def learn(net, ram, action, next_ram):\n")
learner = rep(learner, "    candidates: jax.Array       # (N_LTUS, N_INPUTS + N_LTUS): sum of dL/dz * source since each unit's last new connection\n",
              "    candidates: jax.Array       # (N_LTUS, N_INPUTS + N_LTUS): sum of dL/dz * source since each unit's last new connection\n"
              "    ltu_gain: jax.Array         # (N_OUTPUTS,): sum, this search period, of the loss the LTUs save on each output\n"
              "    prev_gain: jax.Array        # (N_OUTPUTS,): its mean over the previous search period\n"
              "    loss_sum: jax.Array         # (N_OUTPUTS,): sum, this search period, of each output's loss\n"
              "    curv_sum: jax.Array         # (N_OUTPUTS,): sum, this search period, of p (1 - p), the loss's curvature in the logit\n"
              "    y_sum: jax.Array            # (N_OUTPUTS,): sum, this search period, of each output's next bit\n"
              "    prev_use: jax.Array         # (N_LTUS,): each LTU's usefulness at the last search step (0 when it was wired)\n"
              "    birth: jax.Array            # (N_LTUS,): its parent's loss when it was wired by search (NaN: not a trial)\n"
              "    birth_fraction: jax.Array   # (N_LTUS,): its parent's explained fraction then\n"
              "    dead_sum: jax.Array         # (N_OUTPUTS,): replaced trials' outcomes (usefulness / birth), summed by parent, fading\n"
              "    dead_count: jax.Array       # (N_OUTPUTS,)\n"
              "    fit_sums: jax.Array         # (5,): replaced trials' n, sum e, sum f, sum e^2, sum e f, fading\n")
learner = rep(learner, "jnp.zeros((N_LTUS, N_INPUTS + N_LTUS)), jnp.full(N_INPUTS, 0.5)",
              "jnp.zeros((N_LTUS, N_INPUTS + N_LTUS)), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_LTUS), jnp.full(N_LTUS, jnp.nan), jnp.zeros(N_LTUS), jnp.zeros(N_OUTPUTS), jnp.zeros(N_OUTPUTS), jnp.zeros(5), jnp.full(N_INPUTS, 0.5)")
learner = rep(learner, "    accuracy = ((logits > 0)",
              "    ltu_part = (net.w['output'][:, N_INPUTS:-1] * net.mask[:, N_INPUTS:-1]) @ source[N_INPUTS:-1]\n"
              "    gain = optax.sigmoid_binary_cross_entropy(logits - ltu_part, y) - optax.sigmoid_binary_cross_entropy(logits, y)\n"
              "    accuracy = ((logits > 0)")
learner = rep(learner, "candidates=net.candidates + dz[:, None] * source[:-1], ",
              "candidates=net.candidates + dz[:, None] * source[:-1], ltu_gain=net.ltu_gain + gain, loss_sum=net.loss_sum + optax.sigmoid_binary_cross_entropy(logits, y), curv_sum=net.curv_sum + jax.nn.sigmoid(logits) * jax.nn.sigmoid(-logits), y_sum=net.y_sum + y, ")
if E.get('UTILITY') == 'curv':   # usefulness = the loss each LTU's removal would add, to second order: sum_o w_oj^2 E[p_o (1 - p_o)]
    learner = rep(learner, """    return (jnp.abs(net.w['output'][:, N_INPUTS:-1]).sum(0)
            + jnp.abs(net.w['hidden'][:, N_INPUTS:-1] * net.reads[:, N_INPUTS:-1]).sum(0))""",
                  """    return (net.w['output'][:, N_INPUTS:-1] ** 2 * net.mask[:, N_INPUTS:-1] * net.curv_sum[:, None]).sum(0) / PERIOD""")
training = cells[3]
if E.get('RESET_AT_SWITCH'):   # diagnostic: every step-size back to Autostep's initial one at the switch
    training = rep(training, "        net, scores = jax.lax.scan(one_step, net, index * WINDOW + jnp.arange(WINDOW))\n",
                   "        net, scores = jax.lax.scan(one_step, net, index * WINDOW + jnp.arange(WINDOW))\n"
                   f"        fresh = net._replace(opt=net.opt._replace(beta=jax.tree.map(lambda b: jnp.full_like(b, net.opt.init_beta), net.opt.beta), h=jax.tree.map(jnp.zeros_like, net.opt.h), v=jax.tree.map(jnp.zeros_like, net.opt.v)))\n"
                   f"        net = jax.tree.map(lambda a, b: jnp.where((index + 1) * WINDOW == {int(E['SWITCH'])}, a, b), fresh, net)\n")
training = rep(training, "        return jax.tree.map(lambda a, b: jnp.where(due, a, b), searched, net)",
               "        searched = searched._replace(prev_gain=searched.ltu_gain / PERIOD, ltu_gain=jnp.zeros_like(searched.ltu_gain), "
               "loss_sum=jnp.zeros_like(searched.loss_sum), curv_sum=jnp.zeros_like(searched.curv_sum), y_sum=jnp.zeros_like(searched.y_sum), "
               "prev_use=jnp.where(searched.age == 0, 0.0, outgoing))\n"
               "        return jax.tree.map(lambda a, b: jnp.where(due, a, b), searched, net)")
if OUT_GROW:
    training = rep(training, "        net = net._replace(mask=net.mask.at[:, N_INPUTS:-1].set((step >= add_step) & active(net)))\n", "")
    training = rep(training, "        net = jax.tree.map(lambda a, b: jnp.where(grows & (since_added > 0), a, b), grow(net), net)\n",
                   "        net = jax.tree.map(lambda a, b: jnp.where(grows & (since_added > 0), a, b), grow(net), net)\n"
                   "        net = jax.tree.map(lambda a, b: jnp.where(grows & (since_added > 0), a, b), grow_outputs(net), net)\n")
training = rep(training, "        return learn(net, *(v[step] for v in data))",
               "        net, scores = learn(net, *(v[step] for v in data))\n        return net._replace(age=net.age + 1), scores")
training = rep(training, "searched, net), scores\n", "searched, net), scores.mean(0)\n")
training = rep(training, "        weakest = jnp.argsort(jnp.where(active(net), outgoing, jnp.inf))[:REPLACED]\n",
               "        eligible = jnp.where(active(net) & (net.age >= MATURITY)"
               + (" & (outgoing <= net.prev_use)" if E.get('PROTECT') == 'rising' else "") + ", outgoing, jnp.inf)\n"
               "        weakest = jnp.argsort(eligible)[:REPLACED]\n")
if PARENTS in ('trials', 'trials_fit', 'fit_line'):   # trial bookkeeping at search steps
    training = rep(training, "        searched = start(net, jax.random.fold_in(wire_key, index), jnp.zeros(N_LTUS, bool).at[weakest].set(True), True,\n",
                   "        replaced = jnp.zeros(N_LTUS, bool).at[weakest].set(jnp.isfinite(eligible[weakest]))\n"
                   "        trial, parent = ~jnp.isnan(net.birth), jnp.argmax(jnp.abs(net.shadow), 0)\n"
                   "        outcome = jnp.where(trial, trial_use(net) / jnp.maximum(jnp.nan_to_num(net.birth), 1e-30), 0.0)\n"
                   "        ended, keep = replaced & trial, 1 - REPLACED_SHARE\n"
                   "        e = jnp.where(ended, net.birth_fraction, 0.0)\n"
                   "        f = jnp.where(ended, outcome, 0.0)\n"
                   "        net = jax.tree.map(lambda a, b: jnp.where(due, a, b), net._replace(\n"
                   "            dead_sum=keep * net.dead_sum + jnp.zeros(N_OUTPUTS).at[parent].add(f),\n"
                   "            dead_count=keep * net.dead_count + jnp.zeros(N_OUTPUTS).at[parent].add(ended * 1.0),\n"
                   "            fit_sums=keep * net.fit_sums + jnp.stack([ended.sum(), e.sum(), f.sum(), (e * e).sum(), (e * f).sum()]),\n"
                   "            birth=jnp.where(replaced, jnp.nan, net.birth)), net)\n"
                   "        searched = start(net, jax.random.fold_in(wire_key, index), jnp.zeros(N_LTUS, bool).at[weakest].set(True), True,\n")
    training = rep(training, "        searched = searched._replace(prev_gain=",
                   "        L, p = net.loss_sum / PERIOD, net.y_sum / PERIOD\n"
                   "        H = -jax.scipy.special.xlogy(p, p) - jax.scipy.special.xlogy(1 - p, 1 - p)\n"
                   "        new_parent = jnp.argmax(jnp.abs(searched.shadow), 0)\n"
                   "        searched = searched._replace(birth=jnp.where(replaced, L[new_parent], searched.birth),\n"
                   "                                     birth_fraction=jnp.where(replaced, (jnp.maximum(H - L, 0.0) / jnp.maximum(H, 1e-30))[new_parent], searched.birth_fraction))\n"
                   "        searched = searched._replace(prev_gain=")
training = rep(training, "jnp.zeros(N_LTUS, bool).at[weakest].set(True), True,",
               "jnp.zeros(N_LTUS, bool).at[weakest].set(jnp.isfinite(eligible[weakest])"
               + (" & (jnp.arange(REPLACED) < jnp.round(REPLACED_SHARE * jnp.isfinite(eligible).sum()))" if E.get('OF_ELIGIBLE') else "")
               + "), True,")   # OF_ELIGIBLE: replace REPLACED_SHARE of the eligible LTUs, not of all
training = rep(training, "scores.reshape(-1, LOG_EVERY, 2).mean(1)", "scores.reshape(-1, LOG_EVERY // WINDOW, N_OUTPUTS + 1).mean(1)")
training = rep(training, "reshape(len(configs), N_SEEDS, -1, 2)", "reshape(len(configs), N_SEEDS, -1, N_OUTPUTS + 1)")
if E.get('DUMP'):   # also save each run's final number of output connections per LTU
    training = rep(training, "    scores = jnp.concatenate([before, jax.lax.scan(window, net, index[added:])[1]])\n",
                   "    net, after = jax.lax.scan(window, net, index[added:])\n    scores = jnp.concatenate([before, after])\n")
    training = rep(training, "    return scores.reshape(-1, LOG_EVERY // WINDOW, N_OUTPUTS + 1).mean(1)",
                   "    return scores.reshape(-1, LOG_EVERY // WINDOW, N_OUTPUTS + 1).mean(1), dict(conn=(net.mask[:, N_INPUTS:-1] & active(net)[None]).sum(0), parent=jnp.argmax(jnp.abs(net.shadow), 0), "
                   "reads=net.reads[:, :N_INPUTS], use=usefulness(net), age=net.age, active=active(net))")
    training = rep(training, "    scores = train_all(nets, keys, sampled, period, fast, grows, random, swaps, DATA)\n",
                   "    scores, final = train_all(nets, keys, sampled, period, fast, grows, random, swaps, DATA)\n"
                   "    pickle.dump(jax.tree.map(np.asarray, final), open(os.environ['LABEL'] + '_final.pkl', 'wb'))\n")
if __name__ == '__main__':
    g = {}
    exec(imports + '\n' + data + '\n' + learner + '\n' + training, g)
    t0 = perf_counter()
    cfg = (0 if E.get('LINEAR') else g['N_LTUS'], True, g['PERIOD'], True, True, False, True)   # LINEAR: no LTUs
    scores = g['run']([cfg])[cfg]
    pickle.dump(scores, open(E['LABEL'] + '.pkl', 'wb'))
    last = slice(-scores.shape[1] // 10, None)
    print(f"{E['LABEL']}: {g['N_SEEDS']} runs x {g['N_STEPS']} steps in {perf_counter() - t0:.0f}s; last 10%: loss "
          f"{scores[:, last, :-1].mean():.4f}, varying bytes right {scores[:, last, -1].mean():.3f}", flush=True)
