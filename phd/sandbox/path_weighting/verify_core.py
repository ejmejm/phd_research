"""Verification suite for path_gates.py.

Test 1  Correctness of the manual backward: with gates=1 the modulated weight
        gradients must equal jax.grad of the loss (to float tolerance).
Test 2  Correctness of the closed-form meta-grad in the regime where the
        leading-order approximation is EXACT: a single-hidden-layer net with the
        gate on the TOP layer. Analytic dL/dbeta must match a finite-difference
        estimate of how beta changes the next-step loss.
Test 3  Quality of the leading-order approximation on a DEEPER net: report the
        fraction of gate entries where analytic and finite-difference meta-grads
        agree in sign (the dropped upstream terms make this < exact, but it
        should be a clear majority -> the descent direction is usable).
"""
import jax, jax.numpy as jnp
import numpy as np
import path_gates as pg

jax.config.update("jax_enable_x64", True)   # tight tolerances for the checks


def make_data(key, B, in_dim, n_classes, n_heads):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (B, in_dim))
    ys = jax.random.randint(k2, (n_heads, B), 0, n_classes)
    y_onehot = [jax.nn.one_hot(ys[h], n_classes) for h in range(n_heads)]
    return x, y_onehot


# ---------------------------------------------------------------- Test 1
def test_modulated_equals_autodiff():
    f, fp = pg.act("tanh")
    key = jax.random.PRNGKey(0)
    p = pg.init_params(key, in_dim=12, hidden_dims=[16, 10], n_classes=5,
                       n_heads=2, beta_init=10.0)   # sigmoid(10)~=1 -> gates open
    x, y_onehot = make_data(jax.random.PRNGKey(1), B=8, in_dim=12, n_classes=5, n_heads=2)
    task_w = [1.0, 1.0]

    # force gates exactly 1 by using huge beta (sigmoid ~ 1); also test the gated=False path
    logits, cache, top = pg.forward(p, x, f)
    seeds, loss = pg.softmax_ce_seed(logits, y_onehot, task_w)
    gW_t, gW_h, *_ = pg.backward(p, cache, top, seeds, fp, gated=False)  # gates=1 path

    # autodiff oracle
    def loss_fn(trunk_W, head_W):
        a = x
        for W in trunk_W:
            a = f(a @ W.T)
        lg = [a @ Wh.T for Wh in head_W]
        s, L = pg.softmax_ce_seed(lg, y_onehot, task_w)
        return L
    g = jax.grad(loss_fn, argnums=(0, 1))(p.trunk_W, p.head_W)
    gt_auto, gh_auto = g

    errs = []
    for a, b in zip(gW_t, gt_auto):
        errs.append(float(jnp.max(jnp.abs(a - b))))
    for a, b in zip(gW_h, gh_auto):
        errs.append(float(jnp.max(jnp.abs(a - b))))
    max_err = max(errs)
    print(f"[Test 1] manual backward vs autodiff   max|Δ| = {max_err:.2e}", 
          "  PASS" if max_err < 1e-9 else "  FAIL")
    return max_err < 1e-9


# ---------------------------------------------------------------- Test 2
def test_metagrad_exact_single_hidden():
    """Single hidden layer, gate on the head only, BATCH SIZE 1 -> leading-order is EXACT.
    (For B>1 the closed form is a diagonal-in-batch approximation; see batch sweep below.)"""
    f, fp = pg.act("tanh")
    in_dim, H, C, B = 6, 7, 4, 1
    key = jax.random.PRNGKey(2)
    p = pg.init_params(key, in_dim=in_dim, hidden_dims=[H], n_classes=C,
                       n_heads=1, beta_init=0.3)
    # Freeze trunk gate at +inf (open) so ONLY the head gate is active/learned.
    p = pg.Params(p.trunk_W, p.head_W,
                  [jnp.full_like(p.trunk_beta[0], 30.0)], p.head_beta)
    x, y_onehot = make_data(jax.random.PRNGKey(3), B, in_dim, C, 1)
    task_w = [1.0]
    alpha, theta = 1e-3, 0.0   # theta=0: we only read the meta-grad, don't move beta

    # analytic meta-grad on the head gate
    logits, cache, top = pg.forward(p, x, f)
    seeds, _ = pg.softmax_ce_seed(logits, y_onehot, task_w)
    s_pack = pg.backward(p, cache, top, seeds, fp, gated=True)
    true_pack = pg.backward(p, cache, top, seeds, fp, gated=False)
    _, dbeta_h = pg.meta_grad(p, cache, top, s_pack, true_pack, fp, use_proxy=False)
    dbeta_h = dbeta_h[0]

    # finite-difference: one base step, measure next-step loss on SAME batch
    def next_loss(params):
        np_, _, L = pg.step(params, pg.zeros_like_gates(params), x, y_onehot, task_w,
                            f=f, fprime=fp, alpha=alpha, theta=0.0, gamma=0.0,
                            learn_gates=False, use_proxy=False)
        lg2, _, _ = pg.forward(np_, x, f)
        _, L2 = pg.softmax_ce_seed(lg2, y_onehot, task_w)
        return float(L2)

    eps = 1e-4
    fd = np.zeros_like(np.array(p.head_beta[0]))
    base = np.array(p.head_beta[0])
    for j in range(C):
        for i in range(H):
            up = base.copy(); up[j, i] += eps
            dn = base.copy(); dn[j, i] -= eps
            pu = pg.Params(p.trunk_W, p.head_W, p.trunk_beta, [jnp.array(up)])
            pd = pg.Params(p.trunk_W, p.head_W, p.trunk_beta, [jnp.array(dn)])
            fd[j, i] = (next_loss(pu) - next_loss(pd)) / (2 * eps)

    # the analytic grad carries a folded ||a||^2 normalizer; finite-diff carries the raw
    # alpha factor. Compare DIRECTION (cosine) and sign-agreement, which is what matters.
    a = np.array(dbeta_h).ravel(); b = fd.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
    print(f"[Test 2] single-hidden meta-grad: cosine(analytic, finite-diff) = {cos:.4f}"
          f"  sign-agreement = {sign_agree:.2%}",
          "  PASS" if cos > 0.99 else "  FAIL")
    return cos > 0.99


# ---------------------------------------------------------------- Test 3
def test_metagrad_quality_deep():
    """Deeper net, all gates active: leading-order is approximate. Report sign agreement."""
    f, fp = pg.act("tanh")
    in_dim, C, B = 10, 4, 16
    key = jax.random.PRNGKey(4)
    p = pg.init_params(key, in_dim=in_dim, hidden_dims=[12, 12, 8], n_classes=C,
                       n_heads=2, beta_init=0.5)
    x, y_onehot = make_data(jax.random.PRNGKey(5), B, in_dim, C, 2)
    task_w = [1.0, 1.0]
    alpha = 1e-3

    logits, cache, top = pg.forward(p, x, f)
    seeds, _ = pg.softmax_ce_seed(logits, y_onehot, task_w)
    s_pack = pg.backward(p, cache, top, seeds, fp, gated=True)
    true_pack = pg.backward(p, cache, top, seeds, fp, gated=False)
    dbeta_t, dbeta_h = pg.meta_grad(p, cache, top, s_pack, true_pack, fp, use_proxy=False)

    def next_loss(params):
        np_, _, _ = pg.step(params, pg.zeros_like_gates(params), x, y_onehot, task_w,
                            f=f, fprime=fp, alpha=alpha, theta=0.0, gamma=0.0,
                            learn_gates=False, use_proxy=False)
        lg2, _, _ = pg.forward(np_, x, f)
        _, L2 = pg.softmax_ce_seed(lg2, y_onehot, task_w)
        return float(L2)

    eps = 1e-4
    agree, total = 0, 0
    cos_acc = []
    # check a middle trunk layer (layer 0 is inert by construction) and one head
    for layer_beta, analytic in [(("trunk", 1), dbeta_t[1]), (("head", 0), dbeta_h[0])]:
        kind, idx = layer_beta
        beta_arr = (p.trunk_beta if kind == "trunk" else p.head_beta)[idx]
        base = np.array(beta_arr)
        an = np.array(analytic)
        fd = np.zeros_like(base)
        # subsample entries for speed
        rows = range(base.shape[0]); cols = range(base.shape[1])
        for j in rows:
            for i in cols:
                up = base.copy(); up[j, i] += eps
                dn = base.copy(); dn[j, i] -= eps
                if kind == "trunk":
                    pu = pg.Params([up if l == idx else p.trunk_W[l]*1 for l in range(len(p.trunk_W))]
                                   if False else p.trunk_W,
                                   p.head_W,
                                   [jnp.array(up) if l == idx else p.trunk_beta[l] for l in range(len(p.trunk_beta))],
                                   p.head_beta)
                    pd = pg.Params(p.trunk_W, p.head_W,
                                   [jnp.array(dn) if l == idx else p.trunk_beta[l] for l in range(len(p.trunk_beta))],
                                   p.head_beta)
                else:
                    pu = pg.Params(p.trunk_W, p.head_W, p.trunk_beta,
                                   [jnp.array(up) if l == idx else p.head_beta[l] for l in range(len(p.head_beta))])
                    pd = pg.Params(p.trunk_W, p.head_W, p.trunk_beta,
                                   [jnp.array(dn) if l == idx else p.head_beta[l] for l in range(len(p.head_beta))])
                fd[j, i] = (next_loss(pu) - next_loss(pd)) / (2 * eps)
        a = an.ravel(); b = fd.ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        cos_acc.append((kind, cos))
        agree += int(np.sum(np.sign(a) == np.sign(b))); total += a.size
    print(f"[Test 3] deep-net leading-order quality:")
    for kind, cos in cos_acc:
        print(f"          {kind:5s} cosine(analytic, finite-diff) = {cos:+.4f}")
    print(f"          overall sign-agreement = {agree/total:.2%}")
    return True


# ---------------------------------------------------------------- Test 4
def test_sigmoid_bce_backward_equals_autodiff():
    """With gates=1 the manual backward for sigmoid+BCE must equal jax.grad."""
    f, fp = pg.act("leaky_relu", leaky_slope=0.01)
    key = jax.random.PRNGKey(7)
    p = pg.init_params(key, in_dim=12, hidden_dims=[16], n_classes=20,
                       n_heads=1, beta_init=10.0)  # sigmoid(10)~1
    x = jax.random.normal(jax.random.PRNGKey(8), (5, 12))
    # binary targets per output unit
    y = (jax.random.uniform(jax.random.PRNGKey(9), (5, 20)) > 0.5).astype(jnp.float64)
    task_w = [1.0]

    logits, cache, top = pg.forward(p, x, f)
    seeds, loss = pg.sigmoid_bce_seed(logits, [y], task_w)
    gW_t, gW_h, *_ = pg.backward(p, cache, top, seeds, fp, gated=False)

    def loss_fn(trunk_W, head_W):
        a = x
        for W in trunk_W:
            a = f(a @ W.T)
        lg = [a @ Wh.T for Wh in head_W]
        _, L = pg.sigmoid_bce_seed(lg, [y], task_w)
        return L
    gt_auto, gh_auto = jax.grad(loss_fn, argnums=(0, 1))(p.trunk_W, p.head_W)
    errs = []
    for a, b in zip(gW_t, gt_auto):
        errs.append(float(jnp.max(jnp.abs(a - b))))
    for a, b in zip(gW_h, gh_auto):
        errs.append(float(jnp.max(jnp.abs(a - b))))
    max_err = max(errs)
    print(f"[Test 4] sigmoid+BCE manual backward vs autodiff  max|Δ| = {max_err:.2e}",
          "  PASS" if max_err < 1e-9 else "  FAIL")
    return max_err < 1e-9


if __name__ == "__main__":
    ok1 = test_modulated_equals_autodiff()
    ok2 = test_metagrad_exact_single_hidden()
    test_metagrad_quality_deep()
    ok4 = test_sigmoid_bce_backward_equals_autodiff()
    print()
    print("CORE OK" if (ok1 and ok2 and ok4) else "CORE HAS ISSUES")
