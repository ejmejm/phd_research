# -*- coding: utf-8 -*-
r"""
Learned backward path-gates for structural credit assignment.
=============================================================

Core math (manual forward + two manual backward passes + closed-form meta-update).

A standard backprop edge passes  delta_j * w_ji  from node j back to node i.
We insert a per-weight multiplicative gate g_ji = sigmoid(beta_ji) on the
BACKWARD path only:

    s_i = f'(net_i) * sum_j  sigmoid(beta_ji) * w_ji * s_j           (modulated backprop)

The base learner trains on this modulated pseudo-gradient. The gates beta are
themselves learned by (leading-order) gradient descent on the true loss:

    dL/dbeta_ji  ~=  - sigma'(beta_ji) * [ f'(net_i) * w_ji * s_j ] * delta_i_true
                     \_______________/   \________ c_ji _________/   \__________/
                       gate slope          edge's backward push      true error at i

i.e. OPEN the gate on edges whose backward contribution aligns with the true
error at the receiving node; CLOSE gates on edges that carry misleading credit.

This module is framework-light on purpose (jax + numpy only) so it can be
verified in isolation and dropped into the lab's jax-experiment scaffolding.

Conventions
-----------
- A layer maps a_in [B, in] -> net [B, out] -> a_out = f(net) [B, out].
  Weight W has shape [out, in]; forward is  net = a_in @ W.T.
- Network = a TRUNK (list of weight layers, all with activation f) followed by
  K HEADS (each a single linear layer trunk_top -> logits, no activation).
- Every weight matrix (trunk + heads) gets its own gate matrix of the same shape.
- We compute, in one backward sweep each:
    * the MODULATED signals s (gates applied)  -> used for base weight updates
    * the TRUE signals delta_true (gates = 1)   -> used for the honest meta-update
- All grads are minibatch-averaged.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    trunk_W: List[jnp.ndarray]   # each [out, in]
    head_W: List[jnp.ndarray]    # each [n_classes, trunk_top_dim]
    trunk_beta: List[jnp.ndarray]  # gates, same shapes as trunk_W
    head_beta: List[jnp.ndarray]   # gates, same shapes as head_W


def init_params(key, in_dim, hidden_dims, n_classes, n_heads, beta_init):
    """Glorot-ish init for weights; constant beta_init for gates (sigmoid space)."""
    keys = jax.random.split(key, 32)
    ki = 0
    trunk_W, trunk_beta = [], []
    prev = in_dim
    for h in hidden_dims:
        scale = jnp.sqrt(2.0 / prev)
        trunk_W.append(jax.random.normal(keys[ki], (h, prev)) * scale); ki += 1
        trunk_beta.append(jnp.full((h, prev), beta_init))
        prev = h
    head_W, head_beta = [], []
    for _ in range(n_heads):
        scale = jnp.sqrt(2.0 / prev)
        head_W.append(jax.random.normal(keys[ki], (n_classes, prev)) * scale); ki += 1
        head_beta.append(jnp.full((n_classes, prev), beta_init))
    return Params(trunk_W, head_W, trunk_beta, head_beta)


# --------------------------------------------------------------------------- #
#  Activations
# --------------------------------------------------------------------------- #
def act(name, leaky_slope: float = 0.01):
    if name == "tanh":
        return jnp.tanh, lambda net: 1.0 - jnp.tanh(net) ** 2
    if name == "relu":
        return (lambda net: jnp.maximum(net, 0.0)), (lambda net: (net > 0).astype(net.dtype))
    if name == "leaky_relu":
        f = lambda net: jnp.where(net > 0, net, leaky_slope * net)
        fp = lambda net: jnp.where(net > 0, jnp.ones_like(net), leaky_slope * jnp.ones_like(net))
        return f, fp
    raise ValueError(name)


# --------------------------------------------------------------------------- #
#  Forward (stores everything backward needs)
# --------------------------------------------------------------------------- #
def forward(params: Params, x, f):
    """Returns logits per head and a cache of (a_in, net) per trunk layer + trunk_top."""
    a = x
    cache = []  # list of (a_in, net) for each trunk layer
    for W in params.trunk_W:
        net = a @ W.T
        cache.append((a, net))
        a = f(net)
    trunk_top = a
    logits = [trunk_top @ Wh.T for Wh in params.head_W]
    return logits, cache, trunk_top


def softmax_ce_seed(logits, y_onehot, task_w):
    """Per-head seed delta at the logits = task_w * (softmax - onehot) / B.
    Returns list of seeds and the (task-weighted) scalar loss."""
    seeds, loss = [], 0.0
    B = logits[0].shape[0]
    for lg, yo, w in zip(logits, y_onehot, task_w):
        p = jax.nn.softmax(lg, axis=-1)
        seeds.append(w * (p - yo) / B)
        ce = -jnp.sum(yo * jax.nn.log_softmax(lg, axis=-1), axis=-1)   # [B]
        loss = loss + w * jnp.mean(ce)
    return seeds, loss


def sigmoid_bce_seed(logits, y_binary, task_w):
    """Per-head sigmoid + per-output BCE seed = task_w * (sigmoid(logits) - y) / B.

    Each output unit is an independent binary classifier (no softmax coupling
    across outputs). Returns list of seeds (one per head) and the scalar loss.

    y_binary[k]: [B, C_k] binary targets (e.g. one-hot per task collapsed into a
    single head's output, or any {0,1}-valued tensor)."""
    seeds, loss = [], 0.0
    B = logits[0].shape[0]
    for lg, y, w in zip(logits, y_binary, task_w):
        p = jax.nn.sigmoid(lg)
        seeds.append(w * (p - y) / B)
        # numerically stable BCE: log(1+exp(-|z|)) + max(z,0) - z*y
        bce = jnp.maximum(lg, 0.0) - lg * y + jnp.log1p(jnp.exp(-jnp.abs(lg)))
        loss = loss + w * jnp.mean(jnp.sum(bce, axis=-1))
    return seeds, loss


# --------------------------------------------------------------------------- #
#  Backward sweep (one function, gates optional)
# --------------------------------------------------------------------------- #
def backward(params, cache, trunk_top, seeds, fprime, *, gated):
    """One backward sweep.

    If gated=True, backward matmuls use (sigmoid(beta) ⊙ W)  -> modulated signals s.
    If gated=False, they use W                              -> true signals delta_true.

    Returns:
      gW_trunk, gW_head : minibatch-averaged weight gradients (what the base learner uses)
      s_net_trunk       : signal at each trunk layer's *pre-activation* net  (list, low->high)
      s_in_trunk        : signal projected down to each trunk layer's *input* a_in (list)
                          (s_in_trunk[0] is the signal at the network input x)
      s_logits          : the seed signals (unchanged)
    """
    def gate(W, beta):
        return jax.nn.sigmoid(beta) * W if gated else W

    # ---- heads: trunk_top is the shared activation receiving all heads ----
    gW_head = []
    da_top = jnp.zeros_like(trunk_top)         # signal arriving at trunk_top activation
    for Wh, bh, seed in zip(params.head_W, params.head_beta, seeds):
        gW_head.append(seed.T @ trunk_top)     # [C, top] grad for this head
        da_top = da_top + seed @ gate(Wh, bh)  # project logit-error down to trunk_top
    # ---- trunk: walk top -> bottom ----
    gW_trunk = [None] * len(params.trunk_W)
    s_net_trunk = [None] * len(params.trunk_W)
    s_in_trunk = [None] * len(params.trunk_W)
    da = da_top
    for l in reversed(range(len(params.trunk_W))):
        a_in, net = cache[l]
        s_net = da * fprime(net)               # signal at this layer's pre-activation
        s_net_trunk[l] = s_net
        gW_trunk[l] = s_net.T @ a_in           # [out, in] grad for this layer
        W, beta = params.trunk_W[l], params.trunk_beta[l]
        da = s_net @ gate(W, beta)             # project down to this layer's input
        s_in_trunk[l] = da                     # signal at a_in of layer l (= x when l==0)
    return gW_trunk, gW_head, s_net_trunk, s_in_trunk, seeds


# --------------------------------------------------------------------------- #
#  Meta-gradient for the gates (leading-order; honest or proxy)
# --------------------------------------------------------------------------- #
def meta_grad_layer(beta, W, s_out, fprime_in, delta_in_true_postfp, norm_input):
    """dL/dbeta for one gate matrix (closed form, leading order).

        c_ji         = fprime_in_i * W_ji * s_out_j            (this edge's backward push)
        dL/dbeta_ji  = - sigma'(beta_ji) * E_b[ c_ji * delta_in_true_i ] * ||norm_input_i||^2

    where delta_in_true is the POST-f' true error at the input node i (standard delta),
    and ||norm_input_i||^2 = E_b[ a_i^2 ] is the input energy of the weights that node i's
    signal actually updates (the IDBD x^2 normalizer). Returns dL/dbeta, shape == beta.

    s_out                 : [B, out]  modulated signal s_j at the gate's OUTPUT node
    fprime_in             : [B, in]   f'(net_i) at the gate's INPUT node
    delta_in_true_postfp  : [B, in]   TRUE (or proxy) delta at net_i (post-f')
    norm_input            : [B, in]   activations feeding node i's incoming weights
    """
    sigma = jax.nn.sigmoid(beta)
    sigma_prime = sigma * (1.0 - sigma)                       # [out, in]
    u = fprime_in * delta_in_true_postfp                      # [B, in]  (= f'(net_i)*delta_i)
    B = s_out.shape[0]
    corr = (s_out.T @ u) / B                                  # [out, in]  E_b[s_j u_i]
    # ||a^{(l-1)}||^2 is summed over the incoming-weight inputs -> a per-sample SCALAR
    # (the leading-order coefficient; only rescales per-layer magnitude, not direction).
    energy = jnp.mean(jnp.sum(norm_input ** 2, axis=1))       # scalar
    return -(sigma_prime * W * corr) * energy


def meta_grad(params, cache, trunk_top, s_pack, true_pack, fprime, *, use_proxy):
    """Assemble dL/dbeta for every (meaningful) gate matrix.

    s_pack    : modulated backward outputs (gates applied) -> provides s_j at outputs.
    true_pack : true backward outputs (gates = 1) -> provides post-f' true deltas.
                If use_proxy, the modulated signals are reused in place of the true ones.

    Note: the input-layer gate (trunk layer 0) is INERT -- it modulates credit flowing
    to the network inputs, which update no weights -- so its meta-grad is identically 0.
    """
    (_, _, s_net_t, s_in_t, s_logits) = s_pack
    (_, _, t_net_t, t_in_t, _) = s_pack if use_proxy else true_pack
    T = len(params.trunk_W)

    # ----- trunk gates -----
    # Gate on layer l modulates flow from net[l] (node j) down to a^{(l)} (node i),
    # whose pre-activation is net[l-1]; that signal updates W[l-1] (input a^{(l-1)}).
    dbeta_trunk = []
    for l in range(T):
        if l == 0:
            dbeta_trunk.append(jnp.zeros_like(params.trunk_beta[0]))   # inert
            continue
        a_prev, net_prev = cache[l - 1]            # node i's incoming-weight input & pre-act
        dbeta_trunk.append(
            meta_grad_layer(
                params.trunk_beta[l], params.trunk_W[l],
                s_out=s_net_t[l],                  # s_j at layer-l output
                fprime_in=fprime(net_prev),        # f'(net_i)
                delta_in_true_postfp=t_net_t[l - 1],  # true delta at net_i
                norm_input=a_prev,                 # a^{(l-1)} feeds W[l-1]
            )
        )

    # ----- head gates -----
    # Head-k gate modulates flow from logits (node j) down to trunk_top (node i), whose
    # pre-activation is net[T-1]; that signal updates W[T-1] (input a^{(T-1)}).
    a_prev_top, net_prev_top = cache[T - 1]
    delta_top_postfp = t_net_t[T - 1]              # TOTAL true delta at net[T-1]
    dbeta_head = []
    for k in range(len(params.head_W)):
        dbeta_head.append(
            meta_grad_layer(
                params.head_beta[k], params.head_W[k],
                s_out=s_logits[k],                 # s_j at this head's logits
                fprime_in=fprime(net_prev_top),    # f'(net_i)
                delta_in_true_postfp=delta_top_postfp,
                norm_input=a_prev_top,             # a^{(T-1)} feeds W[T-1]
            )
        )
    return dbeta_trunk, dbeta_head


# --------------------------------------------------------------------------- #
#  One full step (base update + meta update), variant-aware
# --------------------------------------------------------------------------- #
@dataclass
class MetaState:
    e_trunk: List[jnp.ndarray]   # eligibility traces (EMA of meta-grad); used if gamma>0
    e_head: List[jnp.ndarray]


def zeros_like_gates(params):
    return MetaState([jnp.zeros_like(b) for b in params.trunk_beta],
                     [jnp.zeros_like(b) for b in params.head_beta])


def step(params, meta_state, x, y_onehot, task_w, *,
         f, fprime, alpha, theta, gamma, learn_gates, use_proxy,
         loss_kind="softmax_ce"):
    """Single training step.

    learn_gates : if False, gates are frozen (baseline / fixed-gate ablation).
    use_proxy   : if True, meta-update uses s in place of delta_true (no 2nd backward).
    gamma       : EMA coefficient for the eligibility trace on the meta-grad (0 => none).
    loss_kind   : "softmax_ce" or "sigmoid_bce".
    """
    logits, cache, trunk_top = forward(params, x, f)
    if loss_kind == "softmax_ce":
        seeds, loss = softmax_ce_seed(logits, y_onehot, task_w)
    elif loss_kind == "sigmoid_bce":
        seeds, loss = sigmoid_bce_seed(logits, y_onehot, task_w)
    else:
        raise ValueError(loss_kind)

    # modulated backward -> base updates
    s_pack = backward(params, cache, trunk_top, seeds, fprime, gated=True)
    gW_t_s, gW_h_s = s_pack[0], s_pack[1]

    new_trunk_W = [W - alpha * g for W, g in zip(params.trunk_W, gW_t_s)]
    new_head_W = [W - alpha * g for W, g in zip(params.head_W, gW_h_s)]

    new_trunk_beta = params.trunk_beta
    new_head_beta = params.head_beta
    new_meta = meta_state

    if learn_gates:
        if use_proxy:
            true_pack = s_pack
        else:
            true_pack = backward(params, cache, trunk_top, seeds, fprime, gated=False)
        dbeta_t, dbeta_h = meta_grad(params, cache, trunk_top, s_pack, true_pack,
                                     fprime, use_proxy=use_proxy)
        if gamma > 0.0:
            e_t = [gamma * e + (1 - gamma) * g for e, g in zip(meta_state.e_trunk, dbeta_t)]
            e_h = [gamma * e + (1 - gamma) * g for e, g in zip(meta_state.e_head, dbeta_h)]
            new_meta = MetaState(e_t, e_h)
            step_t, step_h = e_t, e_h
        else:
            step_t, step_h = dbeta_t, dbeta_h
        # gradient DESCENT on loss: beta <- beta - theta * dL/dbeta
        new_trunk_beta = [b - theta * g for b, g in zip(params.trunk_beta, step_t)]
        new_head_beta = [b - theta * g for b, g in zip(params.head_beta, step_h)]

    new_params = Params(new_trunk_W, new_head_W, new_trunk_beta, new_head_beta)
    return new_params, new_meta, loss


# --------------------------------------------------------------------------- #
#  Pytree registration (so Params / MetaState work with jax.jit and lax.scan)
# --------------------------------------------------------------------------- #
from jax import tree_util as _tu

_tu.register_pytree_node(
    Params,
    lambda p: ((p.trunk_W, p.head_W, p.trunk_beta, p.head_beta), None),
    lambda aux, ch: Params(*ch),
)
_tu.register_pytree_node(
    MetaState,
    lambda m: ((m.e_trunk, m.e_head), None),
    lambda aux, ch: MetaState(*ch),
)
