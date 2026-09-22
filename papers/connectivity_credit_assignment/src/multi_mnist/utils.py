"""JAX configuration, pytree helpers, seeding, and OmegaConf resolvers."""

from ctypes import c_int32
from functools import partial
import hashlib
import random
from typing import Any, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import omegaconf
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig


DEFAULT_JIT_CACHE_DIR = '/tmp/jax_cache'


# ---------------------------------------------------------------------------
# OmegaConf resolvers
#
# `eval` evaluates a Python expression, so a config can say
# `${eval:2**-10}` or `${eval:2000 // ${task.n_tasks}}`.
#
# `switch` looks a value up in a flat (key, value, key, value, ...) list,
# which lets one sweep config carry a different best step-size per budget:
#     ${switch:${model.initial_hidden_units}, 6, -11, 12, -10, ...}
# Keys are matched as strings, so list scalars unquoted.
# ---------------------------------------------------------------------------

def register_resolvers() -> None:
    if not omegaconf.OmegaConf.has_resolver('eval'):
        omegaconf.OmegaConf.register_new_resolver('eval', lambda x: eval(str(x)))
    if not omegaconf.OmegaConf.has_resolver('as_tuple'):
        omegaconf.OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))
    if not omegaconf.OmegaConf.has_resolver('switch'):
        omegaconf.OmegaConf.register_new_resolver(
            'switch', lambda key, *pairs: dict(zip(pairs[::2], pairs[1::2]))[key]
        )


register_resolvers()


# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------

def configure_jax(cfg: DictConfig):
    """Configure the JAX compilation cache and device."""
    cache_dir = cfg.get('jax_jit_cache_dir', DEFAULT_JIT_CACHE_DIR)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update(
        'jax_persistent_cache_enable_xla_caches',
        'xla_gpu_per_fusion_autotune_cache_dir',
    )

    if cfg.get('device') is not None:
        jax.config.update('jax_platform_name', cfg.device)
        print(f'JAX device: {jax.devices(cfg.device)[0]}')
    else:
        print(f'JAX device not specified, using default: {jax.devices()[0]}')


# ---------------------------------------------------------------------------
# Pytree helpers
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    """Count trainable parameters, including masked-out slots."""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree.leaves(params))


def stack_pytrees(pytrees: List[Any]) -> Any:
    """Stack identically-structured pytrees along a new leading axis."""
    treedef = jax.tree.structure(pytrees[0])
    all_leaves = [jax.tree.leaves(pt) for pt in pytrees]
    stacked = [jnp.stack(xs) for xs in zip(*all_leaves)]
    return jax.tree.unflatten(treedef, stacked)


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Return a copy of ``tree`` with the named fields replaced."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def is_array_sequence(x: Any) -> bool:
    return isinstance(x, Sequence) and len(x) > 0 and isinstance(x[0], Array)


@partial(jax.jit, static_argnames=('n',))
def tree_unzip(tree: eqx.Module, n: int) -> Tuple[eqx.Module, ...]:
    """Unzip a pytree of n-tuples into a tuple of n pytrees."""
    pick_i = lambda i: jax.tree.map(lambda xs: xs[i], tree, is_leaf=is_array_sequence)
    return tuple(pick_i(i) for i in range(n))


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: Optional[int]):
    """Seed the Python and NumPy global RNGs."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def seed_from_string(seed: Optional[int], string: str) -> Optional[int]:
    """Derive a stable sub-seed from a base seed and a label.

    Lets independent components (model init, data stream, ...) draw
    reproducible but uncorrelated seeds from one config-level seed.
    """
    if seed is None:
        return random.randint(0, 2**32)
    return seed + int(hashlib.md5(string.encode()).hexdigest(), 16) % (2**32)


def rng_from_string(rng: Optional[PRNGKeyArray], string: str) -> PRNGKeyArray:
    """Derive a JAX PRNG key from a base key and a label.

    The int32 wrap of the md5 digest is load-bearing: it is what the
    published runs folded in, so changing it would change every model
    initialization in the repository.
    """
    if rng is None:
        return jax.random.key(random.randint(0, 2**31))
    string_int = c_int32(int(hashlib.md5(string.encode()).hexdigest(), 16))
    return jax.random.fold_in(rng, string_int)
