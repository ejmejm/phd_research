from functools import partial
from typing import Any, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import DictKey, GetAttrKey, KeyPath, SequenceKey
from jaxtyping import Array
from omegaconf import DictConfig


def configure_jax(cfg: DictConfig):
    """Configure JAX compilation cache and device."""
    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update(
        'jax_persistent_cache_enable_xla_caches',
        'xla_gpu_per_fusion_autotune_cache_dir',
    )
    jax.config.update('jax_platform_name', cfg.device)
    print(f'JAX device: {jax.devices(cfg.device)[0]}')


def count_params(model) -> int:
    """Count total trainable parameters in a model."""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree.leaves(params))


def stack_pytrees(pytrees: List[Any]) -> Any:
    """Stack a list of identically-structured pytrees along a new leading axis."""
    treedef = jax.tree.structure(pytrees[0])
    all_leaves = [jax.tree.leaves(pt) for pt in pytrees]
    stacked = [jnp.stack(xs) for xs in zip(*all_leaves)]
    return jax.tree.unflatten(treedef, stacked)


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces the values of a tree with the provided keyword arguments."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def is_array_sequence(x: Any) -> bool:
    """Checks if `x` is a sequence of JAX arrays."""
    return isinstance(x, Sequence) and len(x) > 0 and isinstance(x[0], Array)


@partial(jax.jit, static_argnames=('n',))
def tree_unzip(tree: eqx.Module, n: int) -> Tuple[eqx.Module]:
    "Unzips a pytree of tuples into a tuple of `n` pytrees."
    pick_i = lambda i: jax.tree.map(lambda xs: xs[i], tree, is_leaf=is_array_sequence)
    return tuple(pick_i(i) for i in range(n))


def tree_auto_unzip(tree: eqx.Module) -> Tuple[eqx.Module]:
    "Unzips a pytree of tuples into a tuple of pytrees."
    # Get number of items per inner sequence
    lengths = jax.tree.leaves(
        jax.tree.map(lambda xs: len(xs), tree, is_leaf=is_array_sequence)
    )
    
    if not lengths:
        return []  # no sequence leaves
    n = lengths[0]
    if any(L != n for L in lengths):
        raise ValueError(f"Mismatched leaf lengths: {lengths}!")

    return tree_unzip(tree, n)


def get_val_at_key_path(tree: eqx.Module, key_path: KeyPath) -> Any:
    for key in key_path:
        if isinstance(key, DictKey):
            tree = tree[key.key]
        elif isinstance(key, GetAttrKey):
            tree = getattr(tree, key.name)
        elif isinstance(key, SequenceKey):
            tree = tree[key.idx]
        else:
            raise ValueError(f"Invalid key: {key}")
    return tree