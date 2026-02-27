"""Shared utilities for scaling_rl experiments."""
from typing import Any, List

import jax
import jax.numpy as jnp
from omegaconf import DictConfig


def configure_jax(cfg: DictConfig) -> None:
    """Configure JAX compilation cache and device."""
    jax.config.update('jax_compilation_cache_dir', cfg.jax_jit_cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_platform_name', cfg.device)
    print(f'JAX is using device: {jax.devices(cfg.device)[0]}')


def stack_pytrees(pytrees: List[Any]) -> Any:
    """Stack a list of identically-structured pytrees along a new leading axis."""
    treedef = jax.tree.structure(pytrees[0])
    all_leaves = [jax.tree.leaves(p) for p in pytrees]
    new_pytrees = [jax.tree.unflatten(treedef, leaves) for leaves in all_leaves]
    return jax.tree.map(lambda *args: jnp.stack(args), *new_pytrees)
