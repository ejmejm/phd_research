from functools import partial
from typing import Any, Sequence, Tuple

import equinox as eqx
import jax
from jaxtyping import Array


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
