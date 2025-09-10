from typing import Tuple

import jax
import equinox as eqx


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces the values of a tree with the provided keyword arguments."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def tree_unzip(tree: eqx.Module, n: int) -> Tuple[eqx.Module]:
    "Unzips a pytree of tuples into a tuple of pytrees."
    return jax.tree.transpose(
        jax.tree.structure(0),
        jax.tree.structure((0,) * n),
        tree,
    )