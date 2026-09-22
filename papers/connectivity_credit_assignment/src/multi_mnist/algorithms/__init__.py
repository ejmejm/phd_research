"""Connectivity algorithms, and the registry that builds one from config.

Adding a method means writing one module here and adding a line to
``build_algorithm``. See ``base.ConnectivityAlgorithm`` for the interface and
``set.SET`` for a worked example.

SET and DEEP-R each have two implementations, picked by ``model.type``:

``padded_mlp``
    ``set.py`` / ``deep_r.py``. Connectivity is a mask over a dense matrix, so
    the connection budget is preserved per *layer* as both papers specify, and
    ``path_purity`` is available. This is what the experiments run on.

``dynamic_network``
    ``dynamic/``. Preallocated per-unit sparse storage, for networks large
    enough that the dense matmul is wasteful. The budget is preserved per unit
    instead, which is a documented deviation.
"""

from typing import Any, Dict

from omegaconf import DictConfig

from .base import ConnectivityAlgorithm
from .deep_r import DeepR
from .dense_transition import DenseTransition
from .set import SET
from .static import StaticConnectivity


def build_algorithm(cfg: DictConfig) -> ConnectivityAlgorithm:
    """Construct the connectivity algorithm named by ``algorithm.name``."""
    name = str(cfg.algorithm.name)
    model_type = str(cfg.model.get('type', 'padded_mlp'))
    on_dynamic = model_type == 'dynamic_network'

    if name == 'static':
        return StaticConnectivity()

    if name == 'dense_transition':
        step = cfg.algorithm.get('transition_step', None)
        # A sweep backend that cannot express null in a numeric grid passes
        # the string "None"; a step beyond total_steps is the other way to
        # say "never", and both must mean the block-sparse baseline.
        if isinstance(step, str):
            step = None if step.strip().lower() in ('none', 'null', '') else int(step)
        if step is not None and int(step) > int(cfg.train.total_steps):
            step = None
        return DenseTransition(
            transition_step=None if step is None else int(step),
            initial_hidden_units=int(cfg.model.initial_hidden_units),
        )

    if name == 'set':
        kwargs = dict(
            evolve_frequency=int(cfg.algorithm.evolve_frequency),
            zeta=float(cfg.algorithm.zeta),
            prune_metric=str(cfg.algorithm.get('prune_metric', 'magnitude')),
            utility_decay=float(cfg.algorithm.get('utility_decay', 0.999)),
        )
        if on_dynamic:
            from .dynamic import SETDynamic
            return SETDynamic(**kwargs)
        return SET(**kwargs)

    if name == 'deep_r':
        kwargs = dict(
            learning_rate=float(cfg.optimizer.learning_rate),
            l1=float(cfg.algorithm.get('l1', 1e-3)),
            temperature=float(cfg.algorithm.get('temperature', 1e-5)),
            evolve_w2=bool(cfg.algorithm.get('evolve_w2', True)),
            event_period=int(cfg.algorithm.get('event_period', 25)),
        )
        if on_dynamic:
            from .dynamic import DeepRDynamic
            return DeepRDynamic(**kwargs)
        return DeepR(**kwargs)

    raise ValueError(
        f'Unknown algorithm.name: {name!r}. '
        f'Known: static, dense_transition, set, deep_r.'
    )


__all__ = [
    'ConnectivityAlgorithm', 'DeepR', 'DenseTransition', 'SET',
    'StaticConnectivity', 'build_algorithm',
]
