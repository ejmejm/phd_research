"""Fixed connectivity: the dense and block-sparse baselines.

Connectivity is decided at initialization by ``model.init_strategy`` and never
changes, so there is nothing to do per step. This is the algorithm behind both
curves of Figure 1, every point of Figure 2, and both panels of Figure 4.
"""

from typing import Dict

import equinox as eqx

from .base import ConnectivityAlgorithm


class StaticConnectivity(ConnectivityAlgorithm):

    name = 'static'
    event_period = 0

    def diagnostics(self, model: eqx.Module, n_tasks: int) -> Dict[str, float]:
        # Empty tells the trainer the structure is fixed, so it computes the
        # diagnostics once before the loop instead of every log period.
        return {}
