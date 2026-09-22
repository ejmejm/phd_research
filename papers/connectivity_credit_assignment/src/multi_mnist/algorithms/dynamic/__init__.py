"""SET and DEEP-R on ``DynamicNetwork``, the preallocated sparse representation.

These are the implementations to use when the network is large enough that a
masked dense matmul is wasteful. They are not what the 16-task experiments run
on -- see ``algorithms/set.py`` and ``algorithms/deep_r.py`` for the
``PaddedMLP`` versions, which preserve the connection budget per layer as both
papers specify and expose ``path_purity``.

Kept in sync deliberately: if you change the algorithm in one place, change it
in the other, or say in the config which one produced a result.
"""

from .deep_r import DeepR as DeepRDynamic
from .set import SET as SETDynamic

__all__ = ['DeepRDynamic', 'SETDynamic']
