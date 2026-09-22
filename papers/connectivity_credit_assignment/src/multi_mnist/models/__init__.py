"""Models.

``PaddedMLP``       masked dense; what every paper figure uses.
``BlockSparseMLP``  K independent sub-MLPs; fast for small block-sparse runs.
``DynamicNetwork``  preallocated sparse storage; what SET and DEEP-R need to
                    stay cheap as the network grows.
"""

from .block_sparse_mlp import BlockSparseMLP, compute_hidden_dim_for_params
from .dynamic_network import (
    DynamicNetwork, build_outgoing_indices, count_active_connections,
    count_active_units, init_random_dynamic_network, sync_outgoing_weights,
)
from .sparse_init import (
    StructureModel, derive_sizes, init_sparse_model,
)
from .padded_mlp import (
    PaddedMLP, fill_masks_to_dense, init_model, model_filter_spec,
    structure_diagnostics,
)

__all__ = [
    'BlockSparseMLP', 'DynamicNetwork', 'PaddedMLP',
    'build_outgoing_indices', 'compute_hidden_dim_for_params',
    'count_active_connections', 'count_active_units', 'fill_masks_to_dense',
    'init_model', 'init_random_dynamic_network', 'model_filter_spec',
    'structure_diagnostics', 'sync_outgoing_weights',
    'StructureModel', 'derive_sizes', 'init_sparse_model',
]
