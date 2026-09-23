"""Dense transition: train block-sparse, then make the network dense.

This is the intervention behind Figure 3. A block-sparse network is trained
until performance stabilizes, and then the connection masks are flipped to
all-ones so the cross-task weights become trainable. Those weights are
already exactly 0, so the flip does not change the function the network
computes -- it changes only which weights gradients can reach.

The point is to separate two explanations for the dense/block-sparse gap. If
dense networks merely fail to *find* the block-sparse solution, then handing
one the solution and letting it continue should be harmless. It is not: the
network degrades after the flip, which is what implicates credit assignment
rather than optimization luck.

``forward_only_distractors`` splits that degradation in two. A new cross-task
connection hurts in two ways at once: forward, it feeds another task's signal
into a unit or an output; backward, a new W2 connection returns another task's
error to a hidden unit, so that unit's incoming weights are now trained on a
loss that is not its own. With the flag set, the connections added by the flip
still feed forward and still learn their own weights, but send no error back
into the hidden layer. Whatever degradation remains is the forward channel;
the gap to the ordinary transition is the backward one.

Only W2 needs the treatment. A new W1 connection's backward path ends at an
input, which is not a parameter, so W1 distractors are already forward-only.

Implemented as a host-side one-shot rather than an in-scan event: it fires
once per run, so a Python flag is clearer than a traced step counter, and the
host round-trip costs nothing at this frequency.
"""

from typing import Any, Dict, Optional, Tuple

import equinox as eqx

from ..models.padded_mlp import fill_masks_to_dense, freeze_gradient_mask
from .base import ConnectivityAlgorithm


class DenseTransition(ConnectivityAlgorithm):

    name = 'dense_transition'
    event_period = 0

    def __init__(
        self,
        transition_step: Optional[int],
        initial_hidden_units: int,
        forward_only_distractors: bool = False,
    ):
        """
        Args:
            transition_step: Step at or after which the masks are filled.
                ``None`` never fills, which reproduces the block-sparse
                baseline through exactly the same code path.
            initial_hidden_units: Width of the active region to fill. Padded
                unit slots beyond this stay masked out.
            forward_only_distractors: Bar the connections the flip adds from
                carrying gradient back into the hidden layer, leaving them
                active in the forward pass. Isolates the forward half of the
                degradation.
        """
        self.transition_step = transition_step
        self.initial_hidden_units = int(initial_hidden_units)
        self.forward_only_distractors = bool(forward_only_distractors)
        self.applied = False

    def prepare_model(self, model: eqx.Module) -> eqx.Module:
        # Pinned at setup, not at the flip: the extra mask is a pytree leaf,
        # and adding one mid-run would invalidate the optimizer's filter spec.
        if not self.forward_only_distractors:
            return model
        return freeze_gradient_mask(model)

    def on_log_period(
        self, model: eqx.Module, optimizer: Any, algo_state: Any, step: int,
    ) -> Tuple[eqx.Module, Any, Any, Dict[str, float]]:
        if self.transition_step is None or self.applied or step < self.transition_step:
            return model, optimizer, algo_state, {}

        model = fill_masks_to_dense(model, self.initial_hidden_units)
        self.applied = True
        mode = 'forward-only' if self.forward_only_distractors else 'trainable'
        print(f'[dense_transition] filled masks to dense at step {step} '
              f'(target {self.transition_step}, new connections {mode})')
        return model, optimizer, algo_state, {'dense_transition_applied_at_step': float(step)}

    @property
    def structure_is_static(self) -> bool:
        # With no transition step this is exactly the block-sparse baseline,
        # and the trainer can cache the structure diagnostics for the run.
        return self.transition_step is None
