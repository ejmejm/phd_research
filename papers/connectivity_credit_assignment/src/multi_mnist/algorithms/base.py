"""The connectivity-algorithm interface.

A connectivity algorithm decides how a network's connections change during
training. The paper's two baselines never change them; SET and DEEP-R prune
and regrow on a schedule. Everything they need is expressed here so that
adding a method means writing one file rather than a second training script.

An algorithm object is *static*: it holds configuration (periods, rates) and
is captured in the training loop's closure, so it must not hold JAX arrays.
Mutable per-seed state lives in ``TrainState.algo`` as a pytree, which is what
gets vmapped over seeds and threaded through the scan.

There are three places structure can change, and they are not
interchangeable:

``event``
    Runs *inside* the jitted, vmapped scan, every ``event_period`` steps. It
    is traced, so it must be shape-stable -- no Python branching on array
    values, no changing array sizes. This is where a prune-and-regrow rule
    belongs, because it fires thousands of times per run and must not pay a
    host round-trip.

``step_update``
    Runs *inside* the scan on every step, and unlike ``step_hook`` it may
    modify the model. It exists for methods whose weight dynamics and whose
    structural bookkeeping run at different rates: DEEP-R applies its L1 pull
    and Langevin noise every step, but rebuilding the sparse index tables is
    expensive and is deferred to ``event``.

``on_log_period``
    Runs on the host between log periods, with concrete values. Use it for
    one-shot interventions where a host round-trip is irrelevant and a Python
    flag is clearer than a traced counter -- the dense transition of Figure 3
    is the motivating case.

An algorithm may use either, both, or neither.
"""

from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray


class ConnectivityAlgorithm:
    """Base class; the default behaviour is to never change connectivity."""

    #: Steps between in-scan structure events. 0 disables ``event`` entirely,
    #: which lets the trainer scan a whole log period in one go.
    event_period: int = 0

    #: Human-readable name, used in logs and run summaries.
    name: str = 'static'

    def init_state(self, model: eqx.Module, *, key: PRNGKeyArray) -> Any:
        """Build the algorithm's per-seed state. ``None`` means stateless."""
        return None

    def step_hook(
        self, algo_state: Any, model_pre: eqx.Module, model_post: eqx.Module, aux: Any,
    ) -> Any:
        """Update per-step statistics. Runs inside the scan on every step.

        Both the pre-update and post-update model are passed because the
        choice matters: a contribution-utility trace pairs ``|weight|`` with
        the activations that weight produced, which means the *pre-update*
        weights.

        ``aux`` is ``(images, hidden)`` -- whatever the forward pass already
        computed and would otherwise discard, so tracking utility costs no
        extra forward pass. The exact meaning of ``hidden`` is the model's:
        ``PaddedMLP`` gives the hidden activations, ``DynamicNetwork`` gives
        its whole value buffer.
        """
        return algo_state

    #: True when ``step_update`` needs a fresh PRNG key on every step. The
    #: trainer only splits ``TrainState.rng`` per step when some algorithm
    #: asks for it, so algorithms that do not pay nothing and keep their
    #: random stream identical to before this hook existed.
    needs_step_key: bool = False

    def step_update(
        self, model: eqx.Module, algo_state: Any, *, key: Optional[PRNGKeyArray],
    ) -> Tuple[eqx.Module, Any]:
        """Modify the model every step. Traced; must be shape-stable.

        Called after the optimizer step and before the model reconciles its
        derived state, so a model like ``DynamicNetwork`` mirrors whatever
        this leaves behind.

        Use it only for work that is cheap enough to run on every step. A
        prune-and-regrow that rebuilds index tables belongs in ``event``.

        ``key`` is ``None`` unless the algorithm sets ``needs_step_key``.
        """
        return model, algo_state

    def event(
        self,
        model: eqx.Module,
        optimizer: Any,
        algo_state: Any,
        *,
        key: PRNGKeyArray,
    ) -> Tuple[eqx.Module, Any, Any, Dict[str, jax.Array]]:
        """Apply one structure update. Traced; must be shape-stable.

        Returns the new model, optimizer, algorithm state, and a dict of
        scalar JAX arrays to accumulate (e.g. counts of pruned connections).
        """
        return model, optimizer, algo_state, {}

    def on_log_period(
        self,
        model: eqx.Module,
        optimizer: Any,
        algo_state: Any,
        step: int,
    ) -> Tuple[eqx.Module, Any, Any, Dict[str, float]]:
        """Host-side hook, called after each log period with a concrete step."""
        return model, optimizer, algo_state, {}

    def diagnostics(self, model: eqx.Module, n_tasks: int) -> Dict[str, float]:
        """Structure statistics to log. Called on the host.

        Return ``{}`` when the structure is fixed for the whole run -- the
        trainer then computes these once instead of syncing the GPU every log
        period.
        """
        return {}

    @property
    def structure_is_static(self) -> bool:
        """True when connectivity cannot change after initialization."""
        return self.event_period == 0
