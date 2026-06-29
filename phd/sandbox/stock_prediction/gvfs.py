"""General Value Functions (GVFs) for the sequential stock predictor.

A GVF defines *what to predict* as a (cumulant, gamma) pair, following the standard
predictive-knowledge framing (Sutton et al., 2011): the value of a state estimates the
discounted sum of future cumulants,

    v(s_t) ~= E[ c_{t+1} + gamma * c_{t+2} + gamma^2 * c_{t+3} + ... ].

With ``gamma = 0`` this collapses to a one-step prediction of the next cumulant -- e.g. a
``close``/``gamma=0`` GVF predicts next step's close, reproducing the original predictor.

Each GVF is *per stock*: its ``cumulant`` reads a field of the agent state (one value per
stock) and returns a ``(num_stocks,)`` vector, so a single GVF spec instantiates one
prediction per stock. Listing several GVFs in the config stacks their cumulants, giving
``n_gvfs * num_stocks`` prediction targets in total.

The agent state is a ``{field: (num_stocks,) array}`` dict holding the latest observation;
GVF cumulants are defined in terms of it. To add a non-field cumulant later, subclass
``GVF`` and override :meth:`cumulant` (the rest of the pipeline is agnostic to how the
cumulant is computed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

# Agent state passed to cumulants: {field_name: (num_stocks,) array}.
State = Mapping[str, jax.Array]


@dataclass(frozen=True)
class GVF:
    """A per-stock GVF whose cumulant is a single observation field.

    Attributes:
        name: identifier used to label logged metrics (e.g. ``"close_g0"``).
        gamma: continuation/discount factor in ``[0, 1)``. ``0`` => one-step prediction.
        field: agent-state field read as the cumulant.
    """

    name: str
    gamma: float
    field: str

    def cumulant(self, state: State) -> jax.Array:
        """Per-stock cumulant for this step, shape ``(num_stocks,)``."""
        return state[self.field]


def build_gvfs(specs: Sequence[Mapping]) -> list[GVF]:
    """Instantiate GVFs from a config list, e.g. ``[{field: close, gamma: 0.0}, ...]``.

    Each spec needs a ``field``; ``gamma`` defaults to 0 (one-step) and ``name`` defaults
    to ``"<field>_g<gamma>"``.
    """
    gvfs: list[GVF] = []
    for spec in specs:
        field = spec["field"]
        gamma = float(spec.get("gamma", 0.0))
        name = spec.get("name") or f"{field}_g{gamma:g}"
        gvfs.append(GVF(name=name, gamma=gamma, field=field))
    if not gvfs:
        raise ValueError("No GVFs specified (config 'gvfs' is empty).")
    return gvfs


def required_fields(gvfs: Sequence[GVF], extra: Sequence[str] = ("close",)) -> list[str]:
    """Observation fields needed to run: every cumulant field plus the input features
    (``close`` by default). Returned sorted so the data load order is deterministic."""
    fields = set(extra)
    for g in gvfs:
        fields.add(g.field)
    return sorted(fields)


def cumulant_vector(gvfs: Sequence[GVF], state: State) -> jax.Array:
    """Stack every GVF's per-stock cumulant into one ``(n_gvfs * num_stocks,)`` vector.

    Layout is GVF-major: ``[gvf0 over all stocks, gvf1 over all stocks, ...]``, matching
    :func:`gamma_vector` and the model's output ordering.
    """
    return jnp.concatenate([g.cumulant(state) for g in gvfs])


def gamma_vector(gvfs: Sequence[GVF], num_stocks: int) -> jax.Array:
    """Per-target discount factors, shape ``(n_gvfs * num_stocks,)``, GVF-major to match
    :func:`cumulant_vector`."""
    return jnp.concatenate([jnp.full(num_stocks, g.gamma) for g in gvfs])
