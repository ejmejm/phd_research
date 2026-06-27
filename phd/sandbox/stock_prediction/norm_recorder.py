"""Optional recording + plotting of raw vs. normalized inputs/targets.

Kept separate from ``linear_predictor.py`` so the training script stays focused on training:
the only coupling is a ``NormalizationRecorder`` that the loop feeds one step at a time and
then asks to ``save`` a figure. Use it to sanity-check that the Welford normalizer is
centering/scaling each stock's series the way you expect.
"""

from __future__ import annotations

import numpy as np


class NormalizationRecorder:
    """Accumulates per-step raw vs. normalized values for every stock, then plots them.

    Feed each step's vectors via :meth:`record` (captured exactly as the model sees them, i.e.
    normalized with pre-update stats), then call :meth:`save` once training is done.
    """

    def __init__(self) -> None:
        self._x_raw: list[np.ndarray] = []
        self._x_norm: list[np.ndarray] = []
        self._y_raw: list[np.ndarray] = []
        self._y_norm: list[np.ndarray] = []

    def record(
        self,
        x_raw: np.ndarray,
        x_norm: np.ndarray,
        y_raw: np.ndarray,
        y_norm: np.ndarray,
    ) -> None:
        """Store one step's input/target vectors (each shape ``(num_stocks,)``)."""
        self._x_raw.append(np.asarray(x_raw, dtype=np.float64).copy())
        self._x_norm.append(np.asarray(x_norm, dtype=np.float64).copy())
        self._y_raw.append(np.asarray(y_raw, dtype=np.float64).copy())
        self._y_norm.append(np.asarray(y_norm, dtype=np.float64).copy())

    def save(self, out_path: str) -> None:
        """Render the recorded series to ``out_path`` (one row per stock)."""
        if not self._x_raw:
            print("NormalizationRecorder: nothing recorded, skipping plot.")
            return
        _plot_normalization(
            np.stack(self._x_raw),
            np.stack(self._x_norm),
            np.stack(self._y_raw),
            np.stack(self._y_norm),
            out_path,
        )


def _robust_ylim(values: np.ndarray, pad: float = 0.10) -> tuple[float, float] | None:
    """y-limits clipped to the 1st-99th percentile (with padding) so a few extreme
    outliers don't squash the rest of the series. Returns None to keep autoscale."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        return None
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _plot_normalization(
    x_raw: np.ndarray,
    x_norm: np.ndarray,
    y_raw: np.ndarray,
    y_norm: np.ndarray,
    out_path: str,
) -> None:
    """Plot raw vs. normalized inputs and targets for every stock on twin y-axes.

    Grid of ``num_stocks`` rows by two columns (inputs, targets). In each cell the raw price
    uses the left y-axis and the normalized value (as fed to the model) the right y-axis. All
    input arrays have shape ``(num_steps, num_stocks)``.
    """
    import matplotlib.pyplot as plt

    num_stocks = x_raw.shape[1]
    steps = range(x_raw.shape[0])
    fig, axes = plt.subplots(
        num_stocks, 2, figsize=(14, 3.2 * num_stocks), squeeze=False, sharex=True
    )

    for s in range(num_stocks):
        for col, (raw, norm, kind) in enumerate(
            (
                (x_raw[:, s], x_norm[:, s], "Input"),
                (y_raw[:, s], y_norm[:, s], "Target"),
            )
        ):
            ax = axes[s, col]
            ax2 = ax.twinx()
            (l_raw,) = ax.plot(steps, raw, color="tab:blue", lw=1.0, label="raw price")
            (l_norm,) = ax2.plot(
                steps, norm, color="tab:orange", lw=1.0, alpha=0.7, label="normalized"
            )
            ax.set_ylabel("raw price", color="tab:blue")
            ax2.set_ylabel("normalized", color="tab:orange")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.axhline(0.0, color="tab:orange", lw=0.5, ls="--")  # zero ref for normalized

            # Clip both axes to robust percentile ranges so outliers don't dominate the view.
            if (lim := _robust_ylim(raw)) is not None:
                ax.set_ylim(*lim)
            if (lim := _robust_ylim(norm)) is not None:
                ax2.set_ylim(*lim)

            ax.set_title(f"{kind} (stock {s})")
            if s == 0 and col == 0:
                ax.legend(handles=[l_raw, l_norm], loc="upper left")

    for col in range(2):
        axes[-1, col].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved normalization plot to {out_path}")
