"""Sequential linear predictor for multi-stock closing prices.

Loads the GGLabYale/MTBench_finance_stock dataset (one row per stock, each field a
list of ~191k per-timestep values) and trains an online linear model with pure numpy
that, at every timestep, predicts the closing prices of the selected stocks.

Inputs to the model are produced by ``agent_state_update``: a persistent state that is
updated at the *end* of each learning step from that step's full data. The starter
implementation simply stores the closing prices, so the model effectively predicts each
close from the previous step's close. Both inputs and targets are normalized with an
incremental Welford (running mean/variance) normalizer, each tracking its own stats.

Data loading lives in ``data.py``. Small mode (default) is fast: it reads only the first
few stocks truncated to a few thousand steps.
"""

from __future__ import annotations

import argparse

import numpy as np

from data import FIELDS, SMALL_NUM_STEPS, SMALL_NUM_STOCKS, load_stock_data
from norm_recorder import NormalizationRecorder


# --------------------------------------------------------------------------------------
# Incremental Welford normalization (per-feature running mean / variance)
# --------------------------------------------------------------------------------------
class WelfordNormalizer:
    """Vectorized online normalizer with exponentially weighted mean/variance.

    Each feature keeps an EWMA of its mean and variance, so recent observations dominate the
    statistics -- useful for non-stationary series. ``alpha`` is the weight given to each new
    observation; the effective horizon is ~1/alpha steps (smaller alpha = longer memory).
    During the first ~1/alpha steps the weight is capped at ``1/count``, i.e. an equal-weight
    running mean that smoothly hands off to the EWMA, so early estimates are not biased toward
    the zero initialization.

    ``normalize`` returns zeros for the first ``warmup`` samples: until then the variance
    estimate is near-degenerate (e.g. nearly-identical consecutive prices), and dividing by it
    would produce explosive values. A handful of samples is enough for the estimate to settle.

    Two further guards keep degenerate series from blowing up the metrics (and the model):
    a feature whose std has collapsed below ``std_floor`` (a perfectly flat series pins it at
    ~0, so the first later move would standardize to ~Δprice/eps ~ 1e7) is treated as carrying
    no signal and emits 0; and the standardized output is clipped to +/-``clip``, a
    one-in-a-billion tail of N(0,1) (6 sigma), so any surviving outlier can't dominate.
    """

    # Alpha default is effectively 1 month of data.
    def __init__(
        self,
        dim: int,
        alpha: float = 0.001,
        warmup: int = 10,
        eps: float = 1e-8,
        std_floor: float = 1e-5,
        clip: float = 6.0,
    ):
        self.count = 0
        self.alpha = alpha
        self.warmup = warmup
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.zeros(dim, dtype=np.float64)  # exponentially weighted variance
        self.eps = eps
        self.std_floor = std_floor  # below this, std is treated as degenerate (no signal)
        self.clip = clip  # |z| cap; 6 sigma ~ one-in-a-billion tail of N(0, 1)

    def update(self, x: np.ndarray) -> None:
        """Exponentially weighted Welford update (Finch 2009) with warmup debiasing."""
        self.count += 1
        a = max(self.alpha, 1.0 / self.count)  # equal-weight until ~1/alpha samples seen
        delta = x - self.mean
        incr = a * delta
        self.mean += incr
        self.var = (1.0 - a) * (self.var + delta * incr)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < self.warmup:
            return np.zeros_like(x)  # variance estimate not yet settled -> no scale to apply
        std = self.std
        z = (x - self.mean) / (std + self.eps)
        # Degenerate variance (e.g. a perfectly flat series) makes z ill-defined: the first
        # later move standardizes to ~Delta/eps ~ 1e7. Emit 0 for those features (no signal).
        z = np.where(std <= self.std_floor, 0.0, z)
        # Clip to a one-in-a-billion N(0, 1) tail so any remaining outlier can't dominate.
        return np.clip(z, -self.clip, self.clip)

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        return z * (self.std + self.eps) + self.mean


# --------------------------------------------------------------------------------------
# Agent state: produces the model's input features from the persistent state
# --------------------------------------------------------------------------------------
def init_agent_state(num_stocks: int) -> np.ndarray:
    """Initial persistent state. Starter state holds one value (close) per stock."""
    return np.zeros(num_stocks, dtype=np.float64)


def agent_state_update(state: np.ndarray, step_data: dict[str, np.ndarray]) -> np.ndarray:
    """Update the persistent state at the end of a learning step.

    Args:
        state: previous persistent state.
        step_data: ``{field: array of shape (n_stocks,)}`` for the current step.

    Returns the new persistent state. Starter version simply stores the closing prices,
    so next step's input is this step's close (predict close_t from close_{t-1}).
    """
    return step_data["close"].copy()


def features_from_state(state: np.ndarray) -> np.ndarray:
    """Map the persistent state to the model's input feature vector (identity for now)."""
    return np.nan_to_num(state, nan=0.0)


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
def train(
    data: dict[str, np.ndarray],
    lr: float,
    log_every: int,
    norm_alpha: float = 0.001,
    plot: bool = False,
    plot_path: str = "normalization.png",
) -> None:
    num_stocks, num_steps = data["close"].shape
    print(f"Training: {num_stocks} stocks, {num_steps} steps, lr={lr}, norm_alpha={norm_alpha}")

    state = init_agent_state(num_stocks)
    in_dim = features_from_state(state).shape[0]
    out_dim = num_stocks

    # Pure-numpy linear model: pred = W @ x + b.
    W = np.zeros((out_dim, in_dim), dtype=np.float64)
    b = np.zeros(out_dim, dtype=np.float64)

    in_norm = WelfordNormalizer(in_dim, alpha=norm_alpha)
    tgt_norm = WelfordNormalizer(out_dim, alpha=norm_alpha)

    # Running metrics over a trailing window.
    win_norm_sse = 0.0
    win_raw_sse = 0.0
    win_count = 0

    # Optional sanity-check recording of raw vs. normalized values (see norm_recorder.py).
    recorder = NormalizationRecorder() if plot else None

    for t in range(num_steps):
        step_data = {f: data[f][:, t] for f in data}

        # Inputs come from the persistent state (set at the end of the previous step).
        x_raw = features_from_state(state)
        y_raw = step_data["close"]

        # Normalize with statistics from prior steps only -- the current target is not yet
        # "revealed", so it must not feed its own normalization (no look-ahead bias).
        x = in_norm.normalize(x_raw)
        y = tgt_norm.normalize(y_raw)

        if recorder is not None:
            recorder.record(x_raw, x, y_raw, y)

        # Forward / loss / SGD step (MSE in normalized space).
        pred = W @ x + b
        err = pred - y  # gradient of 1/2 * ||err||^2 w.r.t. pred
        W -= lr * np.outer(err, x)
        b -= lr * err

        # Track normalized MSE and de-normalized (raw price) RMSE. Denormalize with the same
        # pre-update stats used for ``y`` so the raw metric stays leakage-free too.
        win_norm_sse += float(np.mean(err**2))
        pred_raw = tgt_norm.denormalize(pred)
        win_raw_sse += float(np.mean((pred_raw - y_raw) ** 2))
        win_count += 1

        if (t + 1) % log_every == 0:
            print(
                f"step {t + 1:>7d} | norm MSE {win_norm_sse / win_count:.5f} "
                f"| raw RMSE {np.sqrt(win_raw_sse / win_count):.4f}"
            )
            win_norm_sse = win_raw_sse = 0.0
            win_count = 0

        # Now reveal this step's observation: update the normalizers and the persistent
        # state for the next step.
        in_norm.update(x_raw)
        tgt_norm.update(y_raw)
        state = agent_state_update(state, step_data)

    if recorder is not None:
        recorder.save(plot_path)


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Load the entire dataset (all stocks, all steps; downloads everything). "
        "Default is small mode: a few streamed stocks truncated to a few thousand steps.",
    )
    p.add_argument(
        "--num-stocks",
        type=int,
        default=None,
        help=f"Number of stocks to load (default: {SMALL_NUM_STOCKS} small / all if --full).",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=f"Timesteps per stock (default: {SMALL_NUM_STEPS} small / all if --full).",
    )
    p.add_argument("--lr", type=float, default=0.01, help="SGD learning rate.")
    p.add_argument(
        "--norm-alpha",
        type=float,
        default=1/8640,
        help="EWMA weight for the input/target normalizers (~1/alpha-step horizon; "
        "smaller = longer memory).",
    )
    p.add_argument("--log-every", type=int, default=1000, help="Log interval in steps.")
    p.add_argument(
        "--plot-norm",
        action="store_true",
        help="Record every stock's raw vs. normalized inputs/targets and save twin-axis plots "
        "to --plot-path (to sanity-check the Welford normalizer).",
    )
    p.add_argument(
        "--plot-path",
        type=str,
        default="normalization.png",
        help="Output path for the --plot-norm figure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve sizes: small mode caps both; full mode defaults to everything.
    if args.full:
        num_stocks = args.num_stocks  # None -> all
        num_steps = args.num_steps  # None -> common max length
    else:
        num_stocks = args.num_stocks if args.num_stocks is not None else SMALL_NUM_STOCKS
        num_steps = args.num_steps if args.num_steps is not None else SMALL_NUM_STEPS

    data = load_stock_data(
        fields=FIELDS, num_stocks=num_stocks, num_steps=num_steps, full=args.full
    )
    train(
        data,
        lr=args.lr,
        log_every=args.log_every,
        norm_alpha=args.norm_alpha,
        plot=args.plot_norm,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
