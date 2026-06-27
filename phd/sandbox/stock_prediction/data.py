"""Data loading for the multi-stock prediction experiments.

Loads the GGLabYale/MTBench_finance_stock dataset (one row per stock, each field a list of
~191k per-timestep values) into aligned ``{field: array of shape (n_stocks, n_steps)}``.

The dataset is shipped as Arrow shards that the HF datasets library prepares into a local
cache. When that cache is present (the common case), ``load_stock_data`` reads the rows it
needs straight from the shards via memory-mapping: no network, and a partial load of 50
stocks truncated to a few thousand steps is ~0.05s warm. It falls back to the datasets
library (streaming small / downloading full) only when the dataset is not prepared locally.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pyarrow as pa
from datasets import load_dataset

DATASET_NAME = "GGLabYale/MTBench_finance_stock"
FIELDS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "otc",
    "real_timestamp",
]

# Small-mode defaults (fast, no full download).
SMALL_NUM_STOCKS = 10
SMALL_NUM_STEPS = 10_000


def _local_arrow_shards(dataset_name: str) -> list[str]:
    """Sorted paths to the locally-prepared Arrow shards for ``dataset_name``.

    Looks in the HF datasets cache for the prepared (non-streaming) build. Returns an empty
    list if the dataset has not been downloaded/prepared locally. Sorted shard order
    (``...-train-00000-of-00071``) matches the dataset's row (stock) order, so reading the
    first ``num_stocks`` rows here yields the same stocks the streaming reader would.
    """
    from datasets import config
    from datasets.naming import camelcase_to_snakecase

    if "/" in dataset_name:
        namespace, name = dataset_name.split("/", 1)
        dirname = f"{namespace}___{camelcase_to_snakecase(name)}"
    else:
        dirname = camelcase_to_snakecase(dataset_name)
    pattern = os.path.join(config.HF_DATASETS_CACHE, dirname, "**", "*.arrow")
    return sorted(glob.glob(pattern, recursive=True))


def _collect_from_arrow(shards, fields, num_stocks, num_steps, verbose):
    """Collect per-stock series by memory-mapping local Arrow shards.

    Memory-mapping is lazy, so this touches only the bytes for the kept rows/steps: a
    partial load of 50 stocks truncated to a few thousand steps is ~0.05s (warm), needs no
    network, and sidesteps the datasets streaming reader entirely.
    """
    collected: dict[str, list] = {f: [] for f in fields}
    n_kept = 0
    for shard in shards:
        with pa.memory_map(shard, "r") as source:
            table = pa.ipc.open_stream(source).read_all()
            cols = {f: table.column(f) for f in fields}
            close = cols["close"]
            for i in range(table.num_rows):
                length = len(close[i])
                if num_steps is not None and length < num_steps:
                    continue  # too short to keep the array rectangular
                take = length if num_steps is None else num_steps
                for f in fields:
                    values = cols[f][i].values
                    if take < length:
                        values = values.slice(0, take)  # only materialize the kept steps
                    collected[f].append(values.to_numpy(zero_copy_only=False))
                n_kept += 1
                if verbose and n_kept % 50 == 0:
                    print(f"  ...loaded {n_kept} stocks")
                if num_stocks is not None and n_kept >= num_stocks:
                    return collected, n_kept, "local arrow (mmap)"
    return collected, n_kept, "local arrow (mmap)"


def _collect_via_datasets(fields, num_stocks, num_steps, full, verbose):
    """Fallback collector when the dataset is not prepared locally.

    Streams in small mode (only the consumed rows are fetched) or downloads + memory-maps
    the cache in full mode, via the datasets library.
    """
    streaming = not full
    ds = load_dataset(DATASET_NAME, split="train", streaming=streaming)
    collected: dict[str, list] = {f: [] for f in fields}
    n_kept = 0
    for ex in ds:
        length = len(ex["close"])
        if num_steps is not None and length < num_steps:
            continue  # too short to keep the array rectangular
        take = length if num_steps is None else num_steps
        for f in fields:
            collected[f].append(ex[f][:take])
        n_kept += 1
        if verbose and n_kept % 50 == 0:
            print(f"  ...loaded {n_kept} stocks")
        if num_stocks is not None and n_kept >= num_stocks:
            break
    return collected, n_kept, "full" if full else "small/streamed"


def load_stock_data(
    fields: list[str],
    num_stocks: int | None,
    num_steps: int | None,
    full: bool,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Load aligned per-stock series into ``{field: array of shape (n_stocks, n_steps)}``.

    Args:
        fields: dataset fields to load (each becomes one array).
        num_stocks: number of stocks (rows) to keep, or ``None`` for all.
        num_steps: timesteps to keep per stock, or ``None`` to use the common max
            (min length across the kept stocks).
        full: hint for the fallback path only -- if ``True`` download/cache the whole
            dataset, otherwise stream. When the dataset is already prepared locally, reads
            go straight from the Arrow cache regardless and ``num_stocks``/``num_steps``
            alone decide how much is read.

    Stocks shorter than ``num_steps`` are skipped so the result stays rectangular. ``None``
    values in the source (e.g. the all-null ``otc`` field) become ``np.nan``.
    """
    t0 = time.time()

    # Fast path: read the needed rows directly from the locally-prepared Arrow shards via
    # memory-mapping (no network, no datasets streaming reader). Falls back to the datasets
    # library only when the dataset has not been prepared locally.
    shards = _local_arrow_shards(DATASET_NAME)
    if shards:
        collected, n_kept, source = _collect_from_arrow(
            shards, fields, num_stocks, num_steps, verbose
        )
    else:
        collected, n_kept, source = _collect_via_datasets(
            fields, num_stocks, num_steps, full, verbose
        )

    if n_kept == 0:
        raise RuntimeError("No stocks loaded (all candidates shorter than --num-steps?).")

    # Truncate every stock to the shortest kept series so all fields stack rectangularly.
    common_len = min(len(col) for col in collected["close"])
    data: dict[str, np.ndarray] = {}
    for f in fields:
        data[f] = np.asarray(
            [col[:common_len] for col in collected[f]], dtype=np.float64
        )  # None -> nan via float cast

    if verbose:
        print(
            f"Loaded {n_kept} stocks x {common_len} steps "
            f"({source}) in {time.time() - t0:.2f}s"
        )
    return data
