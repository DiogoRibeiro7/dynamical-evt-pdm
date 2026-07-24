"""Leakage-safe temporal and entity-aware split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Index arrays for ordered train, validation and test partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def ordered_split_indices(
    n_rows: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TemporalSplit:
    """Create contiguous indices while preserving time order."""

    if n_rows < 3:
        raise ValueError("at least three rows are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test partition")

    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    return TemporalSplit(
        train=np.arange(0, train_end, dtype=np.int64),
        validation=np.arange(train_end, validation_end, dtype=np.int64),
        test=np.arange(validation_end, n_rows, dtype=np.int64),
    )


def sort_by_time(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Parse timestamps and return a stable, chronologically ordered frame."""

    if timestamp_column not in frame:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], utc=True)
    return result.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
