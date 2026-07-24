"""Data contracts for industrial time-series adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeSeriesContract:
    """Required columns and ordering assumptions for a dataset."""

    timestamp_column: str
    sensor_columns: tuple[str, ...]
    regime_columns: tuple[str, ...] = ()
    entity_column: str | None = None

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise a descriptive error when a frame violates the contract."""

        required = {
            self.timestamp_column,
            *self.sensor_columns,
            *self.regime_columns,
        }
        if self.entity_column is not None:
            required.add(self.entity_column)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        if frame.empty:
            raise ValueError("dataset must not be empty")
        if frame[self.timestamp_column].isna().any():
            raise ValueError("timestamp column contains missing values")


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate a lightweight column subset."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
