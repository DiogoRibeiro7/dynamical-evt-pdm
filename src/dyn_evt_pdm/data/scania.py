"""SCANIA Component X entity-level validation helpers."""

from __future__ import annotations

import pandas as pd

from dyn_evt_pdm.data.contracts import require_columns


def validate_operational_readouts(frame: pd.DataFrame) -> None:
    """Validate minimal SCANIA operational-readout identifiers."""

    require_columns(frame, ["vehicle_id", "time_step"])
    if frame["vehicle_id"].isna().any():
        raise ValueError("vehicle_id contains missing values")
    if (pd.to_numeric(frame["time_step"], errors="coerce") < 0).any():
        raise ValueError("time_step must be non-negative")


def compute_counter_increments(
    frame: pd.DataFrame,
    *,
    counter_columns: list[str],
) -> pd.DataFrame:
    """Convert cumulative counters to non-negative per-readout increments.

    Counter resets are represented as missing increments and must be handled by an
    explicit imputation policy downstream rather than clipped silently.
    """

    validate_operational_readouts(frame)
    sorted_frame = frame.sort_values(["vehicle_id", "time_step"], kind="stable").copy()
    grouped = sorted_frame.groupby("vehicle_id", sort=False)[counter_columns]
    increments = grouped.diff()
    increments = increments.mask(increments < 0)
    increments.columns = [f"{column}__increment" for column in counter_columns]
    return pd.concat([sorted_frame.reset_index(drop=True), increments.reset_index(drop=True)], axis=1)
