"""MetroPT-specific schema helpers and documented failure intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dyn_evt_pdm.data.io import read_table


@dataclass(frozen=True, slots=True)
class TimestampedFailure:
    """A documented failure interval in UTC-naive source time."""

    start: pd.Timestamp
    end: pd.Timestamp
    failure_type: str


# The exact timestamps must be checked against the downloaded dataset version.
# These labels are intentionally maintained as configuration-ready metadata rather
# than silently embedded in modelling code.
METROPT2_FAILURES: tuple[TimestampedFailure, ...] = (
    TimestampedFailure(
        start=pd.Timestamp("2022-06-04 10:19:24.300"),
        end=pd.Timestamp("2022-06-04 14:22:39.188"),
        failure_type="air_leak",
    ),
    TimestampedFailure(
        start=pd.Timestamp("2022-07-11 10:10:18.948"),
        end=pd.Timestamp("2022-07-14 10:22:08.046"),
        failure_type="oil_leak",
    ),
)


def load_metropt(path: Path, *, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Load MetroPT data and normalize its timestamp column.

    Column names differ across released files. The dataset configuration must map
    the source timestamp column to ``timestamp`` before the research pipeline is run.
    """

    frame = read_table(path)
    if timestamp_column not in frame:
        raise ValueError(
            f"timestamp column {timestamp_column!r} not found; update the dataset config"
        )
    frame = frame.copy()
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column], errors="raise", format="mixed"
    )
    return frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)


def annotate_failures(
    timestamps: pd.Series,
    failures: tuple[TimestampedFailure, ...],
) -> pd.DataFrame:
    """Create failure labels without modifying the original data."""

    parsed = pd.to_datetime(timestamps, errors="raise", format="mixed")
    result = pd.DataFrame(
        {
            "is_failure": False,
            "failure_type": pd.Series([None] * len(parsed), dtype="string"),
        },
        index=timestamps.index,
    )
    for failure in failures:
        mask = parsed.between(failure.start, failure.end, inclusive="both")
        result.loc[mask, "is_failure"] = True
        result.loc[mask, "failure_type"] = failure.failure_type
    return result
