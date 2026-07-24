"""Empirical hitting and return-time utilities."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.types import BoolArray, IntArray


def first_hitting_time(indicator: BoolArray, *, start: int = 0) -> int | None:
    """Return samples until the first target hit at or after ``start``."""

    flags = np.asarray(indicator, dtype=bool)
    if not 0 <= start < len(flags):
        raise ValueError("start must index the indicator array")
    relative = np.flatnonzero(flags[start:])
    return None if len(relative) == 0 else int(relative[0])


def return_times(indicator: BoolArray) -> IntArray:
    """Return positive intervals between successive target hits."""

    hit_indices = np.flatnonzero(np.asarray(indicator, dtype=bool))
    return np.diff(hit_indices).astype(np.int64)


def empirical_hit_probability(
    indicator: BoolArray,
    *,
    horizon: int,
    eligible_starts: IntArray | None = None,
) -> float:
    """Estimate probability of at least one hit within a forward horizon."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    flags = np.asarray(indicator, dtype=bool)
    max_start = len(flags) - horizon
    if max_start <= 0:
        raise ValueError("series is shorter than the requested horizon")

    starts: IntArray
    if eligible_starts is None:
        starts = np.arange(max_start, dtype=np.int64)
    else:
        starts = np.asarray(eligible_starts, dtype=np.int64)
        if np.any((starts < 0) | (starts >= max_start)):
            raise ValueError("eligible start indices are outside the valid range")

    cumulative = np.concatenate(([0], np.cumsum(flags.astype(np.int64))))
    counts = cumulative[starts + horizon + 1] - cumulative[starts + 1]
    return float(np.mean(counts > 0))
