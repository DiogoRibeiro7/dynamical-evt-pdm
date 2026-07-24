"""Lagged multivariate extreme signatures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.types import BoolArray


@dataclass(frozen=True, slots=True)
class LaggedExtremePattern:
    """Ordered component exceedances with non-negative relative lags."""

    components: tuple[int, ...]
    lags: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("at least one component is required")
        if len(self.components) != len(self.lags):
            raise ValueError("components and lags must have equal length")
        if any(component < 0 for component in self.components):
            raise ValueError("component indices must be non-negative")
        if any(lag < 0 for lag in self.lags):
            raise ValueError("lags must be non-negative")


def detect_lagged_pattern(
    exceedances: BoolArray,
    pattern: LaggedExtremePattern,
) -> BoolArray:
    """Detect an exact lagged logical pattern in a component-by-time matrix.

    Parameters
    ----------
    exceedances:
        Boolean matrix with shape ``(n_samples, n_components)``.
    pattern:
        Components and relative lags. A detection at time ``t`` means every
        requested component exceeds at ``t + lag``.
    """

    matrix = np.asarray(exceedances, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("exceedances must be a two-dimensional matrix")
    if max(pattern.components) >= matrix.shape[1]:
        raise ValueError("pattern references an unavailable component")

    max_lag = max(pattern.lags)
    valid_length = matrix.shape[0] - max_lag
    if valid_length <= 0:
        return np.zeros(matrix.shape[0], dtype=bool)

    result = np.ones(valid_length, dtype=bool)
    for component, lag in zip(pattern.components, pattern.lags, strict=True):
        result &= matrix[lag : lag + valid_length, component]
    return np.pad(result, (0, max_lag), constant_values=False)
