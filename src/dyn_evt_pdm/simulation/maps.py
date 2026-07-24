"""Canonical deterministic maps for EVT estimator validation."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.types import FloatArray


def logistic_map(
    n_steps: int,
    *,
    x0: float = 0.123456789,
    parameter: float = 4.0,
    burn_in: int = 1_000,
) -> FloatArray:
    """Simulate the logistic map ``x[t+1] = r*x[t]*(1-x[t])``."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 must be in (0, 1)")
    if not 0.0 < parameter <= 4.0:
        raise ValueError("parameter must be in (0, 4]")

    total = n_steps + burn_in
    values = np.empty(total, dtype=np.float64)
    values[0] = x0
    for index in range(1, total):
        values[index] = parameter * values[index - 1] * (1.0 - values[index - 1])
    return values[burn_in:]
