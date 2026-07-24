"""Calibration diagnostics for event-risk probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One reliability-diagram bin."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    event_rate: float


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the mean squared probability error."""

    probability_array = np.asarray(probabilities, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=float)
    if probability_array.shape != outcome_array.shape:
        raise ValueError("probabilities and outcomes must have equal shape")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any((outcome_array != 0.0) & (outcome_array != 1.0)):
        raise ValueError("outcomes must be binary")
    return float(np.mean((probability_array - outcome_array) ** 2))


def reliability_bins(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Aggregate predicted probabilities into equal-width reliability bins."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    probability_array = np.asarray(probabilities, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=float)
    if probability_array.shape != outcome_array.shape:
        raise ValueError("probabilities and outcomes must have equal shape")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probability_array, edges[1:-1]), 0, n_bins - 1)
    bins: list[CalibrationBin] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        bins.append(
            CalibrationBin(
                lower=float(edges[bin_id]),
                upper=float(edges[bin_id + 1]),
                count=int(np.count_nonzero(mask)),
                mean_probability=float(np.mean(probability_array[mask])),
                event_rate=float(np.mean(outcome_array[mask])),
            )
        )
    return bins
