"""Observables that convert state-space recurrence into extreme values."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.types import FloatArray


def nearest_reference_distance(
    states: FloatArray,
    references: FloatArray,
    *,
    chunk_size: int = 50_000,
    metric: str = "euclidean",
) -> FloatArray:
    """Return Euclidean distance to the nearest dangerous reference state."""

    states_array = np.asarray(states, dtype=np.float64)
    references_array = np.asarray(references, dtype=np.float64)
    if states_array.ndim != 2 or references_array.ndim != 2:
        raise ValueError("states and references must be two-dimensional")
    if states_array.shape[1] != references_array.shape[1]:
        raise ValueError("states and references must share the same feature dimension")
    if len(references_array) == 0:
        raise ValueError("at least one reference state is required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if metric != "euclidean":
        raise ValueError("only euclidean metric is supported by this distance helper")

    output = np.empty(len(states_array), dtype=np.float64)
    for start in range(0, len(states_array), chunk_size):
        stop = min(start + chunk_size, len(states_array))
        differences = states_array[start:stop, None, :] - references_array[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        output[start:stop] = np.min(distances, axis=1)
    return output


def negative_log_distance(
    states: FloatArray,
    references: FloatArray,
    *,
    epsilon: float = 1e-12,
    chunk_size: int = 50_000,
) -> FloatArray:
    """Compute ``-log(distance)`` to a dangerous state-space region.

    Larger values correspond to closer approaches and therefore define extremes.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    distances = nearest_reference_distance(states, references, chunk_size=chunk_size)
    return np.asarray(-np.log(np.maximum(distances, epsilon)), dtype=np.float64)


def distance_observable_report(
    states: FloatArray,
    references: FloatArray,
    *,
    epsilon: float = 1e-12,
    small_scale_threshold: float = 1e-6,
    chunk_size: int = 50_000,
) -> dict[str, object]:
    """Compute distance observable and diagnose exact or near-target matches."""

    distances = nearest_reference_distance(states, references, chunk_size=chunk_size)
    observable = np.asarray(-np.log(np.maximum(distances, epsilon)), dtype=np.float64)
    return {
        "observable": observable,
        "distance": distances,
        "exact_duplicate_count": int(np.count_nonzero(distances <= epsilon)),
        "small_metric_scale_count": int(np.count_nonzero(distances <= small_scale_threshold)),
        "min_distance": float(np.min(distances)) if len(distances) else np.nan,
    }
