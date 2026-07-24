"""Observables that convert state-space recurrence into extreme values."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.types import FloatArray


def nearest_reference_distance(states: FloatArray, references: FloatArray) -> FloatArray:
    """Return Euclidean distance to the nearest dangerous reference state."""

    states_array = np.asarray(states, dtype=np.float64)
    references_array = np.asarray(references, dtype=np.float64)
    if states_array.ndim != 2 or references_array.ndim != 2:
        raise ValueError("states and references must be two-dimensional")
    if states_array.shape[1] != references_array.shape[1]:
        raise ValueError("states and references must share the same feature dimension")
    if len(references_array) == 0:
        raise ValueError("at least one reference state is required")

    # Chunking should be added for very large reference sets. Failure windows are
    # expected to be summarized into prototypes before this function is used.
    differences = states_array[:, None, :] - references_array[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return np.asarray(np.min(distances, axis=1), dtype=np.float64)


def negative_log_distance(
    states: FloatArray,
    references: FloatArray,
    *,
    epsilon: float = 1e-12,
) -> FloatArray:
    """Compute ``-log(distance)`` to a dangerous state-space region.

    Larger values correspond to closer approaches and therefore define extremes.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    distances = nearest_reference_distance(states, references)
    return np.asarray(-np.log(np.maximum(distances, epsilon)), dtype=np.float64)
