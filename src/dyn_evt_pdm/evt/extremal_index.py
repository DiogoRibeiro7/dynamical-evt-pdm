"""Finite-sample extremal-index estimators."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.evt.clusters import extract_clusters


def runs_extremal_index(exceedances: np.ndarray, *, run_length: int) -> float:
    """Estimate the extremal index as clusters divided by exceedances.

    This estimator is transparent and operationally interpretable, but sensitive to
    the run parameter. Threshold and run-length stability analyses are mandatory.
    """

    flags = np.asarray(exceedances, dtype=bool)
    exceedance_count = int(np.count_nonzero(flags))
    if exceedance_count == 0:
        raise ValueError("at least one exceedance is required")
    cluster_count = len(extract_clusters(flags, run_length=run_length))
    return float(cluster_count / exceedance_count)


def intervals_extremal_index(exceedance_indices: np.ndarray) -> float:
    """Estimate the extremal index with the Ferro-Segers intervals estimator.

    The estimator is bounded to ``(0, 1]``. At least two exceedance times are
    required. When all inter-exceedance times are at most two samples, the finite
    sample convention returns one.
    """

    indices = np.asarray(exceedance_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("exceedance_indices must be one-dimensional")
    if len(indices) < 2:
        raise ValueError("at least two exceedance times are required")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("exceedance indices must be strictly increasing")

    inter_exceedance = np.diff(indices).astype(np.float64)
    if float(np.max(inter_exceedance)) <= 2.0:
        return 1.0

    shifted = inter_exceedance - 1.0
    denominator_sum = float(np.sum(shifted * (shifted - 1.0)))
    if denominator_sum <= 0.0:
        return 1.0
    numerator = 2.0 * float(np.sum(shifted)) ** 2
    denominator = float(len(inter_exceedance)) * denominator_sum
    estimate = numerator / denominator
    return float(np.clip(estimate, np.finfo(float).eps, 1.0))


def mean_cluster_size_from_extremal_index(theta: float) -> float:
    """Return the asymptotic mean cluster-size interpretation ``1 / theta``."""

    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must be in (0, 1]")
    return 1.0 / theta
