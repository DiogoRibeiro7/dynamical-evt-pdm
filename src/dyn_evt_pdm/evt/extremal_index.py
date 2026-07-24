"""Finite-sample extremal-index estimators."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.types import BoolArray, IntArray


def runs_extremal_index(exceedances: BoolArray, *, run_length: int) -> float:
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


def disjoint_blocks_extremal_index(exceedances: BoolArray, *, block_size: int) -> float:
    """Estimate the extremal index using disjoint no-exceedance block probability.

    For a high threshold, ``P(max block <= u)`` is approximated by
    ``exp(-theta * block_size * exceedance_rate)``. The estimate is transparent
    and useful for stability checks, but it is sensitive to block size.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    flags = np.asarray(exceedances, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    if len(flags) < block_size:
        raise ValueError("series must contain at least one complete block")
    exceedance_rate = float(np.mean(flags))
    if exceedance_rate <= 0.0:
        raise ValueError("at least one exceedance is required")
    n_blocks = len(flags) // block_size
    blocks = flags[: n_blocks * block_size].reshape(n_blocks, block_size)
    no_exceedance_probability = float(np.mean(~np.any(blocks, axis=1)))
    if no_exceedance_probability <= 0.0:
        return 1.0
    estimate = -np.log(no_exceedance_probability) / (block_size * exceedance_rate)
    return float(np.clip(estimate, np.finfo(float).eps, 1.0))


def intervals_extremal_index(exceedance_indices: IntArray) -> float:
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
