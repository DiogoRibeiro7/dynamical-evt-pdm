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


def k_gaps_extremal_index(
    exceedance_indices: IntArray, *, run_length: int, n_samples: int
) -> float:
    """Estimate the extremal index using a K-gaps likelihood sensitivity estimator.

    Inter-exceedance gaps with at most ``run_length`` intervening non-exceedances
    are treated as within-cluster gaps. Longer gaps contribute to a simple
    threshold-exceedance mixture likelihood with the empirical exceedance rate as
    the normalizing tail probability.
    """

    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    indices = np.asarray(exceedance_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("exceedance_indices must be one-dimensional")
    if len(indices) < 2:
        raise ValueError("at least two exceedance times are required")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("exceedance indices must be strictly increasing")
    if int(indices[-1]) >= n_samples:
        raise ValueError("exceedance indices must be inside the sample")

    gaps = np.diff(indices).astype(np.float64) - 1.0
    truncated = np.maximum(gaps - float(run_length), 0.0)
    positive = truncated > 0.0
    n_between = int(np.count_nonzero(positive))
    n_within = int(len(truncated) - n_between)
    if n_between == 0:
        raise ValueError("K-gaps estimator has no between-cluster gaps")
    tail_probability = len(indices) / n_samples
    weighted_gap_sum = float(tail_probability * np.sum(truncated[positive]))
    if weighted_gap_sum <= 0.0:
        raise ValueError("K-gaps estimator has zero positive gap mass")

    if n_within == 0:
        estimate = 2.0 * n_between / weighted_gap_sum
        return float(np.clip(estimate, np.finfo(float).eps, 1.0))

    # Score equation for the censored mixture likelihood:
    # -n0/(1-theta) + 2*n1/theta - A = 0, A = p_u * sum(K-gaps).
    coefficients = (
        weighted_gap_sum,
        -(n_within + 2.0 * n_between + weighted_gap_sum),
        2.0 * n_between,
    )
    roots = np.roots(coefficients)
    candidates = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= 1e-10 and 0.0 < float(root.real) <= 1.0
    ]
    if not candidates:
        raise ValueError("K-gaps likelihood score has no admissible root")
    return float(np.clip(min(candidates), np.finfo(float).eps, 1.0))


def mean_cluster_size_from_extremal_index(theta: float) -> float:
    """Return the asymptotic mean cluster-size interpretation ``1 / theta``."""

    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must be in (0, 1]")
    return 1.0 / theta
