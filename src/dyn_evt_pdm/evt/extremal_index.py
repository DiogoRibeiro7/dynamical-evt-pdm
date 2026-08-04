"""Finite-sample extremal-index estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.types import BoolArray, IntArray

ESTIMATOR_NAMES: tuple[str, ...] = (
    "runs",
    "ferro_segers_intervals",
    "k_gaps",
    "reciprocal_mean_cluster",
    "block",
    "no_declustering",
)

#: Estimators whose tuning parameter is the declustering run length.
RUN_LENGTH_ESTIMATORS: frozenset[str] = frozenset({"runs", "k_gaps"})

#: Estimators whose tuning parameter is the disjoint block size.
BLOCK_SIZE_ESTIMATORS: frozenset[str] = frozenset({"block", "reciprocal_mean_cluster"})


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


def reciprocal_block_cluster_extremal_index(exceedances: BoolArray, *, block_size: int) -> float:
    """Estimate the extremal index as the reciprocal mean block-cluster size.

    Clusters are identified as the exceedance groups falling inside disjoint blocks
    of length ``block_size``. This is deliberately distinct from the runs estimator:
    when clusters are taken from runs declustering, the reciprocal mean cluster size
    reduces algebraically to the runs estimator and carries no separate information.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    flags = np.asarray(exceedances, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    if len(flags) < block_size:
        raise ValueError("series must contain at least one complete block")
    n_blocks = len(flags) // block_size
    blocks = flags[: n_blocks * block_size].reshape(n_blocks, block_size)
    per_block = blocks.sum(axis=1)
    occupied = per_block[per_block > 0]
    if occupied.size == 0:
        raise ValueError("at least one exceedance is required")
    mean_cluster_size = float(np.mean(occupied))
    if mean_cluster_size <= 0.0:
        raise ValueError("mean block-cluster size must be positive")
    return float(np.clip(1.0 / mean_cluster_size, np.finfo(float).eps, 1.0))


def default_block_size(n_samples: int) -> int:
    """Return the declared disjoint-block size ``round(sqrt(n))`` used by the study."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return max(2, int(round(float(np.sqrt(float(n_samples))))))


def default_bootstrap_block_length(n_samples: int) -> int:
    """Return the declared circular-bootstrap block length ``round(n ** (1/3))``."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return max(2, int(round(float(n_samples) ** (1.0 / 3.0))))


def extremal_index_point_estimate(
    name: str,
    exceedances: BoolArray,
    *,
    run_length: int,
    block_size: int,
) -> float:
    """Return the named extremal-index estimate for one exceedance indicator series.

    Every estimator in :data:`ESTIMATOR_NAMES` is reachable through this dispatch so
    that point estimation and bootstrap resampling share a single code path.
    """

    if name not in ESTIMATOR_NAMES:
        raise ValueError(f"unknown estimator: {name}")
    flags = np.asarray(exceedances, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    indices = np.flatnonzero(flags).astype(np.int64)

    if name == "no_declustering":
        if indices.size == 0:
            raise ValueError("at least one exceedance is required")
        return 1.0
    if name == "runs":
        return runs_extremal_index(flags, run_length=run_length)
    if name == "ferro_segers_intervals":
        return intervals_extremal_index(indices)
    if name == "k_gaps":
        return k_gaps_extremal_index(indices, run_length=run_length, n_samples=int(len(flags)))
    if name == "block":
        return disjoint_blocks_extremal_index(flags, block_size=block_size)
    return reciprocal_block_cluster_extremal_index(flags, block_size=block_size)


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """Bootstrap interval for one extremal-index estimate."""

    lower: float
    upper: float
    n_resamples: int
    n_valid: int
    n_clipped: int

    @property
    def width(self) -> float:
        """Return the interval width, or NaN when the interval is not estimable."""

        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            return float("nan")
        return float(self.upper - self.lower)

    def covers(self, value: float) -> bool:
        """Return whether the interval covers ``value``."""

        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            return False
        return bool(self.lower <= value <= self.upper)


def circular_block_bootstrap(
    exceedances: BoolArray, *, block_length: int, rng: np.random.Generator
) -> BoolArray:
    """Resample an exceedance indicator series with a circular block bootstrap.

    Resampling whole blocks preserves the short-range clustering that the extremal
    index measures. An IID bootstrap would destroy exactly the dependence structure
    under study and produce intervals centred on one.
    """

    if block_length < 1:
        raise ValueError("block_length must be positive")
    flags = np.asarray(exceedances, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    n_samples = int(len(flags))
    if n_samples < block_length:
        raise ValueError("series must be at least as long as the block length")
    n_blocks = int(np.ceil(n_samples / block_length))
    starts = rng.integers(0, n_samples, size=n_blocks)
    offsets = np.arange(block_length, dtype=np.int64)
    positions = (starts[:, None] + offsets[None, :]) % n_samples
    resampled = flags[positions.reshape(-1)][:n_samples]
    return np.asarray(resampled, dtype=bool)


def bootstrap_extremal_index_interval(
    name: str,
    exceedances: BoolArray,
    *,
    run_length: int,
    block_size: int,
    n_resamples: int,
    block_length: int,
    level: float,
    rng: np.random.Generator,
) -> IntervalEstimate:
    """Return a percentile bootstrap interval for the named estimator.

    Resamples that raise :class:`ValueError` are counted as invalid rather than
    silently dropped, so the caller can report an honest resample success rate.
    """

    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")

    estimates: list[float] = []
    n_clipped = 0
    epsilon = float(np.finfo(float).eps)
    for _ in range(n_resamples):
        resampled = circular_block_bootstrap(exceedances, block_length=block_length, rng=rng)
        try:
            estimate = extremal_index_point_estimate(
                name, resampled, run_length=run_length, block_size=block_size
            )
        except ValueError:
            continue
        if not np.isfinite(estimate):
            continue
        if estimate <= epsilon or estimate >= 1.0:
            n_clipped += 1
        estimates.append(float(estimate))

    if len(estimates) < 2:
        return IntervalEstimate(
            lower=float("nan"),
            upper=float("nan"),
            n_resamples=n_resamples,
            n_valid=len(estimates),
            n_clipped=n_clipped,
        )

    tail = (1.0 - level) / 2.0
    values = np.asarray(estimates, dtype=np.float64)
    lower = float(np.quantile(values, tail))
    upper = float(np.quantile(values, 1.0 - tail))
    return IntervalEstimate(
        lower=lower,
        upper=upper,
        n_resamples=n_resamples,
        n_valid=len(estimates),
        n_clipped=n_clipped,
    )
