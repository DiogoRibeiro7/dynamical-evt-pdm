"""Structured univariate peaks-over-threshold analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from dyn_evt_pdm.evt.clusters import cluster_maxima, extract_clusters
from dyn_evt_pdm.evt.extremal_index import (
    disjoint_blocks_extremal_index,
    intervals_extremal_index,
    k_gaps_extremal_index,
    runs_extremal_index,
)
from dyn_evt_pdm.evt.hitting_times import return_times
from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold
from dyn_evt_pdm.types import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Simple percentile confidence interval."""

    lower: float
    upper: float
    level: float


@dataclass(frozen=True, slots=True)
class GPDFit:
    """Generalized Pareto fit to threshold excesses."""

    shape: float
    scale: float
    n_excesses: int


@dataclass(frozen=True, slots=True)
class ThresholdDiagnostic:
    """Diagnostics for one candidate threshold."""

    quantile: float
    threshold: float
    exceedance_count: int
    mean_excess: float


@dataclass(frozen=True, slots=True)
class ExtremalIndexEstimates:
    """Multiple extremal-index estimates for the same exceedance series."""

    runs: float | None
    intervals: float | None
    disjoint_blocks: float | None
    k_gaps: float | None


@dataclass(frozen=True, slots=True)
class EstimatorDiagnostic:
    """Status row for an estimator used in one EVT fit."""

    estimator_name: str
    estimator_version: str
    estimate: float | None
    status: str
    warning: str | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnivariateEVTResult:
    """Complete univariate EVT result with estimates, diagnostics and warnings."""

    threshold_quantile: float
    threshold: float
    n_samples: int
    n_finite: int
    n_exceedances: int
    n_clusters: int
    cluster_sizes: tuple[int, ...]
    cluster_durations: tuple[int, ...]
    cluster_maxima: tuple[float, ...]
    extremal_index: ExtremalIndexEstimates
    gpd: GPDFit | None
    return_time_mean: float | None
    inter_exceedance_mean: float | None
    runs_theta_ci: ConfidenceInterval | None
    diagnostics: tuple[ThresholdDiagnostic, ...]
    estimator_diagnostics: tuple[EstimatorDiagnostic, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert nested dataclasses to plain dictionaries."""

        return asdict(self)


def fit_univariate_evt(
    values: FloatArray,
    *,
    threshold_quantile: float = 0.98,
    run_length: int = 10,
    block_size: int | None = None,
    diagnostic_quantiles: tuple[float, ...] = (0.95, 0.97, 0.98, 0.99),
    min_exceedances: int = 30,
    bootstrap_repetitions: int = 0,
    random_seed: int = 42,
) -> UnivariateEVTResult:
    """Fit a univariate threshold model using training values only."""

    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    if min_exceedances < 3:
        raise ValueError("min_exceedances must be at least three")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("no finite values are available")

    threshold = fit_quantile_threshold(finite, quantile=threshold_quantile)
    exceedances = np.isfinite(array) & (array > threshold)
    exceedance_indices = np.flatnonzero(exceedances).astype(np.int64)
    clusters = extract_clusters(exceedances, run_length=run_length)
    maxima = cluster_maxima(array, clusters) if clusters else np.asarray([], dtype=np.float64)
    cluster_sizes = tuple(cluster.size for cluster in clusters)
    cluster_durations = tuple(cluster.interval.duration for cluster in clusters)
    warnings: list[str] = []
    if len(exceedance_indices) < min_exceedances:
        warnings.append(
            f"only {len(exceedance_indices)} exceedances; minimum diagnostic target is {min_exceedances}"
        )

    estimator_reports: list[EstimatorDiagnostic] = []
    runs_theta = _reported_estimate(
        "runs",
        lambda: runs_extremal_index(exceedances, run_length=run_length),
        assumptions=(
            "declustering run length is predeclared",
            "cluster count divided by exceedance count is used as a diagnostic estimate",
        ),
        reports=estimator_reports,
        warnings=warnings,
    )
    intervals_theta = _reported_estimate(
        "ferro_segers_intervals",
        lambda: intervals_extremal_index(exceedance_indices),
        assumptions=(
            "exceedance times are strictly ordered",
            "limiting inter-exceedance mixture approximation is informative",
        ),
        reports=estimator_reports,
        warnings=warnings,
    )
    block_theta = _reported_estimate(
        "disjoint_blocks",
        lambda: disjoint_blocks_extremal_index(
            exceedances,
            block_size=block_size or max(run_length + 1, int(np.sqrt(len(finite)))),
        ),
        assumptions=(
            "block size is predeclared or derived without labels",
            "no-exceedance block probability approximates the high-threshold limit",
        ),
        reports=estimator_reports,
        warnings=warnings,
    )
    k_gaps_theta = _reported_estimate(
        "k_gaps",
        lambda: k_gaps_extremal_index(
            exceedance_indices,
            run_length=run_length,
            n_samples=len(array),
        ),
        assumptions=(
            "run length K is predeclared",
            "truncated inter-exceedance gaps follow the K-gaps mixture approximation",
        ),
        reports=estimator_reports,
        warnings=warnings,
    )
    gpd_fit = _fit_gpd_excesses(finite, threshold=threshold, warnings=warnings, min_excesses=3)
    gaps = return_times(exceedances)
    diagnostics = tuple(threshold_diagnostics(finite, diagnostic_quantiles=diagnostic_quantiles))
    runs_ci = (
        bootstrap_runs_extremal_index(
            exceedances,
            run_length=run_length,
            repetitions=bootstrap_repetitions,
            random_seed=random_seed,
        )
        if bootstrap_repetitions
        else None
    )
    return UnivariateEVTResult(
        threshold_quantile=threshold_quantile,
        threshold=threshold,
        n_samples=int(len(array)),
        n_finite=int(len(finite)),
        n_exceedances=int(len(exceedance_indices)),
        n_clusters=int(len(clusters)),
        cluster_sizes=cluster_sizes,
        cluster_durations=cluster_durations,
        cluster_maxima=tuple(float(value) for value in maxima),
        extremal_index=ExtremalIndexEstimates(
            runs=runs_theta,
            intervals=intervals_theta,
            disjoint_blocks=block_theta,
            k_gaps=k_gaps_theta,
        ),
        gpd=gpd_fit,
        return_time_mean=float(np.mean(gaps)) if len(gaps) else None,
        inter_exceedance_mean=float(np.mean(gaps)) if len(gaps) else None,
        runs_theta_ci=runs_ci,
        diagnostics=diagnostics,
        estimator_diagnostics=tuple(estimator_reports),
        warnings=tuple(warnings),
    )


def threshold_diagnostics(
    values: FloatArray,
    *,
    diagnostic_quantiles: tuple[float, ...],
) -> list[ThresholdDiagnostic]:
    """Compute mean-excess diagnostics for candidate empirical thresholds."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("no finite values are available")
    diagnostics: list[ThresholdDiagnostic] = []
    for quantile in diagnostic_quantiles:
        threshold = fit_quantile_threshold(finite, quantile=quantile)
        excesses = finite[finite > threshold] - threshold
        diagnostics.append(
            ThresholdDiagnostic(
                quantile=quantile,
                threshold=threshold,
                exceedance_count=int(len(excesses)),
                mean_excess=float(np.mean(excesses)) if len(excesses) else np.nan,
            )
        )
    return diagnostics


def threshold_run_stability(
    values: FloatArray,
    *,
    threshold_quantiles: tuple[float, ...],
    run_lengths: tuple[int, ...],
    min_exceedances: int = 10,
) -> pd.DataFrame:
    """Evaluate univariate estimates over threshold and run-length grids."""

    records: list[dict[str, object]] = []
    for quantile in threshold_quantiles:
        for run_length in run_lengths:
            result = fit_univariate_evt(
                values,
                threshold_quantile=quantile,
                run_length=run_length,
                min_exceedances=min_exceedances,
            )
            records.append(
                {
                    "threshold_quantile": quantile,
                    "run_length": run_length,
                    "threshold": result.threshold,
                    "n_exceedances": result.n_exceedances,
                    "n_clusters": result.n_clusters,
                    "runs_theta": result.extremal_index.runs,
                    "intervals_theta": result.extremal_index.intervals,
                    "blocks_theta": result.extremal_index.disjoint_blocks,
                    "k_gaps_theta": result.extremal_index.k_gaps,
                    "gpd_shape": result.gpd.shape if result.gpd else np.nan,
                    "gpd_scale": result.gpd.scale if result.gpd else np.nan,
                    "warning_count": len(result.warnings),
                }
            )
    return pd.DataFrame.from_records(records)


def bootstrap_runs_extremal_index(
    exceedances: BoolArray,
    *,
    run_length: int,
    repetitions: int,
    random_seed: int,
    block_length: int | None = None,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Percentile block bootstrap interval for the runs estimator."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    flags = np.asarray(exceedances, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("exceedances must be one-dimensional")
    if not np.any(flags):
        raise ValueError("at least one exceedance is required")
    length = len(flags)
    block_length = block_length or max(run_length + 1, int(np.sqrt(length)))
    if block_length < 1:
        raise ValueError("block_length must be positive")
    rng = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = _moving_block_sample(flags, block_length=block_length, rng=rng, length=length)
        if np.any(sample):
            estimates.append(runs_extremal_index(sample, run_length=run_length))
    if not estimates:
        raise ValueError("bootstrap produced no samples with exceedances")
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        level=level,
    )


def _fit_gpd_excesses(
    finite: FloatArray,
    *,
    threshold: float,
    warnings: list[str],
    min_excesses: int,
) -> GPDFit | None:
    excesses = finite[finite > threshold] - threshold
    if len(excesses) < min_excesses:
        warnings.append("too few excesses for GPD fit")
        return None
    shape, _location, scale = genpareto.fit(excesses, floc=0.0)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
        warnings.append("GPD fit failed finite positive-scale diagnostics")
        return None
    return GPDFit(shape=float(shape), scale=float(scale), n_excesses=int(len(excesses)))


def _moving_block_sample(
    values: BoolArray,
    *,
    block_length: int,
    rng: np.random.Generator,
    length: int,
) -> BoolArray:
    starts = rng.integers(
        0, max(1, length - block_length + 1), size=int(np.ceil(length / block_length))
    )
    blocks = [values[start : start + block_length] for start in starts]
    return np.asarray(np.concatenate(blocks)[:length], dtype=np.bool_)


def _reported_estimate(
    estimator_name: str,
    estimator: Callable[[], float],
    *,
    assumptions: tuple[str, ...],
    reports: list[EstimatorDiagnostic],
    warnings: list[str],
) -> float | None:
    try:
        value = estimator()
    except ValueError as exc:
        warning = str(exc)
        warnings.append(warning)
        reports.append(
            EstimatorDiagnostic(
                estimator_name=estimator_name,
                estimator_version="1",
                estimate=None,
                status="failed",
                warning=warning,
                assumptions=assumptions,
            )
        )
        return None
    estimate = float(value)
    reports.append(
        EstimatorDiagnostic(
            estimator_name=estimator_name,
            estimator_version="1",
            estimate=estimate,
            status="succeeded",
            warning=None,
            assumptions=assumptions,
        )
    )
    return estimate
