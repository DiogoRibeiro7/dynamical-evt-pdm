"""Dangerous-region definitions and horizon-risk scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from dyn_evt_pdm.evaluation.calibration import brier_score, reliability_bins
from dyn_evt_pdm.evt.observables import distance_observable_report
from dyn_evt_pdm.types import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class StateProvenance:
    """Provenance metadata for one state vector."""

    index: int
    source: str
    failure_id: str | None = None
    split: str = "train"


@dataclass(frozen=True, slots=True)
class DangerousRegion:
    """Reference states and metadata defining a dangerous region."""

    method: str
    references: FloatArray
    provenance: tuple[StateProvenance, ...]
    metric: str = "euclidean"
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class HorizonRiskReport:
    """Empirical risk and calibration diagnostics for region entry."""

    horizons: tuple[int, ...]
    probabilities: dict[int, float]
    brier_scores: dict[int, float]
    calibration_bins: dict[int, list[dict[str, object]]]


def dangerous_region_from_training_failures(
    states: FloatArray,
    provenance: tuple[StateProvenance, ...],
    *,
    allowed_failure_ids: set[str],
    max_references: int | None = None,
    method: str = "training_failure_prototypes",
) -> DangerousRegion:
    """Build references from allowed training failures only."""

    matrix = np.asarray(states, dtype=np.float64)
    _validate_state_matrix(matrix)
    indices: list[int] = []
    selected_provenance: list[StateProvenance] = []
    for item in provenance:
        if item.split != "train":
            continue
        if item.failure_id is None or item.failure_id not in allowed_failure_ids:
            continue
        indices.append(item.index)
        selected_provenance.append(item)
    if not indices:
        raise ValueError("no allowed training-failure states available")
    references = matrix[indices]
    if max_references is not None:
        references, selected_provenance = _select_spread_prototypes(
            references,
            selected_provenance,
            max_references=max_references,
        )
    _assert_allowed_provenance(selected_provenance, allowed_failure_ids=allowed_failure_ids)
    return DangerousRegion(
        method=method,
        references=np.asarray(references, dtype=np.float64),
        provenance=tuple(selected_provenance),
        metadata={"allowed_failure_ids": sorted(allowed_failure_ids)},
    )


def dangerous_region_from_constraints(
    states: FloatArray,
    mask: BoolArray,
    provenance: tuple[StateProvenance, ...],
) -> DangerousRegion:
    """Build references from an auditable engineering constraint mask."""

    matrix = np.asarray(states, dtype=np.float64)
    flags = np.asarray(mask, dtype=bool)
    _validate_state_matrix(matrix)
    if len(flags) != len(matrix):
        raise ValueError("mask length must match states")
    indices = np.flatnonzero(flags)
    if len(indices) == 0:
        raise ValueError("constraint mask selects no states")
    return DangerousRegion(
        method="engineering_constraints",
        references=np.asarray(matrix[indices], dtype=np.float64),
        provenance=tuple(provenance[int(index)] for index in indices),
    )


def dangerous_region_from_density_level(
    states: FloatArray,
    *,
    quantile: float = 0.98,
) -> DangerousRegion:
    """Define dangerous references as robust-distance outliers in training state space."""

    matrix = np.asarray(states, dtype=np.float64)
    _validate_state_matrix(matrix)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0)
    scale = np.where(scale <= 0.0, 1.0, scale)
    robust_distance = np.linalg.norm((matrix - center) / scale, axis=1)
    threshold = float(np.quantile(robust_distance, quantile))
    indices = np.flatnonzero(robust_distance >= threshold)
    provenance = tuple(
        StateProvenance(index=int(index), source="density_level_set", split="train")
        for index in indices
    )
    return DangerousRegion(
        method="density_level_set",
        references=np.asarray(matrix[indices], dtype=np.float64),
        provenance=provenance,
        metadata={"robust_distance_threshold": threshold},
    )


def score_dangerous_region(
    states: FloatArray,
    region: DangerousRegion,
    *,
    epsilon: float = 1e-12,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Score distance, observable and diagnostic flags against a dangerous region."""

    if region.metric != "euclidean":
        raise ValueError("only euclidean dangerous-region scoring is currently supported")
    report = distance_observable_report(
        states,
        region.references,
        epsilon=epsilon,
        chunk_size=chunk_size,
    )
    distance = np.asarray(report["distance"], dtype=np.float64)
    observable = np.asarray(report["observable"], dtype=np.float64)
    return pd.DataFrame(
        {
            "distance_to_dangerous_region": distance,
            "dangerous_region_observable": observable,
            "exact_reference_duplicate": distance <= epsilon,
            "small_metric_scale": distance <= 1e-6,
        }
    )


def horizon_entry_outcomes(hits: BoolArray, *, horizon: int) -> BoolArray:
    """Return whether a hit occurs strictly after each index within a horizon."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    flags = np.asarray(hits, dtype=bool)
    outcomes = np.zeros(len(flags), dtype=bool)
    cumulative = np.concatenate(([0], np.cumsum(flags.astype(np.int64))))
    for index in range(len(flags)):
        stop = min(len(flags), index + horizon + 1)
        outcomes[index] = (cumulative[stop] - cumulative[index + 1]) > 0
    return outcomes


def estimate_horizon_risk(
    distances: FloatArray,
    *,
    hit_radius: float,
    horizons: tuple[int, ...],
) -> HorizonRiskReport:
    """Estimate empirical probability of entering the region within horizons."""

    if hit_radius < 0.0:
        raise ValueError("hit_radius must be non-negative")
    distance_array = np.asarray(distances, dtype=np.float64)
    hits = distance_array <= hit_radius
    probabilities: dict[int, float] = {}
    brier_scores: dict[int, float] = {}
    calibration: dict[int, list[dict[str, object]]] = {}
    for horizon in horizons:
        outcomes = horizon_entry_outcomes(hits, horizon=horizon)
        risk = _rolling_hit_rate(hits, horizon=horizon)
        probabilities[horizon] = float(np.mean(outcomes))
        brier_scores[horizon] = brier_score(risk, outcomes)
        calibration[horizon] = [asdict(item) for item in reliability_bins(risk, outcomes, n_bins=5)]
    return HorizonRiskReport(
        horizons=horizons,
        probabilities=probabilities,
        brier_scores=brier_scores,
        calibration_bins=calibration,
    )


def mahalanobis_transform(states: FloatArray, *, regularization: float = 1e-6) -> FloatArray:
    """Whiten states using a training covariance estimate before Euclidean scoring."""

    matrix = np.asarray(states, dtype=np.float64)
    _validate_state_matrix(matrix)
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    covariance = np.cov(matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * regularization
    inverse = np.linalg.inv(covariance)
    center = np.mean(matrix, axis=0)
    return np.asarray((matrix - center) @ np.linalg.cholesky(inverse).T, dtype=np.float64)


def nearest_medoid_references(states: FloatArray, *, n_medoids: int) -> FloatArray:
    """Select deterministic medoid-like references by farthest-first traversal."""

    matrix = np.asarray(states, dtype=np.float64)
    _validate_state_matrix(matrix)
    if n_medoids < 1:
        raise ValueError("n_medoids must be positive")
    if n_medoids >= len(matrix):
        return matrix.copy()
    selected = [0]
    min_distances = cdist(matrix, matrix[[0]], metric="euclidean").ravel()
    while len(selected) < n_medoids:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        min_distances = np.minimum(
            min_distances,
            cdist(matrix, matrix[[next_index]], metric="euclidean").ravel(),
        )
    return np.asarray(matrix[selected], dtype=np.float64)


def _rolling_hit_rate(hits: BoolArray, *, horizon: int) -> FloatArray:
    flags = np.asarray(hits, dtype=bool)
    rates = np.zeros(len(flags), dtype=np.float64)
    cumulative = np.concatenate(([0], np.cumsum(flags.astype(np.int64))))
    for index in range(len(flags)):
        stop = min(len(flags), index + horizon + 1)
        denominator = max(1, stop - index - 1)
        rates[index] = (cumulative[stop] - cumulative[index + 1]) / denominator
    return rates


def _select_spread_prototypes(
    references: FloatArray,
    provenance: list[StateProvenance],
    *,
    max_references: int,
) -> tuple[FloatArray, list[StateProvenance]]:
    if max_references < 1:
        raise ValueError("max_references must be positive")
    if len(references) <= max_references:
        return references, provenance
    medoids = nearest_medoid_references(references, n_medoids=max_references)
    selected: list[int] = []
    for medoid in medoids:
        distances = np.linalg.norm(references - medoid, axis=1)
        selected.append(int(np.argmin(distances)))
    unique = list(dict.fromkeys(selected))
    return np.asarray(references[unique], dtype=np.float64), [provenance[index] for index in unique]


def _assert_allowed_provenance(
    provenance: list[StateProvenance],
    *,
    allowed_failure_ids: set[str],
) -> None:
    forbidden = [
        item
        for item in provenance
        if item.split != "train"
        or item.failure_id is None
        or item.failure_id not in allowed_failure_ids
    ]
    if forbidden:
        raise ValueError(f"dangerous-region provenance contains held-out states: {forbidden}")


def _validate_state_matrix(matrix: FloatArray) -> None:
    if matrix.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if len(matrix) == 0:
        raise ValueError("states must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("states contain non-finite values")
