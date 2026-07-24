"""Finite-sample diagnostics for multivariate and lagged extremes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.thresholds import (
    RegimeThresholds,
    apply_regime_thresholds,
    fit_regime_thresholds,
)
from dyn_evt_pdm.types import BoolArray, EventInterval, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class LaggedExtremePattern:
    """Ordered component exceedances with non-negative relative lags."""

    components: tuple[int, ...]
    lags: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("at least one component is required")
        if len(self.components) != len(self.lags):
            raise ValueError("components and lags must have equal length")
        if any(component < 0 for component in self.components):
            raise ValueError("component indices must be non-negative")
        if any(lag < 0 for lag in self.lags):
            raise ValueError("lags must be non-negative")
        if tuple(sorted(self.lags)) != self.lags:
            raise ValueError("lags must be non-decreasing")

    @property
    def window(self) -> int:
        """Number of samples covered by the lagged pattern window."""

        return max(self.lags) + 1

    def label(self, component_names: tuple[str, ...] | None = None) -> str:
        """Return a compact, deterministic pattern label."""

        parts: list[str] = []
        for component, lag in zip(self.components, self.lags, strict=True):
            name = component_names[component] if component_names is not None else f"c{component}"
            parts.append(f"{name}@+{lag}")
        return " -> ".join(parts)


@dataclass(frozen=True, slots=True)
class ComponentRegimeThresholds:
    """Component-specific thresholds conditioned on operating regime."""

    components: tuple[str, ...]
    quantile: float
    thresholds: dict[str, RegimeThresholds]


@dataclass(frozen=True, slots=True)
class LaggedPatternCluster:
    """One overlap-aware cluster of lagged-pattern detections."""

    interval: EventInterval
    pattern_start_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.pattern_start_indices)


@dataclass(frozen=True, slots=True)
class LaggedPatternReport:
    """Finite-sample evidence for one lagged pattern."""

    pattern: LaggedExtremePattern
    count: int
    cluster_count: int
    empirical_extremal_index: float
    null_model: str
    null_counts: tuple[int, ...]
    p_value: float
    adjusted_p_value: float
    split: str
    exploratory: bool

    def to_dict(self, component_names: tuple[str, ...] | None = None) -> dict[str, object]:
        return {
            "pattern": self.pattern.label(component_names),
            "components": self.pattern.components,
            "lags": self.pattern.lags,
            "count": self.count,
            "cluster_count": self.cluster_count,
            "empirical_extremal_index": self.empirical_extremal_index,
            "null_model": self.null_model,
            "null_counts": self.null_counts,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "split": self.split,
            "exploratory": self.exploratory,
        }


@dataclass(frozen=True, slots=True)
class NestedLagSelectionResult:
    """Lag patterns selected on validation and optionally scored on held-out data."""

    validation_reports: tuple[LaggedPatternReport, ...]
    selected_reports: tuple[LaggedPatternReport, ...]
    heldout_reports: tuple[LaggedPatternReport, ...]
    selected_indices: tuple[int, ...]


def detect_lagged_pattern(
    exceedances: BoolArray,
    pattern: LaggedExtremePattern,
) -> BoolArray:
    """Detect an exact lagged logical pattern in a component-by-time matrix.

    A detection at time ``t`` means every requested component exceeds at
    ``t + lag``. The padded tail is always false where the full pattern window is
    unavailable.
    """

    matrix = np.asarray(exceedances, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("exceedances must be a two-dimensional matrix")
    if max(pattern.components) >= matrix.shape[1]:
        raise ValueError("pattern references an unavailable component")

    max_lag = max(pattern.lags)
    valid_length = matrix.shape[0] - max_lag
    if valid_length <= 0:
        return np.zeros(matrix.shape[0], dtype=bool)

    result = np.ones(valid_length, dtype=bool)
    for component, lag in zip(pattern.components, pattern.lags, strict=True):
        result &= matrix[lag : lag + valid_length, component]
    return np.pad(result, (0, max_lag), constant_values=False)


def fit_component_regime_thresholds(
    frame: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    regime_column: str,
    quantile: float,
    train_mask: BoolArray | None = None,
    min_regime_samples: int = 100,
) -> ComponentRegimeThresholds:
    """Fit component-wise regime thresholds from training rows only."""

    _validate_columns(frame, (*component_columns, regime_column))
    mask: BoolArray
    if train_mask is None:
        mask = np.asarray(np.ones(len(frame), dtype=bool), dtype=bool)
    else:
        mask = _checked_mask(train_mask, len(frame), "train_mask")
    training = frame.loc[mask]
    thresholds = {
        column: fit_regime_thresholds(
            training[column],
            training[regime_column],
            quantile=quantile,
            min_regime_samples=min_regime_samples,
        )
        for column in component_columns
    }
    return ComponentRegimeThresholds(
        components=component_columns,
        quantile=quantile,
        thresholds=thresholds,
    )


def apply_component_regime_thresholds(
    frame: pd.DataFrame,
    *,
    regime_column: str,
    thresholds: ComponentRegimeThresholds,
) -> pd.DataFrame:
    """Apply fitted component-specific regime thresholds to all rows."""

    _validate_columns(frame, (*thresholds.components, regime_column))
    columns: dict[str, pd.Series] = {}
    for component in thresholds.components:
        columns[component] = apply_regime_thresholds(
            frame[component],
            frame[regime_column],
            thresholds.thresholds[component],
        )
    return pd.DataFrame(columns, index=frame.index)


def simultaneous_exceedance_vectors(exceedances: BoolArray) -> IntArray:
    """Return each row's simultaneous exceedance vector as a 0/1 matrix."""

    matrix = np.asarray(exceedances, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("exceedances must be two-dimensional")
    return matrix.astype(np.int64)


def simultaneous_vector_counts(
    exceedances: BoolArray,
    *,
    component_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Count observed simultaneous exceedance combinations."""

    vectors = simultaneous_exceedance_vectors(exceedances)
    if component_names is not None and len(component_names) != vectors.shape[1]:
        raise ValueError("component_names length must match exceedance columns")
    counts: dict[str, int] = {}
    for row in vectors:
        active = tuple(int(index) for index in np.flatnonzero(row))
        if not active:
            label = "none"
        elif component_names is None:
            label = "+".join(f"c{index}" for index in active)
        else:
            label = "+".join(component_names[index] for index in active)
        counts[label] = counts.get(label, 0) + 1
    return counts


def empirical_stable_tail_dependence(
    exceedances: BoolArray,
    *,
    component_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Estimate pairwise empirical tail dependence from exceedance indicators.

    The returned ``chi_hat_i_given_j`` is the finite-sample conditional
    probability ``P(i exceeds | j exceeds)``. It is labelled empirical because no
    asymptotic stable-tail limit is asserted here.
    """

    matrix = _checked_exceedance_matrix(exceedances)
    if component_names is not None and len(component_names) != matrix.shape[1]:
        raise ValueError("component_names length must match exceedance columns")
    result: dict[str, dict[str, float | int]] = {}
    for left, right in combinations(range(matrix.shape[1]), 2):
        both = int(np.logical_and(matrix[:, left], matrix[:, right]).sum())
        left_count = int(matrix[:, left].sum())
        right_count = int(matrix[:, right].sum())
        left_name = component_names[left] if component_names else f"c{left}"
        right_name = component_names[right] if component_names else f"c{right}"
        key = f"{left_name}|{right_name}"
        result[key] = {
            "joint_count": both,
            "left_count": left_count,
            "right_count": right_count,
            "chi_hat_left_given_right": both / right_count if right_count else 0.0,
            "chi_hat_right_given_left": both / left_count if left_count else 0.0,
        }
    return result


def construct_lagged_pattern_clusters(
    detections: BoolArray,
    pattern: LaggedExtremePattern,
    *,
    run_length: int = 0,
) -> list[LaggedPatternCluster]:
    """Cluster lagged detections whose pattern windows overlap or nearly touch."""

    flags = np.asarray(detections, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("detections must be one-dimensional")
    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    starts = np.flatnonzero(flags)
    if len(starts) == 0:
        return []

    grouped: list[list[int]] = [[int(starts[0])]]
    previous_end = int(starts[0] + pattern.window - 1)
    for start_value in starts[1:]:
        start = int(start_value)
        gap = start - previous_end - 1
        if gap <= run_length:
            grouped[-1].append(start)
        else:
            grouped.append([start])
        previous_end = max(previous_end, start + pattern.window - 1)

    return [
        LaggedPatternCluster(
            interval=EventInterval(
                start=group[0],
                end=group[-1] + pattern.window - 1,
                label="lagged_pattern_cluster",
            ),
            pattern_start_indices=tuple(group),
        )
        for group in grouped
    ]


def empirical_extremal_index_analogue(
    detections: BoolArray,
    pattern: LaggedExtremePattern,
    *,
    run_length: int = 0,
) -> float:
    """Return clusters / detections for a lagged pattern as an empirical analogue."""

    count = int(np.asarray(detections, dtype=bool).sum())
    if count == 0:
        return 0.0
    clusters = construct_lagged_pattern_clusters(detections, pattern, run_length=run_length)
    return float(len(clusters) / count)


def multivariate_extremal_index_surface(
    exceedances: BoolArray,
    *,
    max_lag: int,
    run_length: int = 0,
) -> pd.DataFrame:
    """Compute a pairwise empirical extremal-index analogue over lags."""

    matrix = _checked_exceedance_matrix(exceedances)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    rows: list[dict[str, float | int]] = []
    for source in range(matrix.shape[1]):
        for target in range(matrix.shape[1]):
            if source == target:
                continue
            for lag in range(max_lag + 1):
                pattern = LaggedExtremePattern(components=(source, target), lags=(0, lag))
                detections = detect_lagged_pattern(matrix, pattern)
                rows.append(
                    {
                        "source_component": source,
                        "target_component": target,
                        "lag": lag,
                        "count": int(detections.sum()),
                        "cluster_count": len(
                            construct_lagged_pattern_clusters(
                                detections,
                                pattern,
                                run_length=run_length,
                            )
                        ),
                        "empirical_extremal_index": empirical_extremal_index_analogue(
                            detections,
                            pattern,
                            run_length=run_length,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def candidate_pair_lag_patterns(
    *,
    n_components: int,
    max_lag: int,
    include_simultaneous: bool = True,
) -> tuple[LaggedExtremePattern, ...]:
    """Enumerate ordered pair patterns for nested lag selection."""

    if n_components < 2:
        raise ValueError("at least two components are required")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    start_lag = 0 if include_simultaneous else 1
    patterns: list[LaggedExtremePattern] = []
    for source in range(n_components):
        for target in range(n_components):
            if source == target:
                continue
            for lag in range(start_lag, max_lag + 1):
                patterns.append(LaggedExtremePattern(components=(source, target), lags=(0, lag)))
    return tuple(patterns)


def evaluate_lagged_patterns(
    exceedances: BoolArray,
    patterns: tuple[LaggedExtremePattern, ...],
    *,
    mask: BoolArray | None = None,
    n_null: int = 199,
    null_model: str = "time_shift",
    random_state: int = 0,
    run_length: int = 0,
    split: str = "validation",
    exploratory: bool = False,
) -> tuple[LaggedPatternReport, ...]:
    """Evaluate patterns against clustering-preserving finite-sample null models."""

    matrix = _checked_exceedance_matrix(exceedances)
    selected: BoolArray
    if mask is None:
        selected = np.asarray(np.ones(matrix.shape[0], dtype=bool), dtype=bool)
    else:
        selected = _checked_mask(mask, matrix.shape[0], "mask")
    if n_null < 0:
        raise ValueError("n_null must be non-negative")
    rng = np.random.default_rng(random_state)
    observed_counts = [_masked_pattern_count(matrix, pattern, selected) for pattern in patterns]
    null_counts_by_pattern: list[list[int]] = [[] for _ in patterns]
    for _ in range(n_null):
        null_matrix = generate_null_exceedances(matrix, method=null_model, rng=rng)
        for index, pattern in enumerate(patterns):
            null_counts_by_pattern[index].append(
                _masked_pattern_count(null_matrix, pattern, selected)
            )

    p_values = tuple(
        _right_tail_p_value(observed, tuple(null_counts))
        for observed, null_counts in zip(observed_counts, null_counts_by_pattern, strict=True)
    )
    adjusted = benjamini_hochberg(p_values)
    reports: list[LaggedPatternReport] = []
    for index, pattern in enumerate(patterns):
        detections = detect_lagged_pattern(matrix, pattern) & selected
        clusters = construct_lagged_pattern_clusters(detections, pattern, run_length=run_length)
        reports.append(
            LaggedPatternReport(
                pattern=pattern,
                count=observed_counts[index],
                cluster_count=len(clusters),
                empirical_extremal_index=empirical_extremal_index_analogue(
                    detections,
                    pattern,
                    run_length=run_length,
                ),
                null_model=null_model,
                null_counts=tuple(int(value) for value in null_counts_by_pattern[index]),
                p_value=p_values[index],
                adjusted_p_value=adjusted[index],
                split=split,
                exploratory=exploratory,
            )
        )
    return tuple(reports)


def select_lagged_patterns_nested(
    exceedances: BoolArray,
    patterns: tuple[LaggedExtremePattern, ...],
    *,
    validation_mask: BoolArray,
    heldout_mask: BoolArray | None = None,
    max_selected: int = 5,
    alpha: float = 0.1,
    n_null: int = 199,
    null_model: str = "time_shift",
    random_state: int = 0,
    run_length: int = 0,
) -> NestedLagSelectionResult:
    """Select lag patterns on validation and label held-out rediscovery honestly."""

    if max_selected < 1:
        raise ValueError("max_selected must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    validation_reports = evaluate_lagged_patterns(
        exceedances,
        patterns,
        mask=validation_mask,
        n_null=n_null,
        null_model=null_model,
        random_state=random_state,
        run_length=run_length,
        split="validation",
        exploratory=False,
    )
    ordered = sorted(
        range(len(validation_reports)),
        key=lambda index: (
            validation_reports[index].adjusted_p_value,
            -validation_reports[index].count,
            validation_reports[index].pattern.lags,
            validation_reports[index].pattern.components,
        ),
    )
    selected_indices = tuple(
        index
        for index in ordered
        if validation_reports[index].adjusted_p_value <= alpha
        and validation_reports[index].count > 0
    )[:max_selected]
    selected_reports = tuple(validation_reports[index] for index in selected_indices)
    if heldout_mask is None or not selected_indices:
        heldout_reports: tuple[LaggedPatternReport, ...] = ()
    else:
        selected_patterns = tuple(patterns[index] for index in selected_indices)
        heldout_reports = evaluate_lagged_patterns(
            exceedances,
            selected_patterns,
            mask=heldout_mask,
            n_null=n_null,
            null_model=null_model,
            random_state=random_state + 1,
            run_length=run_length,
            split="heldout",
            exploratory=True,
        )
    return NestedLagSelectionResult(
        validation_reports=validation_reports,
        selected_reports=selected_reports,
        heldout_reports=heldout_reports,
        selected_indices=selected_indices,
    )


def generate_null_exceedances(
    exceedances: BoolArray,
    *,
    method: str,
    rng: np.random.Generator,
) -> BoolArray:
    """Generate null exceedance matrices while preserving marginal clustering."""

    matrix = _checked_exceedance_matrix(exceedances)
    if method == "time_shift":
        return _time_shift_null(matrix, rng)
    if method == "cluster_permutation":
        return _cluster_permutation_null(matrix, rng)
    raise ValueError("method must be time_shift or cluster_permutation")


def benjamini_hochberg(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Benjamini-Hochberg adjusted p-values in original order."""

    if not p_values:
        return ()
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_sorted = np.empty_like(ranked)
    running = 1.0
    total = len(values)
    for reverse_rank in range(total - 1, -1, -1):
        candidate = ranked[reverse_rank] * total / (reverse_rank + 1)
        running = min(running, candidate)
        adjusted_sorted[reverse_rank] = running
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return tuple(float(value) for value in adjusted)


def multivariate_evt_report(
    frame: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    regime_column: str,
    train_mask: BoolArray,
    validation_mask: BoolArray,
    heldout_mask: BoolArray | None,
    quantile: float = 0.98,
    max_lag: int = 10,
    n_null: int = 199,
    null_model: str = "time_shift",
    alpha: float = 0.1,
    min_regime_samples: int = 100,
    random_state: int = 0,
) -> dict[str, object]:
    """Fit thresholds on train rows and report multivariate lag diagnostics."""

    thresholds = fit_component_regime_thresholds(
        frame,
        component_columns=component_columns,
        regime_column=regime_column,
        quantile=quantile,
        train_mask=train_mask,
        min_regime_samples=min_regime_samples,
    )
    exceedance_frame = apply_component_regime_thresholds(
        frame,
        regime_column=regime_column,
        thresholds=thresholds,
    )
    exceedances = exceedance_frame.to_numpy(dtype=bool)
    patterns = candidate_pair_lag_patterns(
        n_components=len(component_columns),
        max_lag=max_lag,
        include_simultaneous=True,
    )
    selection = select_lagged_patterns_nested(
        exceedances,
        patterns,
        validation_mask=validation_mask,
        heldout_mask=heldout_mask,
        alpha=alpha,
        n_null=n_null,
        null_model=null_model,
        random_state=random_state,
    )
    component_names = tuple(component_columns)
    return {
        "component_columns": component_columns,
        "regime_column": regime_column,
        "threshold_quantile": quantile,
        "thresholds": {
            component: {
                "fallback": thresholds.thresholds[component].fallback,
                "regime_values": thresholds.thresholds[component].values,
            }
            for component in component_columns
        },
        "simultaneous_vectors": simultaneous_vector_counts(
            exceedances,
            component_names=component_names,
        ),
        "empirical_tail_dependence": empirical_stable_tail_dependence(
            exceedances,
            component_names=component_names,
        ),
        "extremal_index_surface": multivariate_extremal_index_surface(
            exceedances,
            max_lag=max_lag,
        ).to_dict(orient="records"),
        "validation_patterns": [
            report.to_dict(component_names) for report in selection.validation_reports
        ],
        "selected_patterns": [
            report.to_dict(component_names) for report in selection.selected_reports
        ],
        "heldout_patterns": [
            report.to_dict(component_names) for report in selection.heldout_reports
        ],
        "multiple_comparison_control": {
            "method": "benjamini_hochberg",
            "alpha": alpha,
        },
        "null_model": null_model,
        "heldout_policy": (
            "Patterns are selected on validation only; held-out reports are marked "
            "exploratory when the same held-out event influenced discovery."
        ),
    }


def _checked_exceedance_matrix(exceedances: BoolArray) -> BoolArray:
    matrix = np.asarray(exceedances, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("exceedances must be two-dimensional")
    return matrix


def _checked_mask(mask: BoolArray, expected_length: int, name: str) -> BoolArray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(f"{name} must be one-dimensional with length {expected_length}")
    return values


def _validate_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _masked_pattern_count(
    exceedances: BoolArray,
    pattern: LaggedExtremePattern,
    mask: BoolArray,
) -> int:
    detections = detect_lagged_pattern(exceedances, pattern)
    valid_mask = _window_contained_mask(mask, pattern.window)
    return int(np.logical_and(detections, valid_mask).sum())


def _window_contained_mask(mask: BoolArray, window: int) -> BoolArray:
    if window <= 1:
        return mask.copy()
    result = np.zeros_like(mask, dtype=bool)
    valid_length = len(mask) - window + 1
    if valid_length <= 0:
        return result
    contained = np.ones(valid_length, dtype=bool)
    for offset in range(window):
        contained &= mask[offset : offset + valid_length]
    result[:valid_length] = contained
    return result


def _right_tail_p_value(observed: int, null_counts: tuple[int, ...]) -> float:
    if not null_counts:
        return 1.0
    null = np.asarray(null_counts, dtype=np.int64)
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _time_shift_null(matrix: BoolArray, rng: np.random.Generator) -> BoolArray:
    shifted = np.empty_like(matrix, dtype=bool)
    n_rows = matrix.shape[0]
    for column in range(matrix.shape[1]):
        offset = int(rng.integers(0, n_rows)) if n_rows else 0
        shifted[:, column] = np.roll(matrix[:, column], offset)
    return shifted


def _cluster_permutation_null(matrix: BoolArray, rng: np.random.Generator) -> BoolArray:
    result = np.zeros_like(matrix, dtype=bool)
    n_rows = matrix.shape[0]
    for column in range(matrix.shape[1]):
        clusters = extract_clusters(matrix[:, column], run_length=0)
        if not clusters:
            continue
        lengths = np.asarray([cluster.interval.duration for cluster in clusters], dtype=np.int64)
        shuffled_lengths = rng.permutation(lengths)
        starts = _random_non_overlapping_starts(
            n_rows=n_rows,
            lengths=shuffled_lengths,
            rng=rng,
        )
        for start, length in zip(starts, shuffled_lengths, strict=True):
            result[start : start + int(length), column] = True
    return result


def _random_non_overlapping_starts(
    *,
    n_rows: int,
    lengths: IntArray,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    total_extreme = int(np.sum(lengths))
    if total_extreme > n_rows:
        raise ValueError("cluster lengths cannot exceed series length")
    n_gaps = len(lengths) + 1
    gap_total = n_rows - total_extreme
    gap_weights = rng.multinomial(gap_total, np.full(n_gaps, 1.0 / n_gaps))
    starts: list[int] = []
    cursor = int(gap_weights[0])
    for index, length in enumerate(lengths):
        starts.append(cursor)
        cursor += int(length) + int(gap_weights[index + 1])
    return tuple(starts)


def component_exceedance_rates(exceedances: BoolArray) -> FloatArray:
    """Return component-wise exceedance rates."""

    matrix = _checked_exceedance_matrix(exceedances)
    return cast(FloatArray, np.mean(matrix, axis=0).astype(np.float64))
