"""Genuine cross-dataset transfer of target regions between compressor datasets.

A transfer experiment applies a region fitted on one dataset to another. Reporting the
same dataset's own metrics under a different label is a projection, not a transfer, so
this module fits every component on the source and freezes it before the destination is
touched.

Four protocols are distinguished and never conflated:

``direct``
    Scaling, prototypes, threshold and distance metric all frozen from the source.
``recalibrated_threshold``
    Frozen region, threshold re-fitted on the destination's validation split only.
``recalibrated_scaling``
    Frozen prototypes, scaling re-fitted on the destination's training split only.
``destination_refit``
    Everything fitted on the destination; the within-dataset reference point.

Transfer degradation is measured against ``destination_refit``, so a degraded transfer
is visible as a number rather than as an absence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from dyn_evt_pdm.types import FloatArray

#: Protocols in the order they are reported.
TRANSFER_PROTOCOLS: tuple[str, ...] = (
    "direct",
    "recalibrated_threshold",
    "recalibrated_scaling",
    "destination_refit",
)

#: Largest tolerable standardised median shift before a feature is called incompatible.
MAX_MEDIAN_SHIFT = 3.0

#: Smallest tolerable overlap between source and destination feature distributions.
MIN_DISTRIBUTION_OVERLAP = 0.3

#: Largest tolerable relative difference in sampling cadence.
MAX_CADENCE_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class FeatureCompatibility:
    """Compatibility verdict for one candidate transfer feature."""

    feature: str
    present_in_both: bool
    source_median: float
    destination_median: float
    standardised_shift: float
    distribution_overlap: float
    source_missing: float
    destination_missing: float
    compatible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SchemaCompatibilityReport:
    """Whether two datasets can support a transfer at all, and on which features."""

    source: str
    destination: str
    features: tuple[FeatureCompatibility, ...]
    source_cadence_seconds: float
    destination_cadence_seconds: float
    cadence_compatible: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def compatible_features(self) -> tuple[str, ...]:
        """Return the features that may be transferred."""

        return tuple(item.feature for item in self.features if item.compatible)

    @property
    def transferable(self) -> bool:
        """Return whether any transfer is admissible."""

        return bool(self.compatible_features) and self.cadence_compatible

    def to_frame(self) -> pd.DataFrame:
        """Return the per-feature verdicts as a table."""

        return pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "present_in_both": item.present_in_both,
                    "source_median": item.source_median,
                    "destination_median": item.destination_median,
                    "standardised_shift": item.standardised_shift,
                    "distribution_overlap": item.distribution_overlap,
                    "source_missing": item.source_missing,
                    "destination_missing": item.destination_missing,
                    "compatible": item.compatible,
                    "reason": item.reason,
                }
                for item in self.features
            ]
        )


@dataclass(frozen=True, slots=True)
class TargetRegion:
    """A target region fitted on one dataset, expressed so it can cross to another.

    Prototypes are stored in raw feature units rather than standardised units. If they
    were stored standardised, recalibrating the scaling on the destination would leave
    the prototypes silently anchored to the source's spread, and the recalibrated
    protocol would not be a recalibration at all.
    """

    features: tuple[str, ...]
    prototypes_raw: FloatArray
    center: pd.Series
    scale: pd.Series
    threshold: float
    fitted_on: str

    def scaled_prototypes(self, center: pd.Series, scale: pd.Series) -> FloatArray:
        """Express the prototypes in the standardised space defined by ``center``."""

        columns = list(self.features)
        centered = self.prototypes_raw - center.loc[columns].to_numpy(dtype=float)
        return np.asarray(centered / scale.loc[columns].to_numpy(dtype=float), dtype=float)


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Event-level outcome of applying one region to one destination dataset."""

    source: str
    destination: str
    protocol: str
    features: tuple[str, ...]
    scaling_fitted_on: str
    prototypes_fitted_on: str
    threshold_fitted_on: str
    threshold: float
    region_occupancy: float
    failure_state_coverage: float
    event_recall: float
    event_precision: float
    false_alarm_events_per_day: float
    median_warning_lead_time: float | None
    time_under_warning: int
    alarm_coverage_fraction: float
    predicted_alarm_events: int
    distance_shift: float
    score_distribution_shift: float
    status: str
    note: str = ""


def _robust_center_scale(
    frame: pd.DataFrame, features: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """Return a median centre and interquartile scale over the given features."""

    numeric = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    center = numeric.median()
    spread = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0.0, np.nan).fillna(1.0)
    return center, spread


def _standardise(
    frame: pd.DataFrame, features: tuple[str, ...], *, center: pd.Series, scale: pd.Series
) -> FloatArray:
    numeric = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    standardised = (numeric - center.loc[list(features)]) / scale.loc[list(features)]
    cleaned = np.nan_to_num(standardised.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(cleaned, dtype=np.float64)


def _minimum_distance(values: FloatArray, references: FloatArray) -> FloatArray:
    """Return each row's distance to the nearest reference point."""

    if references.size == 0:
        return np.full(len(values), np.inf, dtype=float)
    best = np.full(len(values), np.inf, dtype=float)
    for reference in references:
        distance = np.sqrt(np.sum((values - reference) ** 2, axis=1))
        best = np.minimum(best, distance)
    return best


def fit_target_region(
    train_frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    dataset: str,
    failure_column: str = "is_failure",
    max_prototypes: int = 16,
    threshold_quantile: float = 0.98,
) -> TargetRegion | None:
    """Fit a failure-prototype region on training rows only.

    Returns ``None`` when the training split contains no failure, since a prototype
    region has nothing to be built from and reporting a fitted region would be false.
    """

    center, scale = _robust_center_scale(train_frame, features)
    labels = train_frame[failure_column].astype(bool).to_numpy()
    if not labels.any():
        return None

    raw = train_frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    failure_raw = raw.to_numpy(dtype=float)[labels]
    failure_raw = failure_raw[np.isfinite(failure_raw).all(axis=1)]
    if failure_raw.size == 0:
        return None
    if len(failure_raw) > max_prototypes:
        picks = np.linspace(0, len(failure_raw) - 1, max_prototypes, dtype=int)
        failure_raw = failure_raw[picks]

    standardised = _standardise(train_frame, features, center=center, scale=scale)
    prototypes_scaled = (
        failure_raw - center.loc[list(features)].to_numpy(dtype=float)
    ) / scale.loc[list(features)].to_numpy(dtype=float)
    distances = _minimum_distance(standardised, np.asarray(prototypes_scaled, dtype=float))
    finite = distances[np.isfinite(distances)]
    threshold = (
        float(np.quantile(finite, 1.0 - threshold_quantile)) if finite.size else float("nan")
    )

    return TargetRegion(
        features=features,
        prototypes_raw=np.asarray(failure_raw, dtype=float),
        center=center,
        scale=scale,
        threshold=threshold,
        fitted_on=dataset,
    )


def _histogram_overlap(source: FloatArray, destination: FloatArray, *, bins: int = 50) -> float:
    """Return the overlapping area of two empirical distributions, in ``[0, 1]``.

    Two features can share a name and a unit and still describe different operating
    conditions. Overlap measures whether the destination ever visits the region the
    source prototypes occupy; without it a frozen region can be vacuously inapplicable.
    """

    a = np.asarray(source, dtype=float)
    b = np.asarray(destination, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 1.0 if np.isclose(low, high) else 0.0
    edges = np.linspace(low, high, bins + 1)
    a_hist, _ = np.histogram(a, bins=edges, density=False)
    b_hist, _ = np.histogram(b, bins=edges, density=False)
    a_share = a_hist / max(1, a_hist.sum())
    b_share = b_hist / max(1, b_hist.sum())
    return float(np.sum(np.minimum(a_share, b_share)))


def _cadence_seconds(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> float:
    if timestamp_column not in frame.columns:
        return float("nan")
    stamps = pd.to_datetime(frame[timestamp_column], errors="coerce").dropna().sort_values()
    if len(stamps) < 2:
        return float("nan")
    deltas = stamps.diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    if positive.empty:
        return float("nan")
    return float(positive.median())


def _quantile_shift(source: FloatArray, destination: FloatArray) -> float:
    """Return the mean absolute difference between matched distribution quantiles.

    Reported so a degraded transfer can be attributed to distribution movement rather
    than left as an unexplained drop in performance.
    """

    a = np.asarray(source, dtype=float)
    b = np.asarray(destination, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    levels = np.linspace(0.05, 0.95, 19)
    return float(np.mean(np.abs(np.quantile(a, levels) - np.quantile(b, levels))))


def evaluate_transfer(
    region: TargetRegion,
    destination_test: pd.DataFrame,
    *,
    source: str,
    destination: str,
    protocol: str,
    center: pd.Series,
    scale: pd.Series,
    threshold: float,
    scaling_fitted_on: str,
    threshold_fitted_on: str,
    source_train_distances: FloatArray,
    samples_per_day: int,
    merge_gap: int,
    horizon: int,
    matching_tolerance_after: int,
    failure_column: str = "is_failure",
) -> TransferResult:
    """Apply a region to a destination test split and evaluate it at event level."""

    from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, flags_to_events
    from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions

    features = region.features
    standardised = _standardise(destination_test, features, center=center, scale=scale)
    prototypes = region.scaled_prototypes(center, scale)
    distances = _minimum_distance(standardised, prototypes)

    inside = np.asarray(distances <= threshold, dtype=bool)
    target_flags = destination_test[failure_column].astype(bool).to_numpy()
    occupancy = float(np.mean(inside)) if inside.size else float("nan")
    coverage = float(np.mean(inside[target_flags])) if target_flags.any() else float("nan")

    failures = flags_to_events(target_flags, label="failure")
    alarms = flags_to_events(inside, label="alarm", merge_gap=merge_gap)
    if not failures:
        return TransferResult(
            source=source,
            destination=destination,
            protocol=protocol,
            features=features,
            scaling_fitted_on=scaling_fitted_on,
            prototypes_fitted_on=region.fitted_on,
            threshold_fitted_on=threshold_fitted_on,
            threshold=threshold,
            region_occupancy=occupancy,
            failure_state_coverage=coverage,
            event_recall=float("nan"),
            event_precision=float("nan"),
            false_alarm_events_per_day=float("nan"),
            median_warning_lead_time=None,
            time_under_warning=0,
            alarm_coverage_fraction=float("nan"),
            predicted_alarm_events=len(alarms),
            distance_shift=float("nan"),
            score_distribution_shift=float("nan"),
            status="not_estimable",
            note="destination test split contains no labelled failure",
        )

    evaluation = evaluate_event_predictions(
        alarms,
        failures,
        policy=EarlyWarningPolicy(horizon=horizon, tolerance_after=matching_tolerance_after),
        method="optimal",
        total_operating_time=len(target_flags),
        samples_per_day=samples_per_day,
    )
    return TransferResult(
        source=source,
        destination=destination,
        protocol=protocol,
        features=features,
        scaling_fitted_on=scaling_fitted_on,
        prototypes_fitted_on=region.fitted_on,
        threshold_fitted_on=threshold_fitted_on,
        threshold=float(threshold),
        region_occupancy=occupancy,
        failure_state_coverage=coverage,
        event_recall=evaluation.recall,
        event_precision=evaluation.precision,
        false_alarm_events_per_day=float(evaluation.false_alarm_events_per_operating_day or 0.0),
        median_warning_lead_time=evaluation.median_warning_lead_time,
        time_under_warning=evaluation.time_under_warning,
        alarm_coverage_fraction=(
            float(evaluation.time_under_warning) / float(len(target_flags))
            if len(target_flags)
            else float("nan")
        ),
        predicted_alarm_events=len(alarms),
        distance_shift=_quantile_shift(source_train_distances, distances),
        score_distribution_shift=_quantile_shift(
            -source_train_distances.astype(float), -distances.astype(float)
        ),
        status="completed",
        note="",
    )


@dataclass(frozen=True, slots=True)
class TransferSplits:
    """Train, validation and test splits for one dataset in a transfer experiment."""

    dataset: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def run_transfer_protocols(
    source: TransferSplits,
    destination: TransferSplits,
    features: tuple[str, ...],
    *,
    samples_per_day: int = 86_400,
    merge_gap: int = 60,
    horizon: int = 3_600,
    matching_tolerance_after: int = 0,
    threshold_quantile: float = 0.98,
    failure_column: str = "is_failure",
) -> list[TransferResult]:
    """Run every declared protocol for one source and destination pair.

    Each protocol states which component was frozen and which was refitted, so a reader
    can tell a genuine transfer from a destination refit without inspecting the code.
    """

    source_region = fit_target_region(
        source.train,
        features,
        dataset=source.dataset,
        failure_column=failure_column,
        threshold_quantile=threshold_quantile,
    )
    results: list[TransferResult] = []
    if source_region is None:
        return [
            TransferResult(
                source=source.dataset,
                destination=destination.dataset,
                protocol="direct",
                features=features,
                scaling_fitted_on=source.dataset,
                prototypes_fitted_on=source.dataset,
                threshold_fitted_on=source.dataset,
                threshold=float("nan"),
                region_occupancy=float("nan"),
                failure_state_coverage=float("nan"),
                event_recall=float("nan"),
                event_precision=float("nan"),
                false_alarm_events_per_day=float("nan"),
                median_warning_lead_time=None,
                time_under_warning=0,
                alarm_coverage_fraction=float("nan"),
                predicted_alarm_events=0,
                distance_shift=float("nan"),
                score_distribution_shift=float("nan"),
                status="not_estimable",
                note="source training split contains no labelled failure to build prototypes from",
            )
        ]

    source_train_standardised = _standardise(
        source.train, features, center=source_region.center, scale=source_region.scale
    )
    source_train_distances = _minimum_distance(
        source_train_standardised,
        source_region.scaled_prototypes(source_region.center, source_region.scale),
    )

    def _run(
        region: TargetRegion,
        *,
        protocol: str,
        center: pd.Series,
        scale: pd.Series,
        threshold: float,
        scaling_fitted_on: str,
        threshold_fitted_on: str,
        source_name: str | None = None,
    ) -> TransferResult:
        return evaluate_transfer(
            region,
            destination.test,
            source=source_name or source.dataset,
            destination=destination.dataset,
            protocol=protocol,
            center=center,
            scale=scale,
            threshold=threshold,
            scaling_fitted_on=scaling_fitted_on,
            threshold_fitted_on=threshold_fitted_on,
            source_train_distances=source_train_distances,
            samples_per_day=samples_per_day,
            merge_gap=merge_gap,
            horizon=horizon,
            matching_tolerance_after=matching_tolerance_after,
            failure_column=failure_column,
        )

    # 1. Direct: nothing about the destination is used to fit anything.
    results.append(
        _run(
            source_region,
            protocol="direct",
            center=source_region.center,
            scale=source_region.scale,
            threshold=source_region.threshold,
            scaling_fitted_on=source.dataset,
            threshold_fitted_on=source.dataset,
        )
    )

    # 2. Threshold recalibrated on the destination validation split only.
    validation_standardised = _standardise(
        destination.validation, features, center=source_region.center, scale=source_region.scale
    )
    validation_distances = _minimum_distance(
        validation_standardised,
        source_region.scaled_prototypes(source_region.center, source_region.scale),
    )
    finite_validation = validation_distances[np.isfinite(validation_distances)]
    recalibrated_threshold = (
        float(np.quantile(finite_validation, 1.0 - threshold_quantile))
        if finite_validation.size
        else source_region.threshold
    )
    results.append(
        _run(
            source_region,
            protocol="recalibrated_threshold",
            center=source_region.center,
            scale=source_region.scale,
            threshold=recalibrated_threshold,
            scaling_fitted_on=source.dataset,
            threshold_fitted_on=f"{destination.dataset} validation",
        )
    )

    # 3. Scaling recalibrated on the destination training split, prototypes still the
    #    source's. This is why prototypes are carried in raw units.
    destination_center, destination_scale = _robust_center_scale(destination.train, features)
    destination_train_standardised = _standardise(
        destination.train, features, center=destination_center, scale=destination_scale
    )
    rescaled_distances = _minimum_distance(
        destination_train_standardised,
        source_region.scaled_prototypes(destination_center, destination_scale),
    )
    finite_rescaled = rescaled_distances[np.isfinite(rescaled_distances)]
    rescaled_threshold = (
        float(np.quantile(finite_rescaled, 1.0 - threshold_quantile))
        if finite_rescaled.size
        else source_region.threshold
    )
    results.append(
        _run(
            source_region,
            protocol="recalibrated_scaling",
            center=destination_center,
            scale=destination_scale,
            threshold=rescaled_threshold,
            scaling_fitted_on=f"{destination.dataset} train",
            threshold_fitted_on=f"{destination.dataset} train",
        )
    )

    # 4. Destination refit: the within-dataset reference the transfers are judged against.
    destination_region = fit_target_region(
        destination.train,
        features,
        dataset=destination.dataset,
        failure_column=failure_column,
        threshold_quantile=threshold_quantile,
    )
    if destination_region is not None:
        results.append(
            _run(
                destination_region,
                protocol="destination_refit",
                center=destination_region.center,
                scale=destination_region.scale,
                threshold=destination_region.threshold,
                scaling_fitted_on=f"{destination.dataset} train",
                threshold_fitted_on=f"{destination.dataset} train",
                source_name=destination.dataset,
            )
        )
    return results


def transfer_distance_samples(
    region: TargetRegion,
    source_train: pd.DataFrame,
    destination_test: pd.DataFrame,
    *,
    source: str,
    destination: str,
    center: pd.Series,
    scale: pd.Series,
    threshold: float,
    failure_column: str = "is_failure",
    max_samples: int = 4_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample the distance distributions a transfer figure needs.

    Three populations are separated because a transfer can fail in two different ways:
    the destination's normal operation may sit closer to the source's failure
    prototypes than the destination's own failure does, or the whole destination may
    drift away so that nothing crosses the threshold.
    """

    rng = np.random.default_rng(seed)
    prototypes = region.scaled_prototypes(region.center, region.scale)
    source_distances = _minimum_distance(
        _standardise(source_train, region.features, center=region.center, scale=region.scale),
        prototypes,
    )

    destination_scaled = _standardise(destination_test, region.features, center=center, scale=scale)
    destination_distances = _minimum_distance(
        destination_scaled, region.scaled_prototypes(center, scale)
    )
    labels = destination_test[failure_column].astype(bool).to_numpy()

    def _sample(values: FloatArray) -> FloatArray:
        finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
        if finite.size <= max_samples:
            return finite
        return np.asarray(rng.choice(finite, size=max_samples, replace=False), dtype=float)

    populations = {
        "source_train": _sample(source_distances),
        "destination_normal": _sample(destination_distances[~labels]),
        "destination_failure": _sample(destination_distances[labels]),
    }
    records: list[dict[str, object]] = []
    for population, values in populations.items():
        for value in values:
            records.append(
                {
                    "source": source,
                    "destination": destination,
                    "population": population,
                    "distance": float(value),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame.from_records(records)


def transfer_results_frame(results: list[TransferResult]) -> pd.DataFrame:
    """Return transfer results as a table, with degradation against the refit."""

    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([asdict(result) for result in results])
    frame["features"] = frame["features"].map(lambda value: ", ".join(value))

    # Degradation is measured against the within-dataset refit for the same destination,
    # so a transfer that fails is reported as a magnitude rather than as a gap.
    frame["transfer_degradation_precision"] = float("nan")
    for destination, group in frame.groupby("destination"):
        refit = group[group["protocol"] == "destination_refit"]
        if refit.empty:
            continue
        reference = float(refit.iloc[0]["event_precision"])
        if not np.isfinite(reference):
            continue
        mask = frame["destination"] == destination
        frame.loc[mask, "transfer_degradation_precision"] = (
            reference - frame.loc[mask, "event_precision"]
        )
    return frame


def assess_schema_compatibility(
    source_frame: pd.DataFrame,
    destination_frame: pd.DataFrame,
    *,
    source: str,
    destination: str,
    candidate_features: tuple[str, ...],
) -> SchemaCompatibilityReport:
    """Decide which features may cross between two datasets, and why.

    A feature is admissible only if it exists in both, is numeric in both, has a
    comparable missingness profile, and its destination distribution overlaps the
    source. Silently mapping an incompatible feature would produce a transfer number
    that means nothing, so incompatible features are excluded and the reason recorded.
    """

    verdicts: list[FeatureCompatibility] = []
    for feature in candidate_features:
        in_source = feature in source_frame.columns
        in_destination = feature in destination_frame.columns
        if not (in_source and in_destination):
            verdicts.append(
                FeatureCompatibility(
                    feature=feature,
                    present_in_both=False,
                    source_median=float("nan"),
                    destination_median=float("nan"),
                    standardised_shift=float("nan"),
                    distribution_overlap=float("nan"),
                    source_missing=float("nan"),
                    destination_missing=float("nan"),
                    compatible=False,
                    reason="absent from one dataset",
                )
            )
            continue

        a = pd.to_numeric(source_frame[feature], errors="coerce")
        b = pd.to_numeric(destination_frame[feature], errors="coerce")
        a_values = a.to_numpy(dtype=float)
        b_values = b.to_numpy(dtype=float)
        a_median = float(np.nanmedian(a_values)) if np.isfinite(a_values).any() else float("nan")
        b_median = float(np.nanmedian(b_values)) if np.isfinite(b_values).any() else float("nan")

        finite_a = a_values[np.isfinite(a_values)]
        spread = float(np.subtract(*np.nanpercentile(finite_a, [75, 25]))) if finite_a.size else 0.0
        spread = abs(spread) if spread else 1.0
        shift = abs(b_median - a_median) / spread if spread else float("inf")
        overlap = _histogram_overlap(a_values, b_values)
        a_missing = float(np.mean(~np.isfinite(a_values))) if a_values.size else 1.0
        b_missing = float(np.mean(~np.isfinite(b_values))) if b_values.size else 1.0

        reasons: list[str] = []
        if not np.isfinite(shift) or shift > MAX_MEDIAN_SHIFT:
            reasons.append(f"median shift {shift:.2f} exceeds {MAX_MEDIAN_SHIFT:.1f} IQR")
        if overlap < MIN_DISTRIBUTION_OVERLAP:
            reasons.append(
                f"distribution overlap {overlap:.2f} below {MIN_DISTRIBUTION_OVERLAP:.2f}"
            )
        if abs(a_missing - b_missing) > 0.25:
            reasons.append("missingness profiles differ by more than 25 percentage points")

        verdicts.append(
            FeatureCompatibility(
                feature=feature,
                present_in_both=True,
                source_median=a_median,
                destination_median=b_median,
                standardised_shift=shift,
                distribution_overlap=overlap,
                source_missing=a_missing,
                destination_missing=b_missing,
                compatible=not reasons,
                reason="; ".join(reasons) if reasons else "compatible",
            )
        )

    source_cadence = _cadence_seconds(source_frame)
    destination_cadence = _cadence_seconds(destination_frame)
    cadence_ok = True
    notes: list[str] = []
    if np.isfinite(source_cadence) and np.isfinite(destination_cadence) and source_cadence > 0:
        ratio = max(source_cadence, destination_cadence) / min(source_cadence, destination_cadence)
        cadence_ok = ratio <= MAX_CADENCE_RATIO
        if not cadence_ok:
            notes.append(
                f"sampling cadence differs by {ratio:.2f}x "
                f"({source_cadence:.3f}s against {destination_cadence:.3f}s)"
            )
    else:
        notes.append("sampling cadence could not be established for at least one dataset")

    incompatible = [item.feature for item in verdicts if not item.compatible]
    if incompatible:
        notes.append(f"excluded features: {', '.join(incompatible)}")
    return SchemaCompatibilityReport(
        source=source,
        destination=destination,
        features=tuple(verdicts),
        source_cadence_seconds=source_cadence,
        destination_cadence_seconds=destination_cadence,
        cadence_compatible=cadence_ok,
        notes=tuple(notes),
    )
