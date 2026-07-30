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

from dataclasses import dataclass, field

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
