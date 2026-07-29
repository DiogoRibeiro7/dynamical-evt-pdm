"""Aggregate replicate-level simulation output into estimator performance summaries.

Every reported quantity is derived here from the replicate-level artifact so that the
manuscript tables can be checked against the raw draws. Failed and non-estimable
replicates are counted explicitly rather than dropped, because an estimator that
silently declines to produce a value on half its replicates is not comparable to one
that always answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dyn_evt_pdm.evt.extremal_index import ESTIMATOR_NAMES
from dyn_evt_pdm.simulation.study import ESTIMATOR_COLUMN_STEMS
from dyn_evt_pdm.types import FloatArray

#: Grouping keys that hold the reference extremal index fixed within a summary row.
GROUP_COLUMNS: tuple[str, ...] = (
    "system",
    "n_steps",
    "threshold_quantile",
    "run_length",
    "noise_scale",
    "missing_rate",
)

#: Columns produced for each estimator and configuration.
SUMMARY_COLUMNS: tuple[str, ...] = (
    *GROUP_COLUMNS,
    "estimator",
    "reference_theta",
    "reference_kind",
    "tuning_parameter",
    "replicates",
    "estimable_replicates",
    "mean_estimate",
    "bias",
    "bias_standard_error",
    "rmse",
    "rmse_standard_error",
    "median_absolute_error",
    "coverage",
    "coverage_standard_error",
    "coverage_replicates",
    "mean_interval_width",
    "failure_probability",
    "invalid_probability",
    "clipping_probability",
)

_BOUNDARY_EPSILON = float(np.finfo(float).eps)


def _clipping_probability(estimates: FloatArray) -> float:
    """Return the share of finite estimates sitting on the admissible boundary."""

    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan")
    on_boundary = (finite <= _BOUNDARY_EPSILON) | (finite >= 1.0)
    return float(np.mean(on_boundary))


def _rmse_standard_error(errors: FloatArray, *, n_resamples: int = 500) -> float:
    """Return a bootstrap standard error for RMSE over the replicate errors.

    RMSE has no closed-form standard error here, but the manuscript must show
    uncertainty rather than a bare point, so it is resampled directly.
    """

    if errors.size < 2:
        return float("nan")
    rng = np.random.default_rng(20260728)
    draws = rng.integers(0, errors.size, size=(n_resamples, errors.size))
    resampled = np.sqrt(np.mean(errors[draws] ** 2, axis=1))
    return float(np.std(resampled, ddof=1))


def _coverage_statistics(covered: pd.Series) -> tuple[float, float, int]:
    """Return coverage, its Monte Carlo standard error, and the denominator.

    The denominator counts only replicates where an interval and a reference both
    exist. Treating a missing interval as a non-coverage would understate coverage for
    exactly the estimators that fail most often.
    """

    known = covered.dropna()
    n_known = int(len(known))
    if n_known == 0:
        return float("nan"), float("nan"), 0
    coverage = float(np.mean(known.astype(bool).to_numpy()))
    standard_error = float(np.sqrt(coverage * (1.0 - coverage) / n_known))
    return coverage, standard_error, n_known


def summarise_estimator_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise replicate rows into one record per configuration and estimator."""

    required = {*GROUP_COLUMNS, "reference_theta", "reference_kind"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing replicate columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        key_values = dict(zip(GROUP_COLUMNS, keys, strict=True))
        reference_series = group["reference_theta"].dropna()
        reference = float(reference_series.iloc[0]) if len(reference_series) else float("nan")
        kinds = sorted(set(group["reference_kind"].astype(str)))
        if len(kinds) != 1:
            raise ValueError(
                f"reference kinds are mixed within one configuration: {kinds}. "
                "Theoretical and numerical references must never be pooled."
            )
        reference_kind = kinds[0]

        for name in ESTIMATOR_NAMES:
            stem = ESTIMATOR_COLUMN_STEMS[name]
            estimates = group[f"{stem}_theta"].to_numpy(dtype=np.float64)
            estimable = np.isfinite(estimates)
            errors = estimates - reference
            valid_errors = errors[estimable & np.isfinite(errors)]

            coverage, coverage_error, coverage_n = _coverage_statistics(group[f"{stem}_covered"])
            widths = group[f"{stem}_interval_width"].to_numpy(dtype=np.float64)
            finite_widths = widths[np.isfinite(widths)]
            tuning = group[f"{stem}_tuning_parameter"].to_numpy(dtype=np.float64)
            finite_tuning = tuning[np.isfinite(tuning)]

            records.append(
                {
                    **key_values,
                    "estimator": name,
                    "reference_theta": reference,
                    "reference_kind": reference_kind,
                    "tuning_parameter": (
                        float(finite_tuning[0]) if finite_tuning.size else float("nan")
                    ),
                    "replicates": int(len(group)),
                    "estimable_replicates": int(np.count_nonzero(estimable)),
                    "mean_estimate": (
                        float(np.mean(estimates[estimable])) if estimable.any() else float("nan")
                    ),
                    "bias": float(np.mean(valid_errors)) if valid_errors.size else float("nan"),
                    "bias_standard_error": (
                        float(np.std(valid_errors, ddof=1) / np.sqrt(valid_errors.size))
                        if valid_errors.size > 1
                        else float("nan")
                    ),
                    "rmse": (
                        float(np.sqrt(np.mean(valid_errors**2)))
                        if valid_errors.size
                        else float("nan")
                    ),
                    "rmse_standard_error": _rmse_standard_error(valid_errors),
                    "median_absolute_error": (
                        float(np.median(np.abs(valid_errors)))
                        if valid_errors.size
                        else float("nan")
                    ),
                    "coverage": coverage,
                    "coverage_standard_error": coverage_error,
                    "coverage_replicates": coverage_n,
                    "mean_interval_width": (
                        float(np.mean(finite_widths)) if finite_widths.size else float("nan")
                    ),
                    "failure_probability": float(
                        np.mean(group[f"{stem}_failed"].to_numpy(dtype=bool))
                    ),
                    "invalid_probability": float(np.mean(~estimable)),
                    "clipping_probability": _clipping_probability(estimates),
                }
            )

    summary = pd.DataFrame.from_records(records)
    return summary.loc[:, list(SUMMARY_COLUMNS)]


def reference_coverage_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Report how much of the grid carries each kind of reference.

    This is the evidence for stating plainly which regimes the study can speak about.
    """

    if summary.empty:
        return pd.DataFrame(columns=["reference_kind", "configurations", "share"])
    per_configuration = summary.drop_duplicates(subset=list(GROUP_COLUMNS))
    counts = per_configuration["reference_kind"].value_counts()
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "reference_kind": counts.index.astype(str),
            "configurations": counts.to_numpy(dtype=int),
            "share": counts.to_numpy(dtype=float) / float(total),
        }
    ).reset_index(drop=True)


def estimable_regime_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Classify each referenced configuration by how close its reference is to one.

    The distinction matters because an extremal index near one describes a process with
    almost no extremal clustering. A grid populated by such processes cannot discriminate
    estimators in the clustered regime, whatever its replicate count.
    """

    referenced = summary[summary["reference_theta"].notna()].drop_duplicates(
        subset=list(GROUP_COLUMNS)
    )
    if referenced.empty:
        return pd.DataFrame(columns=["regime", "configurations", "share"])
    theta = referenced["reference_theta"].to_numpy(dtype=np.float64)
    regime = np.where(
        theta >= 0.9,
        "near_independent",
        np.where(theta >= 0.5, "moderately_clustered", "strongly_clustered"),
    )
    counts = pd.Series(regime).value_counts()
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "regime": counts.index.astype(str),
            "configurations": counts.to_numpy(dtype=int),
            "share": counts.to_numpy(dtype=float) / float(total),
        }
    ).reset_index(drop=True)
