"""Tests for estimator performance aggregation and the main-paper figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.evt.extremal_index import ESTIMATOR_NAMES
from dyn_evt_pdm.simulation.estimator_figures import (
    assign_regime,
    build_estimator_figures,
    plot_coverage_by_family,
    plot_failure_by_threshold,
    plot_rmse_by_sample_size,
    plot_sensitivity,
)
from dyn_evt_pdm.simulation.estimator_summary import (
    GROUP_COLUMNS,
    SUMMARY_COLUMNS,
    estimable_regime_report,
    reference_coverage_report,
    summarise_estimator_performance,
)
from dyn_evt_pdm.simulation.study import ESTIMATOR_COLUMN_STEMS


def _replicate_frame(
    *,
    n_replicates: int = 20,
    reference: float = 0.5,
    reference_kind: str = "theoretical",
    system: str = "logistic_periodic_target",
    n_steps: int = 2000,
    threshold_quantile: float = 0.98,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic replicate frame with known bias and coverage."""

    rng = np.random.default_rng(seed)
    payload: dict[str, object] = {
        "system": [system] * n_replicates,
        "n_steps": [n_steps] * n_replicates,
        "threshold_quantile": [threshold_quantile] * n_replicates,
        "run_length": [5] * n_replicates,
        "noise_scale": [0.0] * n_replicates,
        "missing_rate": [0.0] * n_replicates,
        "reference_theta": [reference] * n_replicates,
        "reference_kind": [reference_kind] * n_replicates,
    }
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        estimates = reference + rng.normal(0.0, 0.02, size=n_replicates)
        payload[f"{stem}_theta"] = estimates
        payload[f"{stem}_lower"] = estimates - 0.05
        payload[f"{stem}_upper"] = estimates + 0.05
        payload[f"{stem}_interval_width"] = np.full(n_replicates, 0.10)
        payload[f"{stem}_covered"] = pd.array(
            [(est - 0.05) <= reference <= (est + 0.05) for est in estimates], dtype="boolean"
        )
        payload[f"{stem}_tuning_parameter"] = np.full(n_replicates, 5.0)
        payload[f"{stem}_failed"] = [False] * n_replicates
    return pd.DataFrame(payload)


def test_summary_emits_every_declared_column() -> None:
    summary = summarise_estimator_performance(_replicate_frame())
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert len(summary) == len(ESTIMATOR_NAMES)


def test_summary_recovers_known_bias_and_rmse() -> None:
    frame = _replicate_frame(reference=0.5)
    summary = summarise_estimator_performance(frame)
    row = summary[summary["estimator"] == "runs"].iloc[0]

    estimates = frame["runs_theta"].to_numpy(dtype=float)
    expected_bias = float(np.mean(estimates - 0.5))
    expected_rmse = float(np.sqrt(np.mean((estimates - 0.5) ** 2)))
    assert row["bias"] == pytest.approx(expected_bias)
    assert row["rmse"] == pytest.approx(expected_rmse)
    assert row["median_absolute_error"] == pytest.approx(float(np.median(np.abs(estimates - 0.5))))


def test_coverage_standard_error_matches_binomial_formula() -> None:
    summary = summarise_estimator_performance(_replicate_frame())
    row = summary[summary["estimator"] == "runs"].iloc[0]
    coverage = float(row["coverage"])
    n = int(row["coverage_replicates"])
    expected = float(np.sqrt(coverage * (1.0 - coverage) / n))
    assert row["coverage_standard_error"] == pytest.approx(expected)


def test_coverage_denominator_excludes_replicates_without_intervals() -> None:
    frame = _replicate_frame(n_replicates=10)
    frame.loc[:4, "runs_covered"] = pd.NA
    summary = summarise_estimator_performance(frame)
    row = summary[summary["estimator"] == "runs"].iloc[0]
    assert int(row["coverage_replicates"]) == 5


def test_failed_replicates_are_counted_not_dropped() -> None:
    frame = _replicate_frame(n_replicates=10)
    frame.loc[:1, "runs_theta"] = np.nan
    frame.loc[:1, "runs_failed"] = True
    summary = summarise_estimator_performance(frame)
    row = summary[summary["estimator"] == "runs"].iloc[0]
    assert row["failure_probability"] == pytest.approx(0.2)
    assert row["invalid_probability"] == pytest.approx(0.2)
    assert int(row["estimable_replicates"]) == 8
    assert int(row["replicates"]) == 10


def test_clipping_probability_counts_boundary_estimates() -> None:
    frame = _replicate_frame(n_replicates=10)
    frame.loc[:4, "runs_theta"] = 1.0
    summary = summarise_estimator_performance(frame)
    row = summary[summary["estimator"] == "runs"].iloc[0]
    assert row["clipping_probability"] == pytest.approx(0.5)


def test_mixed_reference_kinds_are_rejected() -> None:
    """Theoretical and numerical references must never be pooled into one row."""

    frame = _replicate_frame(n_replicates=10)
    frame.loc[:4, "reference_kind"] = "numerical"
    with pytest.raises(ValueError, match="reference kinds are mixed"):
        summarise_estimator_performance(frame)


def test_summary_requires_reference_columns() -> None:
    frame = _replicate_frame().drop(columns=["reference_kind"])
    with pytest.raises(ValueError, match="missing replicate columns"):
        summarise_estimator_performance(frame)


def test_empty_frame_returns_declared_schema() -> None:
    empty = pd.DataFrame(columns=[*GROUP_COLUMNS, "reference_theta", "reference_kind"])
    summary = summarise_estimator_performance(empty)
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert summary.empty


def test_reference_and_regime_reports_sum_to_one() -> None:
    combined = pd.concat(
        [
            _replicate_frame(reference=1.0, system="iid_light_tail", seed=1),
            _replicate_frame(
                reference=0.27, reference_kind="numerical", system="smoothing", seed=2
            ),
        ],
        ignore_index=True,
    )
    summary = summarise_estimator_performance(combined)

    references = reference_coverage_report(summary)
    assert float(references["share"].sum()) == pytest.approx(1.0)
    assert set(references["reference_kind"]) == {"theoretical", "numerical"}

    regimes = estimable_regime_report(summary)
    assert float(regimes["share"].sum()) == pytest.approx(1.0)
    assert set(regimes["regime"]) == {"near_independent", "strongly_clustered"}


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (1.0, "near_independent"),
        (0.9, "near_independent"),
        (0.7, "moderately_clustered"),
        (0.5, "moderately_clustered"),
        (0.27, "strongly_clustered"),
        (float("nan"), "unreferenced"),
    ],
)
def test_regime_assignment_boundaries(theta: float, expected: str) -> None:
    assert assign_regime(theta) == expected


def _figure_summary() -> pd.DataFrame:
    frames = [
        _replicate_frame(reference=1.0, system="iid_light_tail", seed=1),
        _replicate_frame(reference=1.0, system="iid_pareto", n_steps=5000, seed=3),
        _replicate_frame(reference=0.27, reference_kind="numerical", system="smoothing", seed=2),
        _replicate_frame(
            reference=0.6,
            reference_kind="numerical",
            system="regime_mixture",
            threshold_quantile=0.95,
            seed=4,
        ),
    ]
    return summarise_estimator_performance(pd.concat(frames, ignore_index=True))


def test_every_figure_is_written(tmp_path: Path) -> None:
    figures = build_estimator_figures(_figure_summary(), tmp_path)
    for path in figures.as_tuple():
        assert path.exists()
        assert path.stat().st_size > 5_000


@pytest.mark.parametrize(
    "builder",
    [
        plot_rmse_by_sample_size,
        plot_coverage_by_family,
        plot_failure_by_threshold,
        plot_sensitivity,
    ],
)
def test_individual_figure_builders(builder, tmp_path: Path) -> None:
    path = builder(_figure_summary(), tmp_path / "figure.png")
    assert path.exists()


def test_figures_reject_a_summary_without_references(tmp_path: Path) -> None:
    summary = _figure_summary()
    summary["reference_theta"] = np.nan
    with pytest.raises(ValueError, match="no referenced configurations"):
        plot_rmse_by_sample_size(summary, tmp_path / "figure.png")
