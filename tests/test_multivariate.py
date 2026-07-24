import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.evt.multivariate import (
    LaggedExtremePattern,
    benjamini_hochberg,
    candidate_pair_lag_patterns,
    construct_lagged_pattern_clusters,
    detect_lagged_pattern,
    empirical_stable_tail_dependence,
    evaluate_lagged_patterns,
    fit_component_regime_thresholds,
    generate_null_exceedances,
    multivariate_evt_report,
    multivariate_extremal_index_surface,
    select_lagged_patterns_nested,
    simultaneous_vector_counts,
)


def test_detect_lagged_pattern() -> None:
    matrix = np.zeros((8, 2), dtype=bool)
    matrix[2, 0] = True
    matrix[4, 1] = True
    pattern = LaggedExtremePattern(components=(0, 1), lags=(0, 2))

    detected = detect_lagged_pattern(matrix, pattern)

    assert detected[2]
    assert int(detected.sum()) == 1


def test_component_regime_thresholds_are_fit_on_training_only() -> None:
    frame = pd.DataFrame(
        {
            "current": [1.0, 2.0, 100.0, 3.0],
            "temperature": [2.0, 3.0, 200.0, 4.0],
            "regime": ["idle", "idle", "idle", "idle"],
        }
    )
    train_mask = np.array([True, True, False, True])

    thresholds = fit_component_regime_thresholds(
        frame,
        component_columns=("current", "temperature"),
        regime_column="regime",
        quantile=0.8,
        train_mask=train_mask,
        min_regime_samples=1,
    )

    assert thresholds.thresholds["current"].fallback < 100.0
    assert thresholds.thresholds["temperature"].fallback < 200.0


def test_tail_dependence_vectors_clusters_and_surface() -> None:
    matrix = np.zeros((12, 3), dtype=bool)
    matrix[[1, 2, 8], 0] = True
    matrix[[1, 3, 9], 1] = True
    matrix[[5, 10], 2] = True
    pattern = LaggedExtremePattern(components=(0, 1), lags=(0, 1))
    detections = detect_lagged_pattern(matrix, pattern)

    counts = simultaneous_vector_counts(matrix, component_names=("a", "b", "c"))
    tail = empirical_stable_tail_dependence(matrix, component_names=("a", "b", "c"))
    clusters = construct_lagged_pattern_clusters(detections, pattern, run_length=0)
    surface = multivariate_extremal_index_surface(matrix, max_lag=2)

    assert counts["a+b"] == 1
    assert tail["a|b"]["joint_count"] == 1
    assert [cluster.interval.start for cluster in clusters] == [2, 8]
    assert {"source_component", "target_component", "lag", "empirical_extremal_index"}.issubset(
        surface.columns
    )


def test_null_models_preserve_component_counts() -> None:
    matrix = np.zeros((30, 2), dtype=bool)
    matrix[2:5, 0] = True
    matrix[14:17, 0] = True
    matrix[8:12, 1] = True
    rng = np.random.default_rng(7)

    shifted = generate_null_exceedances(matrix, method="time_shift", rng=rng)
    permuted = generate_null_exceedances(matrix, method="cluster_permutation", rng=rng)

    np.testing.assert_array_equal(shifted.sum(axis=0), matrix.sum(axis=0))
    np.testing.assert_array_equal(permuted.sum(axis=0), matrix.sum(axis=0))


def test_nested_lag_selection_labels_heldout_as_exploratory() -> None:
    matrix = np.zeros((80, 2), dtype=bool)
    for start in (10, 20, 30, 50, 60):
        matrix[start, 0] = True
        matrix[start + 2, 1] = True
    validation_mask = np.zeros(80, dtype=bool)
    validation_mask[:40] = True
    heldout_mask = np.zeros(80, dtype=bool)
    heldout_mask[40:] = True
    patterns = (
        LaggedExtremePattern(components=(0, 1), lags=(0, 2)),
        LaggedExtremePattern(components=(0, 1), lags=(0, 1)),
    )

    selection = select_lagged_patterns_nested(
        matrix,
        patterns,
        validation_mask=validation_mask,
        heldout_mask=heldout_mask,
        max_selected=1,
        alpha=1.0,
        n_null=9,
        random_state=4,
    )

    assert selection.selected_indices == (0,)
    assert selection.heldout_reports
    assert selection.heldout_reports[0].exploratory


def test_evaluate_patterns_and_bh_adjustment_are_bounded() -> None:
    matrix = np.zeros((20, 2), dtype=bool)
    matrix[[3, 10], 0] = True
    matrix[[5, 12], 1] = True
    reports = evaluate_lagged_patterns(
        matrix,
        candidate_pair_lag_patterns(n_components=2, max_lag=2),
        n_null=5,
        random_state=1,
    )

    assert len(reports) == 6
    assert all(0.0 <= report.adjusted_p_value <= 1.0 for report in reports)
    assert benjamini_hochberg((0.01, 0.04, 0.03)) == pytest.approx((0.03, 0.04, 0.04))


def test_multivariate_report_and_cli(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "current": np.r_[np.linspace(0, 1, 40), [10, 0, 9, 0, 8, 0, 7, 0]],
            "temperature": np.r_[np.linspace(0, 1, 40), [0, 10, 0, 9, 0, 8, 0, 7]],
            "pressure": np.r_[np.linspace(1, 0, 40), [0, 1, 0, 1, 0, 1, 0, 1]],
            "regime": ["steady"] * 48,
            "split": ["train"] * 30 + ["validation"] * 10 + ["heldout"] * 8,
        }
    )
    train_mask = frame["split"].eq("train").to_numpy()
    validation_mask = frame["split"].eq("validation").to_numpy()
    heldout_mask = frame["split"].eq("heldout").to_numpy()

    report = multivariate_evt_report(
        frame,
        component_columns=("current", "temperature", "pressure"),
        regime_column="regime",
        train_mask=train_mask,
        validation_mask=validation_mask,
        heldout_mask=heldout_mask,
        quantile=0.8,
        max_lag=2,
        n_null=3,
        min_regime_samples=1,
    )

    assert report["multiple_comparison_control"]["method"] == "benjamini_hochberg"
    assert report["validation_patterns"]

    input_path = tmp_path / "series.csv"
    output_path = tmp_path / "multi.json"
    frame.to_csv(input_path, index=False)
    result = CliRunner().invoke(
        app,
        [
            "analyse-multivariate-extremes",
            "--input",
            str(input_path),
            "--component-columns",
            "current,temperature,pressure",
            "--regime-column",
            "regime",
            "--split-column",
            "split",
            "--output",
            str(output_path),
            "--quantile",
            "0.8",
            "--max-lag",
            "2",
            "--n-null",
            "3",
            "--min-regime-samples",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
