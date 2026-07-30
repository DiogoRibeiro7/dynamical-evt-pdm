"""Tests for cross-dataset schema compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.pipelines.cross_dataset_transfer import (
    MAX_CADENCE_RATIO,
    MIN_DISTRIBUTION_OVERLAP,
    TRANSFER_PROTOCOLS,
    TransferSplits,
    assess_schema_compatibility,
    fit_target_region,
    run_transfer_protocols,
    transfer_results_frame,
)


def _frame(
    *,
    rows: int = 2000,
    shift: float = 0.0,
    cadence_seconds: float = 1.0,
    seed: int = 0,
    drop: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq=pd.Timedelta(seconds=cadence_seconds)
            ),
            "alpha": rng.normal(0.0 + shift, 1.0, rows),
            "beta": rng.normal(10.0, 2.0, rows),
        }
    )
    if drop is not None:
        frame = frame.drop(columns=[drop])
    return frame


def test_identical_datasets_are_fully_compatible() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2),
        source="a",
        destination="b",
        candidate_features=("alpha", "beta"),
    )
    assert report.transferable
    assert set(report.compatible_features) == {"alpha", "beta"}


def test_disjoint_distributions_are_rejected() -> None:
    """A feature whose distributions do not overlap cannot be silently mapped."""

    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2, shift=200.0),
        source="a",
        destination="b",
        candidate_features=("alpha",),
    )
    verdict = report.features[0]
    assert not verdict.compatible
    assert verdict.distribution_overlap < MIN_DISTRIBUTION_OVERLAP
    assert "overlap" in verdict.reason


def test_absent_feature_is_reported_not_assumed() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2, drop="beta"),
        source="a",
        destination="b",
        candidate_features=("alpha", "beta"),
    )
    beta = next(item for item in report.features if item.feature == "beta")
    assert not beta.present_in_both
    assert beta.reason == "absent from one dataset"
    assert report.compatible_features == ("alpha",)


def test_incompatible_cadence_blocks_transfer() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1, cadence_seconds=1.0),
        _frame(seed=2, cadence_seconds=1.0 * (MAX_CADENCE_RATIO + 1.0)),
        source="a",
        destination="b",
        candidate_features=("alpha",),
    )
    assert not report.cadence_compatible
    assert not report.transferable
    assert any("cadence" in note for note in report.notes)


def test_report_frame_lists_every_candidate() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2),
        source="a",
        destination="b",
        candidate_features=("alpha", "beta", "missing_entirely"),
    )
    frame = report.to_frame()
    assert len(frame) == 3
    assert set(frame["feature"]) == {"alpha", "beta", "missing_entirely"}


def test_excluded_features_are_named_in_the_notes() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2, shift=200.0),
        source="a",
        destination="b",
        candidate_features=("alpha", "beta"),
    )
    assert any("alpha" in note for note in report.notes)


def test_no_compatible_feature_means_not_transferable() -> None:
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2, shift=500.0),
        source="a",
        destination="b",
        candidate_features=("alpha",),
    )
    assert report.compatible_features == ()
    assert not report.transferable


@pytest.mark.parametrize("direction", [("a", "b"), ("b", "a")])
def test_compatibility_is_reported_for_both_directions(direction: tuple[str, str]) -> None:
    source, destination = direction
    report = assess_schema_compatibility(
        _frame(seed=1),
        _frame(seed=2),
        source=source,
        destination=destination,
        candidate_features=("alpha",),
    )
    assert report.source == source
    assert report.destination == destination


def _labelled_frame(
    *, rows: int = 4000, shift: float = 0.0, seed: int = 0, failure_at: slice | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="1s"),
            "alpha": rng.normal(shift, 1.0, rows),
            "beta": rng.normal(10.0 + shift, 1.0, rows),
            "is_failure": False,
        }
    )
    # A transfer needs a failure in the source training split to build prototypes from
    # and a failure in the destination test split to evaluate against, so both are
    # present unless the caller asks for one specific window.
    windows = (
        [failure_at]
        if failure_at is not None
        else [slice(int(rows * 0.2), int(rows * 0.25)), slice(int(rows * 0.85), int(rows * 0.9))]
    )
    for window in windows:
        frame.loc[window, "alpha"] += 6.0
        frame.loc[window, "beta"] += 6.0
        frame.loc[window, "is_failure"] = True
    return frame


def _splits(frame: pd.DataFrame, dataset: str) -> TransferSplits:
    n = len(frame)
    a, b = int(n * 0.6), int(n * 0.8)
    return TransferSplits(
        dataset=dataset, train=frame.iloc[:a], validation=frame.iloc[a:b], test=frame.iloc[b:]
    )


def test_region_is_not_fitted_without_a_training_failure() -> None:
    """A prototype region has nothing to build from and must not be invented."""

    clean = _labelled_frame()
    clean["is_failure"] = False
    assert fit_target_region(clean, ("alpha", "beta"), dataset="a") is None


def test_prototypes_are_stored_in_raw_units() -> None:
    """Storing them standardised would make scaling recalibration a no-op."""

    frame = _labelled_frame(failure_at=slice(100, 300))
    region = fit_target_region(frame, ("alpha", "beta"), dataset="a")
    assert region is not None
    raw_alpha = frame.loc[frame["is_failure"], "alpha"]
    assert float(region.prototypes_raw[:, 0].min()) >= float(raw_alpha.min()) - 1e-9
    assert float(region.prototypes_raw[:, 0].max()) <= float(raw_alpha.max()) + 1e-9


def test_rescaling_prototypes_changes_their_standardised_position() -> None:
    frame = _labelled_frame(failure_at=slice(100, 300))
    region = fit_target_region(frame, ("alpha", "beta"), dataset="a")
    assert region is not None
    other_center = region.center + 5.0
    original = region.scaled_prototypes(region.center, region.scale)
    moved = region.scaled_prototypes(other_center, region.scale)
    assert not np.allclose(original, moved)


def test_every_protocol_is_reported_for_a_transferable_pair() -> None:
    source = _splits(_labelled_frame(seed=1), "a")
    destination = _splits(_labelled_frame(seed=2), "b")
    results = run_transfer_protocols(source, destination, ("alpha", "beta"))
    assert [r.protocol for r in results] == list(TRANSFER_PROTOCOLS)


def test_each_protocol_declares_what_it_fitted_where() -> None:
    source = _splits(_labelled_frame(seed=1), "a")
    destination = _splits(_labelled_frame(seed=2), "b")
    by_protocol = {r.protocol: r for r in run_transfer_protocols(source, destination, ("alpha",))}

    direct = by_protocol["direct"]
    assert direct.scaling_fitted_on == "a"
    assert direct.prototypes_fitted_on == "a"
    assert direct.threshold_fitted_on == "a"

    assert by_protocol["recalibrated_threshold"].threshold_fitted_on == "b validation"
    assert by_protocol["recalibrated_scaling"].scaling_fitted_on == "b train"
    assert by_protocol["recalibrated_scaling"].prototypes_fitted_on == "a"
    assert by_protocol["destination_refit"].prototypes_fitted_on == "b"


def test_transfer_without_a_source_failure_is_not_estimable() -> None:
    clean = _labelled_frame(seed=1)
    clean["is_failure"] = False
    source = _splits(clean, "a")
    destination = _splits(_labelled_frame(seed=2), "b")
    results = run_transfer_protocols(source, destination, ("alpha",))
    assert len(results) == 1
    assert results[0].status == "not_estimable"
    assert "no labelled failure" in results[0].note


def test_degradation_is_zero_for_the_destination_refit() -> None:
    source = _splits(_labelled_frame(seed=1), "a")
    destination = _splits(_labelled_frame(seed=2), "b")
    frame = transfer_results_frame(run_transfer_protocols(source, destination, ("alpha",)))
    refit = frame[frame["protocol"] == "destination_refit"].iloc[0]
    assert float(refit["transfer_degradation_precision"]) == pytest.approx(0.0)


def test_results_frame_is_empty_for_no_results() -> None:
    assert transfer_results_frame([]).empty
