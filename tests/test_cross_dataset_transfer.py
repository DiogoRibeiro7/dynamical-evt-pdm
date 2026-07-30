"""Tests for cross-dataset schema compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.pipelines.cross_dataset_transfer import (
    MAX_CADENCE_RATIO,
    MIN_DISTRIBUTION_OVERLAP,
    assess_schema_compatibility,
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
