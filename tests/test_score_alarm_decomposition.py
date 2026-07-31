"""Regression tests for the score-threshold-cluster-alarm decomposition.

The defect these guard against is a label asserting a computation that does not
happen. Two earlier versions of this decomposition looked correct because their
numbers agreed for structural rather than empirical reasons: declustering methods
reported post-declustering onsets in the exceedance column, so ``exceedances`` and
``episodes`` were the same number by construction, and the labelled failures were
drawn as the terminus of the detector flow, which reads as a derivation from
failures back to failures. Neither is visible from the values alone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dyn_evt_pdm.paper.assets import _write_latex_table, classify_failure_layer

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "real_data_matrix"
    / "score_threshold_alarm_decomposition.csv"
)

#: Methods whose event policy declusters, and which therefore must reduce a larger
#: raw exceedance count to a smaller cluster count rather than reporting onsets twice.
DECLUSTERING_METHODS = (
    "fixed_run_declustering",
    "ferro_segers_event_policy",
    "k_gaps_event_policy",
    "dynamical_evt_robust_score",
)


@pytest.fixture(scope="module")
def decomposition() -> pd.DataFrame:
    if not ARTIFACT.exists():
        pytest.skip("real-data decomposition artifact is not present")
    return pd.read_csv(ARTIFACT)


def test_detector_track_is_monotone(decomposition: pd.DataFrame) -> None:
    """Each stage can only discard evidence, never create it."""
    assert (
        decomposition["threshold_exceedances_samples"] >= decomposition["extreme_clusters_count"]
    ).all()
    assert (decomposition["extreme_clusters_count"] >= decomposition["alarm_episodes_count"]).all()
    assert (
        decomposition["alarm_episodes_count"] >= decomposition["matched_alarm_episodes_count"]
    ).all()


def test_declustering_reduces_raw_exceedances(decomposition: pd.DataFrame) -> None:
    """Declustering methods must report raw exceedances, not post-declustering onsets.

    When the exceedance column carried onsets, every declustering row had
    ``exceedances == episodes``, which made the declustering stage invisible.
    """
    rows = decomposition[decomposition["method"].isin(DECLUSTERING_METHODS)]
    assert not rows.empty
    assert (rows["threshold_exceedances_samples"] > rows["extreme_clusters_count"]).all()
    assert (rows["threshold_exceedances_samples"] > rows["alarm_episodes_count"]).all()


def test_ground_truth_is_not_a_detector_stage(decomposition: pd.DataFrame) -> None:
    """Labelled failures are constant per dataset and independent of the method.

    They are ground truth, so a method that alarms more cannot change them.
    """
    for _dataset, group in decomposition.groupby("dataset_id"):
        assert group["labelled_failure_events_count"].nunique() == 1
        assert group["labelled_failure_samples"].nunique() == 1


def test_failure_layer_names_the_earliest_broken_stage() -> None:
    no_exceedance = pd.Series(
        {
            "threshold_exceedances_samples": 0,
            "extreme_clusters_count": 0,
            "alarm_episodes_count": 0,
            "matched_alarm_episodes_count": 0,
            "false_alarm_episodes_count": 0,
            "labelled_failure_events_count": 1,
            "event_precision": 0.0,
            "event_recall": 0.0,
        }
    )
    assert classify_failure_layer(no_exceedance)[0] == "score or threshold failure"

    alarms_never_match = pd.Series(
        {
            "threshold_exceedances_samples": 618,
            "extreme_clusters_count": 29,
            "alarm_episodes_count": 29,
            "matched_alarm_episodes_count": 0,
            "false_alarm_episodes_count": 29,
            "labelled_failure_events_count": 1,
            "event_precision": 0.0,
            "event_recall": 0.0,
        }
    )
    assert classify_failure_layer(alarms_never_match)[0] == "event-matching failure"

    detects_with_burden = pd.Series(
        {
            "threshold_exceedances_samples": 55_622,
            "extreme_clusters_count": 3_363,
            "alarm_episodes_count": 2_631,
            "matched_alarm_episodes_count": 1,
            "false_alarm_episodes_count": 2_627,
            "labelled_failure_events_count": 1,
            "event_precision": 1.0 / 2_631,
            "event_recall": 1.0,
        }
    )
    assert classify_failure_layer(detects_with_burden)[0] == "alarm-burden failure"

    # Without a labelled failure in the split no layer is identifiable, and saying so
    # is different from saying the method succeeded.
    unlabelled = no_exceedance.copy()
    unlabelled["labelled_failure_events_count"] = 0
    assert classify_failure_layer(unlabelled)[0] == "not estimable"


def test_every_row_is_classified(decomposition: pd.DataFrame) -> None:
    layers = {classify_failure_layer(row)[0] for _index, row in decomposition.iterrows()}
    assert layers <= {
        "score or threshold failure",
        "event-matching failure",
        "alarm-burden failure",
        "no operational failure at this layer",
    }


def test_column_weights_must_sum_to_the_column_count(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="must sum to 2"):
        _write_latex_table(
            frame,
            tmp_path / "t.tex",
            caption="c",
            label="tab:t",
            column_weights=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="entries for 2 columns"):
        _write_latex_table(
            frame,
            tmp_path / "t.tex",
            caption="c",
            label="tab:t",
            column_weights=(1.0, 0.5, 0.5),
        )


def test_long_tables_break_across_pages(tmp_path: Path) -> None:
    """A float that overruns the page silently drops its last rows."""
    frame = pd.DataFrame({"a": range(3), "b": range(3)})
    path = tmp_path / "long.tex"
    _write_latex_table(frame, path, caption="c", label="tab:long", long=True)
    text = path.read_text(encoding="utf-8")
    assert "\\begin{longtable}" in text
    assert "\\endhead" in text
    assert "\\begin{table}" not in text
