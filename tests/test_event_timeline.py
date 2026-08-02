"""Reconciliation tests for the two-scale failure timeline.

The defect these guard against is a local figure that cannot be checked against the
global burden. A window around the failure shows the alarms nearest it and none of the
thousands elsewhere, so a reader sees a matched alarm and has no way to discover the
false-alarm rate the event tables report. Every check here ties one number in the
figure to the same number in the event tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.pipelines.event_timeline import (
    TimelineInputs,
    build_global_overview,
    build_local_trace,
    build_timeline_metadata,
    reconcile_timeline,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "real_data_matrix"


def _synthetic_inputs(
    *,
    total: int = 4_000,
    failure_start: int = 3_000,
    failure_end: int = 3_200,
    alarm_starts: tuple[int, ...] = (500, 1_500, 2_800),
    alarm_length: int = 40,
    horizon: int = 300,
    tolerance_after: int = 50,
    samples_per_day: int = 2_400,
) -> TimelineInputs:
    # 2,400 samples/day gives 100 samples/hour, so the 6h/2h local window spans
    # [failure_start - 600, failure_start + 200) and the fixtures can place alarms
    # inside and outside it deliberately.
    scores = np.zeros(total, dtype=float)
    exceedances = np.zeros(total, dtype=bool)
    onsets = np.zeros(total, dtype=bool)
    for start in alarm_starts:
        exceedances[start : start + alarm_length] = True
        onsets[start : start + alarm_length] = True
        scores[start : start + alarm_length] = 10.0
    targets = np.zeros(total, dtype=bool)
    targets[failure_start : failure_end + 1] = True
    return TimelineInputs(
        dataset_id="synthetic",
        failure_id="synthetic_failure_001",
        method="dynamical_evt_robust_score",
        timestamps=np.arange(total).astype(str),
        scores=scores,
        threshold=5.0,
        exceedance_flags=exceedances,
        alarm_onset_flags=onsets,
        target_flags=targets,
        regime_edges=np.array([1.0, 5.0], dtype=float),
        samples_per_day=samples_per_day,
        merge_gap=0,
        horizon=horizon,
        matching_tolerance_after=tolerance_after,
        threshold_quantile=0.98,
        run_length=3,
        target_region_variant="robust_max_abs_z",
    )


def test_global_bins_account_for_every_alarm_episode() -> None:
    """No episode may be dropped by binning; the density panel is the burden."""
    inputs = _synthetic_inputs()
    overview = build_global_overview(inputs, bins=40)
    assert int(overview["alarm_episodes"].sum()) == 3


def test_local_window_is_a_subset_of_the_global_series() -> None:
    inputs = _synthetic_inputs()
    overview = build_global_overview(inputs, bins=40)
    local = build_local_trace(inputs)
    assert not local.empty
    assert int(local["test_index"].min()) >= 0
    assert int(local["test_index"].max()) <= int(overview["bin_end_index"].max())

    local_ids = {int(value) for value in local["alarm_episode_id"].unique() if int(value) >= 0}
    assert local_ids <= {0, 1, 2}
    # The alarm at 2800 is inside the window; the ones at 500 and 1500 are not.
    assert local_ids == {2}


def test_reconciliation_passes_when_the_views_agree() -> None:
    inputs = _synthetic_inputs()
    overview = build_global_overview(inputs, bins=40)
    local = build_local_trace(inputs)
    metadata = build_timeline_metadata(
        inputs,
        local_trace=local,
        matched_alarm_count=1,
        false_alarm_count=2,
        false_alarms_per_day=0.5,
        lead_time_samples=200.0,
        time_under_warning=120,
    )
    problems = reconcile_timeline(
        global_overview=overview,
        local_trace=local,
        metadata=metadata,
        event_table_alarm_count=3,
        event_table_false_alarms_per_day=0.5,
    )
    assert problems == []


def test_reconciliation_reports_a_drifted_alarm_count() -> None:
    """The check has to fail when the figure and the table disagree."""
    inputs = _synthetic_inputs()
    overview = build_global_overview(inputs, bins=40)
    local = build_local_trace(inputs)
    metadata = build_timeline_metadata(
        inputs,
        local_trace=local,
        matched_alarm_count=1,
        false_alarm_count=2,
        false_alarms_per_day=0.5,
        lead_time_samples=200.0,
        time_under_warning=120,
    )
    problems = reconcile_timeline(
        global_overview=overview,
        local_trace=local,
        metadata=metadata,
        event_table_alarm_count=99,
        event_table_false_alarms_per_day=0.5,
    )
    assert any("event table reports 99" in problem for problem in problems)

    problems = reconcile_timeline(
        global_overview=overview,
        local_trace=local,
        metadata=metadata,
        event_table_alarm_count=3,
        event_table_false_alarms_per_day=7.5,
    )
    assert any("false alarms per day" in problem for problem in problems)


def test_matching_window_is_the_policy_window_not_the_failure_interval() -> None:
    """An alarm during a long failure interval is not a useful warning.

    Counting the whole interval as matched inflated the matched count for MetroPT,
    whose labelled failure spans days.
    """
    # Both alarms fall in the plotted window [2400, 3200); only 2,900 falls in the
    # matching window [2700, 3050]. The 3,100 alarm is inside the failure interval.
    inputs = _synthetic_inputs(
        failure_start=3_000,
        failure_end=3_900,
        alarm_starts=(2_900, 3_100),
        alarm_length=20,
        horizon=300,
        tolerance_after=50,
    )
    local = build_local_trace(inputs)
    metadata = build_timeline_metadata(
        inputs,
        local_trace=local,
        matched_alarm_count=1,
        false_alarm_count=1,
        false_alarms_per_day=0.25,
        lead_time_samples=100.0,
        time_under_warning=80,
    )
    assert metadata["local_alarm_episodes"] == 2
    # The alarm raised inside the failure interval is not a warning, so it counts as
    # false. Matching against the whole interval would have called it matched.
    assert metadata["local_false_alarm_episodes"] == 1


def test_threshold_in_the_trace_is_the_frozen_test_threshold() -> None:
    inputs = _synthetic_inputs()
    local = build_local_trace(inputs)
    assert local["threshold"].nunique() == 1
    assert float(local["threshold"].iloc[0]) == pytest.approx(inputs.threshold)


def test_maintenance_boundaries_are_the_ends_of_labelled_failures() -> None:
    inputs = _synthetic_inputs(failure_start=3_000, failure_end=3_150)
    local = build_local_trace(inputs)
    resets = local.loc[local["is_maintenance_reset"].astype(bool), "test_index"]
    assert list(resets) == [3_150]


@pytest.mark.skipif(
    not (ARTIFACTS / "event_timeline_metadata.csv").exists(),
    reason="real-data timeline artifacts are not present",
)
def test_real_timeline_reconciles_with_the_event_table() -> None:
    """The published figures must agree with the published tables."""
    metadata = pd.read_csv(ARTIFACTS / "event_timeline_metadata.csv")
    overview = pd.read_csv(ARTIFACTS / "event_timeline_overview.csv")
    local = pd.read_csv(ARTIFACTS / "event_timeline_trace.csv")
    baselines = pd.read_csv(ARTIFACTS / "event_baseline_comparison.csv")

    for _index, row in metadata.iterrows():
        dataset_id = row["dataset_id"]
        table_row = baselines[
            (baselines["dataset_id"] == dataset_id) & (baselines["method"] == row["method"])
        ]
        assert not table_row.empty, f"no event-table row for {dataset_id}"
        problems = reconcile_timeline(
            # The overview carries every benchmark method for the supplementary figures,
            # so reconciling one figure means selecting its method as well as its dataset.
            global_overview=overview[
                (overview["dataset_id"] == dataset_id) & (overview["method"] == row["method"])
            ],
            local_trace=local[local["dataset_id"] == dataset_id],
            metadata=dict(row),
            event_table_alarm_count=int(table_row["predicted_alarm_events"].iloc[0]),
            event_table_false_alarms_per_day=float(
                table_row["false_alarm_events_per_day"].iloc[0]
            ),
        )
        assert problems == [], f"{dataset_id}: {problems}"


@pytest.mark.skipif(
    not (ARTIFACTS / "event_timeline_overview.csv").exists(),
    reason="real-data timeline artifacts are not present",
)
def test_every_benchmark_method_has_a_global_overview() -> None:
    """The supplementary figures must cover the methods the benchmark reports."""
    overview = pd.read_csv(ARTIFACTS / "event_timeline_overview.csv")
    baselines = pd.read_csv(ARTIFACTS / "event_baseline_comparison.csv")
    for dataset_id in overview["dataset_id"].unique():
        expected = set(baselines.loc[baselines["dataset_id"] == dataset_id, "method"])
        covered = set(overview.loc[overview["dataset_id"] == dataset_id, "method"])
        assert expected <= covered, f"{dataset_id} missing {sorted(expected - covered)}"


@pytest.mark.skipif(
    not (ARTIFACTS / "event_timeline_overview.csv").exists(),
    reason="real-data timeline artifacts are not present",
)
def test_each_method_overview_matches_its_event_table_row() -> None:
    """Every supplementary strip's episode total must equal the table's alarm count."""
    overview = pd.read_csv(ARTIFACTS / "event_timeline_overview.csv")
    baselines = pd.read_csv(ARTIFACTS / "event_baseline_comparison.csv")
    merged = (
        overview.groupby(["dataset_id", "method"])["alarm_episodes"].sum().reset_index()
    )
    for _index, row in merged.iterrows():
        table_row = baselines[
            (baselines["dataset_id"] == row["dataset_id"])
            & (baselines["method"] == row["method"])
        ]
        assert not table_row.empty
        assert int(row["alarm_episodes"]) == int(table_row["predicted_alarm_events"].iloc[0]), (
            f"{row['dataset_id']}/{row['method']} timeline and table disagree"
        )


@pytest.mark.skipif(
    not (ARTIFACTS / "event_timeline_metadata.csv").exists(),
    reason="real-data timeline artifacts are not present",
)
def test_local_window_understates_the_real_burden() -> None:
    """The reason the global scale exists, asserted rather than assumed.

    If these ever became equal the global panel would be redundant; while they differ
    by two orders of magnitude, a local-only figure misrepresents the method.
    """
    metadata = pd.read_csv(ARTIFACTS / "event_timeline_metadata.csv")
    for _index, row in metadata.iterrows():
        assert int(row["local_alarm_episodes"]) < int(row["global_alarm_episodes"])
