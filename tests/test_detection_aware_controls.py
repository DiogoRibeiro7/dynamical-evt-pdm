"""Tests for detection-aware control comparison and the predeclared utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.pipelines.detection_aware_controls import (
    CONTROL_CONDITIONS,
    UtilityWeights,
    add_detection_columns,
    detection_aware_summary,
    joint_utility,
    joint_utility_array,
    utility_sensitivity,
)


def test_detection_dominates_alarm_burden() -> None:
    """A missed failure must not be purchasable with a lower alarm rate.

    This is the defect the phase exists to fix: comparing burden without detection
    rewards a region for raising no alarms while missing the event.
    """

    missed_but_quiet = joint_utility(
        detected=False,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=None,
    )
    detected_but_noisy = joint_utility(
        detected=True,
        false_alarm_events_per_day=50.0,
        alarm_coverage_fraction=0.2,
        median_warning_lead_time=1_000.0,
    )
    assert detected_but_noisy > missed_but_quiet


def test_missed_failure_earns_no_lead_time_credit() -> None:
    with_lead = joint_utility(
        detected=False,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=7_200.0,
    )
    without_lead = joint_utility(
        detected=False,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=None,
    )
    assert with_lead == pytest.approx(without_lead)


def test_exposure_is_charged_separately_from_episode_count() -> None:
    """A detector held permanently on must be penalised even with no false episodes."""

    quiet = joint_utility(
        detected=True,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=None,
    )
    always_on = joint_utility(
        detected=True,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=1.0,
        median_warning_lead_time=None,
    )
    assert always_on < quiet


def test_lead_time_reward_is_capped() -> None:
    weights = UtilityWeights()
    at_cap = joint_utility(
        detected=True,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=weights.lead_time_cap_samples,
    )
    beyond_cap = joint_utility(
        detected=True,
        false_alarm_events_per_day=0.0,
        alarm_coverage_fraction=0.0,
        median_warning_lead_time=weights.lead_time_cap_samples * 10.0,
    )
    assert at_cap == pytest.approx(beyond_cap)


def test_weights_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="detection weight"):
        UtilityWeights(detection=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        UtilityWeights(false_alarm_per_day=-1.0)
    with pytest.raises(ValueError, match="lead_time_cap_samples"):
        UtilityWeights(lead_time_cap_samples=0.0)


def _draws(n: int = 40, *, detect_fraction: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    detected = rng.random(n) < detect_fraction
    return pd.DataFrame(
        {
            "dataset_id": "metropt",
            "failure_id": "metropt_test_failure_001",
            "observed_method": "failure_prototype_region",
            "control_family": "random_occupancy",
            "draw": np.arange(n),
            "event_recall": detected.astype(float),
            "event_precision": rng.random(n) * 0.01,
            "false_alarm_events_per_day": rng.random(n) * 40.0,
            "median_warning_lead_time": np.where(detected, rng.random(n) * 3_000.0, np.nan),
            "time_under_warning": rng.integers(0, 5_000, n),
            "control_occupancy": rng.random(n) * 0.1,
        }
    )


def test_vectorised_utility_matches_the_scalar_form() -> None:
    frame = add_detection_columns(_draws(), total_samples=100_000)
    vectorised = joint_utility_array(frame)
    scalar = [
        joint_utility(
            detected=bool(row["detected"]),
            false_alarm_events_per_day=float(row["false_alarm_events_per_day"]),
            alarm_coverage_fraction=float(row["alarm_coverage_fraction"]),
            median_warning_lead_time=(
                float(row["median_warning_lead_time"])
                if pd.notna(row["median_warning_lead_time"])
                else None
            ),
        )
        for _index, row in frame.iterrows()
    ]
    assert np.allclose(vectorised, scalar)


def test_summary_reports_every_declared_condition() -> None:
    frame = add_detection_columns(_draws(), total_samples=100_000)
    summary = detection_aware_summary(
        frame,
        observed_detected=True,
        observed_recall=1.0,
        observed_occupancy=0.05,
        observed_false_alarms_per_day=9.4,
        observed_alarm_coverage=0.17,
        observed_lead_time=3_435.0,
    )
    assert list(summary["condition"]) == list(CONTROL_CONDITIONS)


def test_control_detection_rate_is_reported() -> None:
    frame = add_detection_columns(_draws(detect_fraction=0.8, seed=3), total_samples=100_000)
    summary = detection_aware_summary(
        frame,
        observed_detected=False,
        observed_recall=0.0,
        observed_occupancy=0.0,
        observed_false_alarms_per_day=0.0,
        observed_alarm_coverage=0.0,
        observed_lead_time=None,
    )
    rate = float(summary["control_detection_rate"].iloc[0])
    assert rate > 0.5


def test_a_missed_observed_region_is_never_credited() -> None:
    """MetroPT's zero-burden result is trivial because the event was missed."""

    frame = add_detection_columns(_draws(detect_fraction=0.9, seed=5), total_samples=100_000)
    summary = detection_aware_summary(
        frame,
        observed_detected=False,
        observed_recall=0.0,
        observed_occupancy=0.0,
        observed_false_alarms_per_day=0.0,
        observed_alarm_coverage=0.0,
        observed_lead_time=None,
    )
    for interpretation in summary["interpretation"]:
        assert "missed the failure" in str(interpretation)
        assert "exceeds" not in str(interpretation)


def test_detecting_condition_excludes_non_detecting_draws() -> None:
    frame = add_detection_columns(_draws(detect_fraction=0.5, seed=7), total_samples=100_000)
    summary = detection_aware_summary(
        frame,
        observed_detected=True,
        observed_recall=1.0,
        observed_occupancy=0.05,
        observed_false_alarms_per_day=10.0,
        observed_alarm_coverage=0.1,
        observed_lead_time=1_000.0,
    )
    all_draws = int(summary[summary["condition"] == "all_draws"]["control_draws"].iloc[0])
    detecting = int(summary[summary["condition"] == "detecting_draws"]["control_draws"].iloc[0])
    assert 0 < detecting < all_draws


def test_occupancy_matching_uses_the_control_occupancy() -> None:
    """Recording the observed occupancy on every draw makes this comparison impossible."""

    frame = add_detection_columns(_draws(seed=11), total_samples=100_000)
    assert frame["control_occupancy"].nunique() > 1
    summary = detection_aware_summary(
        frame,
        observed_detected=True,
        observed_recall=1.0,
        observed_occupancy=0.05,
        observed_false_alarms_per_day=10.0,
        observed_alarm_coverage=0.1,
        observed_lead_time=1_000.0,
    )
    matched = int(summary[summary["condition"] == "occupancy_matched"]["control_draws"].iloc[0])
    assert 0 < matched <= len(frame)


def test_sensitivity_sweeps_every_weight() -> None:
    frame = add_detection_columns(_draws(seed=13), total_samples=100_000)
    sweep = utility_sensitivity(
        frame,
        observed_detected=True,
        observed_false_alarms_per_day=10.0,
        observed_alarm_coverage=0.1,
        observed_lead_time=1_000.0,
    )
    assert set(sweep["varied_weight"]) == {
        "detection",
        "false_alarm_per_day",
        "time_under_warning",
        "lead_time",
    }
    assert sweep["multiplier"].nunique() == 5


def test_empty_draws_are_handled() -> None:
    empty = pd.DataFrame()
    assert add_detection_columns(empty, total_samples=10).empty
    assert detection_aware_summary(
        empty,
        observed_detected=True,
        observed_recall=1.0,
        observed_occupancy=0.0,
        observed_false_alarms_per_day=0.0,
        observed_alarm_coverage=0.0,
        observed_lead_time=None,
    ).empty
