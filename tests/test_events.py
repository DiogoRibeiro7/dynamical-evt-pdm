import numpy as np

from dyn_evt_pdm.evaluation.events import (
    EarlyWarningPolicy,
    bootstrap_event_indices,
    early_warning_window,
    flags_to_events,
    match_event_intervals,
)
from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions, event_metrics
from dyn_evt_pdm.types import EventInterval


def test_flags_to_events() -> None:
    flags = np.array([False, True, True, False, True])
    events = flags_to_events(flags)
    assert [(event.start, event.end) for event in events] == [(1, 2), (4, 4)]


def test_event_metrics_penalise_duplicate_alarms() -> None:
    observed = [EventInterval(10, 20)]
    predicted = [EventInterval(10, 12), EventInterval(16, 18)]
    metrics = event_metrics(predicted, observed)
    assert metrics.recall == 1.0
    assert metrics.duplicate_alarm_rate == 1.0


def test_flags_to_events_merges_touching_and_nearby_intervals() -> None:
    flags = np.array([False, True, True, False, True, False, False, True])

    events = flags_to_events(flags, merge_gap=1)

    assert [(event.start, event.end) for event in events] == [(1, 4), (7, 7)]


def test_early_warning_policy_rejects_alarms_before_window() -> None:
    failure = EventInterval(100, 120, label="failure")
    policy = EarlyWarningPolicy(horizon=20, tolerance_before=0, tolerance_after=5)
    before_window = [EventInterval(70, 75)]
    in_window = [EventInterval(80, 82)]

    assert early_warning_window(failure, policy) == EventInterval(
        80, 105, label="failure_warning_window"
    )
    assert not match_event_intervals(before_window, [failure], policy=policy).assignments
    assert match_event_intervals(in_window, [failure], policy=policy).assignments[0].lead_time == 20


def test_optimal_matching_prevents_one_alarm_from_satisfying_multiple_failures() -> None:
    failures = [EventInterval(100, 110), EventInterval(130, 140)]
    alarms = [EventInterval(95, 96), EventInterval(125, 126)]
    policy = EarlyWarningPolicy(horizon=40, allow_prediction_reuse=False)

    result = match_event_intervals(alarms, failures, policy=policy, method="optimal")

    assert len(result.assignments) == 2
    assert {assignment.prediction_index for assignment in result.assignments} == {0, 1}
    assert {assignment.failure_index for assignment in result.assignments} == {0, 1}


def test_reuse_policy_can_assign_one_alarm_to_multiple_failures() -> None:
    failures = [EventInterval(100, 110), EventInterval(108, 120)]
    alarms = [EventInterval(99, 101)]
    policy = EarlyWarningPolicy(horizon=10, tolerance_after=2, allow_prediction_reuse=True)

    result = match_event_intervals(alarms, failures, policy=policy, method="optimal")

    assert len(result.assignments) == 2
    assert {assignment.failure_index for assignment in result.assignments} == {0, 1}


def test_event_evaluation_handles_nested_duplicates_and_false_alarms() -> None:
    failures = [EventInterval(100, 130)]
    alarms = [
        EventInterval(70, 75),
        EventInterval(88, 90),
        EventInterval(95, 105),
        EventInterval(200, 205),
    ]
    policy = EarlyWarningPolicy(horizon=20, tolerance_before=0, tolerance_after=10)
    evaluation = evaluate_event_predictions(
        alarms,
        failures,
        policy=policy,
        method="optimal",
        total_operating_time=240,
        samples_per_day=24,
        horizon_probabilities=np.array([0.1, 0.8]),
        horizon_outcomes=np.array([False, True]),
    )

    assert evaluation.true_positive_events == 1
    assert evaluation.duplicate_alarm_events == 1
    assert evaluation.false_alarm_events == 2
    assert evaluation.false_alarm_events_per_operating_day == 0.2
    assert evaluation.earliest_warning_lead_time == 12
    assert evaluation.median_warning_lead_time == 12
    assert evaluation.time_under_warning == 26
    assert evaluation.horizon_brier_score is not None
    assert evaluation.utility < 1.0


def test_event_evaluation_no_alarms_and_no_failures() -> None:
    policy = EarlyWarningPolicy(horizon=5)

    no_alarm = evaluate_event_predictions([], [EventInterval(10, 20)], policy=policy)
    no_failure = evaluate_event_predictions([EventInterval(1, 2)], [], policy=policy)

    assert no_alarm.recall == 0.0
    assert no_alarm.missed_failure_events == 1
    assert no_failure.precision == 0.0
    assert no_failure.false_alarm_events == 1


def test_bootstrap_samples_complete_groups() -> None:
    groups = np.array([1, 1, 2, 2, 3])

    samples = bootstrap_event_indices(groups, repetitions=3, random_seed=2)

    assert len(samples) == 3
    for sample in samples:
        assert len(sample) >= 3
        for group in set(groups[index] for index in sample):
            members = set(np.flatnonzero(groups == group))
            assert members.issubset(sample)
