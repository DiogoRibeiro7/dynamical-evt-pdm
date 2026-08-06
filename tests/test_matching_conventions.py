"""Pin the event-matching conventions the manuscript states.

Review asked for these to be formal, because the reported precision depends on choices
that are invisible in a metric name: whether matching is one-to-one, whether duplicates
enter the precision denominator, whether an alarm raised inside a long failure counts as
a warning, and what happens to an alarm eligible for two failures. The manuscript now
states each of them, and these tests hold the implementation to what it states.
"""

from __future__ import annotations

from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, match_event_intervals
from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions
from dyn_evt_pdm.types import EventInterval

POLICY = EarlyWarningPolicy(horizon=100, tolerance_after=10)


def _alarm(start: int, end: int | None = None) -> EventInterval:
    return EventInterval(start=start, end=start if end is None else end, label="alarm")


def _failure(start: int, end: int) -> EventInterval:
    return EventInterval(start=start, end=end, label="failure")


def test_window_ends_at_onset_plus_tolerance_not_at_the_failure_end() -> None:
    """An alarm deep inside a long failure is not an early warning."""
    failure = _failure(1_000, 5_000)
    inside = match_event_intervals([_alarm(3_000)], [failure], policy=POLICY)
    assert not inside.assignments
    assert inside.false_alarm_indices == (0,)

    before = match_event_intervals([_alarm(950)], [failure], policy=POLICY)
    assert len(before.assignments) == 1


def test_an_alarm_starting_before_the_window_and_extending_into_it_is_eligible() -> None:
    failure = _failure(1_000, 2_000)
    result = match_event_intervals([_alarm(800, 950)], [failure], policy=POLICY)
    assert len(result.assignments) == 1


def test_matching_is_one_to_one_and_the_surplus_becomes_duplicates() -> None:
    failure = _failure(1_000, 2_000)
    alarms = [_alarm(920), _alarm(940), _alarm(960)]
    result = match_event_intervals(alarms, [failure], policy=POLICY)
    assert len(result.assignments) == 1
    assert len(result.duplicate_alarm_indices) == 2
    assert result.false_alarm_indices == ()


def test_duplicates_enter_the_precision_denominator() -> None:
    """Precision is matched alarms over all alarms raised, as the manuscript states."""
    failure = _failure(1_000, 2_000)
    alarms = [_alarm(920), _alarm(940), _alarm(960), _alarm(50_000)]
    evaluation = evaluate_event_predictions(
        alarms, [failure], policy=POLICY, method="optimal", total_operating_time=100_000
    )
    assert evaluation.duplicate_alarm_events == 2
    # One matched of four raised: 1/4, not 1/2 as excluding duplicates would give.
    assert evaluation.precision == 0.25
    assert evaluation.recall == 1.0


def test_an_alarm_eligible_for_two_failures_is_assigned_to_only_one() -> None:
    failures = [_failure(1_000, 1_100), _failure(1_050, 1_200)]
    result = match_event_intervals([_alarm(990)], failures, policy=POLICY)
    assert len(result.assignments) == 1
    assert len(result.missed_failure_indices) == 1


def test_two_alarms_and_two_failures_match_pairwise_rather_than_collapsing() -> None:
    failures = [_failure(1_000, 1_100), _failure(5_000, 5_100)]
    alarms = [_alarm(950), _alarm(4_950)]
    result = match_event_intervals(alarms, failures, policy=POLICY, method="optimal")
    assert len(result.assignments) == 2
    assert result.missed_failure_indices == ()
    assert result.duplicate_alarm_indices == ()


def test_a_permanently_on_detector_attains_perfect_episode_metrics() -> None:
    """The degeneracy the manuscript warns about, asserted rather than described.

    One merged episode covering everything is matched, leaves nothing unassigned and
    nothing ineligible, so episode-level precision and recall are both one. Only the
    exposure axis distinguishes it from a useful detector.
    """
    total = 100_000
    evaluation = evaluate_event_predictions(
        [_alarm(0, total - 1)],
        [_failure(50_000, 50_100)],
        policy=POLICY,
        method="optimal",
        total_operating_time=total,
    )
    assert evaluation.precision == 1.0
    assert evaluation.recall == 1.0
    assert evaluation.false_alarm_events_per_operating_day == 0.0
    assert evaluation.time_under_warning == total
