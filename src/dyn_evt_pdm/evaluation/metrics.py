"""Operational event-level metrics for predictive maintenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.evaluation.calibration import CalibrationBin, brier_score, reliability_bins
from dyn_evt_pdm.evaluation.events import (
    EarlyWarningPolicy,
    MatchingMethod,
    MatchResult,
    assignment_utility,
    match_event_intervals,
    match_events,
)
from dyn_evt_pdm.types import BoolArray, EventInterval, FloatArray


@dataclass(frozen=True, slots=True)
class EventUtilityCosts:
    """Cost/reward weights for event-level utility."""

    true_positive_reward: float = 1.0
    false_alarm_cost: float = 0.2
    missed_failure_cost: float = 2.0
    duplicate_alarm_cost: float = 0.1
    late_alarm_cost: float = 0.05
    warning_time_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class EventMetrics:
    """Event-level precision, recall and alarm burden."""

    precision: float
    recall: float
    f1: float
    false_alarm_events: int
    missed_events: int
    duplicate_alarm_rate: float
    mean_lead_time: float | None


@dataclass(frozen=True, slots=True)
class EventEvaluation:
    """Rigorous event-level evaluation under an explicit matching protocol."""

    precision: float
    recall: float
    f1: float
    true_positive_events: int
    false_alarm_events: int
    missed_failure_events: int
    duplicate_alarm_events: int
    false_alarm_events_per_operating_day: float | None
    duplicate_alarms_per_failure: float
    earliest_warning_lead_time: float | None
    median_warning_lead_time: float | None
    time_under_warning: int
    normalized_time_to_detection: float | None
    horizon_brier_score: float | None
    calibration_bins: tuple[CalibrationBin, ...]
    utility: float
    matching: MatchResult


def event_metrics(
    predicted: list[EventInterval],
    observed: list[EventInterval],
) -> EventMetrics:
    """Backward-compatible event metrics using overlap-only greedy matching."""

    true_positives, false_positives, false_negatives = match_events(predicted, observed)
    precision = true_positives / (true_positives + false_positives) if predicted else 0.0
    recall = true_positives / len(observed) if observed else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    overlaps_per_observed = [
        sum(prediction.overlaps(actual) for prediction in predicted) for actual in observed
    ]
    duplicates = sum(max(0, count - 1) for count in overlaps_per_observed)
    lead_times: list[float] = []
    for actual in observed:
        starts = [prediction.start for prediction in predicted if prediction.overlaps(actual)]
        if starts:
            lead_times.append(float(actual.start - min(starts)))
    return EventMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_alarm_events=false_positives,
        missed_events=false_negatives,
        duplicate_alarm_rate=duplicates / len(observed) if observed else 0.0,
        mean_lead_time=float(np.mean(lead_times)) if lead_times else None,
    )


def evaluate_event_predictions(
    predicted: list[EventInterval],
    failures: list[EventInterval],
    *,
    policy: EarlyWarningPolicy,
    method: MatchingMethod = "optimal",
    total_operating_time: int | None = None,
    samples_per_day: int = 86_400,
    horizon_probabilities: FloatArray | None = None,
    horizon_outcomes: BoolArray | None = None,
    utility_costs: EventUtilityCosts | None = None,
) -> EventEvaluation:
    """Evaluate alarm episodes against failure events, not individual timestamps."""

    if samples_per_day <= 0:
        raise ValueError("samples_per_day must be positive")
    if total_operating_time is not None and total_operating_time < 0:
        raise ValueError("total_operating_time must be non-negative")
    costs = utility_costs or EventUtilityCosts()
    matching = match_event_intervals(predicted, failures, policy=policy, method=method)
    true_positive = len(matching.assignments)
    false_alarm = len(matching.false_alarm_indices)
    duplicates = len(matching.duplicate_alarm_indices)
    missed = len(matching.missed_failure_indices)
    alarm_denominator = true_positive + false_alarm + duplicates
    precision = true_positive / alarm_denominator if alarm_denominator else 0.0
    recall = true_positive / len(failures) if failures else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    leads = [assignment.lead_time for assignment in matching.assignments]
    normalized = [assignment.normalized_time_to_detection for assignment in matching.assignments]
    false_alarm_rate = (
        false_alarm / (total_operating_time / samples_per_day)
        if total_operating_time and total_operating_time > 0
        else None
    )
    brier, bins = _risk_metrics(horizon_probabilities, horizon_outcomes)
    warning_time = time_under_warning(predicted)
    utility = event_utility(
        matching,
        warning_time=warning_time,
        costs=costs,
    )
    return EventEvaluation(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive_events=true_positive,
        false_alarm_events=false_alarm,
        missed_failure_events=missed,
        duplicate_alarm_events=duplicates,
        false_alarm_events_per_operating_day=(
            float(false_alarm_rate) if false_alarm_rate is not None else None
        ),
        duplicate_alarms_per_failure=duplicates / len(failures) if failures else 0.0,
        earliest_warning_lead_time=float(max(leads)) if leads else None,
        median_warning_lead_time=float(np.median(leads)) if leads else None,
        time_under_warning=warning_time,
        normalized_time_to_detection=float(np.mean(normalized)) if normalized else None,
        horizon_brier_score=brier,
        calibration_bins=tuple(bins),
        utility=utility,
        matching=matching,
    )


def time_under_warning(predicted: list[EventInterval]) -> int:
    """Total unique samples covered by alarm episodes."""

    if not predicted:
        return 0
    max_end = max(event.end for event in predicted)
    flags = np.zeros(max_end + 1, dtype=bool)
    for event in predicted:
        flags[event.start : event.end + 1] = True
    return int(flags.sum())


def event_utility(
    matching: MatchResult,
    *,
    warning_time: int,
    costs: EventUtilityCosts,
) -> float:
    """Compute configurable event-level utility."""

    utility = 0.0
    for assignment in matching.assignments:
        utility += assignment_utility(
            assignment,
            true_positive_reward=costs.true_positive_reward,
            late_penalty=costs.late_alarm_cost,
        )
    utility -= costs.false_alarm_cost * len(matching.false_alarm_indices)
    utility -= costs.missed_failure_cost * len(matching.missed_failure_indices)
    utility -= costs.duplicate_alarm_cost * len(matching.duplicate_alarm_indices)
    utility -= costs.warning_time_cost * warning_time
    return float(utility)


def _risk_metrics(
    probabilities: FloatArray | None,
    outcomes: BoolArray | None,
) -> tuple[float | None, list[CalibrationBin]]:
    if probabilities is None and outcomes is None:
        return None, []
    if probabilities is None or outcomes is None:
        raise ValueError("horizon probabilities and outcomes must be provided together")
    return brier_score(probabilities, outcomes), reliability_bins(probabilities, outcomes)
