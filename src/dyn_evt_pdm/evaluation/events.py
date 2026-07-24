"""Event construction and matching protocols for predictive maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from dyn_evt_pdm.types import BoolArray, EventInterval, IntArray

MatchingMethod = Literal["greedy", "optimal"]


@dataclass(frozen=True, slots=True)
class EarlyWarningPolicy:
    """Explicit interval policy for assigning alarms to failures."""

    horizon: int = 0
    tolerance_before: int = 0
    tolerance_after: int = 0
    allow_prediction_reuse: bool = False

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.tolerance_before < 0:
            raise ValueError("tolerance_before must be non-negative")
        if self.tolerance_after < 0:
            raise ValueError("tolerance_after must be non-negative")


@dataclass(frozen=True, slots=True)
class EventAssignment:
    """One alarm-to-failure assignment under a stated protocol."""

    prediction_index: int
    failure_index: int
    lead_time: int
    normalized_time_to_detection: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Complete event-matching result, including duplicate and false-alarm sets."""

    assignments: tuple[EventAssignment, ...]
    eligible_pairs: tuple[tuple[int, int], ...]
    false_alarm_indices: tuple[int, ...]
    duplicate_alarm_indices: tuple[int, ...]
    missed_failure_indices: tuple[int, ...]
    method: MatchingMethod
    policy: EarlyWarningPolicy


def flags_to_events(
    flags: BoolArray,
    *,
    label: str = "alarm",
    merge_gap: int = 0,
) -> list[EventInterval]:
    """Convert pointwise flags into inclusive events with a configurable merge gap."""

    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    values = np.asarray(flags, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    raw_events = [
        EventInterval(start=int(start), end=int(end), label=label)
        for start, end in zip(starts, ends, strict=True)
    ]
    return merge_events(raw_events, merge_gap=merge_gap)


def merge_events(events: list[EventInterval], *, merge_gap: int) -> list[EventInterval]:
    """Merge events separated by at most ``merge_gap`` non-event samples."""

    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if not events:
        return []
    ordered = sorted(events, key=lambda event: (event.start, event.end))
    merged: list[EventInterval] = [ordered[0]]
    for event in ordered[1:]:
        previous = merged[-1]
        gap = event.start - previous.end - 1
        if gap <= merge_gap:
            merged[-1] = EventInterval(
                start=previous.start,
                end=max(previous.end, event.end),
                label=previous.label,
            )
        else:
            merged.append(event)
    return merged


def early_warning_window(failure: EventInterval, policy: EarlyWarningPolicy) -> EventInterval:
    """Return the assignment window ending at failure onset plus explicit tolerance."""

    start = max(0, failure.start - policy.horizon - policy.tolerance_before)
    end = failure.start + policy.tolerance_after
    return EventInterval(start=start, end=end, label=f"{failure.label}_warning_window")


def match_events(
    predicted: list[EventInterval],
    observed: list[EventInterval],
) -> tuple[int, int, int]:
    """Backward-compatible greedy overlap matching."""

    unmatched_predictions = set(range(len(predicted)))
    true_positives = 0
    for actual in observed:
        candidates = [index for index in unmatched_predictions if predicted[index].overlaps(actual)]
        if candidates:
            best = min(candidates, key=lambda index: abs(predicted[index].start - actual.start))
            unmatched_predictions.remove(best)
            true_positives += 1
    false_positives = len(unmatched_predictions)
    false_negatives = len(observed) - true_positives
    return true_positives, false_positives, false_negatives


def match_event_intervals(
    predictions: list[EventInterval],
    failures: list[EventInterval],
    *,
    policy: EarlyWarningPolicy,
    method: MatchingMethod = "greedy",
) -> MatchResult:
    """Match alarm events to failure events using an explicit early-warning policy."""

    if method not in {"greedy", "optimal"}:
        raise ValueError("method must be greedy or optimal")
    eligible = _eligible_pair_map(predictions, failures, policy)
    if policy.allow_prediction_reuse:
        assignments = _reuse_assignments(predictions, failures, eligible, policy)
    elif method == "greedy":
        assignments = _greedy_assignments(predictions, failures, eligible, policy)
    else:
        assignments = _optimal_assignments(predictions, failures, eligible, policy)

    assigned_failures = {assignment.failure_index for assignment in assignments}
    eligible_predictions = {prediction_index for prediction_index, _failure_index in eligible}
    duplicate_indices = _duplicate_alarm_indices(eligible, assignments)
    false_indices = tuple(
        index for index in range(len(predictions)) if index not in eligible_predictions
    )
    missed_indices = tuple(
        index for index in range(len(failures)) if index not in assigned_failures
    )
    return MatchResult(
        assignments=tuple(assignments),
        eligible_pairs=tuple(sorted(eligible)),
        false_alarm_indices=false_indices,
        duplicate_alarm_indices=duplicate_indices,
        missed_failure_indices=missed_indices,
        method=method,
        policy=policy,
    )


def assignment_utility(
    assignment: EventAssignment,
    *,
    true_positive_reward: float = 1.0,
    late_penalty: float = 1.0,
) -> float:
    """Utility for one assigned event under configurable costs."""

    lateness = max(0, -assignment.lead_time)
    return float(true_positive_reward - late_penalty * lateness)


def bootstrap_event_indices(
    groups: IntArray,
    *,
    repetitions: int,
    random_seed: int = 42,
) -> tuple[tuple[int, ...], ...]:
    """Bootstrap complete failures, days or vehicles by group id, never individual rows."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    group_values = np.asarray(groups)
    if group_values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique_groups = np.unique(group_values)
    if len(unique_groups) == 0:
        raise ValueError("at least one group is required")
    rng = np.random.default_rng(random_seed)
    samples: list[tuple[int, ...]] = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices: list[int] = []
        for group in sampled_groups:
            indices.extend(int(index) for index in np.flatnonzero(group_values == group))
        samples.append(tuple(indices))
    return tuple(samples)


def _eligible_pair_map(
    predictions: list[EventInterval],
    failures: list[EventInterval],
    policy: EarlyWarningPolicy,
) -> set[tuple[int, int]]:
    eligible: set[tuple[int, int]] = set()
    for prediction_index, prediction in enumerate(predictions):
        for failure_index, failure in enumerate(failures):
            if prediction.overlaps(early_warning_window(failure, policy)):
                eligible.add((prediction_index, failure_index))
    return eligible


def _reuse_assignments(
    predictions: list[EventInterval],
    failures: list[EventInterval],
    eligible: set[tuple[int, int]],
    policy: EarlyWarningPolicy,
) -> list[EventAssignment]:
    return [
        _make_assignment(
            predictions[prediction_index],
            failures[failure_index],
            prediction_index,
            failure_index,
            policy,
        )
        for prediction_index, failure_index in sorted(
            eligible,
            key=lambda pair: _assignment_cost(predictions[pair[0]], failures[pair[1]], policy),
        )
    ]


def _greedy_assignments(
    predictions: list[EventInterval],
    failures: list[EventInterval],
    eligible: set[tuple[int, int]],
    policy: EarlyWarningPolicy,
) -> list[EventAssignment]:
    assignments: list[EventAssignment] = []
    used_predictions: set[int] = set()
    used_failures: set[int] = set()
    ordered_pairs = sorted(
        eligible,
        key=lambda pair: (
            _assignment_cost(predictions[pair[0]], failures[pair[1]], policy),
            failures[pair[1]].start,
            predictions[pair[0]].start,
        ),
    )
    for prediction_index, failure_index in ordered_pairs:
        if prediction_index in used_predictions or failure_index in used_failures:
            continue
        assignments.append(
            _make_assignment(
                predictions[prediction_index],
                failures[failure_index],
                prediction_index,
                failure_index,
                policy,
            )
        )
        used_predictions.add(prediction_index)
        used_failures.add(failure_index)
    return assignments


def _optimal_assignments(
    predictions: list[EventInterval],
    failures: list[EventInterval],
    eligible: set[tuple[int, int]],
    policy: EarlyWarningPolicy,
) -> list[EventAssignment]:
    if not eligible:
        return []
    cost = np.full((len(predictions), len(failures)), 1e9, dtype=np.float64)
    for prediction_index, failure_index in eligible:
        cost[prediction_index, failure_index] = _assignment_cost(
            predictions[prediction_index],
            failures[failure_index],
            policy,
        )
    row_ind, col_ind = linear_sum_assignment(cost)
    assignments: list[EventAssignment] = []
    for prediction_index, failure_index in zip(row_ind, col_ind, strict=True):
        if cost[prediction_index, failure_index] >= 1e9:
            continue
        assignments.append(
            _make_assignment(
                predictions[int(prediction_index)],
                failures[int(failure_index)],
                int(prediction_index),
                int(failure_index),
                policy,
            )
        )
    return sorted(assignments, key=lambda assignment: assignment.failure_index)


def _make_assignment(
    prediction: EventInterval,
    failure: EventInterval,
    prediction_index: int,
    failure_index: int,
    policy: EarlyWarningPolicy,
) -> EventAssignment:
    lead_time = failure.start - prediction.start
    denominator = max(1, policy.horizon + policy.tolerance_before + policy.tolerance_after)
    normalized = np.clip(lead_time / denominator, -1.0, 1.0)
    return EventAssignment(
        prediction_index=prediction_index,
        failure_index=failure_index,
        lead_time=int(lead_time),
        normalized_time_to_detection=float(normalized),
    )


def _assignment_cost(
    prediction: EventInterval,
    failure: EventInterval,
    policy: EarlyWarningPolicy,
) -> float:
    window = early_warning_window(failure, policy)
    if prediction.start < window.start:
        return float(window.start - prediction.start + 10_000)
    if prediction.start > window.end:
        return float(prediction.start - window.end + 10_000)
    lead_time = failure.start - prediction.start
    return float(-lead_time)


def _duplicate_alarm_indices(
    eligible: set[tuple[int, int]],
    assignments: list[EventAssignment],
) -> tuple[int, ...]:
    by_failure: dict[int, list[int]] = {}
    for prediction_index, failure_index in eligible:
        by_failure.setdefault(failure_index, []).append(prediction_index)
    assigned_by_failure = {
        assignment.failure_index: assignment.prediction_index for assignment in assignments
    }
    duplicates: set[int] = set()
    for failure_index, prediction_indices in by_failure.items():
        assigned_prediction = assigned_by_failure.get(failure_index)
        if assigned_prediction is None:
            continue
        for index in prediction_indices:
            if index != assigned_prediction:
                duplicates.add(index)
    return tuple(sorted(duplicates))
