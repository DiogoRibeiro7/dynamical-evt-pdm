"""Operational event-level metrics for predictive maintenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.evaluation.events import match_events
from dyn_evt_pdm.types import EventInterval


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


def event_metrics(
    predicted: list[EventInterval],
    observed: list[EventInterval],
) -> EventMetrics:
    """Evaluate alarm episodes rather than individual samples."""

    true_positives, false_positives, false_negatives = match_events(predicted, observed)
    precision = true_positives / (true_positives + false_positives) if predicted else 0.0
    recall = true_positives / len(observed) if observed else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    overlaps_per_observed = [sum(prediction.overlaps(actual) for prediction in predicted) for actual in observed]
    duplicates = sum(max(0, count - 1) for count in overlaps_per_observed)
    duplicate_rate = duplicates / len(observed) if observed else 0.0

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
        duplicate_alarm_rate=float(duplicate_rate),
        mean_lead_time=float(np.mean(lead_times)) if lead_times else None,
    )
