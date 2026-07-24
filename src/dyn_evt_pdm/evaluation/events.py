"""Conversions between pointwise flags and event intervals."""

from __future__ import annotations

import numpy as np

from dyn_evt_pdm.types import BoolArray, EventInterval


def flags_to_events(flags: BoolArray, *, label: str = "alarm") -> list[EventInterval]:
    """Convert consecutive true samples into inclusive intervals."""

    values = np.asarray(flags, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        EventInterval(start=int(start), end=int(end), label=label)
        for start, end in zip(starts, ends, strict=True)
    ]


def match_events(
    predicted: list[EventInterval],
    observed: list[EventInterval],
) -> tuple[int, int, int]:
    """Greedily match each observed event to at most one overlapping prediction."""

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
