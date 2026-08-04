"""Global and local views of one held-out failure, built from a single alarm series.

The earlier timeline was local only: a three-hour window around the failure onset,
showing the score, the threshold, and the alarms inside that window. A local window
cannot be reconciled with the global alarm burden. The alarms it shows are selected
by proximity to the failure, so a reader sees the episodes that look useful and none
of the several thousand that generate the false-alarm rate reported in the event
tables. A figure that shows one matched alarm next to a failure, with no indication
that 2,626 other episodes exist, overstates the method whatever the caption says.

Both views here are derived from the same per-sample alarm flags, so the local window
is by construction a slice of the global series rather than a separate computation.
:func:`reconcile_timeline` asserts that relationship, along with the agreement between
the counts reported here and the counts in the event-level tables.

The global overview is built for every method in the benchmark, since the binned form
costs a few hundred rows each and the supplement compares burden across methods. The
per-sample local trace is built only for the registered method: it exists to explain
one detection, and nineteen methods of per-sample series is not a reviewable artifact.

Maintenance boundaries are the ends of labelled failure intervals, which is when the
compressor returned to service. The processed schema carries no separate maintenance
or repair log, so those are the only reset points that exist in the data, and the
metadata records how many were found rather than implying a fuller record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dyn_evt_pdm.evaluation.events import flags_to_events
from dyn_evt_pdm.types import BoolArray, EventInterval, FloatArray, IntArray

#: Hours of context shown before and after the failure onset in the local view.
LOCAL_WINDOW_BEFORE_HOURS = 6.0
LOCAL_WINDOW_AFTER_HOURS = 2.0

#: Bins used for the global overview. The full MetroPT test split is ~25 operating
#: days at 1 Hz, so per-sample plotting is neither legible nor necessary; the bin
#: counts are what the density panel draws.
GLOBAL_BIN_COUNT = 240


@dataclass(frozen=True)
class TimelineInputs:
    """One method's per-sample series over a dataset's full test split."""

    dataset_id: str
    failure_id: str
    method: str
    timestamps: np.ndarray[Any, Any]
    scores: FloatArray
    threshold: float
    exceedance_flags: BoolArray
    alarm_onset_flags: BoolArray
    target_flags: BoolArray
    regime_edges: FloatArray
    samples_per_day: int
    merge_gap: int
    horizon: int
    matching_tolerance_after: int
    threshold_quantile: float
    run_length: int
    target_region_variant: str


def _samples_per_hour(samples_per_day: int) -> float:
    return max(1.0, float(samples_per_day) / 24.0)


def _regime_labels(scores: FloatArray, edges: FloatArray) -> IntArray:
    """Label each sample with the regime the regime-conditioned method would use."""
    if edges.size == 0:
        return np.zeros(scores.shape[0], dtype=int)
    return np.asarray(np.digitize(scores, edges), dtype=int)


def build_global_overview(inputs: TimelineInputs, *, bins: int = GLOBAL_BIN_COUNT) -> pd.DataFrame:
    """Bin the full test split so every alarm episode is visible at once.

    The density channel is deliberately two numbers, not one. ``alarm_episodes`` counts
    onsets and ``time_under_warning_samples`` counts samples held under warning; a
    method can be low on one and high on the other, and reporting a single "density"
    would hide exactly the asymmetry the results section turns on.
    """
    total = int(inputs.target_flags.shape[0])
    if total == 0:
        return pd.DataFrame()

    alarms = flags_to_events(inputs.alarm_onset_flags, label="alarm", merge_gap=inputs.merge_gap)
    onset_indices = np.array([int(alarm.start) for alarm in alarms], dtype=int)
    under_warning = np.zeros(total, dtype=bool)
    for alarm in alarms:
        under_warning[int(alarm.start) : int(alarm.end) + 1] = True

    edges = np.linspace(0, total, num=bins + 1, dtype=int)
    samples_per_hour = _samples_per_hour(inputs.samples_per_day)
    failures = flags_to_events(inputs.target_flags, label="failure")
    failure_starts = np.array([int(event.start) for event in failures], dtype=int)
    # The end of a labelled failure interval is the return to service; the processed
    # schema records no other maintenance event.
    maintenance_resets = np.array([int(event.end) for event in failures], dtype=int)

    rows = []
    for index in range(bins):
        start, end = int(edges[index]), int(edges[index + 1])
        if end <= start:
            continue
        rows.append(
            {
                "dataset_id": inputs.dataset_id,
                "failure_id": inputs.failure_id,
                "method": inputs.method,
                "bin_index": index,
                "bin_start_index": start,
                "bin_end_index": end - 1,
                "bin_start_day": float(start / inputs.samples_per_day),
                "bin_end_day": float(end / inputs.samples_per_day),
                "bin_start_hours": float(start / samples_per_hour),
                "alarm_episodes": int(((onset_indices >= start) & (onset_indices < end)).sum()),
                "time_under_warning_samples": int(under_warning[start:end].sum()),
                "exceedance_samples": int(
                    np.asarray(inputs.exceedance_flags[start:end], dtype=bool).sum()
                ),
                "contains_failure_onset": bool(
                    ((failure_starts >= start) & (failure_starts < end)).any()
                ),
                "contains_maintenance_reset": bool(
                    ((maintenance_resets >= start) & (maintenance_resets < end)).any()
                ),
                "failure_samples": int(
                    np.asarray(inputs.target_flags[start:end], dtype=bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_local_trace(inputs: TimelineInputs) -> pd.DataFrame:
    """Per-sample view of the window around the first labelled failure onset."""
    failures = flags_to_events(inputs.target_flags, label="failure")
    if not failures:
        return pd.DataFrame()

    total = int(inputs.target_flags.shape[0])
    center = int(failures[0].start)
    samples_per_hour = _samples_per_hour(inputs.samples_per_day)
    start = max(0, center - int(LOCAL_WINDOW_BEFORE_HOURS * samples_per_hour))
    end = min(total, center + int(LOCAL_WINDOW_AFTER_HOURS * samples_per_hour))
    index = np.arange(start, end, dtype=int)

    alarms = flags_to_events(inputs.alarm_onset_flags, label="alarm", merge_gap=inputs.merge_gap)
    episode_id = np.full(total, -1, dtype=int)
    for position, alarm in enumerate(alarms):
        episode_id[int(alarm.start) : int(alarm.end) + 1] = position

    # Clusters here are runs of exceedances under the method's own event policy, the
    # same construction the decomposition counts, so the two artifacts agree.
    clusters = flags_to_events(
        inputs.exceedance_flags, label="cluster", merge_gap=max(0, inputs.merge_gap)
    )
    cluster_id = np.full(total, -1, dtype=int)
    for position, cluster in enumerate(clusters):
        cluster_id[int(cluster.start) : int(cluster.end) + 1] = position

    # The warning window is the horizon before each labelled failure onset: an alarm
    # inside it is what "matched" means under the event policy.
    warning_window = np.zeros(total, dtype=bool)
    for event in failures:
        warning_window[max(0, int(event.start) - inputs.horizon) : int(event.start) + 1] = True

    maintenance_reset = np.zeros(total, dtype=bool)
    for event in failures:
        maintenance_reset[int(event.end)] = True

    regimes = _regime_labels(np.asarray(inputs.scores, dtype=float), inputs.regime_edges)

    return pd.DataFrame(
        {
            "dataset_id": inputs.dataset_id,
            "failure_id": inputs.failure_id,
            "method": inputs.method,
            "test_index": index,
            "timestamp": inputs.timestamps[index].astype(str),
            "elapsed_hours": (index - center) / samples_per_hour,
            "score": np.asarray(inputs.scores, dtype=float)[index],
            "threshold": inputs.threshold,
            "regime_id": regimes[index],
            "is_exceedance": np.asarray(inputs.exceedance_flags, dtype=bool)[index],
            "cluster_id": cluster_id[index],
            "alarm_episode_id": episode_id[index],
            "in_warning_window": warning_window[index],
            "is_failure": np.asarray(inputs.target_flags, dtype=bool)[index],
            "is_maintenance_reset": maintenance_reset[index],
        }
    )


def build_timeline_metadata(
    inputs: TimelineInputs,
    *,
    local_trace: pd.DataFrame,
    matched_alarm_count: int,
    false_alarm_count: int,
    false_alarms_per_day: float,
    lead_time_samples: float | None,
    time_under_warning: int,
) -> dict[str, object]:
    """Every field the figure caption has to state to identify the configuration."""
    total = int(inputs.target_flags.shape[0])
    alarms = flags_to_events(inputs.alarm_onset_flags, label="alarm", merge_gap=inputs.merge_gap)
    samples_per_hour = _samples_per_hour(inputs.samples_per_day)

    if local_trace.empty:
        local_alarm_ids: set[int] = set()
    else:
        local_alarm_ids = {
            int(value) for value in local_trace["alarm_episode_id"].unique() if int(value) >= 0
        }
    failures = flags_to_events(inputs.target_flags, label="failure")
    local_false = sum(
        1
        for position in local_alarm_ids
        if not _alarm_is_matched(
            alarms[position], failures, inputs.horizon, inputs.matching_tolerance_after
        )
    )

    return {
        "dataset_id": inputs.dataset_id,
        "failure_id": inputs.failure_id,
        "method": inputs.method,
        "target_region_variant": inputs.target_region_variant,
        "threshold_quantile": float(inputs.threshold_quantile),
        "threshold_value": float(inputs.threshold),
        "run_length": int(inputs.run_length),
        "merge_gap": int(inputs.merge_gap),
        "warning_horizon_samples": int(inputs.horizon),
        "warning_horizon_hours": float(inputs.horizon / samples_per_hour),
        "test_duration_samples": total,
        "test_duration_days": float(total / inputs.samples_per_day),
        "global_alarm_episodes": len(alarms),
        "local_alarm_episodes": len(local_alarm_ids),
        "global_false_alarm_episodes": int(false_alarm_count),
        "local_false_alarm_episodes": int(local_false),
        "matched_alarm_episodes": int(matched_alarm_count),
        "false_alarms_per_operating_day": float(false_alarms_per_day),
        "lead_time_samples": "" if lead_time_samples is None else float(lead_time_samples),
        "lead_time_hours": (
            "" if lead_time_samples is None else float(lead_time_samples / samples_per_hour)
        ),
        "time_under_warning_samples": int(time_under_warning),
        "time_under_warning_fraction": float(time_under_warning / total) if total else 0.0,
        "labelled_failure_events": len(failures),
        # Every reset the data records is the end of a labelled failure interval; there
        # is no separate maintenance log, and this field says so by construction.
        "maintenance_resets_recorded": len(failures),
        "local_window_before_hours": LOCAL_WINDOW_BEFORE_HOURS,
        "local_window_after_hours": LOCAL_WINDOW_AFTER_HOURS,
    }


def _alarm_is_matched(
    alarm: EventInterval,
    failures: list[EventInterval],
    horizon: int,
    tolerance_after: int,
) -> bool:
    """Whether an alarm episode falls inside a labelled failure's matching window.

    The window is the one :class:`EarlyWarningPolicy` uses, ``[start - horizon,
    start + tolerance_after]``, not the whole failure interval. Using the interval
    counted every alarm raised during a multi-day failure as useful, which inflated
    the matched count and understated the local false-alarm count by the same amount.
    """
    for event in failures:
        window_start = max(0, int(event.start) - horizon)
        window_end = int(event.start) + tolerance_after
        if int(alarm.start) <= window_end and int(alarm.end) >= window_start:
            return True
    return False


def _as_int(metadata: dict[str, object], key: str) -> int:
    value = metadata[key]
    if not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"metadata field {key!r} is not numeric: {value!r}")
    return int(value)


def _as_float(metadata: dict[str, object], key: str) -> float:
    value = metadata[key]
    if not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"metadata field {key!r} is not numeric: {value!r}")
    return float(value)


def reconcile_timeline(
    *,
    global_overview: pd.DataFrame,
    local_trace: pd.DataFrame,
    metadata: dict[str, object],
    event_table_alarm_count: int,
    event_table_false_alarms_per_day: float,
) -> list[str]:
    """Return the reconciliation failures between the two views and the event tables.

    An empty list means the figure can be believed. Each string names a specific
    disagreement rather than a boolean, so a failure says which quantity drifted.
    """
    problems: list[str] = []

    binned_alarms = int(global_overview["alarm_episodes"].sum()) if not global_overview.empty else 0
    global_alarms = _as_int(metadata, "global_alarm_episodes")
    if binned_alarms != global_alarms:
        problems.append(
            f"global bins hold {binned_alarms} alarm episodes but metadata reports {global_alarms}"
        )

    if global_alarms != event_table_alarm_count:
        problems.append(
            f"timeline reports {global_alarms} alarm episodes but the event table reports "
            f"{event_table_alarm_count}"
        )

    local_alarms = _as_int(metadata, "local_alarm_episodes")
    if local_alarms > global_alarms:
        problems.append(
            f"local window holds {local_alarms} alarm episodes, more than the global "
            f"{global_alarms}; the local view must be a subset"
        )

    if not local_trace.empty and not global_overview.empty:
        local_start = int(local_trace["test_index"].min())
        local_end = int(local_trace["test_index"].max())
        global_end = int(global_overview["bin_end_index"].max())
        if local_start < 0 or local_end > global_end:
            problems.append(
                f"local window spans [{local_start}, {local_end}] outside the test split "
                f"[0, {global_end}]"
            )

    expected_rate = _as_float(metadata, "false_alarms_per_operating_day")
    if abs(expected_rate - event_table_false_alarms_per_day) > 1e-6:
        problems.append(
            f"timeline false alarms per day {expected_rate} does not match the event table "
            f"{event_table_false_alarms_per_day}"
        )

    if not local_trace.empty:
        thresholds = local_trace["threshold"].unique()
        if len(thresholds) != 1:
            problems.append(f"local trace carries {len(thresholds)} thresholds, expected one")
        elif abs(float(thresholds[0]) - _as_float(metadata, "threshold_value")) > 1e-9:
            problems.append("local trace threshold differs from the frozen test threshold")

    return problems
