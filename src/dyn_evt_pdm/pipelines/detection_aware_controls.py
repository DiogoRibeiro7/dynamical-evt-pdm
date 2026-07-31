"""Detection-aware comparison of an observed target region against matched controls.

Comparing alarm burden without conditioning on detection rewards a detector for not
detecting. A region that raises no alarms has a false-alarm rate of zero and looks
better than any control on burden alone, while missing the failure entirely. Every
comparison here therefore carries the detection outcome alongside the burden.

The joint utility is predeclared: weights are fixed in this module, justified below, and
varied in a sensitivity analysis rather than tuned after seeing results. Its components
are always reported separately, so a reader can see whether a utility difference comes
from detection, from burden, or from exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dyn_evt_pdm.types import FloatArray

#: Comparison conditions, in the order they are reported.
CONTROL_CONDITIONS: tuple[str, ...] = (
    "all_draws",
    "detecting_draws",
    "recall_matched",
    "occupancy_matched",
)


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """Predeclared weights for the joint operational utility.

    The default weights encode one maintenance stance, stated so it can be disagreed
    with: detecting the failure is worth a great deal more than any single day of false
    alarms; false alarms are charged per operating day; sustained warning exposure is
    charged separately because a detector can hold the system alarmed without raising
    many episodes; and early warning earns a bounded reward that cannot by itself
    outweigh a missed detection.

    ``detection`` dominates deliberately. With a detection weight of 100 and a
    false-alarm weight of 1, a method must save more than a hundred false alarms per day
    to justify missing a failure, which no method in this study approaches. That is the
    intended ordering: a missed failure is not purchasable with a lower alarm rate.
    """

    detection: float = 100.0
    false_alarm_per_day: float = 1.0
    time_under_warning: float = 10.0
    lead_time: float = 5.0
    lead_time_cap_samples: float = 7_200.0

    def __post_init__(self) -> None:
        if self.detection <= 0.0:
            raise ValueError("detection weight must be positive")
        if self.lead_time_cap_samples <= 0.0:
            raise ValueError("lead_time_cap_samples must be positive")
        for value in (self.false_alarm_per_day, self.time_under_warning, self.lead_time):
            if value < 0.0:
                raise ValueError("cost and reward weights must be non-negative")


def joint_utility(
    *,
    detected: bool,
    false_alarm_events_per_day: float,
    alarm_coverage_fraction: float,
    median_warning_lead_time: float | None,
    weights: UtilityWeights | None = None,
) -> float:
    """Return the predeclared joint utility for one detector outcome.

    Warning exposure enters as a fraction of the evaluated period rather than as a raw
    sample count, so the term is comparable across datasets of different lengths. The
    lead-time reward is capped and only paid when the failure was actually detected,
    which prevents a missed failure from collecting credit for an early false alarm.
    """

    costs = weights or UtilityWeights()
    utility = costs.detection if detected else 0.0
    utility -= costs.false_alarm_per_day * float(false_alarm_events_per_day or 0.0)
    utility -= costs.time_under_warning * float(alarm_coverage_fraction or 0.0)
    if detected and median_warning_lead_time is not None and np.isfinite(median_warning_lead_time):
        capped = min(float(median_warning_lead_time), costs.lead_time_cap_samples)
        utility += costs.lead_time * max(0.0, capped) / costs.lead_time_cap_samples
    return float(utility)


def joint_utility_array(
    frame: pd.DataFrame, *, weights: UtilityWeights | None = None
) -> FloatArray:
    """Return the predeclared joint utility for every row of a control frame."""

    costs = weights or UtilityWeights()
    detected = frame["detected"].astype(bool).to_numpy()
    false_alarms = pd.to_numeric(frame["false_alarm_events_per_day"], errors="coerce").to_numpy(
        dtype=float
    )
    coverage = pd.to_numeric(frame["alarm_coverage_fraction"], errors="coerce").to_numpy(
        dtype=float
    )
    lead = pd.to_numeric(frame["median_warning_lead_time"], errors="coerce").to_numpy(dtype=float)

    utility = np.where(detected, costs.detection, 0.0)
    utility -= costs.false_alarm_per_day * np.nan_to_num(false_alarms, nan=0.0)
    utility -= costs.time_under_warning * np.nan_to_num(coverage, nan=0.0)
    capped = np.clip(np.nan_to_num(lead, nan=0.0), 0.0, costs.lead_time_cap_samples)
    reward = costs.lead_time * capped / costs.lead_time_cap_samples
    # A missed failure collects no lead-time credit; otherwise an early false alarm
    # would be rewarded for arriving before a failure it never detected.
    utility += np.where(detected & np.isfinite(lead), reward, 0.0)
    return np.asarray(utility, dtype=np.float64)


def add_detection_columns(draws: pd.DataFrame, *, total_samples: int) -> pd.DataFrame:
    """Attach detection, exposure and predeclared utility columns to control draws."""

    if draws.empty:
        return draws
    frame = draws.copy()
    frame["detected"] = frame["event_recall"].astype(float) > 0.0
    if "alarm_coverage_fraction" not in frame.columns:
        frame["alarm_coverage_fraction"] = frame["time_under_warning"].astype(float) / float(
            max(1, total_samples)
        )
    frame["joint_utility"] = joint_utility_array(frame)
    return frame


def _select_condition(
    draws: pd.DataFrame,
    condition: str,
    *,
    observed_recall: float,
    observed_occupancy: float,
    occupancy_tolerance: float = 0.25,
) -> pd.DataFrame:
    """Return the control draws admitted under one comparison condition."""

    if condition == "all_draws":
        return draws
    if condition == "detecting_draws":
        return draws[draws["detected"].astype(bool)]
    if condition == "recall_matched":
        return draws[np.isclose(draws["event_recall"].astype(float), observed_recall, atol=1e-9)]
    if condition == "occupancy_matched":
        if "control_occupancy" not in draws.columns or not np.isfinite(observed_occupancy):
            return draws.iloc[0:0]
        occupancy = draws["control_occupancy"].astype(float)
        if observed_occupancy <= 0.0:
            return draws[occupancy <= occupancy_tolerance]
        ratio = occupancy / observed_occupancy
        return draws[(ratio >= 1.0 - occupancy_tolerance) & (ratio <= 1.0 + occupancy_tolerance)]
    raise ValueError(f"unknown control condition: {condition}")


def detection_aware_summary(
    draws: pd.DataFrame,
    *,
    observed_detected: bool,
    observed_recall: float,
    observed_occupancy: float,
    observed_false_alarms_per_day: float,
    observed_alarm_coverage: float,
    observed_lead_time: float | None,
    weights: UtilityWeights | None = None,
) -> pd.DataFrame:
    """Compare the observed region against controls under every declared condition.

    The detection rate of the control population is reported on every row. Without it a
    favourable percentile is ambiguous: a region can beat its controls on burden simply
    because the controls detect the failure and it does not.
    """

    if draws.empty:
        return pd.DataFrame()

    observed_utility = joint_utility(
        detected=observed_detected,
        false_alarm_events_per_day=observed_false_alarms_per_day,
        alarm_coverage_fraction=observed_alarm_coverage,
        median_warning_lead_time=observed_lead_time,
        weights=weights,
    )

    records: list[dict[str, object]] = []
    group_columns = [
        column
        for column in ("dataset_id", "failure_id", "observed_method", "control_family")
        if column in draws.columns
    ]
    for keys, group in draws.groupby(group_columns, dropna=False, sort=True):
        key_values = dict(
            zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=True)
        )
        for condition in CONTROL_CONDITIONS:
            selected = _select_condition(
                group,
                condition,
                observed_recall=observed_recall,
                observed_occupancy=observed_occupancy,
            )
            utilities = selected["joint_utility"].astype(float).to_numpy()
            detection_rate = (
                float(np.mean(group["detected"].astype(bool).to_numpy())) if len(group) else np.nan
            )
            records.append(
                {
                    **key_values,
                    "condition": condition,
                    "control_draws": int(len(selected)),
                    "control_detection_rate": detection_rate,
                    "observed_detected": observed_detected,
                    "observed_utility": observed_utility,
                    "control_utility_median": (
                        float(np.median(utilities)) if utilities.size else float("nan")
                    ),
                    "control_utility_p025": (
                        float(np.quantile(utilities, 0.025)) if utilities.size else float("nan")
                    ),
                    "control_utility_p975": (
                        float(np.quantile(utilities, 0.975)) if utilities.size else float("nan")
                    ),
                    "observed_percentile": (
                        float(np.mean(utilities <= observed_utility)) if utilities.size else np.nan
                    ),
                    "interpretation": _interpret(
                        observed_detected=observed_detected,
                        detection_rate=detection_rate,
                        selected=selected,
                        observed_utility=observed_utility,
                        condition=condition,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _interpret(
    *,
    observed_detected: bool,
    detection_rate: float,
    selected: pd.DataFrame,
    observed_utility: float,
    condition: str,
) -> str:
    """Describe one comparison, refusing to credit a region that did not detect."""

    if selected.empty:
        return "no control draw satisfies this condition"
    if not observed_detected:
        if np.isfinite(detection_rate) and detection_rate > 0.0:
            return (
                "observed region missed the failure while "
                f"{detection_rate * 100:.0f} percent of controls detected it"
            )
        return "observed region missed the failure; controls did not detect it either"
    utilities = selected["joint_utility"].astype(float).to_numpy()
    percentile = float(np.mean(utilities <= observed_utility))
    if percentile >= 0.95:
        return f"observed exceeds {condition.replace('_', ' ')} controls"
    if percentile <= 0.05:
        return f"observed below {condition.replace('_', ' ')} controls"
    return f"observed within the {condition.replace('_', ' ')} control range"


def utility_sensitivity(
    draws: pd.DataFrame,
    *,
    observed_detected: bool,
    observed_false_alarms_per_day: float,
    observed_alarm_coverage: float,
    observed_lead_time: float | None,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> pd.DataFrame:
    """Vary each declared weight and report whether the conclusion survives.

    A conclusion that holds only at one weighting is a property of the weighting, so the
    sensitivity sweep is part of the result rather than an appendix to it.
    """

    if draws.empty:
        return pd.DataFrame()
    # The sweep runs on the detecting subset as well as on all draws. Sweeping only the
    # unconditioned population would test the robustness of a comparison that is not the
    # one the conclusion rests on: a region can look stable against all controls while
    # its standing against controls that also detect moves under reweighting.
    populations: dict[str, pd.DataFrame] = {"all_draws": draws}
    if "detected" in draws.columns:
        detecting = draws[draws["detected"].astype(bool)]
        if not detecting.empty:
            populations["detecting_draws"] = detecting

    records: list[dict[str, object]] = []
    for field in ("detection", "false_alarm_per_day", "time_under_warning", "lead_time"):
        for multiplier in multipliers:
            base = UtilityWeights()
            weights = UtilityWeights(
                detection=base.detection * (multiplier if field == "detection" else 1.0),
                false_alarm_per_day=base.false_alarm_per_day
                * (multiplier if field == "false_alarm_per_day" else 1.0),
                time_under_warning=base.time_under_warning
                * (multiplier if field == "time_under_warning" else 1.0),
                lead_time=base.lead_time * (multiplier if field == "lead_time" else 1.0),
            )
            observed_utility = joint_utility(
                detected=observed_detected,
                false_alarm_events_per_day=observed_false_alarms_per_day,
                alarm_coverage_fraction=observed_alarm_coverage,
                median_warning_lead_time=observed_lead_time,
                weights=weights,
            )
            for condition, population in populations.items():
                control = joint_utility_array(population, weights=weights)
                records.append(
                    {
                        "condition": condition,
                        "varied_weight": field,
                        "multiplier": multiplier,
                        "control_draws": int(len(population)),
                        "observed_utility": observed_utility,
                        "control_utility_median": (
                            float(np.median(control)) if control.size else np.nan
                        ),
                        "observed_percentile": (
                            float(np.mean(control <= observed_utility)) if control.size else np.nan
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)
