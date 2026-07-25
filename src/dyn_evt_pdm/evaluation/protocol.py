"""Frozen event-evaluation protocol specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, MatchingMethod
from dyn_evt_pdm.evaluation.metrics import EventUtilityCosts

BootstrapUnit = Literal["failure", "day", "operating_cycle", "vehicle", "dataset_specific", "none"]


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Versioned, hashable protocol for event-level evaluation."""

    protocol_id: str
    version: str
    dataset_id: str
    independent_unit: str
    train_partition: str
    validation_partition: str
    calibration_partition: str
    test_partition: str
    exclusion_column: str | None
    maintenance_boundary_column: str | None
    warning_horizons: tuple[int, ...]
    event_onset_definition: str
    event_end_definition: str
    alarm_merge_gap: int
    cooldown: int
    matching_tolerance_before: int
    matching_tolerance_after: int
    matching_method: MatchingMethod
    bootstrap_unit: BootstrapUnit
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    utility_costs: EventUtilityCosts
    frozen_timestamp_utc: str

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.independent_unit:
            raise ValueError("independent_unit must not be empty")
        if not self.warning_horizons:
            raise ValueError("warning_horizons must not be empty")
        if any(horizon < 1 for horizon in self.warning_horizons):
            raise ValueError("warning_horizons must be positive")
        if self.alarm_merge_gap < 0:
            raise ValueError("alarm_merge_gap must be non-negative")
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        if self.matching_tolerance_before < 0:
            raise ValueError("matching_tolerance_before must be non-negative")
        if self.matching_tolerance_after < 0:
            raise ValueError("matching_tolerance_after must be non-negative")
        if self.matching_method not in {"greedy", "optimal"}:
            raise ValueError("matching_method must be greedy or optimal")
        if self.bootstrap_unit not in {
            "failure",
            "day",
            "operating_cycle",
            "vehicle",
            "dataset_specific",
            "none",
        }:
            raise ValueError("unsupported bootstrap_unit")
        if not self.primary_metrics:
            raise ValueError("primary_metrics must not be empty")
        if "T" not in self.frozen_timestamp_utc or not self.frozen_timestamp_utc.endswith("Z"):
            raise ValueError("frozen_timestamp_utc must be an ISO UTC timestamp ending in Z")

    @property
    def protocol_hash(self) -> str:
        """Stable hash over every protocol field."""

        return protocol_hash(self)

    def warning_policy(self, horizon: int) -> EarlyWarningPolicy:
        """Create the matching policy for one predeclared warning horizon."""

        if horizon not in self.warning_horizons:
            raise ValueError(f"horizon {horizon} is not predeclared in this protocol")
        return EarlyWarningPolicy(
            horizon=horizon,
            tolerance_before=self.matching_tolerance_before,
            tolerance_after=self.matching_tolerance_after,
            allow_prediction_reuse=False,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the protocol including its hash."""

        payload = asdict(self)
        payload["protocol_hash"] = self.protocol_hash
        return cast(dict[str, object], payload)


def protocol_hash(protocol: EvaluationProtocol) -> str:
    """Return a stable hash for an evaluation protocol."""

    payload = json.dumps(asdict(protocol), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def protocol_from_mapping(raw: dict[str, Any]) -> EvaluationProtocol:
    """Parse and validate a protocol mapping from YAML or JSON."""

    costs = raw.get("utility_costs", {})
    if costs is None:
        costs = {}
    if not isinstance(costs, dict):
        raise ValueError("utility_costs must be a mapping")
    return EvaluationProtocol(
        protocol_id=str(raw["protocol_id"]),
        version=str(raw["version"]),
        dataset_id=str(raw["dataset_id"]),
        independent_unit=str(raw["independent_unit"]),
        train_partition=str(raw.get("train_partition", "train")),
        validation_partition=str(raw.get("validation_partition", "validation")),
        calibration_partition=str(raw.get("calibration_partition", "validation")),
        test_partition=str(raw.get("test_partition", "test")),
        exclusion_column=_optional_str(raw.get("exclusion_column")),
        maintenance_boundary_column=_optional_str(raw.get("maintenance_boundary_column")),
        warning_horizons=tuple(int(value) for value in raw["warning_horizons"]),
        event_onset_definition=str(raw["event_onset_definition"]),
        event_end_definition=str(raw["event_end_definition"]),
        alarm_merge_gap=int(raw.get("alarm_merge_gap", 0)),
        cooldown=int(raw.get("cooldown", 0)),
        matching_tolerance_before=int(raw.get("matching_tolerance_before", 0)),
        matching_tolerance_after=int(raw.get("matching_tolerance_after", 0)),
        matching_method=_matching_method(str(raw.get("matching_method", "optimal"))),
        bootstrap_unit=_bootstrap_unit(str(raw.get("bootstrap_unit", "none"))),
        primary_metrics=tuple(str(value) for value in raw["primary_metrics"]),
        secondary_metrics=tuple(str(value) for value in raw.get("secondary_metrics", ())),
        utility_costs=EventUtilityCosts(
            true_positive_reward=float(costs.get("true_positive_reward", 1.0)),
            false_alarm_cost=float(costs.get("false_alarm_cost", 0.2)),
            missed_failure_cost=float(costs.get("missed_failure_cost", 2.0)),
            duplicate_alarm_cost=float(costs.get("duplicate_alarm_cost", 0.1)),
            late_alarm_cost=float(costs.get("late_alarm_cost", 0.05)),
            warning_time_cost=float(costs.get("warning_time_cost", 0.0)),
        ),
        frozen_timestamp_utc=str(raw["frozen_timestamp_utc"]),
    )


def read_protocol(path: Path) -> EvaluationProtocol:
    """Read a protocol from YAML or JSON."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("protocol file root must be a mapping")
    return protocol_from_mapping(cast(dict[str, Any], raw))


def write_protocol(protocol: EvaluationProtocol, path: Path) -> None:
    """Write a validated protocol artifact atomically enough for local workflows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(protocol.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(path)


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _matching_method(value: str) -> MatchingMethod:
    if value not in {"greedy", "optimal"}:
        raise ValueError("matching_method must be greedy or optimal")
    return cast(MatchingMethod, value)


def _bootstrap_unit(value: str) -> BootstrapUnit:
    if value not in {"failure", "day", "operating_cycle", "vehicle", "dataset_specific", "none"}:
        raise ValueError("unsupported bootstrap_unit")
    return cast(BootstrapUnit, value)
