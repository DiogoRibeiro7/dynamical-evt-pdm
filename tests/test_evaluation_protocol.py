import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.evaluation.protocol import (
    EvaluationProtocol,
    protocol_from_mapping,
    read_protocol,
    write_protocol,
)


def test_evaluation_protocol_hash_and_warning_policy_are_stable() -> None:
    protocol = _fixture_protocol()

    same = _fixture_protocol()
    policy = protocol.warning_policy(20)

    assert protocol.protocol_hash == same.protocol_hash
    assert len(protocol.protocol_hash) == 16
    assert policy.horizon == 20
    assert policy.tolerance_after == 2
    with pytest.raises(ValueError, match="not predeclared"):
        protocol.warning_policy(999)


def test_protocol_from_mapping_validates_required_fields() -> None:
    raw = _fixture_mapping()
    raw["warning_horizons"] = [0]

    with pytest.raises(ValueError, match="positive"):
        protocol_from_mapping(raw)


def test_read_write_protocol_and_cli_freeze(tmp_path: Path) -> None:
    config_path = tmp_path / "protocol.yaml"
    artifact_path = tmp_path / "protocol.json"
    config_path.write_text(
        """
protocol_id: fixture_protocol
version: "1"
dataset_id: fixture_dataset
independent_unit: failure
train_partition: train
validation_partition: validation
calibration_partition: validation
test_partition: test
exclusion_column: exclusion_mask
maintenance_boundary_column: maintenance_boundary
warning_horizons: [10, 20]
event_onset_definition: first labelled failure row
event_end_definition: last labelled failure row
alarm_merge_gap: 3
cooldown: 5
matching_tolerance_before: 1
matching_tolerance_after: 2
matching_method: optimal
bootstrap_unit: failure
primary_metrics: [event_recall]
secondary_metrics: [time_under_warning]
frozen_timestamp_utc: "2026-07-25T00:00:00Z"
""",
        encoding="utf-8",
    )

    protocol = read_protocol(config_path)
    write_protocol(protocol, artifact_path)

    payload = json.loads(artifact_path.read_text())
    assert payload["protocol_hash"] == protocol.protocol_hash
    result = CliRunner().invoke(
        app,
        [
            "freeze-evaluation-protocol",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "cli_protocol.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fixture_protocol" in result.output
    assert (tmp_path / "cli_protocol.json").exists()


def _fixture_protocol() -> EvaluationProtocol:
    return protocol_from_mapping(_fixture_mapping())


def _fixture_mapping() -> dict[str, object]:
    return {
        "protocol_id": "fixture_protocol",
        "version": "1",
        "dataset_id": "fixture_dataset",
        "independent_unit": "failure",
        "warning_horizons": [10, 20],
        "event_onset_definition": "first labelled failure row",
        "event_end_definition": "last labelled failure row",
        "alarm_merge_gap": 3,
        "cooldown": 5,
        "matching_tolerance_before": 1,
        "matching_tolerance_after": 2,
        "matching_method": "optimal",
        "bootstrap_unit": "failure",
        "primary_metrics": ["event_recall"],
        "secondary_metrics": ["time_under_warning"],
        "frozen_timestamp_utc": "2026-07-25T00:00:00Z",
    }
