import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dyn_evt_pdm.data.prepare import (
    prepare_hydraulic_systems,
    prepare_metropt,
    prepare_scania,
    prepare_secom,
)


def test_tracked_metropt_failure_metadata_is_source_verified() -> None:
    failure_yaml = Path(__file__).parents[1] / "docs" / "datasets" / "metropt_failures.yaml"
    metadata = yaml.safe_load(failure_yaml.read_text(encoding="utf-8"))

    assert metadata["dataset"] == "metropt"
    assert metadata["status"] == "source_verified"
    assert [failure["id"] for failure in metadata["failures"]] == [
        "failure_1",
        "failure_2",
        "failure_3",
    ]
    assert [(failure["start"], failure["end"]) for failure in metadata["failures"]] == [
        ("2022-02-28 21:53:00", "2022-03-01 02:00:00"),
        ("2022-03-23 14:54:00", "2022-03-23 15:24:00"),
        ("2022-05-30 12:00:00", "2022-06-02 06:18:00"),
    ]


def test_prepare_metropt_writes_labels_manifest_and_parquet(tmp_path: Path) -> None:
    raw = tmp_path / "metropt.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2022-01-01 00:00:00",
                "2022-01-01 00:00:01",
                "2022-01-01 00:00:01",
                "2022-01-01 00:00:04",
            ],
            "TP2": [1.0, 2.0, 3.0, 4.0],
            "Motor_current": [5.0, 6.0, 7.0, 8.0],
        }
    ).to_csv(raw, index=False)
    failure_yaml = tmp_path / "failures.yaml"
    failure_yaml.write_text(
        """
failures:
  - type: fixture_failure
    start: "2022-01-01 00:00:01"
    end: "2022-01-01 00:00:01"
""",
        encoding="utf-8",
    )

    result = prepare_metropt(
        dataset="metropt",
        raw_path=raw,
        output_root=tmp_path / "processed",
        failure_yaml=failure_yaml,
        chunk_size=2,
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert result.rows == 4
    assert manifest["quality_checks"]["duplicate_timestamps"] == 1
    assert manifest["quality_checks"]["sampling_gaps_gt_1_5s"] == 1
    assert manifest["canonical_aliases"]["TP2"] == "pressure_tp2"
    prepared = pd.read_parquet(result.parquet_files[0])
    assert "pressure_tp2" in prepared
    assert "is_failure" in prepared
    assert prepared["is_failure"].any()


def test_prepare_scania_writes_joined_splits_increments_and_leakage_check(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_scania_split(raw_root, "train", [1, 1, 2], [0.0, 1.0, 0.0], [10.0, 12.0, 4.0])
    _write_scania_split(raw_root, "validation", [3], [0.0], [1.0])
    _write_scania_split(raw_root, "test", [4], [0.0], [2.0])
    pd.DataFrame(
        {"vehicle_id": [1, 2], "length_of_study_time_step": [2.0, 1.0], "in_study_repair": [1, 0]}
    ).to_csv(raw_root / "train_tte.csv", index=False)

    result = prepare_scania(raw_root=raw_root, output_root=tmp_path / "processed", chunk_size=2)

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["entity_leakage"]["train_vs_validation"] == []
    train_manifest = manifest["split_manifests"]["train"]
    assert train_manifest["quality_checks"]["counter_reset_rows"] == 0
    train_part = pd.read_parquet(train_manifest["parquet_files"][0])
    assert "counter__increment" in train_part
    assert "Spec_0" in train_part
    assert "in_study_repair" in train_part


def test_prepare_scania_rejects_vehicle_leakage(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_scania_split(raw_root, "train", [1], [0.0], [1.0])
    _write_scania_split(raw_root, "validation", [1], [0.0], [2.0])
    _write_scania_split(raw_root, "test", [2], [0.0], [3.0])

    with pytest.raises(ValueError, match="vehicle leakage"):
        prepare_scania(raw_root=raw_root, output_root=tmp_path / "processed", chunk_size=2)


def test_prepare_hydraulic_systems_writes_cycle_manifest(tmp_path: Path) -> None:
    raw_zip = tmp_path / "hydraulic.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("profile.txt", "100 100 0 130 1\n20 90 1 115 0\n")
        archive.writestr("PS1.txt", "1 2 3\n4 5 6\n")
        archive.writestr("TS1.txt", "10 11\n12 13\n")

    result = prepare_hydraulic_systems(zip_path=raw_zip, output_root=tmp_path / "processed")

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert result.dataset == "hydraulic_systems"
    assert manifest["quality_checks"]["degraded_cycles"] == 1
    prepared = pd.read_parquet(result.parquet_files[0])
    assert {"ps1_mean", "ts1_last", "is_degraded", "split"}.issubset(prepared.columns)
    assert prepared["is_degraded"].tolist() == [False, True]


def test_prepare_secom_writes_wafer_manifest(tmp_path: Path) -> None:
    raw_zip = tmp_path / "secom.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("secom.data", "1.0 NaN 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n")
        archive.writestr(
            "secom_labels.data",
            "-1 19/07/2008 11:55:00\n1 19/07/2008 12:32:00\n-1 19/07/2008 13:17:00\n",
        )

    result = prepare_secom(zip_path=raw_zip, output_root=tmp_path / "processed")

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert result.dataset == "secom"
    assert manifest["quality_checks"]["failure_examples"] == 1
    assert manifest["quality_checks"]["missing_sensor_values"] == 1
    prepared = pd.read_parquet(result.parquet_files[0])
    assert {"wafer_id", "sensor_000", "is_failure", "source_timestamp", "split"}.issubset(
        prepared.columns
    )


def _write_scania_split(
    raw_root: Path,
    split: str,
    vehicle_ids: list[int],
    time_steps: list[float],
    counters: list[float],
) -> None:
    pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "time_step": time_steps,
            "counter": counters,
            "hist_0": counters,
        }
    ).to_csv(raw_root / f"{split}_operational_readouts.csv", index=False)
    pd.DataFrame(
        {
            "vehicle_id": sorted(set(vehicle_ids)),
            "Spec_0": ["Cat0"] * len(set(vehicle_ids)),
        }
    ).to_csv(raw_root / f"{split}_specifications.csv", index=False)
    if split in {"validation", "test"}:
        pd.DataFrame(
            {
                "vehicle_id": sorted(set(vehicle_ids)),
                "class_label": [0] * len(set(vehicle_ids)),
            }
        ).to_csv(raw_root / f"{split}_labels.csv", index=False)
