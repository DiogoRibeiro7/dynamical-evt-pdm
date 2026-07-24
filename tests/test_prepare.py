import json
from pathlib import Path

import pandas as pd
import pytest

from dyn_evt_pdm.data.prepare import prepare_metropt, prepare_scania


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
