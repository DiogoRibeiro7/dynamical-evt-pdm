import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.data.registry import sha256_file
from dyn_evt_pdm.pipelines.experiment_matrix import (
    ExperimentMatrixConfig,
    run_experiment_matrix,
)


def test_run_experiment_matrix_writes_terminal_manifest(tmp_path: Path) -> None:
    manifest = run_experiment_matrix(
        ExperimentMatrixConfig(
            output_root=tmp_path / "matrix",
            smoke=True,
            n_jobs=1,
        )
    )

    manifest_path = tmp_path / "matrix" / "experiment_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest.status == "succeeded"
    assert payload["status"] == "succeeded"
    assert len({cell["experiment_id"] for cell in payload["cells"]}) == len(payload["cells"])
    assert {cell["status"] for cell in payload["cells"]} == {"succeeded"}
    assert (tmp_path / "matrix" / "evaluation_protocol.json").exists()
    assert (tmp_path / "matrix" / "simulation_study.parquet").exists()
    assert (tmp_path / "matrix" / "simulation_study.parquet.decisions.json").exists()
    assert (tmp_path / "matrix" / "baseline_predictions.parquet").exists()
    assert (tmp_path / "matrix" / "baseline_metadata.json").exists()


def test_run_experiment_matrix_cli(tmp_path: Path) -> None:
    output_root = tmp_path / "cli_matrix"

    result = CliRunner().invoke(
        app,
        [
            "run-experiment-matrix",
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "status=succeeded" in result.output
    assert (output_root / "experiment_manifest.json").exists()


def test_run_experiment_matrix_with_real_data_verification(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    _write_dataset_fixture(raw_root, processed_root, "metropt")
    _write_dataset_fixture(raw_root, processed_root, "metropt2")
    _write_dataset_fixture(raw_root, processed_root, "scania_component_x")

    manifest = run_experiment_matrix(
        ExperimentMatrixConfig(
            output_root=tmp_path / "matrix",
            smoke=True,
            n_jobs=1,
            include_real_data=True,
            raw_root=raw_root,
            processed_root=processed_root,
        )
    )

    assert manifest.status == "succeeded"
    payload = json.loads((tmp_path / "matrix" / "experiment_manifest.json").read_text())
    real_data = [cell for cell in payload["cells"] if cell["name"] == "real_data_verification"]
    assert len(real_data) == 1
    assert real_data[0]["status"] == "succeeded"
    assert (tmp_path / "matrix" / "real_data_report.json").exists()
    assert (tmp_path / "matrix" / "real_data_status.csv").exists()
    assert (tmp_path / "matrix" / "dataset_characteristics.tex").exists()
    assert (tmp_path / "matrix" / "industrial_results_summary.csv").exists()
    assert (tmp_path / "matrix" / "industrial_results.json").exists()
    assert (tmp_path / "matrix" / "industrial_results.tex").exists()


def _write_dataset_fixture(raw_root: Path, processed_root: Path, dataset_id: str) -> None:
    raw_dir = raw_root / dataset_id
    processed_dir = processed_root / dataset_id
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    raw_file = raw_dir / "source.csv"
    raw_file.write_text("timestamp,value\n2024-01-01,1\n", encoding="utf-8")
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": str(raw_file),
                    "size_bytes": raw_file.stat().st_size,
                    "sha256": sha256_file(raw_file),
                }
            ]
        ),
        encoding="utf-8",
    )
    if dataset_id == "scania_component_x":
        parquet_files = _write_scania_processed_fixture(processed_dir)
        schema = {
            "vehicle_id": {"dtype": "int64"},
            "time_step": {"dtype": "float64"},
            "100_0__increment": {"dtype": "float64"},
            "171_0__increment": {"dtype": "float64"},
            "in_study_repair": {"dtype": "int64"},
        }
        processed_manifest: dict[str, object] = {
            "dataset": dataset_id,
            "rows": 9,
            "chunks": 3,
            "parquet_files": [str(path) for path in parquet_files],
            "schema": schema,
            "quality_checks": {},
            "split_manifests": {
                split: {
                    "split": split,
                    "rows": 3,
                    "chunks": 1,
                    "parquet_files": [str(processed_dir / split / "part-00000.parquet")],
                    "schema": schema,
                    "quality_checks": {},
                    "vehicle_ids": [split_index * 10 + 1, split_index * 10 + 2],
                }
                for split_index, split in enumerate(("train", "validation", "test"))
            },
        }
    else:
        processed_file = processed_dir / "part-00000.parquet"
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="s"),
                "pressure_tp2": [0.0, 0.1, 0.2, 0.0, 5.0, 6.0],
                "pressure_tp3": [1.0, 1.1, 1.2, 1.0, 4.5, 7.0],
                "pressure_h1": [2.0, 2.1, 2.2, 2.0, 2.5, 8.0],
                "reservoir_pressure": [3.0, 3.1, 3.2, 3.0, 3.5, 9.0],
                "oil_temperature": [50.0, 50.1, 50.2, 50.0, 55.0, 60.0],
                "motor_current": [4.0, 4.1, 4.2, 4.0, 8.0, 10.0],
                "flowmeter": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
                "is_failure": [False, False, False, False, False, True],
            }
        ).to_parquet(processed_file, index=False)
        schema = {
            "pressure_tp2": {"dtype": "float64"},
            "pressure_tp3": {"dtype": "float64"},
            "pressure_h1": {"dtype": "float64"},
            "reservoir_pressure": {"dtype": "float64"},
            "oil_temperature": {"dtype": "float64"},
            "motor_current": {"dtype": "float64"},
            "flowmeter": {"dtype": "float64"},
            "is_failure": {"dtype": "bool"},
        }
        processed_manifest = {
            "dataset": dataset_id,
            "rows": 6,
            "chunks": 1,
            "parquet_files": [str(processed_file)],
            "schema": schema,
            "quality_checks": {},
        }
    (processed_dir / "manifest.json").write_text(
        json.dumps(processed_manifest),
        encoding="utf-8",
    )


def _write_scania_processed_fixture(processed_dir: Path) -> list[Path]:
    files: list[Path] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / "part-00000.parquet"
        pd.DataFrame(
            {
                "vehicle_id": [split_index * 10 + 1, split_index * 10 + 1, split_index * 10 + 2],
                "time_step": [0.0, 1.0, 0.0],
                "100_0__increment": [0.0, 1.0 + split_index, 5.0 + split_index],
                "171_0__increment": [0.0, 2.0 + split_index, 6.0 + split_index],
                "in_study_repair": [0, 0, 1 if split == "test" else 0],
            }
        ).to_parquet(path, index=False)
        files.append(path)
    return files
