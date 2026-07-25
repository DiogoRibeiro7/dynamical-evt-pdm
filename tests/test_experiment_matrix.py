import json
from pathlib import Path

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


def _write_dataset_fixture(raw_root: Path, processed_root: Path, dataset_id: str) -> None:
    raw_dir = raw_root / dataset_id
    processed_dir = processed_root / dataset_id
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    raw_file = raw_dir / "source.csv"
    processed_file = processed_dir / "part-00000.parquet"
    raw_file.write_text("timestamp,value\n2024-01-01,1\n", encoding="utf-8")
    processed_file.write_text(
        "not parquet; verification only checks manifest integrity\n", encoding="utf-8"
    )
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
    processed_manifest: dict[str, object] = {
        "dataset": dataset_id,
        "rows": 1,
        "chunks": 1,
        "parquet_files": [str(processed_file)],
        "schema": {"value": {"dtype": "int64"}},
        "quality_checks": {},
    }
    if dataset_id == "scania_component_x":
        processed_manifest["split_manifests"] = {"train": {"quality_checks": {}}}
    (processed_dir / "manifest.json").write_text(
        json.dumps(processed_manifest),
        encoding="utf-8",
    )
