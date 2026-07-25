import json
from pathlib import Path

from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
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
