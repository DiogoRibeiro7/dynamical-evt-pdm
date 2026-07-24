import json
from pathlib import Path

from typer.testing import CliRunner

from dyn_evt_pdm.cli import app

runner = CliRunner()


def test_cli_simulate_and_analyse(tmp_path: Path) -> None:
    data_path = tmp_path / "simulation.csv"
    result = runner.invoke(app, ["simulate", "--output", str(data_path), "--n-steps", "1000", "--seed", "5"])
    assert result.exit_code == 0, result.output
    assert data_path.exists()

    summary_path = tmp_path / "summary.json"
    result = runner.invoke(
        app,
        [
            "analyse-series",
            "--input",
            str(data_path),
            "--output",
            str(summary_path),
            "--run-length",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(summary_path.read_text())["n_samples"] == 1000
