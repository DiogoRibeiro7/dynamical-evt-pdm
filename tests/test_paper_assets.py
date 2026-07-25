import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.paper.assets import PaperAssetConfig, build_paper_assets, configuration_hash
from dyn_evt_pdm.paper.claims import verify_paper_assets
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine


def test_build_paper_assets_writes_figures_tables_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "cached_processed.csv"
    study_path = tmp_path / "simulation_study.parquet"
    output_root = tmp_path / "paper"
    config_path = tmp_path / "config.yaml"
    simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=12)).to_csv(
        input_path, index=False
    )
    _write_cached_simulation_study(study_path)
    config_path.write_text("experiment:\n  name: smoke\n", encoding="utf-8")

    manifest = build_paper_assets(
        PaperAssetConfig(
            input_path=input_path,
            output_root=output_root,
            config_paths=(config_path,),
            simulation_study_path=study_path,
        )
    )

    manifest_path = output_root / "asset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == manifest.experiment_id
    assert payload["config_hash"] == configuration_hash((config_path,))
    assert payload["protocol_hash"]
    assert payload["dataset_checksum"]
    assert (output_root / "figures" / "threshold_stability.png").exists()
    assert (output_root / "figures" / "event_timeline.png").exists()
    assert (output_root / "tables" / "dataset_split_summary.csv").exists()
    assert (output_root / "tables" / "computational_benchmark.csv").exists()
    assert (output_root / "tables" / "claim_ledger.csv").exists()
    assert (output_root / "claim_ledger.json").exists()
    assert (output_root / "claim_ledger.md").exists()
    assert (output_root / "results_synthesis.md").exists()
    assert (output_root / "asset_provenance.json").exists()
    assert (output_root / "latex" / "result_macros.tex").exists()
    assert (output_root / "latex" / "threshold_stability.tex").exists()
    assert len(payload["generated_files"]) >= 24
    verification = verify_paper_assets(output_root)
    assert verification.ok, verification.failures
    assert verification.checked_claims >= 7


def test_build_paper_assets_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "cached_processed.csv"
    study_path = tmp_path / "simulation_study.parquet"
    output_root = tmp_path / "paper"
    simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=13)).to_csv(
        input_path, index=False
    )
    _write_cached_simulation_study(study_path)

    result = CliRunner().invoke(
        app,
        [
            "build-paper-assets",
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
            "--simulation-study-path",
            str(study_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_root / "asset_manifest.json").exists()
    verify_result = CliRunner().invoke(
        app,
        [
            "verify-paper-assets",
            "--output-root",
            str(output_root),
        ],
    )
    assert verify_result.exit_code == 0, verify_result.output


def test_verify_paper_assets_detects_manual_modification(tmp_path: Path) -> None:
    input_path = tmp_path / "cached_processed.csv"
    study_path = tmp_path / "simulation_study.parquet"
    output_root = tmp_path / "paper"
    simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=14)).to_csv(
        input_path, index=False
    )
    _write_cached_simulation_study(study_path)
    build_paper_assets(
        PaperAssetConfig(
            input_path=input_path,
            output_root=output_root,
            simulation_study_path=study_path,
        )
    )

    table_path = output_root / "tables" / "dataset_split_summary.csv"
    table_path.write_text(table_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    verification = verify_paper_assets(output_root)
    assert not verification.ok
    assert any("manual modification detected" in failure for failure in verification.failures)


def _write_cached_simulation_study(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "system": ["iid", "iid", "cyclic", "cyclic"],
            "threshold_quantile": [0.95, 0.98, 0.95, 0.98],
            "runs_bias": [0.1, 0.05, -0.2, -0.1],
            "intervals_bias": [0.08, 0.03, -0.15, -0.05],
            "runs_error": [0.1, 0.05, -0.2, -0.1],
            "intervals_error": [0.08, 0.03, -0.15, -0.05],
        }
    )
    frame.to_parquet(path, index=False)
