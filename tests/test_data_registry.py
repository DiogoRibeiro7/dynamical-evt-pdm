import json
from pathlib import Path

from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.data.registry import (
    REGISTRY,
    build_data_report,
    normalize_dataset_ids,
    sha256_file,
)


def test_normalize_dataset_ids_accepts_public_aliases() -> None:
    assert normalize_dataset_ids("scania-component-x") == ("scania_component_x",)
    assert normalize_dataset_ids("all") == ("metropt", "metropt2", "scania_component_x")


def test_metropt_registry_matches_source_verified_failure_metadata() -> None:
    metropt = REGISTRY["metropt"]

    assert metropt.failure_or_repair_event_source == "version-controlled failure metadata YAML"
    assert not any("requires source verification" in item for item in metropt.known_limitations)
    assert any("three source-verified" in item for item in metropt.known_limitations)


def test_build_data_report_verifies_local_manifests(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    _write_dataset_manifests(raw_root, processed_root, "metropt", rows=3)
    _write_dataset_manifests(raw_root, processed_root, "metropt2", rows=4)
    _write_dataset_manifests(raw_root, processed_root, "scania_component_x", rows=5)

    report = build_data_report(raw_root=raw_root, processed_root=processed_root)

    statuses = {
        item["registry"]["dataset_id"]: item["verification"]["status"]
        for item in report["datasets"]
    }
    assert statuses == {
        "metropt": "verified",
        "metropt2": "verified",
        "scania_component_x": "verified",
    }
    assert {item["rows"] for item in report["datasets"]} == {3, 4, 5}


def test_data_cli_list_verify_and_report(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    for dataset in ("metropt", "metropt2", "scania_component_x"):
        _write_dataset_manifests(raw_root, processed_root, dataset, rows=2)
    runner = CliRunner()

    list_result = runner.invoke(app, ["data", "list", "--json"])
    assert list_result.exit_code == 0, list_result.output
    assert {item["dataset_id"] for item in json.loads(list_result.output)} == {
        "metropt",
        "metropt2",
        "scania_component_x",
    }

    verification_path = tmp_path / "verification.json"
    verify_result = runner.invoke(
        app,
        [
            "data",
            "verify",
            "--dataset",
            "metropt",
            "--raw-root",
            str(raw_root),
            "--processed-root",
            str(processed_root),
            "--output",
            str(verification_path),
        ],
    )
    assert verify_result.exit_code == 0, verify_result.output
    assert "metropt: verified" in verify_result.output
    assert json.loads(verification_path.read_text())[0]["status"] == "verified"

    report_path = tmp_path / "dataset_report.json"
    table_path = tmp_path / "dataset_table.tex"
    report_result = runner.invoke(
        app,
        [
            "data",
            "report",
            "--raw-root",
            str(raw_root),
            "--processed-root",
            str(processed_root),
            "--output",
            str(report_path),
            "--latex-output",
            str(table_path),
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    assert "datasets" in json.loads(report_path.read_text())
    assert "\\begin{tabular}" in table_path.read_text()


def _write_dataset_manifests(
    raw_root: Path,
    processed_root: Path,
    dataset: str,
    *,
    rows: int,
) -> None:
    raw_dir = raw_root / dataset
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "source.csv"
    raw_file.write_text("x\n1\n", encoding="utf-8")
    raw_manifest = [
        {
            "dataset": dataset,
            "filename": raw_file.name,
            "path": str(raw_file),
            "url": "https://example.test/source.csv",
            "size_bytes": raw_file.stat().st_size,
            "sha256": sha256_file(raw_file),
            "checksum": None,
            "status": "downloaded",
        }
    ]
    (raw_dir / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")

    processed_dir = processed_root / dataset
    processed_dir.mkdir(parents=True)
    part = processed_dir / "part-00000.parquet"
    part.write_bytes(b"not actually parquet; only checksum verification needs bytes")
    processed_manifest = {
        "dataset": dataset,
        "rows": rows,
        "chunks": 1,
        "parquet_files": [str(part)],
        "quality_checks": {},
        "schema": {"x": {"dtype": "int64"}},
    }
    if dataset == "scania_component_x":
        processed_manifest["split_manifests"] = {
            "train": {"quality_checks": {}, "rows": rows, "parquet_files": [str(part)]}
        }
    (processed_dir / "manifest.json").write_text(json.dumps(processed_manifest), encoding="utf-8")
