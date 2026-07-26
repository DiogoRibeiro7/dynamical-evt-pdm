"""Experiment-matrix orchestration with terminal status manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from dyn_evt_pdm.data.registry import build_data_report, render_dataset_latex_table
from dyn_evt_pdm.evaluation.protocol import read_protocol, write_protocol
from dyn_evt_pdm.models.baseline_runner import (
    BaselineRunConfig,
    baseline_metadata_to_frame,
    run_baseline_experiment,
)
from dyn_evt_pdm.pipelines.industrial_results import (
    IndustrialResultsConfig,
    run_industrial_results,
)
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.study import (
    simulation_study_config_from_mapping,
    write_simulation_study,
)

ExperimentStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    """One planned experiment cell with deterministic identity."""

    name: str
    family: str
    dataset_id: str
    output_paths: tuple[str, ...]
    config: dict[str, object]

    @property
    def experiment_id(self) -> str:
        """Stable ID from the planned cell content."""

        payload = {
            "name": self.name,
            "family": self.family,
            "dataset_id": self.dataset_id,
            "output_paths": self.output_paths,
            "config": self.config,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Terminal status record for one matrix cell."""

    experiment_id: str
    name: str
    family: str
    dataset_id: str
    status: ExperimentStatus
    output_paths: tuple[str, ...]
    started_at: float | None
    finished_at: float | None
    runtime_seconds: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class ExperimentMatrixManifest:
    """Manifest for one experiment-matrix execution."""

    matrix_id: str
    status: ExperimentStatus
    code_hash: str
    protocol_hash: str
    created_at: float
    output_root: str
    cells: tuple[ExperimentRecord, ...]
    environment: dict[str, object]


@dataclass(frozen=True, slots=True)
class ExperimentMatrixConfig:
    """Configuration for the reproducibility-subset experiment matrix."""

    output_root: Path = Path("artifacts/experiment_matrix")
    protocol_config_path: Path = Path("configs/evaluation/base.yaml")
    simulation_config_path: Path = Path("configs/simulation/cyclic_degradation.yaml")
    smoke: bool = True
    n_jobs: int = 1
    include_real_data: bool = False
    raw_root: Path = Path("data/raw")
    processed_root: Path = Path("data/processed")


def run_experiment_matrix(config: ExperimentMatrixConfig) -> ExperimentMatrixManifest:
    """Execute a deterministic experiment matrix and write its manifest."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    protocol = read_protocol(config.protocol_config_path)
    cells = _planned_cells(config, protocol_hash=protocol.protocol_hash)
    _assert_unique_experiment_ids(cells)
    records: list[ExperimentRecord] = []

    executors: dict[str, Callable[[ExperimentCell], None]] = {
        "freeze_protocol": lambda _cell: write_protocol(
            protocol, config.output_root / "evaluation_protocol.json"
        ),
        "simulation_recovery": lambda _cell: _run_simulation_cell(config),
        "baseline_smoke": lambda _cell: _run_baseline_smoke_cell(config),
        "real_data_verification": lambda _cell: _run_real_data_verification_cell(config),
        "industrial_real_data_results": lambda _cell: _run_industrial_results_cell(config),
        "evidence_scope": lambda _cell: _run_evidence_scope_cell(config),
    }
    for cell in cells:
        executor = executors.get(cell.name)
        if executor is None:
            records.append(
                _record(
                    cell,
                    status="skipped",
                    started_at=None,
                    finished_at=None,
                    reason="no executor registered for this cell",
                )
            )
            continue
        started = time.time()
        try:
            executor(cell)
        except Exception as exc:
            finished = time.time()
            records.append(
                _record(
                    cell,
                    status="failed",
                    started_at=started,
                    finished_at=finished,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        finished = time.time()
        missing = [path for path in cell.output_paths if not Path(path).exists()]
        status: ExperimentStatus = "failed" if missing else "succeeded"
        reason = f"missing expected artifacts: {missing}" if missing else ""
        records.append(
            _record(
                cell,
                status=status,
                started_at=started,
                finished_at=finished,
                reason=reason,
            )
        )

    matrix_status: ExperimentStatus = (
        "succeeded" if all(record.status == "succeeded" for record in records) else "failed"
    )
    manifest = ExperimentMatrixManifest(
        matrix_id=_matrix_id(cells, protocol_hash=protocol.protocol_hash),
        status=matrix_status,
        code_hash=_current_commit_hash(),
        protocol_hash=protocol.protocol_hash,
        created_at=time.time(),
        output_root=str(config.output_root),
        cells=tuple(records),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "smoke": config.smoke,
            "n_jobs": config.n_jobs,
        },
    )
    _write_manifest(config.output_root / "experiment_manifest.json", manifest)
    return manifest


def _planned_cells(
    config: ExperimentMatrixConfig,
    *,
    protocol_hash: str,
) -> tuple[ExperimentCell, ...]:
    root = config.output_root
    cells = [
        ExperimentCell(
            name="freeze_protocol",
            family="preflight",
            dataset_id="generic",
            output_paths=(str(root / "evaluation_protocol.json"),),
            config={
                "protocol_config_path": str(config.protocol_config_path),
                "protocol_hash": protocol_hash,
            },
        ),
        ExperimentCell(
            name="simulation_recovery",
            family="simulation",
            dataset_id="synthetic",
            output_paths=(
                str(root / "simulation_study.parquet"),
                str(root / "simulation_study.parquet.manifest.json"),
                str(root / "simulation_study.parquet.decisions.json"),
            ),
            config={
                "simulation_config_path": str(config.simulation_config_path),
                "smoke": config.smoke,
                "n_jobs": config.n_jobs,
            },
        ),
        ExperimentCell(
            name="baseline_smoke",
            family="baseline",
            dataset_id="synthetic_cyclic",
            output_paths=(
                str(root / "baseline_predictions.parquet"),
                str(root / "baseline_metadata.json"),
            ),
            config={
                "n_steps": 1000,
                "seed": 1729,
                "window_size": 2,
                "horizon": 20,
            },
        ),
    ]
    if config.include_real_data:
        cells.append(
            ExperimentCell(
                name="real_data_verification",
                family="real_data",
                dataset_id="metropt+metropt2+scania_component_x",
                output_paths=(
                    str(root / "real_data_report.json"),
                    str(root / "real_data_status.csv"),
                    str(root / "dataset_characteristics.tex"),
                ),
                config={
                    "raw_root": str(config.raw_root),
                    "processed_root": str(config.processed_root),
                    "required_status": "verified",
                },
            )
        )
        cells.append(
            ExperimentCell(
                name="industrial_real_data_results",
                family="real_data",
                dataset_id="metropt+metropt2+scania_component_x",
                output_paths=(
                    str(root / "industrial_results_summary.csv"),
                    str(root / "industrial_results.json"),
                    str(root / "industrial_results.tex"),
                ),
                config={
                    "processed_root": str(config.processed_root),
                    "required_datasets": "metropt,metropt2,scania_component_x",
                    "threshold_quantile": 0.98,
                },
            )
        )
        cells.append(
            ExperimentCell(
                name="evidence_scope",
                family="evidence_scope",
                dataset_id="synthetic+metropt+metropt2+scania_component_x",
                output_paths=(
                    str(root / "evidence_scope.json"),
                    str(root / "evidence_scope.csv"),
                    str(root / "evidence_scope.tex"),
                ),
                config={
                    "includes": (
                        "simulation_recovery,baseline_smoke,real_data_verification,"
                        "industrial_real_data_results"
                    ),
                    "claim_boundary": (
                        "broader industrial performance remains outside the generated evidence"
                    ),
                },
            )
        )
    return tuple(cells)


def _run_simulation_cell(config: ExperimentMatrixConfig) -> None:
    raw = yaml.safe_load(config.simulation_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("simulation config root must be a mapping")
    study_config = simulation_study_config_from_mapping(
        cast(dict[str, Any], raw), smoke=config.smoke, n_jobs=config.n_jobs
    )
    write_simulation_study(study_config, config.output_root / "simulation_study.parquet")


def _run_baseline_smoke_cell(config: ExperimentMatrixConfig) -> None:
    frame = simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=1729))
    result = run_baseline_experiment(
        frame,
        BaselineRunConfig(
            feature_columns=("pressure", "current", "temperature"),
            dataset_id="synthetic_cyclic",
            timestamp_column="time",
            regime_column="regime",
            split_column="split",
            target_column="is_fault",
            window_size=2,
            horizon=20,
            isolation_n_estimators=20,
            random_state=1729,
        ),
    )
    result.predictions.to_parquet(config.output_root / "baseline_predictions.parquet", index=False)
    metadata = baseline_metadata_to_frame(result.metadata)
    (config.output_root / "baseline_metadata.json").write_text(
        json.dumps(metadata.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def _run_real_data_verification_cell(config: ExperimentMatrixConfig) -> None:
    report = build_data_report(raw_root=config.raw_root, processed_root=config.processed_root)
    rows = []
    blocked = []
    for item in report["datasets"]:
        registry = item["registry"]
        verification = item["verification"]
        status = str(verification["status"])
        dataset_id = str(registry["dataset_id"])
        rows.append(
            {
                "dataset_id": dataset_id,
                "status": status,
                "rows": item.get("rows"),
                "chunks": item.get("chunks"),
                "independent_unit": registry["statistical_independence_unit"],
                "raw_manifest_path": verification["raw_manifest_path"],
                "processed_manifest_path": verification["processed_manifest_path"],
            }
        )
        if status != "verified":
            blocked.append(dataset_id)
    (config.output_root / "real_data_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    import pandas as pd

    pd.DataFrame(rows).to_csv(config.output_root / "real_data_status.csv", index=False)
    (config.output_root / "dataset_characteristics.tex").write_text(
        render_dataset_latex_table(report),
        encoding="utf-8",
    )
    if blocked:
        raise FileNotFoundError(
            "real-data verification blocked for "
            f"{blocked}; run `poetry run dyn-evt fetch-data --dataset all` and "
            "`make prepare-data` before the real-data matrix"
        )


def _run_industrial_results_cell(config: ExperimentMatrixConfig) -> None:
    manifest = run_industrial_results(
        IndustrialResultsConfig(
            processed_root=config.processed_root,
            output_root=config.output_root,
        )
    )
    failed = [result.dataset_id for result in manifest.results if result.status == "failed"]
    if failed:
        raise RuntimeError(f"industrial real-data results failed for {failed}")


def _run_evidence_scope_cell(config: ExperimentMatrixConfig) -> None:
    report_path = config.output_root / "real_data_report.json"
    industrial_path = config.output_root / "industrial_results.json"
    if not report_path.exists():
        _run_real_data_verification_cell(config)
    if not industrial_path.exists():
        _run_industrial_results_cell(config)

    report = _read_json_object(report_path)
    industrial = _read_json_object(industrial_path)
    industrial_results = {
        str(item.get("dataset_id")): item
        for item in cast(list[dict[str, Any]], industrial.get("results", []))
    }

    dataset_rows: list[dict[str, object]] = []
    for item in cast(list[dict[str, Any]], report.get("datasets", [])):
        registry = cast(dict[str, Any], item["registry"])
        verification = cast(dict[str, Any], item["verification"])
        dataset_id = str(registry["dataset_id"])
        result = industrial_results.get(dataset_id, {})
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "sampling_structure": str(registry["sampling_structure"]),
                "entity_identifier": str(registry["entity_identifier"]),
                "independent_unit": str(registry["statistical_independence_unit"]),
                "event_source": str(registry["failure_or_repair_event_source"]),
                "rows": item.get("rows"),
                "chunks": item.get("chunks"),
                "schema_columns": item.get("schema_columns"),
                "verification_status": str(verification["status"]),
                "industrial_result_status": str(result.get("status", "missing")),
                "estimand": str(result.get("estimand", "missing")),
                "industrial_independent_units": result.get("independent_units"),
                "target_events_or_units": result.get("target_events_or_units"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1": result.get("f1"),
                "limitation": str(result.get("limitation", "")),
                "known_limitations": "; ".join(
                    str(limit) for limit in registry.get("known_limitations", [])
                ),
            }
        )

    workflow_rows = [
        {
            "workflow_family": "simulation_recovery",
            "evidence_role": "known-ground-truth estimator behavior",
            "status": "included",
        },
        {
            "workflow_family": "baseline_smoke",
            "evidence_role": "shared prediction and evaluation contract",
            "status": "included",
        },
        {
            "workflow_family": "real_data_verification",
            "evidence_role": "dataset provenance, manifest, and checksum coverage",
            "status": "included",
        },
        {
            "workflow_family": "industrial_real_data_results",
            "evidence_role": "conservative diagnostics on prepared real datasets",
            "status": "included",
        },
    ]
    scope = {
        "summary": {
            "dataset_count": len(dataset_rows),
            "verified_dataset_count": sum(
                row["verification_status"] == "verified" for row in dataset_rows
            ),
            "sampling_structure_count": len(
                {str(row["sampling_structure"]) for row in dataset_rows}
            ),
            "sampling_structures": sorted({str(row["sampling_structure"]) for row in dataset_rows}),
            "independent_units": sorted({str(row["independent_unit"]) for row in dataset_rows}),
            "workflow_count": len(workflow_rows),
            "industrial_result_statuses": sorted(
                {str(row["industrial_result_status"]) for row in dataset_rows}
            ),
            "claim_boundary": (
                "This evidence supports claims for the listed datasets and workflows; "
                "broader industrial performance remains outside this generated evidence."
            ),
        },
        "datasets": dataset_rows,
        "workflows": workflow_rows,
    }
    (config.output_root / "evidence_scope.json").write_text(
        json.dumps(scope, indent=2, default=str),
        encoding="utf-8",
    )
    import pandas as pd

    pd.DataFrame(dataset_rows).to_csv(config.output_root / "evidence_scope.csv", index=False)
    (config.output_root / "evidence_scope.tex").write_text(
        _render_evidence_scope_latex(dataset_rows),
        encoding="utf-8",
    )


def _render_evidence_scope_latex(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Dataset & Sampling structure & Unit & Verification & Industrial result \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["dataset_id"])),
                    _latex_escape(str(row["sampling_structure"])),
                    _latex_escape(str(row["independent_unit"])),
                    _latex_escape(str(row["verification_status"])),
                    _latex_escape(str(row["industrial_result_status"])),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _record(
    cell: ExperimentCell,
    *,
    status: ExperimentStatus,
    started_at: float | None,
    finished_at: float | None,
    reason: str,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=cell.experiment_id,
        name=cell.name,
        family=cell.family,
        dataset_id=cell.dataset_id,
        status=status,
        output_paths=cell.output_paths,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=(
            float(finished_at - started_at)
            if started_at is not None and finished_at is not None
            else None
        ),
        reason=reason,
    )


def _matrix_id(cells: tuple[ExperimentCell, ...], *, protocol_hash: str) -> str:
    encoded = json.dumps(
        {
            "protocol_hash": protocol_hash,
            "cells": [cell.experiment_id for cell in cells],
        },
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _assert_unique_experiment_ids(cells: tuple[ExperimentCell, ...]) -> None:
    ids = [cell.experiment_id for cell in cells]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicated experiment IDs: {duplicates}")


def _write_manifest(path: Path, manifest: ExperimentMatrixManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    temporary.replace(path)


def _current_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
