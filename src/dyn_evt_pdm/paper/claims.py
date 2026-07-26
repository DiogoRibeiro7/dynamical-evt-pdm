"""Claim-ledger generation and verification for paper assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ClaimClass = Literal["descriptive", "inferential", "methodological", "operational", "limitation"]
ClaimStatus = Literal[
    "supported",
    "narrowly supported",
    "exploratory",
    "unsupported",
    "contradicted",
    "not estimable",
]

ALLOWED_CLAIM_STATUSES: set[str] = {
    "supported",
    "narrowly supported",
    "exploratory",
    "unsupported",
    "contradicted",
    "not estimable",
}
BLOCKED_CLAIM_STATUSES = {"unsupported", "contradicted"}


@dataclass(frozen=True, slots=True)
class Claim:
    """One artifact-backed claim available, or blocked, for manuscript writing."""

    claim_id: str
    wording: str
    claim_class: ClaimClass
    hypothesis_id: str
    datasets: str
    independent_units: str
    estimand: str
    experiment_ids: str
    table_or_figure_reference: str
    effect_estimate: str
    uncertainty: str
    sensitivity_status: str
    assumptions: str
    counterevidence: str
    permitted_strength: str
    manuscript_sections_allowed: str
    final_status: ClaimStatus


@dataclass(frozen=True, slots=True)
class ClaimLedgerConfig:
    """Inputs required to generate claims and result-synthesis assets."""

    output_root: Path
    experiment_id: str
    code_hash: str
    config_hash: str
    protocol_hash: str
    dataset_checksum: str
    dataset_name: str
    known_experiment_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperAssetVerification:
    """Verification summary for generated paper assets."""

    output_root: str
    checked_files: int
    checked_claims: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return true when no verification failures were found."""

        return not self.failures


def build_claim_ledger(config: ClaimLedgerConfig) -> list[Path]:
    """Build machine-readable and human-readable claim assets from generated tables."""

    tables = config.output_root / "tables"
    latex = config.output_root / "latex"
    tables.mkdir(parents=True, exist_ok=True)
    latex.mkdir(parents=True, exist_ok=True)

    claims = _claims_from_tables(config)
    claim_rows = [asdict(claim) for claim in claims]
    csv_path = tables / "claim_ledger.csv"
    json_path = config.output_root / "claim_ledger.json"
    markdown_path = config.output_root / "claim_ledger.md"
    synthesis_path = config.output_root / "results_synthesis.md"
    macro_path = latex / "result_macros.tex"

    pd.DataFrame(claim_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(claim_rows, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_claim_markdown(claims, config), encoding="utf-8")
    synthesis_path.write_text(_render_results_synthesis(claims), encoding="utf-8")
    macro_path.write_text(_render_result_macros(claims, config), encoding="utf-8")
    return [csv_path, json_path, markdown_path, synthesis_path, macro_path]


def write_asset_provenance(
    path: Path,
    *,
    files: tuple[Path, ...],
    experiment_id: str,
    code_hash: str,
    config_hash: str,
    protocol_hash: str,
    dataset_checksum: str,
    input_path: Path,
) -> None:
    """Write per-file provenance records with hashes for generated assets."""

    records = []
    for asset_path in sorted(files):
        if not asset_path.exists():
            continue
        records.append(
            {
                "path": str(asset_path),
                "sha256": sha256_file(asset_path),
                "experiment_id": experiment_id,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "protocol_hash": protocol_hash,
                "dataset_checksum": dataset_checksum,
                "source_input": str(input_path),
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def verify_paper_assets(output_root: Path) -> PaperAssetVerification:
    """Verify generated paper assets, provenance hashes, and claim injection rules."""

    failures: list[str] = []
    manifest_path = output_root / "asset_manifest.json"
    provenance_path = output_root / "asset_provenance.json"
    claim_path = output_root / "claim_ledger.json"
    manifest: dict[str, Any] = {}
    checked_files = 0
    checked_claims = 0

    if not manifest_path.exists():
        failures.append(f"missing manifest: {manifest_path}")
    else:
        manifest = _read_json_object(manifest_path, failures)
        for raw_path in manifest.get("generated_files", []):
            asset_path = Path(str(raw_path))
            checked_files += 1
            if not asset_path.exists():
                failures.append(f"missing generated file listed in manifest: {asset_path}")

    provenance_records: list[dict[str, Any]] = []
    if not provenance_path.exists():
        failures.append(f"missing provenance file: {provenance_path}")
    else:
        provenance_records = _read_json_list(provenance_path, failures)
        for record in provenance_records:
            asset_path = Path(str(record.get("path", "")))
            expected_hash = str(record.get("sha256", ""))
            if not asset_path.exists():
                failures.append(f"provenance file missing on disk: {asset_path}")
                continue
            actual_hash = sha256_file(asset_path)
            if expected_hash != actual_hash:
                failures.append(f"manual modification detected for {asset_path}")

    if manifest:
        provenance_paths = {str(record.get("path", "")) for record in provenance_records}
        exempt = {str(manifest_path), str(provenance_path)}
        for raw_path in manifest.get("generated_files", []):
            if str(raw_path) not in exempt and str(raw_path) not in provenance_paths:
                failures.append(f"generated file lacks provenance: {raw_path}")

    if not claim_path.exists():
        failures.append(f"missing claim ledger: {claim_path}")
    else:
        known_ids = {str(value) for value in manifest.get("known_experiment_ids", [])}
        known_ids.add(str(manifest.get("experiment_id", "")))
        claims = _read_json_list(claim_path, failures)
        seen_claim_ids: set[str] = set()
        for claim in claims:
            checked_claims += 1
            claim_id = str(claim.get("claim_id", ""))
            if claim_id in seen_claim_ids:
                failures.append(f"duplicated claim_id: {claim_id}")
            seen_claim_ids.add(claim_id)
            status = str(claim.get("final_status", ""))
            if status not in ALLOWED_CLAIM_STATUSES:
                failures.append(f"claim {claim_id} has invalid status: {status}")
            sections = str(claim.get("manuscript_sections_allowed", "")).strip()
            if status in BLOCKED_CLAIM_STATUSES and sections:
                failures.append(f"blocked claim {claim_id} is allowed in manuscript sections")
            for experiment_id in _split_ids(str(claim.get("experiment_ids", ""))):
                if experiment_id not in known_ids:
                    failures.append(
                        f"claim {claim_id} references unknown experiment ID {experiment_id}"
                    )

    return PaperAssetVerification(
        output_root=str(output_root),
        checked_files=checked_files,
        checked_claims=checked_claims,
        failures=tuple(failures),
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claims_from_tables(config: ClaimLedgerConfig) -> tuple[Claim, ...]:
    split_summary = _read_optional_csv(config.output_root / "tables" / "dataset_split_summary.csv")
    frontier = _read_optional_csv(
        config.output_root / "tables" / "lead_time_false_alarm_frontier.csv"
    )
    matching = _read_optional_csv(config.output_root / "tables" / "matching_tolerance_surface.csv")
    simulation = _read_optional_csv(config.output_root / "tables" / "simulation_bias_rmse.csv")
    reliability = _read_optional_csv(config.output_root / "tables" / "reliability_diagram.csv")
    split_calibration = _read_optional_csv(
        config.output_root / "tables" / "split_calibration_intervals.csv"
    )
    split_brier = _read_optional_csv(config.output_root / "tables" / "split_calibration_brier.csv")
    ablations = _read_optional_csv(config.output_root / "tables" / "ablation_summary.csv")

    rows = int(split_summary["rows"].sum()) if "rows" in split_summary else 0
    failures = int(split_summary["failures"].sum()) if "failures" in split_summary else 0
    regimes = (
        int(split_summary["regimes"].max())
        if "regimes" in split_summary and len(split_summary)
        else 0
    )
    best_rmse = _minimum_numeric(simulation, "runs_rmse")
    frontier_recall = _maximum_numeric(frontier, "event_recall")
    frontier_false_alarm = _minimum_numeric(frontier, "false_alarm_events_per_day")
    best_matching_f1 = _maximum_numeric(matching, "event_f1")
    reliability_bins = int(len(reliability))
    split_calibration_bins = int(len(split_calibration))
    test_brier = _split_numeric(split_brier, split="test", column="brier_score")
    ablation_families = (
        int(ablations["ablation_family"].nunique())
        if "ablation_family" in ablations and len(ablations)
        else 0
    )

    return (
        Claim(
            claim_id="CLM-001",
            wording=(
                f"The generated paper assets for {config.dataset_name} cover {rows} rows, "
                f"{failures} labelled failure samples, and {regimes} observed regimes."
            ),
            claim_class="descriptive",
            hypothesis_id="H-DATA-001",
            datasets=config.dataset_name,
            independent_units=_independent_units(split_summary),
            estimand="processed rows, labelled failure samples, and observed regimes",
            experiment_ids=config.experiment_id,
            table_or_figure_reference="tables/dataset_split_summary.csv",
            effect_estimate=f"rows={rows}; failure_samples={failures}; regimes={regimes}",
            uncertainty="not applicable for deterministic data inventory",
            sensitivity_status="source checksum recorded",
            assumptions="processed cache preserves declared split and label columns",
            counterevidence="raw data access restrictions are represented outside this asset cache",
            permitted_strength="descriptive inventory only",
            manuscript_sections_allowed="Data; Experimental Protocol",
            final_status="supported" if rows > 0 else "not estimable",
        ),
        Claim(
            claim_id="CLM-002",
            wording=(
                "Threshold and run-length choices change the empirical number of clusters "
                "and must be presented as sensitivity analyses."
            ),
            claim_class="methodological",
            hypothesis_id="H-EVT-001",
            datasets=config.dataset_name,
            independent_units=_independent_units(split_summary),
            estimand="runs extremal-index and cluster-count sensitivity",
            experiment_ids=config.experiment_id,
            table_or_figure_reference="tables/threshold_stability.csv; figures/threshold_stability.png",
            effect_estimate="see threshold_stability.csv",
            uncertainty="grid sensitivity, not a confidence interval",
            sensitivity_status="threshold and run-length grid generated",
            assumptions="same observable and causal ordering across grid cells",
            counterevidence="small failure counts limit inferential strength",
            permitted_strength="methodological diagnostic",
            manuscript_sections_allowed="Methods; Results; Limitations",
            final_status="narrowly supported",
        ),
        Claim(
            claim_id="CLM-003",
            wording=(
                "The controlled simulation artifact quantifies extremal-index recovery error "
                "across threshold settings."
            ),
            claim_class="inferential",
            hypothesis_id="H-SIM-001",
            datasets="simulation",
            independent_units="simulation repetitions",
            estimand="mean bias and RMSE of runs extremal-index estimates",
            experiment_ids=config.experiment_id,
            table_or_figure_reference="tables/simulation_bias_rmse.csv; figures/simulation_bias_rmse.png",
            effect_estimate=_format_effect("best_runs_rmse", best_rmse),
            uncertainty="summary table preserves grouped RMSE; intervals require full simulation artifact",
            sensitivity_status="threshold grid generated",
            assumptions="simulation config and seeds define the estimand",
            counterevidence="smoke configurations are not full-scale industrial evidence",
            permitted_strength="simulation evidence only",
            manuscript_sections_allowed="Results; Simulation Study",
            final_status="supported" if np.isfinite(best_rmse) else "not estimable",
        ),
        Claim(
            claim_id="CLM-004",
            wording=(
                "The alarm frontier exposes the tradeoff between event recall, lead time, "
                "and false-alarm event rate."
            ),
            claim_class="operational",
            hypothesis_id="H-OPS-001",
            datasets=config.dataset_name,
            independent_units=_independent_units(split_summary),
            estimand="event recall and false-alarm events per operating day by threshold",
            experiment_ids=config.experiment_id,
            table_or_figure_reference=(
                "tables/lead_time_false_alarm_frontier.csv; "
                "figures/lead_time_false_alarm_frontier.png; "
                "tables/matching_tolerance_surface.csv; "
                "figures/matching_tolerance_surface.png"
            ),
            effect_estimate=(
                f"max_event_recall={_format_number(frontier_recall)}; "
                f"min_false_alarm_events_per_day={_format_number(frontier_false_alarm)}; "
                f"best_matching_tolerance_f1={_format_number(best_matching_f1)}"
            ),
            uncertainty="matching tolerance and merge-gap surface generated",
            sensitivity_status="threshold, matching tolerance, and merge-gap grids generated",
            assumptions="failure labels define independent event targets",
            counterevidence="few or no labelled failures make the frontier non-estimable",
            permitted_strength="exploratory operational tradeoff",
            manuscript_sections_allowed="Results; Limitations" if failures > 0 else "",
            final_status="exploratory"
            if failures > 0 and np.isfinite(frontier_recall)
            else "not estimable",
        ),
        Claim(
            claim_id="CLM-005",
            wording=(
                "Risk calibration is only an empirical reliability diagnostic unless evaluated "
                "on a frozen out-of-sample calibration protocol."
            ),
            claim_class="limitation",
            hypothesis_id="H-CAL-001",
            datasets=config.dataset_name,
            independent_units=f"{reliability_bins} reliability bins",
            estimand="observed event rate versus ranked horizon risk",
            experiment_ids=config.experiment_id,
            table_or_figure_reference=(
                "tables/reliability_diagram.csv; figures/reliability_diagram.png; "
                "tables/split_calibration_intervals.csv; "
                "figures/split_calibration_intervals.png"
            ),
            effect_estimate=(
                f"bins={reliability_bins}; split_bins={split_calibration_bins}; "
                f"test_brier={_format_number(test_brier)}"
            ),
            uncertainty="split-aware Wilson binomial intervals generated",
            sensitivity_status="training-ranked split calibration generated",
            assumptions="ranked observable is calibrated from the training split as a risk surrogate",
            counterevidence="full probability calibration still requires a prospective risk model",
            permitted_strength="limitation statement",
            manuscript_sections_allowed="Limitations; Discussion",
            final_status="narrowly supported" if reliability_bins > 0 else "not estimable",
        ),
        Claim(
            claim_id="CLM-006",
            wording=(
                f"The generated ablation table records {ablation_families} ablation families "
                "for sensitivity reporting."
            ),
            claim_class="methodological",
            hypothesis_id="H-ABL-001",
            datasets=config.dataset_name,
            independent_units=_independent_units(split_summary),
            estimand="registered ablation family coverage",
            experiment_ids=config.experiment_id,
            table_or_figure_reference="tables/ablation_summary.csv",
            effect_estimate=f"ablation_families={ablation_families}",
            uncertainty="not applicable for registry coverage",
            sensitivity_status="registered family summary generated",
            assumptions="summary rows are not substitutes for full real-data ablation effects",
            counterevidence="families with NaN summary values still require artifact-level interpretation",
            permitted_strength="coverage claim only",
            manuscript_sections_allowed="Experimental Protocol; Results",
            final_status="supported" if ablation_families > 0 else "not estimable",
        ),
        Claim(
            claim_id="CLM-007",
            wording=(
                "The current generated assets do not support a positive claim that the proposed "
                "method is population-level superior beyond the five registered real datasets."
            ),
            claim_class="limitation",
            hypothesis_id="H-IND-001",
            datasets="Hydraulic Systems; MetroPT; MetroPT2; SCANIA Component X; SECOM",
            independent_units="failure episode or operating day; load cycle; vehicle; wafer",
            estimand="broad industrial performance beyond the evaluated evidence",
            experiment_ids=config.experiment_id,
            table_or_figure_reference="claim_ledger.json",
            effect_estimate="not estimable beyond the registered evidence matrix",
            uncertainty="not estimable",
            sensitivity_status="five-dataset evidence matrix generated; external generalization blocked",
            assumptions="requires additional real datasets or prospective deployment for broader claims",
            counterevidence="registered datasets use different estimands and include weak diagnostic rows",
            permitted_strength="negative/null finding only",
            manuscript_sections_allowed="Limitations",
            final_status="narrowly supported",
        ),
    )


def _render_claim_markdown(claims: tuple[Claim, ...], config: ClaimLedgerConfig) -> str:
    lines = [
        "# Claim Ledger",
        "",
        f"- Experiment ID: `{config.experiment_id}`",
        f"- Code hash: `{config.code_hash}`",
        f"- Config hash: `{config.config_hash}`",
        f"- Protocol hash: `{config.protocol_hash}`",
        f"- Dataset checksum: `{config.dataset_checksum}`",
        "",
        "| Claim | Status | Permitted strength | Reference |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                (
                    claim.claim_id,
                    claim.final_status,
                    claim.permitted_strength,
                    claim.table_or_figure_reference,
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_results_synthesis(claims: tuple[Claim, ...]) -> str:
    lines = ["# Results Synthesis", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim.hypothesis_id}",
                "",
                f"- Claim ID: `{claim.claim_id}`",
                f"- Numerical finding: {claim.effect_estimate}",
                f"- Uncertainty: {claim.uncertainty}",
                f"- Sensitivity: {claim.sensitivity_status}",
                f"- Baseline comparison: {claim.counterevidence}",
                f"- Allowed wording: {claim.wording}",
                f"- Prohibited stronger wording: {claim.permitted_strength} must not be exceeded.",
                "",
            ]
        )
    return "\n".join(lines)


def _render_result_macros(claims: tuple[Claim, ...], config: ClaimLedgerConfig) -> str:
    supported = sum(
        1
        for claim in claims
        if claim.final_status in {"supported", "narrowly supported", "exploratory"}
    )
    return "\n".join(
        [
            "% Generated by dyn-evt-pdm. Do not edit manually.",
            f"\\newcommand{{\\DynEvtExperimentId}}{{{config.experiment_id}}}",
            f"\\newcommand{{\\DynEvtCodeHash}}{{{config.code_hash[:12]}}}",
            f"\\newcommand{{\\DynEvtConfigHash}}{{{config.config_hash}}}",
            f"\\newcommand{{\\DynEvtProtocolHash}}{{{config.protocol_hash}}}",
            f"\\newcommand{{\\DynEvtDatasetChecksum}}{{{config.dataset_checksum[:12]}}}",
            f"\\newcommand{{\\DynEvtClaimCount}}{{{len(claims)}}}",
            f"\\newcommand{{\\DynEvtSupportedClaimCount}}{{{supported}}}",
            "",
        ]
    )


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json_object(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid JSON object {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        failures.append(f"expected JSON object at {path}")
        return {}
    return raw


def _read_json_list(path: Path, failures: list[str]) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid JSON list {path}: {exc}")
        return []
    if not isinstance(raw, list):
        failures.append(f"expected JSON list at {path}")
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
        else:
            failures.append(f"non-object entry in {path}")
    return result


def _split_ids(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())


def _independent_units(split_summary: pd.DataFrame) -> str:
    if split_summary.empty or "split" not in split_summary:
        return "not estimable"
    return f"{len(split_summary)} split-level units"


def _minimum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.min())


def _maximum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.max())


def _split_numeric(frame: pd.DataFrame, *, split: str, column: str) -> float:
    if "split" not in frame or column not in frame:
        return float("nan")
    selected = frame[frame["split"].astype(str).str.lower() == split]
    values = pd.to_numeric(selected[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def _format_effect(name: str, value: float) -> str:
    if not np.isfinite(value):
        return f"{name}=not_estimable"
    return f"{name}={_format_number(value)}"


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "not_estimable"
    return f"{value:.4g}"
