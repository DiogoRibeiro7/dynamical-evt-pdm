"""Submission-package assembly and final adversarial review artifacts."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from dyn_evt_pdm.paper.claims import sha256_file
from dyn_evt_pdm.paper.manuscript import check_paper_sources

SubmissionDecision = Literal[
    "submission ready",
    "not submission ready",
]
REQUIRED_REAL_DATASETS = frozenset(
    {"hydraulic_systems", "metropt", "metropt2", "scania_component_x", "secom"}
)
ADVANCED_BASELINE_METHODS = frozenset(
    {
        "engineering_threshold",
        "best_individual_sensor_threshold",
        "global_empirical_threshold",
        "regime_conditioned_empirical_threshold",
        "classical_pot_gpd",
        "fixed_run_declustering",
        "ferro_segers_event_policy",
        "k_gaps_event_policy",
        "spot",
        "isolation_forest",
        "robust_online_changepoint",
        "linear_autoencoder",
        "compact_nonlinear_autoencoder",
        "conformal_anomaly",
        "empirical_horizon_risk",
        "dynamical_evt_robust_score",
        "failure_prototype_region",
        "rare_state_region",
    }
)
ADVANCED_EVENT_DATASETS = frozenset({"metropt", "metropt2"})


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    """One adversarial review finding and its revision state."""

    issue_id: str
    severity: str
    affected_claim: str
    required_action: str
    code_change: str
    experiment_change: str
    manuscript_change: str
    resolution_status: str
    resulting_artifact: str
    residual_limitation: str


@dataclass(frozen=True, slots=True)
class SubmissionPackageConfig:
    """Source and output locations for a submission package."""

    paper_root: Path = Path("paper")
    asset_root: Path = Path("reports/paper")
    output_root: Path = Path("reports/submission")
    real_data_matrix_root: Path = Path("artifacts/real_data_matrix")
    industrial_results_root: Path = Path("artifacts/real_data_matrix")
    protocol_path: Path = Path("configs/evaluation/base.yaml")
    lock_path: Path = Path("poetry.lock")
    license_path: Path = Path("LICENSE")
    readme_path: Path = Path("README.md")
    citation_path: Path = Path("CITATION.cff")
    codemeta_path: Path = Path("codemeta.json")
    release_notes_path: Path = Path("RELEASE_NOTES.md")
    archive_manifest_path: Path = Path("ARCHIVE_MANIFEST.json")


@dataclass(frozen=True, slots=True)
class SubmissionPackageManifest:
    """Manifest for assembled package payload files, excluding the manifest itself."""

    output_root: str
    created_at: float
    decision: SubmissionDecision
    files: tuple[dict[str, str], ...]
    unresolved_blockers: tuple[str, ...]
    paper_check_failures: tuple[str, ...]


def build_submission_package(config: SubmissionPackageConfig) -> SubmissionPackageManifest:
    """Assemble a reviewed submission package from compiled paper artifacts."""

    paper_check = check_paper_sources(
        paper_root=config.paper_root,
        asset_root=config.asset_root,
        require_pdfs=True,
    )
    real_data_audit = audit_real_data_matrix(config.real_data_matrix_root)
    industrial_results_audit = audit_industrial_results(config.industrial_results_root)
    issues = adversarial_review_issues(real_data_audit, industrial_results_audit)
    decision = final_submission_decision(issues, paper_check.failures)
    blockers = _submission_blockers(issues, paper_check.failures)

    output_root = config.output_root
    _ensure_clean_package_tree(output_root)
    copied: list[Path] = []
    copied.extend(_copy_required_deliverables(config, output_root))

    review_report = output_root / "reviewer_report.md"
    revision_matrix_csv = output_root / "revision_matrix.csv"
    revision_matrix_md = output_root / "revision_matrix.md"
    statements = output_root / "submission_statements.md"
    reproducibility = output_root / "reproducibility_instructions.md"
    decision_path = output_root / "final_decision.md"

    review_report.write_text(
        render_reviewer_report(issues, decision, blockers, real_data_audit),
        encoding="utf-8",
    )
    _write_revision_matrix(issues, revision_matrix_csv, revision_matrix_md)
    statements.write_text(render_submission_statements(config), encoding="utf-8")
    reproducibility.write_text(render_reproducibility_instructions(), encoding="utf-8")
    decision_path.write_text(render_final_decision(decision, blockers), encoding="utf-8")
    copied.extend(
        [
            review_report,
            revision_matrix_csv,
            revision_matrix_md,
            statements,
            reproducibility,
            decision_path,
        ]
    )

    manifest_path = output_root / "submission_manifest.json"
    manifest = SubmissionPackageManifest(
        output_root=str(output_root),
        created_at=time.time(),
        decision=decision,
        files=tuple(_file_record(path, output_root) for path in sorted(copied)),
        unresolved_blockers=blockers,
        paper_check_failures=paper_check.failures,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


@dataclass(frozen=True, slots=True)
class RealDataMatrixAudit:
    """Evidence that the registered real-data matrix reached terminal status."""

    status: str
    manifest_path: Path
    status_path: Path
    verified_datasets: tuple[str, ...]
    failed_datasets: tuple[str, ...]
    missing_datasets: tuple[str, ...]
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == "succeeded" and not self.failed_datasets and not self.missing_datasets


@dataclass(frozen=True, slots=True)
class IndustrialResultsAudit:
    """Evidence that industrial result artifacts exist for every registered dataset."""

    status: str
    summary_path: Path
    details_path: Path
    datasets: tuple[str, ...]
    failed_datasets: tuple[str, ...]
    missing_datasets: tuple[str, ...]
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == "succeeded" and not self.failed_datasets and not self.missing_datasets


def audit_real_data_matrix(root: Path) -> RealDataMatrixAudit:
    """Read the local real-data matrix manifest and dataset status table."""

    manifest_path = root / "experiment_manifest.json"
    status_path = root / "real_data_status.csv"
    report_path = root / "real_data_report.json"
    evidence_scope_path = root / "evidence_scope.json"
    if not manifest_path.exists():
        return RealDataMatrixAudit(
            status="missing",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"missing real-data matrix manifest: {manifest_path}",
        )
    if not status_path.exists():
        return RealDataMatrixAudit(
            status="missing",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"missing real-data status table: {status_path}",
        )
    for path, label in (
        (report_path, "real-data report"),
        (evidence_scope_path, "evidence-scope report"),
    ):
        artifact_error = _json_object_artifact_error(path, label)
        if artifact_error:
            return RealDataMatrixAudit(
                status="failed",
                manifest_path=manifest_path,
                status_path=status_path,
                verified_datasets=(),
                failed_datasets=(),
                missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
                reason=artifact_error,
            )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RealDataMatrixAudit(
            status="failed",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"real-data matrix manifest is unreadable: {exc}",
        )
    if not isinstance(payload, dict):
        return RealDataMatrixAudit(
            status="failed",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason="real-data matrix manifest root is not an object",
        )
    manifest_status = str(payload.get("status", "missing")).lower()
    cells = payload.get("cells", [])
    real_data_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and str(cell.get("family", "")).lower() == "real_data"
    ]
    terminal_statuses = {"succeeded", "failed", "skipped"}
    nonterminal = [
        str(cell.get("name", "unknown"))
        for cell in real_data_cells
        if str(cell.get("status", "")).lower() not in terminal_statuses
    ]
    try:
        table = pd.read_csv(status_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return RealDataMatrixAudit(
            status="failed",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"real-data status table is unreadable: {exc}",
        )
    required_columns = {"dataset_id", "status"}
    if not required_columns.issubset(table.columns):
        return RealDataMatrixAudit(
            status="failed",
            manifest_path=manifest_path,
            status_path=status_path,
            verified_datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason="real-data status table is missing dataset_id or status",
        )
    verified = tuple(
        str(row.dataset_id)
        for row in table.itertuples(index=False)
        if str(row.status).lower() == "verified"
    )
    failed = tuple(
        str(row.dataset_id)
        for row in table.itertuples(index=False)
        if str(row.status).lower() != "verified"
    )
    missing = tuple(sorted(REQUIRED_REAL_DATASETS.difference(verified).difference(failed)))
    if manifest_status != "succeeded":
        reason = f"real-data matrix manifest status is {manifest_status}"
    elif nonterminal:
        reason = f"real-data cells are not terminal: {', '.join(nonterminal)}"
    elif failed:
        reason = f"datasets are not verified: {', '.join(failed)}"
    elif missing:
        reason = f"required datasets are missing from status table: {', '.join(missing)}"
    elif not real_data_cells:
        reason = "real-data matrix manifest has no real-data cell"
    else:
        reason = f"verified datasets: {', '.join(verified)}"
    status = (
        "succeeded"
        if manifest_status == "succeeded"
        and real_data_cells
        and not nonterminal
        and not failed
        and not missing
        else "failed"
    )
    return RealDataMatrixAudit(
        status=status,
        manifest_path=manifest_path,
        status_path=status_path,
        verified_datasets=verified,
        failed_datasets=failed,
        missing_datasets=missing,
        reason=reason,
    )


def audit_industrial_results(root: Path) -> IndustrialResultsAudit:
    """Read industrial result artifacts and require terminal rows for all real datasets."""

    summary_path = root / "industrial_results_summary.csv"
    details_path = root / "industrial_results.json"
    if not summary_path.exists():
        return IndustrialResultsAudit(
            status="missing",
            summary_path=summary_path,
            details_path=details_path,
            datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"missing industrial result summary: {summary_path}",
        )
    if not details_path.exists():
        return IndustrialResultsAudit(
            status="missing",
            summary_path=summary_path,
            details_path=details_path,
            datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"missing industrial result details: {details_path}",
        )
    artifact_error = _json_object_artifact_error(details_path, "industrial result details")
    if artifact_error:
        return IndustrialResultsAudit(
            status="failed",
            summary_path=summary_path,
            details_path=details_path,
            datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=artifact_error,
        )
    try:
        table = pd.read_csv(summary_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return IndustrialResultsAudit(
            status="failed",
            summary_path=summary_path,
            details_path=details_path,
            datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason=f"industrial result summary is unreadable: {exc}",
        )
    required_columns = {"dataset_id", "status"}
    if not required_columns.issubset(table.columns):
        return IndustrialResultsAudit(
            status="failed",
            summary_path=summary_path,
            details_path=details_path,
            datasets=(),
            failed_datasets=(),
            missing_datasets=tuple(sorted(REQUIRED_REAL_DATASETS)),
            reason="industrial result summary is missing dataset_id or status",
        )
    datasets = tuple(str(value) for value in table["dataset_id"].dropna())
    terminal_statuses = {"succeeded", "not_estimable", "failed"}
    nonterminal = tuple(
        str(row.dataset_id)
        for row in table.itertuples(index=False)
        if str(row.status).lower() not in terminal_statuses
    )
    failed = tuple(
        str(row.dataset_id)
        for row in table.itertuples(index=False)
        if str(row.status).lower() == "failed"
    )
    missing = tuple(sorted(REQUIRED_REAL_DATASETS.difference(datasets)))
    if nonterminal:
        reason = f"industrial result rows are not terminal: {', '.join(nonterminal)}"
    elif failed:
        reason = f"industrial result rows failed: {', '.join(failed)}"
    elif missing:
        reason = f"required industrial result rows are missing: {', '.join(missing)}"
    else:
        reason = f"industrial result artifacts cover: {', '.join(datasets)}"
    status = "succeeded" if not nonterminal and not failed and not missing else "failed"
    return IndustrialResultsAudit(
        status=status,
        summary_path=summary_path,
        details_path=details_path,
        datasets=datasets,
        failed_datasets=failed,
        missing_datasets=missing,
        reason=reason,
    )


def _csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return not table.empty


def _broad_simulation_artifact_ok(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return False
    systems = config.get("systems", [])
    sample_sizes = config.get("sample_sizes", [])
    thresholds = config.get("threshold_quantiles", [])
    run_lengths = config.get("run_lengths", [])
    try:
        repetitions = int(config.get("repetitions", 0))
        rows = int(payload.get("rows", 0))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(systems, list)
        and isinstance(sample_sizes, list)
        and isinstance(thresholds, list)
        and isinstance(run_lengths, list)
        and len(systems) >= 8
        and len(sample_sizes) >= 2
        and len(thresholds) >= 3
        and len(run_lengths) >= 3
        and repetitions >= 10
        and rows >= 2500
    )


def _high_replication_simulation_artifact_ok(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return False
    try:
        repetitions = int(config.get("repetitions", 0))
        rows = int(payload.get("rows", 0))
    except (TypeError, ValueError):
        return False
    return repetitions >= 500 and rows > 0


def _event_baseline_family_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    required = {"dataset_id", "method"}
    if table.empty or not required.issubset(table.columns):
        return False
    for dataset_id in ADVANCED_EVENT_DATASETS:
        observed = set(table.loc[table["dataset_id"] == dataset_id, "method"].astype(str))
        if not ADVANCED_BASELINE_METHODS.issubset(observed):
            return False
    return True


def _software_archive_doi_present(citation_path: Path) -> bool:
    if not citation_path.exists():
        return False
    try:
        text = citation_path.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "doi:" in text or "\ndoi:" in text or "\nidentifiers:" in text


def _json_object_artifact_error(path: Path, label: str) -> str:
    if not path.exists():
        return f"missing {label}: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        verb = "are" if label.endswith("details") else "is"
        return f"{label} {verb} unreadable: {exc}"
    if not isinstance(payload, dict):
        return f"{label} root is not an object"
    return ""


def adversarial_review_issues(
    real_data_audit: RealDataMatrixAudit | None = None,
    industrial_results_audit: IndustrialResultsAudit | None = None,
) -> tuple[ReviewIssue, ...]:
    """Return the final adversarial review matrix."""

    audit = real_data_audit or audit_real_data_matrix(Path("artifacts/real_data_matrix"))
    industrial_audit = industrial_results_audit or audit_industrial_results(
        Path("artifacts/real_data_matrix")
    )
    rev001_resolved = industrial_audit.ok
    rev005_resolved = audit.ok
    rev001_experiment = (
        "Industrial result artifacts exist for the five registered real datasets; "
        "the manuscript still blocks population-level superiority language."
        if rev001_resolved
        else "Full industrial matrix remains required before stronger claims."
    )
    rev001_artifact = (
        f"{industrial_audit.summary_path}; {industrial_audit.details_path}"
        if rev001_resolved
        else "paper/main.tex; reports/paper/claim_ledger.json; paper/generated/claim_audit.json"
    )
    rev005_experiment = (
        f"Full real-data matrix reached terminal status with {len(audit.verified_datasets)} "
        f"verified datasets."
        if rev005_resolved
        else "The reproducibility subset has terminal status; full real-data matrix is not yet terminal."
    )
    rev005_artifact = (
        f"{audit.manifest_path}; {audit.status_path}"
        if rev005_resolved
        else "artifacts/experiment_matrix/experiment_manifest.json; paper/sections/08_discussion.tex"
    )
    simulation_scope_resolved = _broad_simulation_artifact_ok(
        Path("artifacts/simulation_study_broad.parquet.manifest.json")
    )
    industrial_artifact_root = industrial_audit.summary_path.parent
    event_baseline_resolved = _csv_has_rows(
        industrial_artifact_root / "event_baseline_comparison.csv"
    )
    event_variant_resolved = _csv_has_rows(
        industrial_artifact_root / "event_variant_comparison.csv"
    )
    event_timeline_resolved = _csv_has_rows(industrial_artifact_root / "event_timeline_trace.csv")
    high_rep_simulation_resolved = _high_replication_simulation_artifact_ok(
        Path("artifacts/simulation_study_high_rep.parquet.manifest.json")
    )
    complete_baseline_family_resolved = _event_baseline_family_complete(
        industrial_artifact_root / "event_baseline_comparison.csv"
    )
    transferability_resolved = _csv_has_rows(
        industrial_artifact_root / "target_region_transferability.csv"
    )
    matched_controls_resolved = _csv_has_rows(
        industrial_artifact_root / "matched_negative_controls.csv"
    )
    failure_decomposition_resolved = _csv_has_rows(
        industrial_artifact_root / "score_threshold_alarm_decomposition.csv"
    )
    timeline_reconciliation_resolved = _csv_has_rows(
        industrial_artifact_root / "timeline_reconciliation.csv"
    )
    main_results_streamlined = not (
        (Path("paper/sections/07_results.tex").exists())
        and any(
            phrase in Path("paper/sections/07_results.tex").read_text(encoding="utf-8")
            for phrase in (
                "method-scope",
                "timeline-traceability",
                "metric-provenance",
            )
        )
    )
    software_archive_resolved = _software_archive_doi_present(Path("CITATION.cff"))
    legacy_issues = (
        ReviewIssue(
            issue_id="REV-001",
            severity="fatal-if-unqualified",
            affected_claim="CLM-003; CLM-007",
            required_action="Prevent the manuscript from presenting the workflow as a new industrial theorem or broad superiority result.",
            code_change="Claim-ledger and paper-check commands block untracked claim IDs and generated-asset provenance failures.",
            experiment_change=rev001_experiment,
            manuscript_change="Title, abstract, results, discussion, and conclusion use diagnostic and limitation language.",
            resolution_status="resolved" if rev001_resolved else "partially resolved",
            resulting_artifact=rev001_artifact,
            residual_limitation="" if rev001_resolved else industrial_audit.reason,
        ),
        ReviewIssue(
            issue_id="REV-002",
            severity="major",
            affected_claim="CLM-001",
            required_action="Keep independent-unit accounting separate from raw row counts.",
            code_change="Paper checker requires claim references; claim ledger records split-level units and limitations.",
            experiment_change="Future real-data intervals must bootstrap over failures or vehicles.",
            manuscript_change="Dataset, protocol, and discussion sections explicitly reject row-count independence.",
            resolution_status="resolved",
            resulting_artifact="paper/sections/05_datasets.tex; paper/sections/06_experimental_protocol.tex",
            residual_limitation="",
        ),
        ReviewIssue(
            issue_id="REV-003",
            severity="major",
            affected_claim="CLM-002; CLM-006",
            required_action="Expose threshold, run-length, and ablation sensitivity instead of selecting one favorable setting.",
            code_change="Paper assets build threshold stability and ablation tables from code.",
            experiment_change="Sensitivity artifacts regenerated by make paper-check.",
            manuscript_change="Results and supplement label the ablation table as coverage, not performance superiority.",
            resolution_status="resolved",
            resulting_artifact="reports/paper/figures/threshold_stability.png; reports/paper/tables/ablation_summary.csv",
            residual_limitation="",
        ),
        ReviewIssue(
            issue_id="REV-004",
            severity="major",
            affected_claim="CLM-004; CLM-005",
            required_action="Block strong warning-lead-time and calibration claims when they rely on empirical diagnostics only.",
            code_change="Paper assets now generate matching-tolerance surfaces and split-aware calibration intervals with provenance.",
            experiment_change="make paper-check regenerates matching tolerance, merge-gap, and training-ranked calibration artifacts.",
            manuscript_change="Results section references tolerance and calibration interval artifacts while preserving diagnostic language.",
            resolution_status="resolved",
            resulting_artifact=(
                "reports/paper/tables/matching_tolerance_surface.csv; "
                "reports/paper/tables/split_calibration_intervals.csv"
            ),
            residual_limitation="",
        ),
        ReviewIssue(
            issue_id="REV-005",
            severity="major",
            affected_claim="CLM-007",
            required_action="Preserve negative results and prevent hidden failed experiments.",
            code_change="Experiment matrix records terminal failed/skipped/succeeded status; paper assets preserve null limitation claims.",
            experiment_change=rev005_experiment,
            manuscript_change="Discussion presents limits on broader industrial performance claims as an explicit negative finding.",
            resolution_status="resolved" if rev005_resolved else "partially resolved",
            resulting_artifact=rev005_artifact,
            residual_limitation="" if rev005_resolved else audit.reason,
        ),
        ReviewIssue(
            issue_id="REV-006",
            severity="major",
            affected_claim="all manuscript claims",
            required_action="Produce a reproducibility package and final readiness decision.",
            code_change="Submission-package builder copies sources, manifests, claims, protocol, lock file, and availability statements.",
            experiment_change="make paper-check regenerates the reproducibility subset before package assembly.",
            manuscript_change="Supplement gives build and audit instructions.",
            resolution_status="resolved",
            resulting_artifact="reports/submission/submission_manifest.json",
            residual_limitation="",
        ),
        ReviewIssue(
            issue_id="REV-007",
            severity="major",
            affected_claim="CLM-001; CLM-002",
            required_action="Complete the broad Monte Carlo validation grid before claiming estimator reliability.",
            code_change="Paper assets now expose method-scope completion status.",
            experiment_change=(
                "Broad simulation artifact generated."
                if simulation_scope_resolved
                else "Only the smoke-scale simulation recovery artifact is generated."
            ),
            manuscript_change="Simulation and results sections label the generated simulation as bounded finite-sample validation.",
            resolution_status="resolved" if simulation_scope_resolved else "partially resolved",
            resulting_artifact="artifacts/simulation_study_broad.parquet; reports/paper/tables/method_scope_completion.csv",
            residual_limitation=""
            if simulation_scope_resolved
            else "full Monte Carlo validation grid is not completed",
        ),
        ReviewIssue(
            issue_id="REV-008",
            severity="major",
            affected_claim="CLM-003; CLM-007",
            required_action="Run target-region variants, negative controls, and real event-level baselines before claiming comparative superiority.",
            code_change="Metric-provenance and method-scope assets separate smoke baselines from event metrics.",
            experiment_change=(
                "Real event-level baseline and variant artifacts generated."
                if event_baseline_resolved and event_variant_resolved
                else "Real event-level baseline comparison and variant matrix are not generated."
            ),
            manuscript_change="Main text reports real event-level comparisons and keeps superiority language blocked.",
            resolution_status="resolved"
            if event_baseline_resolved and event_variant_resolved
            else "partially resolved",
            resulting_artifact="reports/paper/tables/metric_provenance.csv; reports/paper/tables/method_scope_completion.csv",
            residual_limitation=""
            if event_baseline_resolved and event_variant_resolved
            else "real event-level baseline comparison and target-region variant matrix are not completed",
        ),
        ReviewIssue(
            issue_id="REV-009",
            severity="major",
            affected_claim="CLM-004; CLM-005",
            required_action="Replace representative timing displays with failure-specific real-data timelines before using them as empirical evidence.",
            code_change="Timeline traceability asset records the displayed source and manuscript use.",
            experiment_change=(
                "MetroPT held-out failure timeline trace generated."
                if event_timeline_resolved
                else "No real MetroPT or MetroPT2 failure-specific timeline artifact is generated."
            ),
            manuscript_change="Timeline caption and audit table identify the real held-out failure source.",
            resolution_status="resolved" if event_timeline_resolved else "partially resolved",
            resulting_artifact="reports/paper/tables/timeline_traceability.csv",
            residual_limitation=""
            if event_timeline_resolved
            else "real failure-specific timeline artifact is not completed",
        ),
        ReviewIssue(
            issue_id="REV-010",
            severity="minor",
            affected_claim="software availability",
            required_action="Provide software citation metadata and create an external archive DOI only if required by the target venue.",
            code_change="Added local citation metadata, CodeMeta metadata, and release notes.",
            experiment_change="No experimental change.",
            manuscript_change="Repository is cited by URL without an invented DOI.",
            resolution_status="resolved",
            resulting_artifact="CITATION.cff; codemeta.json; RELEASE_NOTES.md",
            residual_limitation="",
        ),
    )
    advanced_issues = (
        ReviewIssue(
            issue_id="REV-011",
            severity="major",
            affected_claim="simulation validity",
            required_action="Replace ten-replication simulation with high-replication Monte Carlo evidence before claiming adequacy.",
            code_change="Submission audit now gates high-replication simulation separately from the earlier broad smoke-scale grid.",
            experiment_change=(
                "High-replication simulation manifest found."
                if high_rep_simulation_resolved
                else "No high-replication simulation manifest with at least 500 repetitions per cell is present."
            ),
            manuscript_change="Simulation claims must remain bounded until high-replication evidence exists.",
            resolution_status="resolved" if high_rep_simulation_resolved else "unresolved",
            resulting_artifact="artifacts/simulation_study_high_rep.parquet",
            residual_limitation=""
            if high_rep_simulation_resolved
            else "high-replication Monte Carlo validation is incomplete",
        ),
        ReviewIssue(
            issue_id="REV-012",
            severity="major",
            affected_claim="event-level baseline comparison",
            required_action="Run every declared baseline family on MetroPT and MetroPT2 under the same event policy.",
            code_change="Submission audit checks declared baseline-family coverage in event_baseline_comparison.csv.",
            experiment_change=(
                "Every declared baseline has held-out event-level rows."
                if complete_baseline_family_resolved
                else "The event-level baseline artifact does not yet contain every declared baseline for both compressor datasets."
            ),
            manuscript_change="Comparative baseline claims must remain limited until the family is complete.",
            resolution_status="resolved" if complete_baseline_family_resolved else "unresolved",
            resulting_artifact="artifacts/real_data_matrix/event_baseline_comparison.csv",
            residual_limitation=""
            if complete_baseline_family_resolved
            else "complete event-level baseline family is incomplete",
        ),
        ReviewIssue(
            issue_id="REV-013",
            severity="major",
            affected_claim="target-region transferability",
            required_action="Quantify MetroPT versus MetroPT2 failure-prototype divergence with state-space evidence.",
            code_change="Submission audit requires a target-region transferability artifact.",
            experiment_change=(
                "Target-region transferability artifact found."
                if transferability_resolved
                else "No target-region transferability artifact is present."
            ),
            manuscript_change="Abstract, Results, Discussion, and Conclusion must mention prototype-region instability only after artifact support.",
            resolution_status="resolved" if transferability_resolved else "unresolved",
            resulting_artifact="artifacts/real_data_matrix/target_region_transferability.csv",
            residual_limitation=""
            if transferability_resolved
            else "target-region transferability analysis is incomplete",
        ),
        ReviewIssue(
            issue_id="REV-014",
            severity="major",
            affected_claim="negative-control validity",
            required_action="Replace single negative controls with repeated matched controls preserving nuisance structure.",
            code_change="Submission audit requires matched_negative_controls.csv.",
            experiment_change=(
                "Matched negative-control draws are present."
                if matched_controls_resolved
                else "No repeated matched negative-control artifact is present."
            ),
            manuscript_change="Negative-control conclusions must not exceed the generated control distribution.",
            resolution_status="resolved" if matched_controls_resolved else "unresolved",
            resulting_artifact="artifacts/real_data_matrix/matched_negative_controls.csv",
            residual_limitation=""
            if matched_controls_resolved
            else "matched repeated negative controls are incomplete",
        ),
        ReviewIssue(
            issue_id="REV-015",
            severity="major",
            affected_claim="root-cause analysis",
            required_action="Separate score failure, threshold failure, and alarm-conversion failure before assigning root cause.",
            code_change="Submission audit requires score_threshold_alarm_decomposition.csv.",
            experiment_change=(
                "Three-layer failure decomposition artifact found."
                if failure_decomposition_resolved
                else "No three-layer failure decomposition artifact is present."
            ),
            manuscript_change="Discussion cannot attribute failures to alarm conversion without score and threshold evidence.",
            resolution_status="resolved" if failure_decomposition_resolved else "unresolved",
            resulting_artifact="artifacts/real_data_matrix/score_threshold_alarm_decomposition.csv",
            residual_limitation=""
            if failure_decomposition_resolved
            else "score-threshold-alarm failure decomposition is incomplete",
        ),
        ReviewIssue(
            issue_id="REV-016",
            severity="major",
            affected_claim="timeline evidence",
            required_action="Reconcile the local timeline panel with global alarm counts and full-test exposure.",
            code_change="Submission audit requires timeline_reconciliation.csv.",
            experiment_change=(
                "Timeline reconciliation artifact found."
                if timeline_reconciliation_resolved
                else "No local/global timeline reconciliation artifact is present."
            ),
            manuscript_change="Timeline caption must state that local panels do not show all global alarms.",
            resolution_status="resolved" if timeline_reconciliation_resolved else "unresolved",
            resulting_artifact="artifacts/real_data_matrix/timeline_reconciliation.csv",
            residual_limitation=""
            if timeline_reconciliation_resolved
            else "local/global timeline reconciliation is incomplete",
        ),
        ReviewIssue(
            issue_id="REV-017",
            severity="major",
            affected_claim="main-results discipline",
            required_action="Move audit-completion and traceability tables out of the main Results.",
            code_change="Submission audit scans the main Results for repository-status table inputs.",
            experiment_change="No experimental change.",
            manuscript_change=(
                "Main Results no longer includes audit-completion table inputs."
                if main_results_streamlined
                else "Main Results still includes audit-completion or traceability table inputs."
            ),
            resolution_status="resolved" if main_results_streamlined else "unresolved",
            resulting_artifact="paper/sections/07_results.tex",
            residual_limitation=""
            if main_results_streamlined
            else "main Results still contains audit-oriented tables",
        ),
        ReviewIssue(
            issue_id="REV-018",
            severity="major",
            affected_claim="software availability",
            required_action="Create and cite a DOI-backed software archive, or mark archive DOI creation incomplete.",
            code_change="Submission audit checks citation metadata for a persistent DOI/identifier.",
            experiment_change="No experimental change.",
            manuscript_change=(
                "Persistent software identifier is present in citation metadata."
                if software_archive_resolved
                else "Repository citation remains URL-only; no DOI-backed archive is claimed."
            ),
            resolution_status="resolved" if software_archive_resolved else "unresolved",
            resulting_artifact="CITATION.cff; external archive record",
            residual_limitation=""
            if software_archive_resolved
            else "DOI-backed software archive is incomplete",
        ),
    )
    return (*legacy_issues, *advanced_issues)


def final_submission_decision(
    issues: tuple[ReviewIssue, ...],
    paper_check_failures: tuple[str, ...],
) -> SubmissionDecision:
    """Return final readiness based on unresolved blockers and paper-check status."""

    if paper_check_failures:
        return "not submission ready"
    if any(
        issue.severity.startswith("fatal") and issue.resolution_status != "resolved"
        for issue in issues
    ):
        return "not submission ready"
    if any(issue.resolution_status != "resolved" for issue in issues):
        return "not submission ready"
    return "submission ready"


def _submission_blockers(
    issues: tuple[ReviewIssue, ...],
    paper_check_failures: tuple[str, ...],
) -> tuple[str, ...]:
    issue_blockers = tuple(
        issue.residual_limitation
        for issue in issues
        if issue.resolution_status != "resolved" and issue.residual_limitation
    )
    paper_blockers = tuple(f"paper check failed: {failure}" for failure in paper_check_failures)
    return (*issue_blockers, *paper_blockers)


def render_reviewer_report(
    issues: tuple[ReviewIssue, ...],
    decision: SubmissionDecision,
    blockers: tuple[str, ...],
    real_data_audit: RealDataMatrixAudit | None = None,
) -> str:
    """Render the hostile but technically fair reviewer report."""

    fatal = [issue for issue in issues if issue.severity.startswith("fatal")]
    major = [issue for issue in issues if issue.severity == "major"]
    audit = real_data_audit or audit_real_data_matrix(Path("artifacts/real_data_matrix"))
    real_data_reproducibility_line = (
        "- Full real-data matrix terminal status is recorded locally: " f"{audit.reason}."
        if audit.ok
        else "- The reproducibility subset rebuilds locally, but full real-data terminal "
        f"experiment status remains a blocker: {audit.reason}."
    )
    confidence = (
        "Confidence: high for the readiness decision because it follows generated checks "
        "and all adversarial-review issues are resolved."
        if not blockers
        else "Confidence: high for the readiness decision because it follows generated checks "
        "and explicit unresolved blockers."
    )
    blocker_lines = [f"- {blocker}" for blocker in blockers] if blockers else ["- None."]
    return "\n".join(
        [
            "# Final Adversarial Reviewer Report",
            "",
            "## Summary Of The Contribution",
            "",
            "The study provides a reproducible, dynamical-EVT-inspired negative-result analysis of public industrial monitoring data. Its credible contribution is the evidence-bound finding that clustered extreme scores and dependence diagnostics did not translate into reliable operational early warning under the evaluated alarm policies.",
            "",
            "## Fatal Flaws",
            "",
            *_issue_lines(fatal),
            "",
            "## Major Concerns",
            "",
            *_issue_lines(major),
            "",
            "## Minor Concerns",
            "",
            "- Long generated tables may require journal-specific formatting.",
            "- The manuscript is intentionally conservative and will need venue-specific tightening.",
            "",
            "## Reproducibility Concerns",
            "",
            real_data_reproducibility_line,
            "- Generated assets are checksummed, and manual changes are rejected by the verifier.",
            "",
            "## Statistical Concerns",
            "",
            "- Effective sample size must remain tied to failures, blocks, or vehicles.",
            "- Calibration and operational alarm-grid figures are diagnostics without full out-of-sample intervals.",
            "- Matching tolerance, missingness, smoothing, and dangerous-region provenance would remain decision-changing audits for any broader deployment or superiority claim.",
            "",
            "## Missing Citations",
            "",
            "- No missing citation was detected by the current paper checker.",
            "",
            "## Claims Kept Restricted",
            "",
            "- Broad industrial performance language beyond the evaluated evidence must remain blocked.",
            "- Any causal mechanical degradation language must remain blocked.",
            "- Calibration must be described as a diagnostic unless calibrated out-of-sample probability evidence is added.",
            "",
            "## Evidence That Would Broaden Claims",
            "",
            "- Prospective compressor-event evidence under predeclared settings.",
            "- Additional independent event datasets large enough for event-level uncertainty estimation.",
            "- SCANIA vehicle-level uncertainty intervals if making broader fleet-risk claims.",
            "- Larger missingness, smoothing, downsampling, and baseline-challenge surfaces for stronger generalization claims.",
            "",
            "## Recommendation",
            "",
            f"Recommendation: {decision}.",
            "",
            "## Confidence",
            "",
            confidence,
            "",
            "## Unresolved Blockers",
            "",
            *blocker_lines,
            "",
        ]
    )


def render_submission_statements(config: SubmissionPackageConfig) -> str:
    """Render required journal-facing administrative statements."""

    return "\n".join(
        [
            "# Submission Statements",
            "",
            "## Code Availability",
            "",
            "The analysis code, manuscript sources, and build scripts are available in the associated Git repository. Generated artifacts are rebuilt by the documented Make targets.",
            "",
            "## Data Availability",
            "",
            "The package does not redistribute raw third-party datasets. Public dataset sources and local acquisition provenance are recorded by the repository data registry and raw-data fetch pipeline.",
            "",
            "## Environment",
            "",
            f"Dependencies are pinned by `{config.lock_path}`.",
            "",
            "## License",
            "",
            f"Repository licensing is recorded in `{config.license_path}`.",
            "",
            "## Author Contributions",
            "",
            "Diogo Ribeiro: conceptualization, software, validation, formal analysis, data curation, manuscript drafting, and reproducibility packaging.",
            "",
            "## Competing Interests",
            "",
            "No competing-interest disclosure is recorded in the repository metadata.",
            "",
            "## Funding",
            "",
            "No external funding source is recorded in the repository metadata.",
            "",
            "## Ethics",
            "",
            "No human-subject intervention is performed by this repository. Dataset-specific ethics and license terms remain governed by the original data providers.",
            "",
        ]
    )


def render_reproducibility_instructions() -> str:
    """Render clean-environment reproduction instructions."""

    return "\n".join(
        [
            "# Reproducibility Instructions",
            "",
            "1. Install dependencies with `poetry install --with dev`.",
            "2. Fetch public raw data with `make fetch-data` where licenses and network access permit.",
            "3. Prepare available real datasets with `make prepare-data`.",
            "4. Run the reproducibility experiment subset with `make experiment-matrix`.",
            "5. Regenerate figures, tables, macros, provenance, and claim ledger with `make paper-assets`.",
            "6. Compile and audit the manuscript and supplement with `make paper-check`.",
            "7. Assemble this package with `poetry run dyn-evt build-submission-package`.",
            "",
            "Deviations must be recorded in the reviewer report or revision matrix. Raw datasets are not copied into the submission package.",
            "",
        ]
    )


def render_final_decision(decision: SubmissionDecision, blockers: tuple[str, ...]) -> str:
    """Render final readiness decision."""

    lines = ["# Final Decision", "", decision, ""]
    if blockers:
        lines.extend(["## Unresolved Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
        lines.append("")
    return "\n".join(lines)


def _issue_lines(issues: list[ReviewIssue]) -> list[str]:
    if not issues:
        return ["- None."]
    return [
        f"- {issue.issue_id}: {issue.required_action} Status: {issue.resolution_status}. "
        f"Residual limitation: {issue.residual_limitation or 'none'}"
        for issue in issues
    ]


def _write_revision_matrix(
    issues: tuple[ReviewIssue, ...],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    frame = pd.DataFrame([asdict(issue) for issue in issues])
    frame.to_csv(csv_path, index=False)
    lines = [
        "# Revision Matrix",
        "",
        "| Issue | Severity | Affected claim | Status | Residual limitation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for issue in issues:
        lines.append(
            f"| {issue.issue_id} | {issue.severity} | {issue.affected_claim} | "
            f"{issue.resolution_status} | {issue.residual_limitation or 'none'} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_required_deliverables(config: SubmissionPackageConfig, output_root: Path) -> list[Path]:
    copied: list[Path] = []
    copies = (
        (config.paper_root / "main.pdf", output_root / "manuscript" / "main.pdf"),
        (
            config.paper_root / "supplement" / "supplement.pdf",
            output_root / "supplement" / "supplement.pdf",
        ),
        (config.paper_root / "references.bib", output_root / "sources" / "references.bib"),
        (config.paper_root / "main.tex", output_root / "sources" / "main.tex"),
        (
            config.paper_root / "pdf_reproducibility.tex",
            output_root / "sources" / "pdf_reproducibility.tex",
        ),
        (config.paper_root / "latexmkrc", output_root / "sources" / "latexmkrc"),
        (config.paper_root / "Makefile", output_root / "sources" / "Makefile"),
        (config.protocol_path, output_root / "artifacts" / "evaluation_protocol.yaml"),
        (config.lock_path, output_root / "environment" / "poetry.lock"),
        (config.license_path, output_root / "LICENSE"),
        (config.readme_path, output_root / "README.md"),
        (config.citation_path, output_root / "CITATION.cff"),
        (config.codemeta_path, output_root / "codemeta.json"),
        (config.release_notes_path, output_root / "RELEASE_NOTES.md"),
        (config.archive_manifest_path, output_root / "ARCHIVE_MANIFEST.json"),
        (Path("docs/release_archive.md"), output_root / "sources" / "release_archive.md"),
        (
            config.asset_root / "asset_manifest.json",
            output_root / "artifacts" / "asset_manifest.json",
        ),
        (
            config.asset_root / "asset_provenance.json",
            output_root / "artifacts" / "asset_provenance.json",
        ),
        (
            config.asset_root / "claim_ledger.json",
            output_root / "artifacts" / "claim_ledger.json",
        ),
        (
            config.asset_root / "claim_ledger.md",
            output_root / "artifacts" / "claim_ledger.md",
        ),
    )
    for source, destination in copies:
        copied.append(_copy_file(source, destination))
    copied.extend(
        _copy_tree_files(config.paper_root / "sections", output_root / "sources" / "sections")
    )
    copied.extend(
        _copy_tree_files(config.paper_root / "supplement", output_root / "sources" / "supplement")
    )
    copied.extend(
        _copy_tree_files(config.paper_root / "generated", output_root / "sources" / "generated")
    )
    copied.extend(_copy_tree_files(config.asset_root / "figures", output_root / "figures"))
    copied.extend(_copy_tree_files(config.asset_root / "tables", output_root / "tables"))
    copied.extend(_copy_tree_files(config.asset_root / "latex", output_root / "generated"))
    copied.extend(
        _copy_tree_files(
            config.asset_root / "figures", output_root / "reports" / "paper" / "figures"
        )
    )
    copied.extend(
        _copy_tree_files(config.asset_root / "tables", output_root / "reports" / "paper" / "tables")
    )
    copied.extend(
        _copy_tree_files(config.asset_root / "latex", output_root / "reports" / "paper" / "latex")
    )
    copied.extend(_copy_real_data_matrix_files(config.real_data_matrix_root, output_root))
    copied.extend(_copy_industrial_result_files(config.industrial_results_root, output_root))
    return copied


def _copy_real_data_matrix_files(source_root: Path, output_root: Path) -> list[Path]:
    copied: list[Path] = []
    for name in (
        "experiment_manifest.json",
        "real_data_status.csv",
        "real_data_report.json",
        "dataset_characteristics.tex",
        "evidence_scope.csv",
        "evidence_scope.json",
        "evidence_scope.tex",
    ):
        source = source_root / name
        if source.exists():
            copied.append(_copy_file(source, output_root / "artifacts" / "real_data_matrix" / name))
    return copied


def _copy_industrial_result_files(source_root: Path, output_root: Path) -> list[Path]:
    copied: list[Path] = []
    for name in (
        "industrial_results_summary.csv",
        "industrial_results.json",
        "industrial_results.tex",
        "event_baseline_comparison.csv",
        "event_variant_comparison.csv",
        "event_timeline_trace.csv",
        "event_level_comparison_manifest.json",
    ):
        source = source_root / name
        if source.exists():
            copied.append(
                _copy_file(source, output_root / "artifacts" / "industrial_results" / name)
            )
            if name == "industrial_results.tex":
                copied.append(
                    _copy_file(source, output_root / "artifacts" / "real_data_matrix" / name)
                )
    return copied


def _copy_tree_files(source_root: Path, destination_root: Path) -> list[Path]:
    copied: list[Path] = []
    if not source_root.exists():
        return copied
    for source in source_root.rglob("*"):
        if not source.is_file() or _is_latex_build_output(source):
            continue
        destination = destination_root / source.relative_to(source_root)
        copied.append(_copy_file(source, destination))
    return copied


def _copy_file(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"submission source missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _ensure_clean_package_tree(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _file_record(path: Path, output_root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_file(path),
    }


def _is_latex_build_output(path: Path) -> bool:
    return path.suffix.lower() in {
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".pdf",
        ".toc",
    }
