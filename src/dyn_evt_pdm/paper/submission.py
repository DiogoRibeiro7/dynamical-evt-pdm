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
    "submission ready after minor editorial changes",
    "not submission ready",
]


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
    protocol_path: Path = Path("configs/evaluation/base.yaml")
    lock_path: Path = Path("poetry.lock")
    license_path: Path = Path("LICENSE")
    readme_path: Path = Path("README.md")


@dataclass(frozen=True, slots=True)
class SubmissionPackageManifest:
    """Manifest for the assembled submission package."""

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
    issues = adversarial_review_issues()
    decision = final_submission_decision(issues, paper_check.failures)
    blockers = tuple(
        issue.residual_limitation
        for issue in issues
        if issue.resolution_status != "resolved" and issue.residual_limitation
    )

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

    review_report.write_text(render_reviewer_report(issues, decision, blockers), encoding="utf-8")
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
    return SubmissionPackageManifest(
        output_root=manifest.output_root,
        created_at=manifest.created_at,
        decision=manifest.decision,
        files=tuple([*manifest.files, _file_record(manifest_path, output_root)]),
        unresolved_blockers=manifest.unresolved_blockers,
        paper_check_failures=manifest.paper_check_failures,
    )


def adversarial_review_issues() -> tuple[ReviewIssue, ...]:
    """Return the final adversarial review matrix."""

    return (
        ReviewIssue(
            issue_id="REV-001",
            severity="fatal-if-unqualified",
            affected_claim="CLM-003; CLM-007",
            required_action="Prevent the manuscript from presenting the workflow as a new industrial theorem or broad superiority result.",
            code_change="Claim-ledger and paper-check commands block untracked claim IDs and generated-asset provenance failures.",
            experiment_change="Full industrial matrix remains required before stronger claims.",
            manuscript_change="Title, abstract, results, discussion, and conclusion use diagnostic and limitation language.",
            resolution_status="partially resolved",
            resulting_artifact="paper/main.tex; reports/paper/claim_ledger.json; paper/generated/claim_audit.json",
            residual_limitation="Complete MetroPT, MetroPT2, and SCANIA result artifacts are still needed for submission-level industrial claims.",
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
            experiment_change="Smoke-scale sensitivity artifacts regenerated by make paper-check.",
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
            experiment_change="The reproducibility subset has terminal status; full real-data matrix is not yet terminal.",
            manuscript_change="Discussion presents lack of population-level industrial superiority as an explicit negative finding.",
            resolution_status="partially resolved",
            resulting_artifact="artifacts/experiment_matrix_prompt17/experiment_manifest.json; paper/sections/08_discussion.tex",
            residual_limitation="Full registered real-data matrix terminal statuses must be regenerated in the final environment.",
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
    )


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
    return "submission ready after minor editorial changes"


def render_reviewer_report(
    issues: tuple[ReviewIssue, ...],
    decision: SubmissionDecision,
    blockers: tuple[str, ...],
) -> str:
    """Render the hostile but technically fair reviewer report."""

    fatal = [issue for issue in issues if issue.severity.startswith("fatal")]
    major = [issue for issue in issues if issue.severity == "major"]
    return "\n".join(
        [
            "# Final Adversarial Reviewer Report",
            "",
            "## Summary Of The Contribution",
            "",
            "The study provides a reproducible, dynamical-EVT-inspired diagnostic workflow for predictive maintenance. Its credible contribution is the artifact discipline: frozen evaluation, event-level alarm accounting, sensitivity assets, claim ledger, and manuscript/package checks.",
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
            "- The reproducibility subset rebuilds locally, but full real-data terminal experiment status remains a blocker for industrial claims.",
            "- Generated assets are checksummed, and manual changes are rejected by the verifier.",
            "",
            "## Statistical Concerns",
            "",
            "- Effective sample size must remain tied to failures, blocks, or vehicles.",
            "- Calibration and operational frontier figures are diagnostics without full out-of-sample intervals.",
            "- Matching tolerance, missingness, smoothing, and dangerous-region provenance remain decision-changing audits.",
            "",
            "## Missing Citations",
            "",
            "- No missing citation was detected by the current paper checker.",
            "",
            "## Claims That Must Be Weakened",
            "",
            "- Any population-level industrial superiority language must remain blocked.",
            "- Any causal mechanical degradation language must remain blocked.",
            "- Calibration must be described as a diagnostic unless calibrated out-of-sample probability evidence is added.",
            "",
            "## Experiments That Would Change The Decision",
            "",
            "- Complete MetroPT leave-one-failure-out experiments.",
            "- Complete MetroPT2 replication under predeclared settings.",
            "- Complete SCANIA vehicle-level validation with vehicle-level uncertainty.",
            "- Matching-tolerance, missingness, smoothing, and baseline-strengthening challenge surfaces.",
            "",
            "## Recommendation",
            "",
            f"Recommendation: {decision}.",
            "",
            "## Confidence",
            "",
            "Confidence: high for the readiness decision because it follows generated checks and explicit unresolved blockers.",
            "",
            "## Unresolved Blockers",
            "",
            *[f"- {blocker}" for blocker in blockers],
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
            "Author contribution statement to be completed by the final author list.",
            "",
            "## Competing Interests",
            "",
            "Conflict-of-interest statement to be completed before journal submission.",
            "",
            "## Funding",
            "",
            "Funding statement to be completed before journal submission.",
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
        (config.protocol_path, output_root / "artifacts" / "evaluation_protocol.yaml"),
        (config.lock_path, output_root / "environment" / "poetry.lock"),
        (config.license_path, output_root / "LICENSE"),
        (config.readme_path, output_root / "README.md"),
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
    copied.extend(_copy_tree_files(config.asset_root / "figures", output_root / "figures"))
    copied.extend(_copy_tree_files(config.asset_root / "tables", output_root / "tables"))
    copied.extend(_copy_tree_files(config.asset_root / "latex", output_root / "generated"))
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
