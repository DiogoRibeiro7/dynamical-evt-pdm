"""Manuscript source and build checks for the LaTeX paper."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dyn_evt_pdm.paper.claims import verify_paper_assets


@dataclass(frozen=True, slots=True)
class PaperCheckReport:
    """Machine-readable paper quality-check result."""

    paper_root: str
    asset_root: str
    checked_sources: int
    checked_asset_references: int
    checked_citations: int
    checked_claim_references: int
    checked_claim_ledger_entries: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return true when no paper-check failures were found."""

        return not self.failures


REQUIRED_SOURCE_FILES = (
    "main.tex",
    "pdf_reproducibility.tex",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_method.tex",
    "sections/04_simulation.tex",
    "sections/05_datasets.tex",
    "sections/06_experimental_protocol.tex",
    "sections/07_results.tex",
    "sections/08_discussion.tex",
    "sections/09_conclusion.tex",
    "supplement/supplement.tex",
    "supplement/extended_methods.tex",
    "supplement/additional_simulations.tex",
    "supplement/additional_real_data_results.tex",
    "supplement/robustness.tex",
    "supplement/reproducibility.tex",
    "generated/result_macros.tex",
    "references.bib",
    "latexmkrc",
    "Makefile",
)
PLACEHOLDER_PATTERN = re.compile(r"\b(TODO|TBD|XX)\b|\bdummy\b", re.IGNORECASE)
CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
BIB_KEY_PATTERN = re.compile(r"@\w+\{([^,\s]+)")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
CLAIM_PATTERN = re.compile(r"\bCLM-\d{3}\b")
LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
REAL_DATA_MATRIX_PREFIX = "../artifacts/real_data_matrix/"
UNSUPPORTED_CLAIM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bvalidated across (?:the )?five datasets\b", re.IGNORECASE),
        "validated-across-five-datasets overclaim",
    ),
    (
        re.compile(r"\bfive[- ]datasets?\b", re.IGNORECASE),
        "dataset-count generalisation",
    ),
    (
        re.compile(r"\bgeneral industrial applicability\b", re.IGNORECASE),
        "general-industrial-applicability overclaim",
    ),
    (
        re.compile(r"\bpopulation-level superiority\b", re.IGNORECASE),
        "population-level-superiority overclaim",
    ),
    (
        re.compile(r"\bevent-level replication on Hydraulic Systems or SECOM\b", re.IGNORECASE),
        "event-level replication overclaim for Hydraulic Systems or SECOM",
    ),
    (
        re.compile(r"\bcommon estimand across all datasets\b", re.IGNORECASE),
        "common-estimand overclaim",
    ),
    (
        re.compile(r"\basset bundle\b", re.IGNORECASE),
        "repository-status result in main narrative",
    ),
    (
        re.compile(r"\bLead-time versus false-alarm frontier\b", re.IGNORECASE),
        "zero-recall frontier framing",
    ),
    (
        re.compile(r"\balarm frontier exposes\b", re.IGNORECASE),
        "frontier framing for alarm grid",
    ),
    (
        re.compile(r"\bbroadly validated\b", re.IGNORECASE),
        "broad-validation overclaim",
    ),
    (
        re.compile(r"\bmulti-dataset validation\b", re.IGNORECASE),
        "dataset-count validation overclaim",
    ),
    (
        re.compile(r"\bgeneralizable industrial use\b", re.IGNORECASE),
        "generalizable-industrial-use overclaim",
    ),
    (
        re.compile(
            r"\bclustered extreme scores can be converted into useful maintenance alarms\b",
            re.IGNORECASE,
        ),
        "unsupported literature assumption",
    ),
    (
        re.compile(r"\b(?:pointwise smoke|smoke-runner|smoke runner)\b", re.IGNORECASE),
        "synthetic smoke audit in main manuscript",
    ),
)


def check_paper_sources(
    paper_root: Path = Path("paper"),
    asset_root: Path = Path("reports/paper"),
    *,
    require_pdfs: bool = True,
    write_reports: bool = True,
) -> PaperCheckReport:
    """Check manuscript sources, generated asset references, citations, and claims."""

    failures: list[str] = []
    source_files = _required_sources(paper_root, failures)
    source_text = {path: path.read_text(encoding="utf-8") for path in source_files}
    auditable_text = _source_text_with_referenced_inputs(source_text, paper_root)
    _check_placeholders(auditable_text, failures)
    _check_unsupported_claim_language(auditable_text, failures)
    _check_duplicate_labels(auditable_text, failures)

    cited_keys, bib_keys = _check_citations(source_text, paper_root / "references.bib", failures)
    asset_references = _check_asset_references(source_text, paper_root, asset_root, failures)
    claim_references = _check_claim_references(source_text, asset_root, failures)

    asset_verification = verify_paper_assets(asset_root)
    if not asset_verification.ok:
        failures.extend(f"asset verification: {failure}" for failure in asset_verification.failures)

    if require_pdfs:
        for pdf_path in (paper_root / "main.pdf", paper_root / "supplement" / "supplement.pdf"):
            if not pdf_path.exists():
                failures.append(f"missing compiled PDF: {pdf_path}")
            else:
                _check_float_after_references(pdf_path, failures)

    _check_latex_logs(paper_root, failures)
    report = PaperCheckReport(
        paper_root=str(paper_root),
        asset_root=str(asset_root),
        checked_sources=len(source_files),
        checked_asset_references=len(asset_references),
        checked_citations=len(cited_keys),
        checked_claim_references=len(claim_references),
        checked_claim_ledger_entries=asset_verification.checked_claims,
        failures=tuple(dict.fromkeys(failures)),
    )
    if write_reports:
        _write_audit_reports(
            paper_root,
            report=report,
            cited_keys=tuple(sorted(cited_keys)),
            bib_keys=tuple(sorted(bib_keys)),
            claim_references=tuple(sorted(claim_references)),
            claim_ledger_entries=asset_verification.checked_claims,
            asset_references=tuple(sorted(asset_references)),
        )
    return report


def _required_sources(paper_root: Path, failures: list[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative in REQUIRED_SOURCE_FILES:
        path = paper_root / relative
        if not path.exists():
            failures.append(f"missing required paper source: {path}")
            continue
        paths.append(path)
    return tuple(paths)


def _check_placeholders(source_text: dict[Path, str], failures: list[str]) -> None:
    for path, text in source_text.items():
        for match in PLACEHOLDER_PATTERN.finditer(text):
            failures.append(f"placeholder marker {match.group(0)!r} in {path}")


def _check_unsupported_claim_language(source_text: dict[Path, str], failures: list[str]) -> None:
    for path, text in source_text.items():
        for pattern, reason in UNSUPPORTED_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                failures.append(
                    f"unsupported manuscript claim language ({reason}) in {path}: "
                    f"{match.group(0)!r}"
                )


def _check_duplicate_labels(source_text: dict[Path, str], failures: list[str]) -> None:
    locations: dict[str, list[Path]] = {}
    for path, text in source_text.items():
        for label in LABEL_PATTERN.findall(text):
            locations.setdefault(label, []).append(path)
    for label, paths in sorted(locations.items()):
        if len(paths) > 1:
            failures.append(
                f"duplicate LaTeX label {label!r} in "
                + ", ".join(str(path) for path in sorted(set(paths)))
            )


def _source_text_with_referenced_inputs(
    source_text: dict[Path, str],
    paper_root: Path,
) -> dict[Path, str]:
    auditable = dict(source_text)
    for text in tuple(source_text.values()):
        for raw_reference in INPUT_PATTERN.findall(text):
            resolved = (paper_root / raw_reference).resolve()
            if resolved in auditable or not resolved.exists() or resolved.suffix.lower() != ".tex":
                continue
            auditable[resolved] = resolved.read_text(encoding="utf-8")
    return auditable


def _check_citations(
    source_text: dict[Path, str],
    bib_path: Path,
    failures: list[str],
) -> tuple[set[str], set[str]]:
    cited_keys: set[str] = set()
    for text in source_text.values():
        for group in CITE_PATTERN.findall(text):
            cited_keys.update(key.strip() for key in group.split(",") if key.strip())

    if not bib_path.exists():
        failures.append(f"missing bibliography: {bib_path}")
        return cited_keys, set()
    bib_text = bib_path.read_text(encoding="utf-8")
    bib_keys = set(BIB_KEY_PATTERN.findall(bib_text))
    duplicate_keys = sorted(
        key for key in bib_keys if BIB_KEY_PATTERN.findall(bib_text).count(key) > 1
    )
    for key in duplicate_keys:
        failures.append(f"duplicated bibliography key: {key}")
    for key in sorted(cited_keys.difference(bib_keys)):
        failures.append(f"citation key has no bibliography entry: {key}")
    for key in sorted(bib_keys.difference(cited_keys)):
        failures.append(f"bibliography entry is not cited: {key}")
    if not cited_keys:
        failures.append("no citations found in manuscript sources")
    return cited_keys, bib_keys


def _check_asset_references(
    source_text: dict[Path, str],
    paper_root: Path,
    asset_root: Path,
    failures: list[str],
) -> set[str]:
    provenance_path = asset_root / "asset_provenance.json"
    provenance_paths = _provenance_paths(provenance_path, failures)
    references: set[str] = set()
    for text in source_text.values():
        for raw_reference in (*GRAPHICS_PATTERN.findall(text), *INPUT_PATTERN.findall(text)):
            if raw_reference.startswith("../reports/paper/"):
                resolved = (paper_root / raw_reference).resolve()
                references.add(str(resolved))
                if not resolved.exists():
                    failures.append(f"missing generated asset reference: {raw_reference}")
                if str(resolved) not in provenance_paths:
                    failures.append(f"generated reference lacks provenance: {raw_reference}")
            elif raw_reference.startswith(REAL_DATA_MATRIX_PREFIX):
                resolved = (paper_root / raw_reference).resolve()
                references.add(str(resolved))
                if not resolved.exists():
                    failures.append(f"missing real-data matrix artifact reference: {raw_reference}")
    if not references:
        failures.append("no generated paper asset references found")
    return references


def _check_claim_references(
    source_text: dict[Path, str],
    asset_root: Path,
    failures: list[str],
) -> set[str]:
    referenced = set()
    for text in source_text.values():
        referenced.update(CLAIM_PATTERN.findall(text))
    claim_path = asset_root / "claim_ledger.json"
    if not claim_path.exists():
        failures.append(f"missing claim ledger: {claim_path}")
        return referenced
    claims = _read_json_list(claim_path, failures)
    known = {str(claim.get("claim_id", "")) for claim in claims}
    for claim_id in sorted(referenced.difference(known)):
        failures.append(f"manuscript references unknown claim: {claim_id}")
    return referenced


def _check_latex_logs(paper_root: Path, failures: list[str]) -> None:
    patterns = (
        "undefined references",
        "undefined citation",
        "citation `",
        "reference `",
        "LaTeX Error",
    )
    for log_path in (paper_root / "main.log", paper_root / "supplement" / "supplement.log"):
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        for pattern in patterns:
            if pattern.lower() in lower:
                failures.append(
                    f"LaTeX log contains unresolved issue marker {pattern!r}: {log_path}"
                )
                break


def _check_float_after_references(pdf_path: Path, failures: list[str]) -> None:
    try:
        text = subprocess.check_output(
            ["pdftotext", str(pdf_path), "-"],
            encoding="utf-8",
            errors="ignore",
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    reference_match = re.search(r"(?m)^References\s*$", text)
    if reference_match is None:
        return
    after_references = text[reference_match.end() :]
    float_match = re.search(r"\b(?:Figure|Table)\s+\d+", after_references)
    if float_match is not None:
        failures.append(
            f"compiled PDF has float caption after References in {pdf_path}: "
            f"{float_match.group(0)!r}"
        )


def _provenance_paths(path: Path, failures: list[str]) -> set[str]:
    if not path.exists():
        failures.append(f"missing provenance file: {path}")
        return set()
    records = _read_json_list(path, failures)
    return {str(Path(str(record.get("path", ""))).resolve()) for record in records}


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
            failures.append(f"non-object JSON entry in {path}")
    return result


def _write_audit_reports(
    paper_root: Path,
    *,
    report: PaperCheckReport,
    cited_keys: tuple[str, ...],
    bib_keys: tuple[str, ...],
    claim_references: tuple[str, ...],
    claim_ledger_entries: int,
    asset_references: tuple[str, ...],
) -> None:
    generated = paper_root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "paper_check_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    (generated / "citation_audit.json").write_text(
        json.dumps(
            {
                "cited_keys": cited_keys,
                "bibliography_keys": bib_keys,
                "uncited_keys": sorted(set(bib_keys).difference(cited_keys)),
                "missing_keys": sorted(set(cited_keys).difference(bib_keys)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (generated / "claim_audit.json").write_text(
        json.dumps(
            {
                "visible_claim_references": claim_references,
                "claim_references": claim_references,
                "claim_ledger_entries": claim_ledger_entries,
                "asset_references": asset_references,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
