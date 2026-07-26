import json
import re
from pathlib import Path

from dyn_evt_pdm.paper.claims import sha256_file
from dyn_evt_pdm.paper.manuscript import check_paper_sources


def test_tracked_bibliography_has_verified_scania_article_metadata() -> None:
    bib_path = Path(__file__).parents[1] / "paper" / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8")
    match = re.search(r"@article\{Kharazian2025Scania,\s*(.*?)\n\}", bib_text, flags=re.DOTALL)

    assert match is not None
    entry = match.group(1)
    assert "volume = {12}" in entry
    assert "number = {493}" in entry
    assert "doi = {10.1038/s41597-025-04802-6}" in entry


def test_check_paper_sources_accepts_provenance_backed_manuscript(tmp_path: Path) -> None:
    paper_root, asset_root = _write_minimal_paper_tree(tmp_path)

    report = check_paper_sources(
        paper_root=paper_root,
        asset_root=asset_root,
        require_pdfs=False,
        write_reports=False,
    )

    assert report.ok, report.failures
    assert report.checked_citations == 1
    assert report.checked_claim_references == 1
    assert report.checked_asset_references == 2


def test_check_paper_sources_rejects_unknown_claim(tmp_path: Path) -> None:
    paper_root, asset_root = _write_minimal_paper_tree(tmp_path)
    section = paper_root / "sections" / "01_introduction.tex"
    section.write_text(section.read_text(encoding="utf-8") + " Unknown CLM-999.", encoding="utf-8")

    report = check_paper_sources(
        paper_root=paper_root,
        asset_root=asset_root,
        require_pdfs=False,
        write_reports=False,
    )

    assert not report.ok
    assert any("unknown claim" in failure for failure in report.failures)


def test_check_paper_sources_rejects_dataset_count_overclaims(tmp_path: Path) -> None:
    paper_root, asset_root = _write_minimal_paper_tree(tmp_path)
    section = paper_root / "sections" / "01_introduction.tex"
    section.write_text(
        section.read_text(encoding="utf-8")
        + " The method is validated across five datasets with general industrial applicability.",
        encoding="utf-8",
    )

    report = check_paper_sources(
        paper_root=paper_root,
        asset_root=asset_root,
        require_pdfs=False,
        write_reports=False,
    )

    assert not report.ok
    assert any("dataset-count generalisation" in failure for failure in report.failures)
    assert any("general-industrial-applicability" in failure for failure in report.failures)


def test_check_paper_sources_audits_real_data_matrix_references(tmp_path: Path) -> None:
    paper_root, asset_root = _write_minimal_paper_tree(tmp_path)
    section = paper_root / "sections" / "05_datasets.tex"
    section.write_text(
        "\\input{../artifacts/real_data_matrix/evidence_scope.tex}\n",
        encoding="utf-8",
    )

    missing_report = check_paper_sources(
        paper_root=paper_root,
        asset_root=asset_root,
        require_pdfs=False,
        write_reports=False,
    )

    assert not missing_report.ok
    assert any(
        "missing real-data matrix artifact reference" in item for item in missing_report.failures
    )

    matrix_table = tmp_path / "artifacts" / "real_data_matrix" / "evidence_scope.tex"
    matrix_table.parent.mkdir(parents=True)
    matrix_table.write_text("\\begin{tabular}{l}ok\\\\\\end{tabular}\n", encoding="utf-8")

    present_report = check_paper_sources(
        paper_root=paper_root,
        asset_root=asset_root,
        require_pdfs=False,
        write_reports=False,
    )

    assert present_report.ok, present_report.failures
    assert present_report.checked_asset_references == 3


def _write_minimal_paper_tree(tmp_path: Path) -> tuple[Path, Path]:
    paper_root = tmp_path / "paper"
    asset_root = tmp_path / "reports" / "paper"
    for directory in (
        paper_root / "sections",
        paper_root / "supplement",
        paper_root / "generated",
        asset_root / "figures",
        asset_root / "latex",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    figure = asset_root / "figures" / "figure.png"
    table = asset_root / "latex" / "table.tex"
    figure.write_bytes(b"not a real png but present")
    table.write_text("\\begin{table}\\caption{Generated}\\end{table}\n", encoding="utf-8")

    claims = [
        {
            "claim_id": "CLM-001",
            "final_status": "supported",
            "experiment_ids": "EXP-1",
            "manuscript_sections_allowed": "Results",
        }
    ]
    (asset_root / "claim_ledger.json").write_text(json.dumps(claims), encoding="utf-8")
    provenance = [
        {
            "path": str(figure.resolve()),
            "sha256": sha256_file(figure),
            "experiment_id": "EXP-1",
        },
        {
            "path": str(table.resolve()),
            "sha256": sha256_file(table),
            "experiment_id": "EXP-1",
        },
    ]
    (asset_root / "asset_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    manifest = {
        "experiment_id": "EXP-1",
        "known_experiment_ids": ["EXP-1"],
        "generated_files": [
            str(figure.resolve()),
            str(table.resolve()),
            str((asset_root / "asset_provenance.json").resolve()),
            str((asset_root / "asset_manifest.json").resolve()),
        ],
    }
    (asset_root / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (paper_root / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}"
        "\\input{sections/01_introduction}"
        "\\bibliography{references}\\end{document}\n",
        encoding="utf-8",
    )
    (paper_root / "sections" / "01_introduction.tex").write_text(
        "See CLM-001 and \\cite{Known}. "
        "\\includegraphics{../reports/paper/figures/figure.png}"
        "\\input{../reports/paper/latex/table.tex}\n",
        encoding="utf-8",
    )
    for relative in (
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
        "latexmkrc",
        "Makefile",
    ):
        (paper_root / relative).write_text("% source\n", encoding="utf-8")
    (paper_root / "references.bib").write_text(
        "@article{Known, author={A. Author}, title={Known title}, journal={Journal}, year={2024}}\n",
        encoding="utf-8",
    )
    return paper_root, asset_root
