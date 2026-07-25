import json
from pathlib import Path

from dyn_evt_pdm.paper.claims import sha256_file
from dyn_evt_pdm.paper.submission import SubmissionPackageConfig, build_submission_package


def test_build_submission_package_writes_review_matrix_and_decision(tmp_path: Path) -> None:
    paper_root, asset_root = _write_package_fixture(tmp_path)
    output_root = tmp_path / "submission"
    protocol_path = tmp_path / "protocol.yaml"
    lock_path = tmp_path / "poetry.lock"
    license_path = tmp_path / "LICENSE"
    readme_path = tmp_path / "README.md"
    protocol_path.write_text("protocol_id: fixture\n", encoding="utf-8")
    lock_path.write_text("# lock\n", encoding="utf-8")
    license_path.write_text("BSD-3-Clause\n", encoding="utf-8")
    readme_path.write_text("# fixture\n", encoding="utf-8")

    manifest = build_submission_package(
        SubmissionPackageConfig(
            paper_root=paper_root,
            asset_root=asset_root,
            output_root=output_root,
            protocol_path=protocol_path,
            lock_path=lock_path,
            license_path=license_path,
            readme_path=readme_path,
        )
    )

    assert manifest.decision == "not submission ready"
    assert manifest.unresolved_blockers
    assert (output_root / "reviewer_report.md").exists()
    assert (output_root / "revision_matrix.csv").exists()
    assert (output_root / "submission_statements.md").exists()
    assert (output_root / "manuscript" / "main.pdf").exists()
    assert not (output_root / "data").exists()
    payload = json.loads((output_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert payload["decision"] == manifest.decision
    assert any(item["path"] == "reviewer_report.md" for item in payload["files"])


def _write_package_fixture(tmp_path: Path) -> tuple[Path, Path]:
    paper_root = tmp_path / "paper"
    asset_root = tmp_path / "reports" / "paper"
    for directory in (
        paper_root / "sections",
        paper_root / "supplement",
        paper_root / "generated",
        asset_root / "figures",
        asset_root / "latex",
        asset_root / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    figure = asset_root / "figures" / "figure.png"
    table = asset_root / "latex" / "table.tex"
    csv_table = asset_root / "tables" / "table.csv"
    figure.write_bytes(b"figure")
    table.write_text("\\begin{table}\\caption{Generated}\\end{table}\n", encoding="utf-8")
    csv_table.write_text("a\n1\n", encoding="utf-8")
    claims = [
        {
            "claim_id": "CLM-001",
            "final_status": "supported",
            "experiment_ids": "EXP-1",
            "manuscript_sections_allowed": "Results",
        }
    ]
    (asset_root / "claim_ledger.json").write_text(json.dumps(claims), encoding="utf-8")
    (asset_root / "claim_ledger.md").write_text("# Claims\n", encoding="utf-8")
    provenance = [
        {"path": str(figure.resolve()), "sha256": sha256_file(figure), "experiment_id": "EXP-1"},
        {"path": str(table.resolve()), "sha256": sha256_file(table), "experiment_id": "EXP-1"},
        {
            "path": str(csv_table.resolve()),
            "sha256": sha256_file(csv_table),
            "experiment_id": "EXP-1",
        },
        {
            "path": str(asset_root / "claim_ledger.json"),
            "sha256": sha256_file(asset_root / "claim_ledger.json"),
            "experiment_id": "EXP-1",
        },
        {
            "path": str(asset_root / "claim_ledger.md"),
            "sha256": sha256_file(asset_root / "claim_ledger.md"),
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
            str(csv_table.resolve()),
            str((asset_root / "claim_ledger.json").resolve()),
            str((asset_root / "claim_ledger.md").resolve()),
            str((asset_root / "asset_provenance.json").resolve()),
            str((asset_root / "asset_manifest.json").resolve()),
        ],
    }
    (asset_root / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (paper_root / "main.pdf").write_bytes(b"pdf")
    (paper_root / "supplement" / "supplement.pdf").write_bytes(b"pdf")
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
