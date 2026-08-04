"""Layout guards for generated LaTeX tables.

A ``table`` float that does not fit a page does not error: LaTeX warns and lets the
content run off the bottom, so the surplus rows vanish from the PDF while the source and
the CSV still contain them. Nothing in the build fails, and the defect is invisible
unless someone reads the built page. It has occurred three times in this project, most
recently in the supplementary benchmark, where the last rows of a 38-row table were
absent from the published PDF.

The first test reads the LaTeX logs, which state the overflow exactly. A row-count
heuristic cannot: height depends on how much each cell wraps, and a 19-row table with
text columns overran while a 21-row numeric one fit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "reports" / "paper" / "latex"
BUILD_LOGS = (
    ROOT / "paper" / "main.log",
    ROOT / "paper" / "supplement" / "supplement.log",
)


def _table_files() -> list[Path]:
    if not LATEX_DIR.exists():
        return []
    return sorted(LATEX_DIR.glob("*.tex"))


@pytest.mark.parametrize("log_path", BUILD_LOGS, ids=lambda p: p.name)
def test_no_float_overruns_its_page(log_path: Path) -> None:
    """The definitive check: LaTeX reports every float it could not fit."""
    if not log_path.exists():
        pytest.skip(f"{log_path.name} is not present; build the document first")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    overruns = [line.strip() for line in lines if "Float too large" in line]
    assert not overruns, (
        f"{log_path.name} reports floats that overrun their page and therefore drop "
        f"rows from the PDF: {overruns}"
    )


@pytest.mark.parametrize("log_path", BUILD_LOGS, ids=lambda p: p.name)
def test_no_unresolved_references(log_path: Path) -> None:
    """Cross-document table references must resolve to real supplementary numbers."""
    if not log_path.exists():
        pytest.skip(f"{log_path.name} is not present; build the document first")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    problems = [
        line.strip()
        for line in lines
        if "undefined on input line" in line or "multiply defined" in line
    ]
    assert not problems, f"{log_path.name} has unresolved or duplicate labels: {problems}"


@pytest.mark.parametrize("path", _table_files(), ids=lambda p: p.name)
def test_percent_signs_are_not_pre_escaped(path: Path) -> None:
    r"""A pre-escaped percent renders as a literal backslash.

    The table writer escapes LaTeX specials itself, so a ``\%`` reaching it becomes
    ``\textbackslash{}%`` in the PDF. This has produced visible defects twice.
    """
    text = path.read_text(encoding="utf-8")
    body = text.split("\\midrule", 1)[-1] if "\\midrule" in text else text
    assert "\\textbackslash" not in body, (
        f"{path.name} renders a literal backslash in its body; a percent or other "
        f"special was escaped before reaching the table writer"
    )


def test_at_least_one_table_was_checked() -> None:
    """Guard against the parametrised tests silently covering nothing."""
    if not LATEX_DIR.exists():
        pytest.skip("generated tables are not present")
    assert _table_files()
