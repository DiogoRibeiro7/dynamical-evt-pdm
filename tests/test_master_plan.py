from pathlib import Path


def test_master_plan_and_end_to_end_target_are_present() -> None:
    plan = Path("docs/implementation_plan.md").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for phrase in [
        "Hypothesis Mapping",
        "Workflow Compliance",
        "make end-to-end-smoke",
        "make real-data-matrix",
        "Submission Readiness",
        "Scientific Boundaries",
    ]:
        assert phrase in plan

    assert "end-to-end-smoke:" in makefile
    assert "real-data-matrix:" in makefile
    assert "industrial-results:" in makefile
    assert "analyse-dangerous-region" in makefile
    assert "analyse-multivariate-extremes" in makefile
    assert "run-baselines" in makefile
    assert "build-paper-assets" in makefile
