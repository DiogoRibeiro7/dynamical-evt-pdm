from pathlib import Path


def test_adversarial_review_contains_required_decision_sections() -> None:
    text = Path("docs/adversarial_review.md").read_text(encoding="utf-8")

    required_sections = [
        "## Fatal Flaws",
        "## Major Revisions",
        "## Minor Revisions",
        "## Missing Experiments",
        "## Claims That Must Be Weakened",
        "## Decision-Changing Tests Or Analyses",
        "## Final Recommendation",
    ]

    for section in required_sections:
        assert section in text
    assert "Recommendation: submission ready." in text
    assert "raw 1 Hz rows are computational scale, not independent evidence" in text
