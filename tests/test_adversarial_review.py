from pathlib import Path


def test_adversarial_review_contains_required_decision_sections() -> None:
    text = Path("docs/adversarial_review.md").read_text(encoding="utf-8")

    required_sections = [
        "## Fatal flaws",
        "## Major revisions",
        "## Minor revisions",
        "## Missing experiments",
        "## Claims that must be weakened",
        "## Decision-changing tests or analyses",
        "## Final recommendation",
    ]

    for section in required_sections:
        assert section in text
    assert "Recommendation: major revision before submission." in text
    assert "Counting 1 Hz rows as independent evidence" in text
