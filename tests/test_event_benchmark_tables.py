"""Tests for the benchmark tables, the fairness audit and the degeneracy guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.pipelines.event_benchmark_tables import (
    CATEGORY_MEMBERS,
    build_compact_benchmark,
    build_fairness_audit,
    build_full_benchmark,
    build_supplementary_detection_table,
    build_supplementary_provenance_table,
    fairness_audit_frame,
    write_benchmark_tables,
)
from dyn_evt_pdm.pipelines.event_method_impl import best_single_sensor_index


def _row(method: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": "metropt",
        "failure_id": "metropt_test_failure_001",
        "method": method,
        "method_family": "baseline",
        "score_kind": "robust_max_abs_z",
        "threshold_rule": "train_quantile",
        "event_policy": "merge_gap",
        "target_events": 1,
        "predicted_alarm_events": 100,
        "detected": True,
        "event_recall": 1.0,
        "event_precision": 0.01,
        "event_f1": 0.02,
        "false_alarm_events_per_day": 50.0,
        "median_warning_lead_time": 100.0,
        "time_under_warning": 500,
        "duplicate_alarm_events": 2,
        "event_utility": -10.0,
        "runtime_seconds": 0.1,
        "peak_memory_bytes": 1000,
        "parameter_count": 1,
        "calibration_status": "uncalibrated",
        "tuning_partition": "train",
        "configuration_hash": "abc123",
        "leakage_control": "train thresholds and references only",
        "alarm_coverage_fraction": 0.05,
        "degenerate_always_on": False,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_fairness_audit_passes_on_a_consistent_run() -> None:
    frame = _frame(
        [_row("engineering_threshold"), _row("global_empirical_threshold", event_f1=0.03)]
    )
    audit = {finding.check: finding.status for finding in build_fairness_audit(frame)}
    assert audit["configuration_hash"] == "pass"
    assert audit["target_events"] == "pass"
    assert audit["no_test_partition_tuning"] == "pass"
    assert audit["declared_specifications"] == "pass"


def test_fairness_audit_flags_mixed_configuration_hashes() -> None:
    """Rows assembled from different runs are not a fair comparison."""

    frame = _frame(
        [
            _row("engineering_threshold", configuration_hash="aaa"),
            _row("global_empirical_threshold", configuration_hash="bbb"),
        ]
    )
    audit = {finding.check: finding.status for finding in build_fairness_audit(frame)}
    assert audit["configuration_hash"] == "fail"


def test_fairness_audit_flags_test_partition_tuning() -> None:
    frame = _frame([_row("engineering_threshold", tuning_partition="test")])
    audit = {finding.check: finding.status for finding in build_fairness_audit(frame)}
    assert audit["no_test_partition_tuning"] == "fail"


def test_fairness_audit_reports_identical_methods_rather_than_hiding_them() -> None:
    """Regression: nine baselines once reported identical results with no disclosure."""

    frame = _frame([_row("engineering_threshold"), _row("global_empirical_threshold")])
    findings = {finding.check: finding for finding in build_fairness_audit(frame)}
    distinctness = findings["method_result_distinctness"]
    assert distinctness.status == "reported"
    assert "engineering_threshold" in distinctness.detail
    assert "global_empirical_threshold" in distinctness.detail


def test_fairness_audit_passes_distinctness_when_results_differ() -> None:
    frame = _frame(
        [
            _row("engineering_threshold", predicted_alarm_events=10),
            _row("global_empirical_threshold", predicted_alarm_events=20),
        ]
    )
    findings = {finding.check: finding for finding in build_fairness_audit(frame)}
    assert findings["method_result_distinctness"].status == "pass"


def test_fairness_audit_flags_undeclared_methods() -> None:
    frame = _frame([_row("a_method_nobody_declared")])
    audit = {finding.check: finding.status for finding in build_fairness_audit(frame)}
    assert audit["declared_specifications"] == "fail"


def test_compact_table_covers_every_category_present() -> None:
    rows = [_row(method) for members in CATEGORY_MEMBERS.values() for method in members[:1]]
    for row in rows:
        if row["method"] == "dynamical_evt_robust_score":
            row["method_family"] = "registered"
    compact = build_compact_benchmark(_frame(rows))
    assert len(compact) == len(CATEGORY_MEMBERS)


def test_compact_table_never_selects_an_always_on_detector() -> None:
    """A permanently-on alarm scores perfectly and must not win its category."""

    frame = _frame(
        [
            _row(
                "engineering_threshold",
                event_f1=1.0,
                event_precision=1.0,
                false_alarm_events_per_day=0.0,
                alarm_coverage_fraction=0.99,
                degenerate_always_on=True,
            ),
            _row("global_empirical_threshold", event_f1=0.02),
        ]
    )
    compact = build_compact_benchmark(frame)
    simple = compact[compact["category"] == "Best simple baseline"].iloc[0]
    assert simple["method"] == "global_empirical_threshold"


def test_degenerate_row_still_appears_in_the_full_table() -> None:
    frame = _frame([_row("engineering_threshold", degenerate_always_on=True)])
    full = build_full_benchmark(frame)
    assert len(full) == 1


def test_compact_selection_breaks_ties_on_alarm_burden() -> None:
    frame = _frame(
        [
            _row("engineering_threshold", event_f1=0.5, false_alarm_events_per_day=90.0),
            _row("global_empirical_threshold", event_f1=0.5, false_alarm_events_per_day=10.0),
        ]
    )
    compact = build_compact_benchmark(frame)
    simple = compact[compact["category"] == "Best simple baseline"].iloc[0]
    assert simple["method"] == "global_empirical_threshold"


def test_full_benchmark_exposes_every_required_column() -> None:
    full = build_full_benchmark(_frame([_row("engineering_threshold")]))
    for column in (
        "detected",
        "event_utility",
        "runtime_seconds",
        "peak_memory_bytes",
        "parameter_count",
        "calibration_status",
        "tuning_partition",
        "configuration_hash",
        "score_kind",
        "threshold_rule",
        "event_policy",
    ):
        assert column in full.columns


def test_write_benchmark_tables_emits_every_artifact(tmp_path: Path) -> None:
    """Three CSVs plus the compact, detection, provenance and fairness LaTeX tables."""

    written = write_benchmark_tables(_frame([_row("engineering_threshold")]), tmp_path)
    names = {path.name for path in written}
    assert names == {
        "event_benchmark_compact.csv",
        "event_benchmark_compact.tex",
        "event_benchmark_full.csv",
        "event_benchmark_fairness_audit.csv",
        "event_benchmark_fairness_audit.tex",
        "event_benchmark_detection.tex",
        "event_benchmark_provenance.tex",
    }
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_supplementary_detection_table_has_a_row_per_method_and_dataset() -> None:
    """The phase requires every declared method to carry a supplementary row."""

    rows = [
        _row(method, dataset_id=dataset)
        for dataset in ("metropt", "metropt2")
        for method in ("engineering_threshold", "global_empirical_threshold", "spot")
    ]
    detection = build_supplementary_detection_table(_frame(rows))
    assert len(detection) == 6
    assert set(detection["Data"]) == {"MetroPT", "MetroPT2"}


def test_provenance_table_is_one_row_per_method() -> None:
    rows = [
        _row(method, dataset_id=dataset)
        for dataset in ("metropt", "metropt2")
        for method in ("engineering_threshold", "spot")
    ]
    provenance = build_supplementary_provenance_table(_frame(rows))
    assert len(provenance) == 2


def test_method_display_names_avoid_raw_identifiers() -> None:
    """Stripping underscores yields captions like "classical pot gpd"."""

    detection = build_supplementary_detection_table(_frame([_row("classical_pot_gpd")]))
    assert detection["Method"].iloc[0] == "Classical POT/GPD"


def test_audit_frame_is_tabular() -> None:
    frame = fairness_audit_frame(build_fairness_audit(_frame([_row("engineering_threshold")])))
    assert list(frame.columns) == ["check", "status", "detail"]
    assert len(frame) > 0


def test_near_constant_channel_is_not_selected_as_the_best_sensor() -> None:
    """A near-constant channel yields a zero threshold and an always-on detector."""

    rng = np.random.default_rng(0)
    rows = 2000
    informative = rng.normal(size=rows)
    near_constant = np.zeros(rows)
    labels = np.zeros(rows, dtype=bool)
    labels[-100:] = True
    informative[-100:] += 2.0

    # Fewer than 2% of rows are non-zero, so the 0.98 training quantile of this channel
    # lands on the constant and cannot discriminate -- which is the real failure mode
    # observed on MetroPT2, where the selected channel produced a threshold of exactly
    # zero. A channel with a larger non-zero share is genuinely usable and must not be
    # excluded, so the guard keys on quantile separation rather than on sparsity.
    near_constant[-20:] = 5.0
    matrix = np.column_stack([informative, near_constant])

    assert best_single_sensor_index(matrix, labels, threshold_quantile=0.98) == 0


def test_a_sparse_but_separating_channel_remains_eligible() -> None:
    """The guard must exclude only channels whose quantile cannot discriminate."""

    rng = np.random.default_rng(1)
    rows = 2000
    weak = rng.normal(scale=0.01, size=rows)
    sparse_but_usable = np.zeros(rows)
    sparse_but_usable[-100:] = 5.0  # 5% non-zero, so the 0.98 quantile is 5.0
    labels = np.zeros(rows, dtype=bool)
    labels[-100:] = True
    matrix = np.column_stack([weak, sparse_but_usable])

    assert best_single_sensor_index(matrix, labels, threshold_quantile=0.98) == 1


def test_sensor_selection_falls_back_when_no_channel_is_eligible() -> None:
    matrix = np.zeros((100, 3))
    index = best_single_sensor_index(matrix, np.zeros(100, dtype=bool))
    assert 0 <= index < 3


@pytest.mark.parametrize("category", list(CATEGORY_MEMBERS))
def test_every_category_has_members(category: str) -> None:
    assert CATEGORY_MEMBERS[category]
