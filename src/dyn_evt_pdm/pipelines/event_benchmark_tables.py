"""Compact and full event-level benchmark tables, plus the fairness audit.

The compact main-paper table and the complete supplementary table are both derived from
one artifact, so the main paper cannot show a selection the supplement contradicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dyn_evt_pdm.pipelines.event_method_specs import SPECS_BY_NAME

#: Categories the compact main table must represent, in presentation order.
COMPACT_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("best_simple_baseline", "Best simple baseline"),
    ("best_classical_evt_baseline", "Best classical EVT baseline"),
    ("best_anomaly_baseline", "Best anomaly baseline"),
    ("best_learned_model", "Best learned model"),
    ("registered", "Registered recurrence score"),
    ("failure_prototype_region", "Failure-prototype region"),
    ("rare_state_region", "Rare-state region"),
)

#: Which methods belong to each selectable category.
CATEGORY_MEMBERS: dict[str, tuple[str, ...]] = {
    "best_simple_baseline": (
        "engineering_threshold",
        "best_individual_sensor_threshold",
        "global_empirical_threshold",
        "regime_conditioned_empirical_threshold",
    ),
    "best_classical_evt_baseline": (
        "classical_pot_gpd",
        "fixed_run_declustering",
        "ferro_segers_event_policy",
        "k_gaps_event_policy",
        "spot",
    ),
    "best_anomaly_baseline": (
        "isolation_forest",
        "robust_online_changepoint",
        "conformal_anomaly",
        "empirical_horizon_risk",
    ),
    "best_learned_model": (
        "linear_autoencoder",
        "compact_nonlinear_autoencoder",
    ),
    "registered": ("dynamical_evt_robust_score",),
    "failure_prototype_region": ("failure_prototype_region",),
    "rare_state_region": ("rare_state_region",),
}

#: Elements of the protocol that must be identical across every method for the
#: comparison to be fair. Event policy is deliberately excluded: several methods exist
#: precisely to vary it, and the audit records that variation rather than forbidding it.
SHARED_PROTOCOL_COLUMNS: tuple[str, ...] = (
    "configuration_hash",
    "target_events",
    "leakage_control",
)


@dataclass(frozen=True, slots=True)
class FairnessFinding:
    """One checked fairness property and its outcome."""

    check: str
    status: str
    detail: str


def _best_row(frame: pd.DataFrame, members: tuple[str, ...]) -> pd.Series | None:
    """Return the member with the highest event F1, breaking ties on alarm burden.

    Selecting on F1 alone would let a method with a marginally better score win despite
    a far worse alarm burden, so burden is the declared tie-break.
    """

    subset = frame[frame["method"].isin(members)]
    if subset.empty:
        return None
    # An always-on detector scores a perfect episode-level precision and recall without
    # detecting anything, so it must not be able to win a category. The row still
    # appears in the full supplementary table, flagged.
    if "degenerate_always_on" in subset.columns:
        usable = subset[~subset["degenerate_always_on"].astype(bool)]
        if not usable.empty:
            subset = usable
    ordered = subset.sort_values(
        by=["event_f1", "false_alarm_events_per_day"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def build_compact_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one representative method per category, per dataset and failure."""

    records: list[dict[str, object]] = []
    for (dataset, failure), group in frame.groupby(["dataset_id", "failure_id"], sort=True):
        for category, label in COMPACT_CATEGORIES:
            row = _best_row(group, CATEGORY_MEMBERS[category])
            if row is None:
                continue
            records.append(
                {
                    "dataset_id": dataset,
                    "failure_id": failure,
                    "category": label,
                    "method": row["method"],
                    "detected": bool(row.get("detected", row["event_recall"] > 0.0)),
                    "event_recall": float(row["event_recall"]),
                    "event_precision": float(row["event_precision"]),
                    "false_alarm_events_per_day": float(row["false_alarm_events_per_day"]),
                    "median_warning_lead_time": row["median_warning_lead_time"],
                    "duplicate_alarm_events": int(row["duplicate_alarm_events"]),
                    "event_utility": float(row.get("event_utility", float("nan"))),
                }
            )
    return pd.DataFrame.from_records(records)


def build_full_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the complete supplementary benchmark with every declared column."""

    columns = [
        "dataset_id",
        "failure_id",
        "method",
        "method_family",
        "score_kind",
        "threshold_rule",
        "event_policy",
        "detected",
        "event_recall",
        "event_precision",
        "event_f1",
        "false_alarm_events_per_day",
        "median_warning_lead_time",
        "time_under_warning",
        "duplicate_alarm_events",
        "event_utility",
        "runtime_seconds",
        "peak_memory_bytes",
        "parameter_count",
        "calibration_status",
        "tuning_partition",
        "configuration_hash",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].sort_values(
        by=["dataset_id", "method_family", "method"], kind="mergesort"
    )


def build_fairness_audit(frame: pd.DataFrame) -> list[FairnessFinding]:
    """Check the properties that must hold for the benchmark to be a fair comparison."""

    findings: list[FairnessFinding] = []

    for column in SHARED_PROTOCOL_COLUMNS:
        if column not in frame.columns:
            findings.append(FairnessFinding(column, "not_checkable", f"column {column} is absent"))
            continue
        per_dataset = frame.groupby("dataset_id")[column].nunique()
        offenders = per_dataset[per_dataset > 1]
        if offenders.empty:
            findings.append(
                FairnessFinding(column, "pass", f"one {column} per dataset across all methods")
            )
        else:
            findings.append(
                FairnessFinding(
                    column,
                    "fail",
                    f"{column} varies within {', '.join(offenders.index.astype(str))}",
                )
            )

    if "tuning_partition" in frame.columns:
        leaked = sorted(
            {
                str(value)
                for value in frame["tuning_partition"].dropna().unique()
                if "test" in str(value).lower()
            }
        )
        findings.append(
            FairnessFinding(
                "no_test_partition_tuning",
                "pass" if not leaked else "fail",
                "no method declares test-partition tuning"
                if not leaked
                else f"test-partition tuning declared by: {', '.join(leaked)}",
            )
        )

    # Method identity: two methods reporting identical results is not a fairness failure
    # by itself, but it must be visible rather than presented as independent evidence.
    metric_columns = [
        column
        for column in (
            "predicted_alarm_events",
            "event_recall",
            "event_precision",
            "false_alarm_events_per_day",
            "duplicate_alarm_events",
        )
        if column in frame.columns
    ]
    duplicate_groups: list[str] = []
    if metric_columns:
        for dataset, group in frame.groupby("dataset_id"):
            signature = group[metric_columns].round(10).astype(str).agg("|".join, axis=1)
            for _key, methods in group.assign(_sig=signature.to_numpy()).groupby("_sig")["method"]:
                names = sorted(methods)
                if len(names) > 1:
                    duplicate_groups.append(f"{dataset}: {', '.join(names)}")
    findings.append(
        FairnessFinding(
            "method_result_distinctness",
            "pass" if not duplicate_groups else "reported",
            "every declared method produced a distinct result set"
            if not duplicate_groups
            else "methods with identical results (reported, not hidden): "
            + "; ".join(duplicate_groups),
        )
    )

    # Event conversion is part of the detector, not the protocol: several methods exist
    # precisely to vary declustering and merging. The audit therefore records which
    # conversions were used instead of requiring one, so a reader can see that the
    # comparison holds matching, horizon and exposure fixed while the methods differ.
    if "event_policy" in frame.columns:
        policies = sorted({str(value) for value in frame["event_policy"].dropna().unique()})
        gaps = (
            sorted({int(value) for value in frame["merge_gap_used"].dropna().unique()})
            if "merge_gap_used" in frame.columns
            else []
        )
        findings.append(
            FairnessFinding(
                "event_conversion_declared",
                "reported",
                f"event policies in use: {', '.join(policies)}"
                + (f"; merge gaps in use: {', '.join(str(gap) for gap in gaps)}" if gaps else ""),
            )
        )

    declared = {spec.name for spec in SPECS_BY_NAME.values()}
    present = set(frame["method"].unique()) if "method" in frame.columns else set()
    undeclared = sorted(present - declared)
    findings.append(
        FairnessFinding(
            "declared_specifications",
            "pass" if not undeclared else "fail",
            "every reported method has a declared specification"
            if not undeclared
            else f"undeclared methods present: {', '.join(undeclared)}",
        )
    )
    return findings


def fairness_audit_frame(findings: list[FairnessFinding]) -> pd.DataFrame:
    """Return the fairness audit as a table."""

    return pd.DataFrame(
        [
            {"check": finding.check, "status": finding.status, "detail": finding.detail}
            for finding in findings
        ]
    )


def write_benchmark_tables(frame: pd.DataFrame, output_root: Path) -> list[Path]:
    """Write the compact table, the full supplementary table and the fairness audit."""

    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    compact_path = tables / "event_benchmark_compact.csv"
    build_compact_benchmark(frame).to_csv(compact_path, index=False)
    written.append(compact_path)

    full_path = tables / "event_benchmark_full.csv"
    build_full_benchmark(frame).to_csv(full_path, index=False)
    written.append(full_path)

    audit_path = tables / "event_benchmark_fairness_audit.csv"
    fairness_audit_frame(build_fairness_audit(frame)).to_csv(audit_path, index=False)
    written.append(audit_path)
    return written
