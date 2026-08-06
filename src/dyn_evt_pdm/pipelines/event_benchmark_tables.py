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


def _dataset_label(value: object) -> str:
    labels = {"metropt": "MetroPT", "metropt2": "MetroPT2"}
    return labels.get(str(value), str(value).replace("_", " "))


#: Readable method names. Stripping underscores from identifiers yields captions like
#: "classical pot gpd", which is not how these methods are named in the literature.
METHOD_DISPLAY_NAMES: dict[str, str] = {
    "engineering_threshold": "Engineering threshold",
    "best_individual_sensor_threshold": "Single-sensor threshold",
    "global_empirical_threshold": "Global empirical threshold",
    "regime_conditioned_empirical_threshold": "Regime-conditioned threshold",
    "classical_pot_gpd": "Classical POT/GPD",
    "fixed_run_declustering": "Fixed-run declustering",
    "ferro_segers_event_policy": "Ferro--Segers policy",
    "k_gaps_event_policy": "K-gaps policy",
    "spot": "SPOT",
    "isolation_forest": "Isolation forest",
    "robust_online_changepoint": "Robust change point",
    "linear_autoencoder": "Linear autoencoder",
    "compact_nonlinear_autoencoder": "Nonlinear autoencoder",
    "conformal_anomaly": "Conformal anomaly",
    "empirical_horizon_risk": "Empirical horizon risk",
    "dynamical_evt_robust_score": "Recurrence score",
    "failure_prototype_region": "Failure-prototype region",
    "rare_state_region": "Rare-state region",
    "negative_control_region": "Negative-control region",
}

#: Short category labels for the compact table, which has no room for the long form.
COMPACT_CATEGORY_SHORT: dict[str, str] = {
    "Best simple baseline": "Simple",
    "Best classical EVT baseline": "Classical EVT",
    "Best anomaly baseline": "Anomaly",
    "Best learned model": "Learned",
    "Registered recurrence score": "Registered",
    "Failure-prototype region": "Prototype region",
    "Rare-state region": "Rare-state region",
}


def _method_label(value: object) -> str:
    key = str(value)
    return METHOD_DISPLAY_NAMES.get(key, key.replace("_", " "))


def _fraction_percent(value: float | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if numeric != numeric:  # NaN
        return ""
    return f"{numeric * 100:.2f}"


def build_supplementary_detection_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-method detection and burden, for every declared method and dataset.

    Alarm burden is reported on both axes because episode counts and warning exposure
    can disagree in direction, and a permanently-on detector scores a perfect
    episode-level precision and recall while holding the system under warning.
    """

    ordered = frame.sort_values(by=["dataset_id", "method_family", "method"], kind="mergesort")
    return pd.DataFrame(
        {
            "Data": ordered["dataset_id"].map(_dataset_label),
            "Method": ordered["method"].map(_method_label),
            "Family": ordered["method_family"].map(_method_label),
            "Det.": ordered["detected"].map(lambda value: "yes" if bool(value) else "no"),
            "Recall": ordered["event_recall"].map(lambda value: f"{float(value):.3g}"),
            "Prec.": ordered["event_precision"].map(lambda value: f"{float(value):.3g}"),
            "FA/day": ordered["false_alarm_events_per_day"].map(
                lambda value: f"{float(value):.3g}"
            ),
            "Expo. %": ordered.get("alarm_coverage_fraction", pd.Series(dtype=float)).map(
                _fraction_percent
            )
            if "alarm_coverage_fraction" in ordered.columns
            else "",
            "Dup.": ordered["duplicate_alarm_events"],
            "Utility": ordered["event_utility"].map(lambda value: f"{float(value):.3g}")
            if "event_utility" in ordered.columns
            else "",
        }
    )


def build_supplementary_provenance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-method cost and provenance: the audit trail behind each benchmark row."""

    ordered = frame.sort_values(
        by=["dataset_id", "method_family", "method"], kind="mergesort"
    ).drop_duplicates(subset=["method"])
    columns: dict[str, object] = {
        "Method": ordered["method"].map(_method_label),
        "Score": ordered.get("score_kind", pd.Series(dtype=str)).map(_method_label),
        "Threshold rule": ordered.get("threshold_rule", pd.Series(dtype=str)).map(_method_label),
        "Event policy": ordered.get("event_policy", pd.Series(dtype=str)).map(_method_label),
        "Merge gap": ordered.get("merge_gap_used", pd.Series(dtype=float)),
        "Params": ordered.get("parameter_count", pd.Series(dtype=float)),
        "Calibration": ordered.get("calibration_status", pd.Series(dtype=str)),
        "Tuned on": ordered.get("tuning_partition", pd.Series(dtype=str)),
    }
    return pd.DataFrame(columns)


#: Column layout for the compact table. Explicit widths and a spanning dataset header
#: keep the category and method names on one line each; equal-width columns wrap them
#: over three or four lines and cost a full page.
_COMPACT_LAYOUT: tuple[tuple[str, str, str], ...] = (
    ("category", "Category", "l"),
    ("method", "Method", "l"),
    ("detected", "Det.", "c"),
    ("event_precision", "Precision", "r"),
    ("false_alarm_events_per_day", "FA/day", "r"),
    ("median_warning_lead_time", "Lead (samples)", "r"),
    ("event_utility", "Utility", "r"),
)


def _write_compact_latex_table(compact: pd.DataFrame, path: Path) -> None:
    """Write the compact benchmark, grouping method rows under each dataset."""

    spec = " ".join(alignment for _key, _header, alignment in _COMPACT_LAYOUT)
    n_columns = len(_COMPACT_LAYOUT)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        (
            "\\caption{Compact held-out event-level benchmark: the best method in each "
            "category per dataset, selected on event $F_1$ with alarm burden as the "
            "declared tie-break. A method flagged as effectively always on cannot win a "
            "category, because episode-level precision and recall are degenerate for a "
            "permanently-on detector. The complete benchmark over all nineteen methods, "
            "including any excluded on that ground, is in Supplementary Table~S2.}"
        ),
        "\\label{tab:event-benchmark-compact}",
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(header for _key, header, _alignment in _COMPACT_LAYOUT) + " \\\\",
    ]

    previous = ""
    for _index, row in compact.iterrows():
        dataset = _dataset_label(row["dataset_id"])
        if dataset != previous:
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{n_columns}}}{{l}}{{\\textit{{{dataset}}}}} \\\\")
            previous = dataset
        cells: list[str] = []
        for key, _header, _alignment in _COMPACT_LAYOUT:
            value = row[key]
            if key == "category":
                cells.append(COMPACT_CATEGORY_SHORT.get(str(value), str(value)))
            elif key == "method":
                cells.append(_method_label(value))
            elif key == "detected":
                cells.append("yes" if bool(value) else "no")
            elif key == "median_warning_lead_time":
                cells.append("" if pd.isna(value) else f"{int(float(value)):,}")
            else:
                cells.append(f"{float(value):.3g}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_tables(frame: pd.DataFrame, output_root: Path) -> list[Path]:
    """Write the compact table, both supplementary tables and the fairness audit."""

    tables = output_root / "tables"
    latex = output_root / "latex"
    tables.mkdir(parents=True, exist_ok=True)
    latex.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Deferred import: the paper asset builder calls into this module.
    from dyn_evt_pdm.paper.assets import _write_latex_table

    compact = build_compact_benchmark(frame)
    compact_path = tables / "event_benchmark_compact.csv"
    compact.to_csv(compact_path, index=False)
    written.append(compact_path)

    compact_tex = latex / "event_benchmark_compact.tex"
    _write_compact_latex_table(compact, compact_tex)
    written.append(compact_tex)

    full_path = tables / "event_benchmark_full.csv"
    build_full_benchmark(frame).to_csv(full_path, index=False)
    written.append(full_path)

    audit = fairness_audit_frame(build_fairness_audit(frame))
    audit_path = tables / "event_benchmark_fairness_audit.csv"
    audit.to_csv(audit_path, index=False)
    written.append(audit_path)

    detection = build_supplementary_detection_table(frame)
    detection_tex = latex / "event_benchmark_detection.tex"
    _write_latex_table(
        detection,
        detection_tex,
        caption=(
            "Complete held-out event-level benchmark under the shared alarm policy: "
            "every declared method on every compressor dataset. Alarm burden is given "
            "as both interruption frequency (false alarms per operating day) and "
            "warning exposure (percentage of the test period under warning), because "
            "the two can disagree in direction."
        ),
        label="tab:supp-event-benchmark",
    )
    written.append(detection_tex)

    provenance = build_supplementary_provenance_table(frame)
    provenance_tex = latex / "event_benchmark_provenance.tex"
    _write_latex_table(
        provenance,
        provenance_tex,
        caption=(
            "Method provenance for the benchmark: score, threshold rule, event policy, "
            "alarm merge gap, parameter count, calibration status and tuning partition. "
            "Methods sharing a score are separated by their threshold rule or event "
            "policy; no two share all three."
        ),
        label="tab:supp-event-provenance",
    )
    written.append(provenance_tex)

    audit_tex = latex / "event_benchmark_fairness_audit.tex"
    _write_latex_table(
        audit,
        audit_tex,
        caption=(
            "Fairness audit for the event-level benchmark. Shared protocol elements are "
            "checked for identity; event conversion is reported rather than required to "
            "be uniform, because several methods exist to vary it."
        ),
        label="tab:supp-event-fairness",
    )
    written.append(audit_tex)
    return written
