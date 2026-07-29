"""Main-paper and supplementary tables for the extremal-index estimator study.

The main table carries one compact row per representative configuration; the full
grid goes to the supplement. Both are generated from the same summary frame so the
compact table can never drift from the evidence behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dyn_evt_pdm.evt.extremal_index import ESTIMATOR_NAMES
from dyn_evt_pdm.simulation.estimator_figures import ESTIMATOR_LABELS, assign_regime

#: Columns of the compact main-paper table, in presentation order.
MAIN_TABLE_COLUMNS: tuple[str, ...] = (
    "process",
    "reference_theta",
    "reference_kind",
    "estimator",
    "n",
    "threshold_range",
    "bias",
    "rmse",
    "coverage",
    "failure_rate",
)

#: Printed header and column alignment for the compact main table.
#:
#: The generic asset writer gives every column an equal share of the text width, which
#: collides the labels once a table reaches ten columns. Text columns are therefore
#: given explicit widths and numeric columns are right-aligned at their natural size.
#: Per-estimator columns. The process family is carried by a spanning group header
#: rather than a column, which avoids a wrapped narrow cell whose lines would sit
#: ambiguously against the estimator rows beside it.
_MAIN_TABLE_LAYOUT: tuple[tuple[str, str, str], ...] = (
    ("estimator", "Estimator", "l"),
    ("n", "$n$", "r"),
    ("threshold_range", "Threshold", "c"),
    ("bias", "Bias", "r"),
    ("rmse", "RMSE", "r"),
    ("coverage", "Coverage", "r"),
    ("failure_rate", "Fail.", "r"),
)


def _write_main_latex_table(frame: pd.DataFrame, path: Path) -> None:
    """Write the compact main table, grouping estimator rows under each process."""

    spec = " ".join(alignment for _key, _header, alignment in _MAIN_TABLE_LAYOUT)
    n_columns = len(_MAIN_TABLE_LAYOUT)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        (
            "\\caption{Finite-sample extremal-index estimator performance for one "
            "representative process per clustering regime, aggregated over the threshold "
            "span shown. Coverage is measured against a nominal 0.95 level; the failure "
            "column is the share of replicates returning no estimate. A dagger marks a "
            "numerically established reference extremal index; unmarked references are "
            "closed-form. The full grid is in the supplement.}"
        ),
        "\\label{tab:estimator-main-summary}",
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(header for _key, header, _alignment in _MAIN_TABLE_LAYOUT) + " \\\\",
    ]

    previous_process = ""
    for _index, row in frame.iterrows():
        process = str(row["process"])
        if process != previous_process:
            reference = float(row["reference_theta"])
            marker = r"$^{\dagger}$" if str(row["reference_kind"]) == "numerical" else ""
            heading = (
                f"\\textit{{{process}}}, "
                f"$\\theta_{{\\mathrm{{ref}}}} = {reference:.3f}${marker}"
            )
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{n_columns}}}{{l}}{{{heading}}} \\\\")
            previous_process = process
        cells = [
            f"{row[key]:.3f}" if isinstance(row[key], float) else str(row[key])
            for key, _header, _alignment in _MAIN_TABLE_LAYOUT
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class TableSet:
    """Paths written by the estimator table builder."""

    main_csv: Path
    main_tex: Path
    supplement_csv: Path
    supplement_tex: Path
    regime_csv: Path

    def as_tuple(self) -> tuple[Path, ...]:
        """Return every generated path."""

        return (
            self.main_csv,
            self.main_tex,
            self.supplement_csv,
            self.supplement_tex,
            self.regime_csv,
        )


def _threshold_range(values: pd.Series) -> str:
    quantiles = sorted({float(value) for value in values})
    if not quantiles:
        return ""
    if len(quantiles) == 1:
        return f"{quantiles[0]:.2f}"
    return f"{quantiles[0]:.2f}-{quantiles[-1]:.2f}"


def select_representative_processes(summary: pd.DataFrame) -> list[tuple[str, float, str]]:
    """Pick one representative process per clustering regime for the main table.

    A closed-form reference is preferred over a numerical one, and among equals the
    configuration with the most replicate evidence wins. Selecting deterministically
    from the data keeps the main table from becoming a hand-picked subset.
    """

    referenced = summary[summary["reference_theta"].notna()]
    if referenced.empty:
        return []

    keys = referenced.drop_duplicates(subset=["system", "reference_theta", "reference_kind"])
    best: dict[str, tuple[str, float, str]] = {}
    best_rank: dict[str, tuple[int, int]] = {}
    triples = zip(
        [str(value) for value in keys["system"].tolist()],
        [float(value) for value in keys["reference_theta"].tolist()],
        [str(value) for value in keys["reference_kind"].tolist()],
        strict=True,
    )
    for system, reference, kind in triples:
        regime = assign_regime(reference)
        matching = referenced[
            (referenced["system"] == system)
            & (referenced["reference_theta"] == reference)
            & (referenced["reference_kind"] == kind)
        ]
        rank = (1 if kind == "theoretical" else 0, int(matching["replicates"].sum()))
        if regime not in best_rank or rank > best_rank[regime]:
            best_rank[regime] = rank
            best[regime] = (system, reference, kind)
    order = ["near_independent", "moderately_clustered", "strongly_clustered"]
    return [best[regime] for regime in order if regime in best]


def build_main_table(summary: pd.DataFrame, *, compact: bool = True) -> pd.DataFrame:
    """Collapse the summary into one row per process and estimator.

    With ``compact`` set, only one representative process per clustering regime is
    kept, which is what the main paper carries; the full referenced grid goes to the
    supplement. Rows are aggregated across thresholds and run lengths and the threshold
    span is reported, so a reader sees the compaction rather than mistaking it for a
    single setting. Configurations without an established reference are excluded, since
    bias, RMSE and coverage are undefined there; the regime table accounts for them.
    """

    referenced = summary[summary["reference_theta"].notna()]
    if referenced.empty:
        return pd.DataFrame(columns=list(MAIN_TABLE_COLUMNS))
    if compact:
        selected = select_representative_processes(summary)
        if selected:
            mask = pd.Series(False, index=referenced.index)
            for system, reference, kind in selected:
                mask |= (
                    (referenced["system"] == system)
                    & (referenced["reference_theta"] == reference)
                    & (referenced["reference_kind"] == kind)
                )
            referenced = referenced[mask]

    records: list[dict[str, object]] = []
    for keys, group in referenced.groupby(
        ["system", "reference_theta", "reference_kind"], sort=True
    ):
        system_key, reference_key, kind_key = keys
        system = str(system_key)
        reference = float(str(reference_key))
        kind = str(kind_key)
        for name in ESTIMATOR_NAMES:
            rows = group[group["estimator"] == name]
            if rows.empty:
                continue
            records.append(
                {
                    "process": system.replace("_", " "),
                    "reference_theta": round(reference, 3),
                    "reference_kind": kind,
                    "estimator": ESTIMATOR_LABELS[name],
                    "n": int(rows["n_steps"].max()),
                    "threshold_range": _threshold_range(rows["threshold_quantile"]),
                    "bias": round(float(rows["bias"].mean()), 3),
                    "rmse": round(float(rows["rmse"].mean()), 3),
                    "coverage": round(float(rows["coverage"].mean()), 3),
                    "failure_rate": round(float(rows["failure_probability"].mean()), 3),
                }
            )
    return pd.DataFrame.from_records(records).loc[:, list(MAIN_TABLE_COLUMNS)]


def build_regime_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Report how many configurations sit in each reference and clustering regime.

    This is the table behind the scope statement: the grid speaks about
    near-independent processes, and the reader should be able to see that directly.
    """

    per_configuration = summary.drop_duplicates(
        subset=[
            "system",
            "n_steps",
            "threshold_quantile",
            "run_length",
            "noise_scale",
            "missing_rate",
        ]
    )
    records: list[dict[str, object]] = []
    for system, group in per_configuration.groupby("system", sort=True):
        references = group["reference_theta"]
        established = references.notna()
        regimes = [assign_regime(float(value)) for value in references[established]]
        records.append(
            {
                "process": str(system).replace("_", " "),
                "configurations": int(len(group)),
                "with_reference": int(established.sum()),
                "reference_kind": (
                    ", ".join(sorted(set(group.loc[established, "reference_kind"].astype(str))))
                    if established.any()
                    else "none established"
                ),
                "reference_theta": (
                    ", ".join(
                        f"{value:.3f}"
                        for value in sorted({round(float(v), 3) for v in references[established]})
                    )
                    if established.any()
                    else ""
                ),
                "regime": ", ".join(sorted(set(regimes))) if regimes else "not established",
            }
        )
    return pd.DataFrame.from_records(records)


def build_estimator_tables(summary: pd.DataFrame, output_root: Path) -> TableSet:
    """Write the compact main table, the full supplementary grid and the regime table."""

    tables = output_root / "tables"
    latex = output_root / "latex"
    tables.mkdir(parents=True, exist_ok=True)
    latex.mkdir(parents=True, exist_ok=True)

    main = build_main_table(summary)
    regime = build_regime_table(summary)

    main_csv = tables / "estimator_main_summary.csv"
    supplement_csv = tables / "estimator_full_grid.csv"
    regime_csv = tables / "estimator_reference_regimes.csv"
    main.to_csv(main_csv, index=False)
    summary.to_csv(supplement_csv, index=False)
    regime.to_csv(regime_csv, index=False)

    # Deferred import: the paper asset builder will call into this module, so a
    # module-level import here would close an import cycle.
    from dyn_evt_pdm.paper.assets import _write_latex_table

    main_tex = latex / "estimator_main_summary.tex"
    supplement_tex = latex / "estimator_full_grid.tex"
    _write_main_latex_table(main, main_tex)
    _write_latex_table(
        regime,
        supplement_tex,
        caption=(
            "Reference availability and clustering regime for every simulated process "
            "family. Configurations without an established reference contribute no "
            "bias, RMSE or coverage evidence."
        ),
        label="tab:estimator-reference-regimes",
    )
    return TableSet(
        main_csv=main_csv,
        main_tex=main_tex,
        supplement_csv=supplement_csv,
        supplement_tex=supplement_tex,
        regime_csv=regime_csv,
    )
