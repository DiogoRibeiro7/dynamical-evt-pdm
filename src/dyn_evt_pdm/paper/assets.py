"""Reproducible paper figures, tables and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, flags_to_events
from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions
from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.extremal_index import runs_extremal_index
from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold
from dyn_evt_pdm.evt.univariate import threshold_run_stability
from dyn_evt_pdm.simulation.study import (
    simulation_study_config_from_mapping,
    write_simulation_study,
)


@dataclass(frozen=True, slots=True)
class PaperAssetConfig:
    """Inputs and output locations for manuscript artifacts."""

    input_path: Path = Path("data/processed/synthetic_cyclic.csv")
    output_root: Path = Path("reports/paper")
    config_paths: tuple[Path, ...] = (
        Path("configs/base.yaml"),
        Path("configs/simulation/cyclic_degradation.yaml"),
        Path("configs/experiments/metropt_primary.yaml"),
    )
    simulation_config_path: Path = Path("configs/simulation/cyclic_degradation.yaml")
    simulation_study_path: Path = Path("artifacts/simulation_study_smoke.parquet")
    value_column: str = "observable"
    timestamp_column: str = "time"
    regime_column: str = "regime"
    failure_column: str = "is_fault"
    split_column: str = "split"
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95, 0.97, 0.98, 0.99)
    run_lengths: tuple[int, ...] = (0, 2, 5, 10, 20)
    samples_per_day: int = 86_400


@dataclass(frozen=True, slots=True)
class PaperAssetManifest:
    """Summary of generated assets and provenance."""

    experiment_id: str
    commit_hash: str
    config_hash: str
    input_path: str
    output_root: str
    generated_files: tuple[str, ...]
    runtime_seconds: float


def build_paper_assets(config: PaperAssetConfig) -> PaperAssetManifest:
    """Build every generated figure/table used by the manuscript."""

    start = time.perf_counter()
    if not config.input_path.exists():
        raise FileNotFoundError(f"processed input not found: {config.input_path}")
    frame = _read_table(config.input_path)
    _require_columns(
        frame,
        (
            config.value_column,
            config.regime_column,
            config.failure_column,
            config.split_column,
        ),
    )
    output_root = config.output_root
    figures = output_root / "figures"
    tables = output_root / "tables"
    latex = output_root / "latex"
    for directory in (figures, tables, latex):
        directory.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    benchmark_rows: list[dict[str, object]] = []

    generated += _timed_asset(
        "threshold_stability",
        lambda: _threshold_stability_assets(frame, config, figures, tables, latex),
        benchmark_rows,
    )
    generated += _timed_asset(
        "extremal_index_by_regime",
        lambda: _extremal_index_by_regime_assets(frame, config, figures, tables),
        benchmark_rows,
    )
    generated += _timed_asset(
        "cluster_size_distribution",
        lambda: _cluster_distribution_assets(frame, config, figures, tables),
        benchmark_rows,
    )
    generated += _timed_asset(
        "event_timeline",
        lambda: _event_timeline_assets(frame, config, figures),
        benchmark_rows,
    )
    generated += _timed_asset(
        "lead_time_false_alarm_frontier",
        lambda: _lead_time_frontier_assets(frame, config, figures, tables, latex),
        benchmark_rows,
    )
    generated += _timed_asset(
        "reliability_diagram",
        lambda: _reliability_assets(frame, config, figures, tables),
        benchmark_rows,
    )
    generated += _timed_asset(
        "dataset_split_summary",
        lambda: _dataset_split_assets(frame, config, tables, latex),
        benchmark_rows,
    )
    generated += _timed_asset(
        "ablation_tables",
        lambda: _ablation_assets(frame, config, tables, latex),
        benchmark_rows,
    )
    generated += _timed_asset(
        "simulation_bias_rmse",
        lambda: _simulation_bias_assets(config, figures, tables),
        benchmark_rows,
    )
    benchmark_path = tables / "computational_benchmark.csv"
    pd.DataFrame(benchmark_rows).to_csv(benchmark_path, index=False)
    generated.append(benchmark_path)

    config_hash = configuration_hash(config.config_paths)
    manifest_path = output_root / "asset_manifest.json"
    manifest = PaperAssetManifest(
        experiment_id=hashlib.sha256(
            f"{config.input_path}|{config_hash}|{len(frame)}".encode()
        ).hexdigest()[:16],
        commit_hash=current_commit_hash(),
        config_hash=config_hash,
        input_path=str(config.input_path),
        output_root=str(output_root),
        generated_files=tuple(str(path) for path in sorted([*generated, manifest_path])),
        runtime_seconds=float(time.perf_counter() - start),
    )
    manifest_payload = asdict(manifest)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest


def configuration_hash(paths: tuple[Path, ...]) -> str:
    """Hash normalized configuration content for asset provenance."""

    payloads: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() in {".yaml", ".yml"}:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            content = path.read_text(encoding="utf-8")
        payloads.append({"path": str(path), "content": content})
    encoded = json.dumps(payloads, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def current_commit_hash() -> str:
    """Return the current Git commit hash, or ``unknown`` outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _threshold_stability_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    result = threshold_run_stability(
        frame[config.value_column].to_numpy(dtype=float),
        threshold_quantiles=config.threshold_quantiles,
        run_lengths=config.run_lengths,
        min_exceedances=3,
    )
    csv_path = tables / "threshold_stability.csv"
    tex_path = latex / "threshold_stability.tex"
    figure_path = figures / "threshold_stability.png"
    result.to_csv(csv_path, index=False)
    _write_latex_table(
        result[["threshold_quantile", "run_length", "n_exceedances", "n_clusters"]].head(24),
        tex_path,
        caption="Threshold and run-length stability diagnostics.",
        label="tab:threshold-stability",
    )
    pivot = result.pivot_table(
        index="threshold_quantile",
        columns="run_length",
        values="runs_theta",
        aggfunc="mean",
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for run_length in pivot.columns:
        axis.plot(pivot.index, pivot[run_length], marker="o", label=f"run={run_length}")
    axis.set_xlabel("Threshold quantile")
    axis.set_ylabel("Runs extremal-index estimate")
    axis.set_ylim(0.0, 1.05)
    axis.legend(frameon=False, ncols=2, fontsize=8)
    _save_figure(figure, figure_path)
    return [csv_path, tex_path, figure_path]


def _extremal_index_by_regime_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
) -> list[Path]:
    rows: list[dict[str, object]] = []
    global_threshold = fit_quantile_threshold(
        frame[config.value_column].to_numpy(dtype=float),
        quantile=max(config.threshold_quantiles),
    )
    for regime, group in frame.groupby(config.regime_column, dropna=False):
        values = group[config.value_column].to_numpy(dtype=float)
        if len(values) < 3:
            continue
        threshold = min(global_threshold, fit_quantile_threshold(values, quantile=0.95))
        flags = np.asarray(values > threshold, dtype=bool)
        theta = runs_extremal_index(flags, run_length=5) if np.any(flags) else np.nan
        rows.append(
            {
                "regime": str(regime),
                "rows": len(group),
                "threshold": threshold,
                "exceedances": int(flags.sum()),
                "runs_theta": theta,
            }
        )
    table = pd.DataFrame(rows)
    csv_path = tables / "extremal_index_by_regime.csv"
    figure_path = figures / "extremal_index_by_regime.png"
    table.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.bar(table["regime"], table["runs_theta"].fillna(0.0), color="#4C78A8")
    axis.set_xlabel("Regime")
    axis.set_ylabel("Empirical runs theta")
    axis.set_ylim(0.0, 1.05)
    _save_figure(figure, figure_path)
    return [csv_path, figure_path]


def _cluster_distribution_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
) -> list[Path]:
    threshold = fit_quantile_threshold(
        frame[config.value_column].to_numpy(dtype=float),
        quantile=0.98,
    )
    clusters = extract_clusters(
        np.asarray(frame[config.value_column].to_numpy(dtype=float) > threshold, dtype=bool),
        run_length=5,
    )
    table = pd.DataFrame(
        {
            "cluster_start": [cluster.interval.start for cluster in clusters],
            "cluster_end": [cluster.interval.end for cluster in clusters],
            "cluster_size": [cluster.size for cluster in clusters],
            "cluster_duration": [cluster.interval.duration for cluster in clusters],
        }
    )
    csv_path = tables / "cluster_size_distribution.csv"
    figure_path = figures / "cluster_size_distribution.png"
    table.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    if table.empty:
        axis.text(0.5, 0.5, "No clusters", ha="center", va="center")
    else:
        axis.hist(table["cluster_size"], bins="auto", color="#59A14F", edgecolor="white")
    axis.set_xlabel("Cluster size")
    axis.set_ylabel("Count")
    _save_figure(figure, figure_path)
    return [csv_path, figure_path]


def _event_timeline_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
) -> list[Path]:
    threshold = fit_quantile_threshold(
        frame[config.value_column].to_numpy(dtype=float),
        quantile=0.98,
    )
    alarms = np.asarray(frame[config.value_column].to_numpy(dtype=float) > threshold, dtype=bool)
    failures = frame[config.failure_column].astype(bool).to_numpy()
    x = (
        frame[config.timestamp_column]
        if config.timestamp_column in frame
        else np.arange(len(frame))
    )
    figure, axis = plt.subplots(figsize=(9.0, 3.6))
    axis.plot(x, frame[config.value_column], color="#4C78A8", linewidth=0.9, label="observable")
    axis.axhline(threshold, color="#E15759", linewidth=0.9, label="threshold")
    _shade_flags(axis, x, failures, color="#F28E2B", alpha=0.20, label="failure")
    _shade_flags(axis, x, alarms, color="#59A14F", alpha=0.13, label="alarm")
    axis.set_xlabel(config.timestamp_column if config.timestamp_column in frame else "sample")
    axis.set_ylabel(config.value_column)
    axis.legend(frameon=False, ncols=4, fontsize=8)
    path = figures / "event_timeline.png"
    _save_figure(figure, path)
    return [path]


def _lead_time_frontier_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    values = frame[config.value_column].to_numpy(dtype=float)
    failures = flags_to_events(
        frame[config.failure_column].astype(bool).to_numpy(), label="failure"
    )
    rows: list[dict[str, object]] = []
    for quantile in config.threshold_quantiles:
        threshold = fit_quantile_threshold(values, quantile=quantile)
        alarms = flags_to_events(values > threshold, label="alarm", merge_gap=5)
        evaluation = evaluate_event_predictions(
            alarms,
            failures,
            policy=EarlyWarningPolicy(horizon=300, tolerance_after=30),
            method="optimal",
            total_operating_time=len(frame),
            samples_per_day=config.samples_per_day,
        )
        rows.append(
            {
                "threshold_quantile": quantile,
                "threshold": threshold,
                "event_recall": evaluation.recall,
                "event_precision": evaluation.precision,
                "false_alarm_events_per_day": evaluation.false_alarm_events_per_operating_day,
                "median_warning_lead_time": evaluation.median_warning_lead_time,
            }
        )
    table = pd.DataFrame(rows)
    csv_path = tables / "lead_time_false_alarm_frontier.csv"
    tex_path = latex / "lead_time_false_alarm_frontier.tex"
    figure_path = figures / "lead_time_false_alarm_frontier.png"
    table.to_csv(csv_path, index=False)
    _write_latex_table(
        table, tex_path, caption="Lead-time versus false-alarm frontier.", label="tab:frontier"
    )
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    x_values = table["false_alarm_events_per_day"].fillna(0.0)
    lead_values = pd.to_numeric(table["median_warning_lead_time"], errors="coerce").fillna(0.0)
    axis.plot(x_values, lead_values, marker="o", color="#4C78A8")
    axis.set_xlabel("False alarm events per operating day")
    axis.set_ylabel("Median warning lead time")
    _save_figure(figure, figure_path)
    return [csv_path, tex_path, figure_path]


def _reliability_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
) -> list[Path]:
    values = frame[config.value_column].to_numpy(dtype=float)
    ranks = pd.Series(values).rank(pct=True).to_numpy(dtype=float)
    outcomes = _future_positive(frame[config.failure_column].astype(bool).to_numpy(), horizon=300)
    bins: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (ranks >= edges[index]) & (
            ranks <= edges[index + 1] if index == 9 else ranks < edges[index + 1]
        )
        if not np.any(mask):
            continue
        bins.append(
            {
                "lower": edges[index],
                "upper": edges[index + 1],
                "count": int(mask.sum()),
                "mean_probability": float(np.mean(ranks[mask])),
                "event_rate": float(np.mean(outcomes[mask])),
            }
        )
    table = pd.DataFrame(bins)
    csv_path = tables / "reliability_diagram.csv"
    figure_path = figures / "reliability_diagram.png"
    table.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    axis.plot(table["mean_probability"], table["event_rate"], marker="o", color="#4C78A8")
    axis.set_xlabel("Mean predicted horizon risk")
    axis.set_ylabel("Observed horizon event rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    _save_figure(figure, figure_path)
    return [csv_path, figure_path]


def _dataset_split_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby(config.split_column, dropna=False):
        rows.append(
            {
                "dataset": config.input_path.stem,
                "split": str(split),
                "rows": len(group),
                "failures": int(group[config.failure_column].astype(bool).sum()),
                "regimes": int(group[config.regime_column].nunique(dropna=False)),
            }
        )
    table = pd.DataFrame(rows)
    csv_path = tables / "dataset_split_summary.csv"
    tex_path = latex / "dataset_split_summary.tex"
    table.to_csv(csv_path, index=False)
    _write_latex_table(
        table, tex_path, caption="Dataset and split summary.", label="tab:dataset-splits"
    )
    return [csv_path, tex_path]


def _ablation_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    rows: list[dict[str, object]] = []
    values = frame[config.value_column].to_numpy(dtype=float)
    for family, candidates in {
        "threshold": config.threshold_quantiles,
        "run_length": config.run_lengths,
        "lag": (0, 1, 2, 5, 10),
        "regime": tuple(str(value) for value in frame[config.regime_column].dropna().unique()),
        "dangerous_region": ("prototype", "constraint", "density"),
        "noise_missingness": ("none", "noise_0.05", "missing_0.05"),
    }.items():
        for candidate in candidates:
            if family == "threshold":
                threshold = fit_quantile_threshold(values, quantile=float(candidate))
                score = float(np.mean(values > threshold))
            elif family == "run_length":
                threshold = fit_quantile_threshold(values, quantile=0.98)
                clusters = extract_clusters(values > threshold, run_length=int(candidate))
                score = float(len(clusters))
            else:
                score = np.nan
            rows.append({"ablation_family": family, "setting": candidate, "summary_value": score})
    table = pd.DataFrame(rows)
    csv_path = tables / "ablation_summary.csv"
    tex_path = latex / "ablation_summary.tex"
    table.to_csv(csv_path, index=False)
    _write_latex_table(
        table.head(30), tex_path, caption="Ablation matrix summary.", label="tab:ablations"
    )
    return [csv_path, tex_path]


def _simulation_bias_assets(
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
) -> list[Path]:
    if not config.simulation_study_path.exists():
        raw = yaml.safe_load(config.simulation_config_path.read_text(encoding="utf-8"))
        study_config = simulation_study_config_from_mapping(raw, smoke=True, n_jobs=1)
        write_simulation_study(study_config, config.simulation_study_path)
    study = pd.read_parquet(config.simulation_study_path)
    if "runs_error" not in study:
        study["runs_error"] = study["runs_bias"]
    if "intervals_error" not in study:
        study["intervals_error"] = study["intervals_bias"]
    summary = (
        study.groupby(["system", "threshold_quantile"], as_index=False)
        .agg(
            runs_bias=("runs_bias", "mean"),
            intervals_bias=("intervals_bias", "mean"),
            runs_rmse=("runs_error", _rmse),
            intervals_rmse=("intervals_error", _rmse),
        )
        .sort_values(["system", "threshold_quantile"])
    )
    csv_path = tables / "simulation_bias_rmse.csv"
    figure_path = figures / "simulation_bias_rmse.png"
    summary.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for system, group in summary.groupby("system"):
        axis.plot(group["threshold_quantile"], group["runs_rmse"], marker="o", label=str(system))
    axis.set_xlabel("Threshold quantile")
    axis.set_ylabel("Runs theta RMSE")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, figure_path)
    return [csv_path, figure_path]


def _timed_asset(
    name: str,
    builder: Callable[[], list[Path]],
    benchmark_rows: list[dict[str, object]],
) -> list[Path]:
    start = time.perf_counter()
    paths = builder()
    benchmark_rows.append(
        {
            "asset_group": name,
            "runtime_seconds": float(time.perf_counter() - start),
            "output_count": len(paths),
        }
    )
    return paths


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("paper asset input must be csv or parquet")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _write_latex_table(frame: pd.DataFrame, path: Path, *, caption: str, label: str) -> None:
    columns = [str(column) for column in frame.columns]
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for _index, row in frame.iterrows():
        values = [_latex_escape(_format_latex_value(row[column])) for column in frame.columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_latex_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Creator": "dyn-evt-pdm paper asset builder"})
    plt.close(figure)


def _shade_flags(
    axis: Axes,
    x_values: pd.Series | np.ndarray[Any, Any],
    flags: np.ndarray[Any, Any],
    *,
    color: str,
    alpha: float,
    label: str,
) -> None:
    events = flags_to_events(flags)
    used_label = False
    x_array = np.asarray(x_values)
    for event in events:
        axis.axvspan(
            x_array[event.start],
            x_array[event.end],
            color=color,
            alpha=alpha,
            label=label if not used_label else None,
        )
        used_label = True


def _future_positive(flags: np.ndarray[Any, Any], *, horizon: int) -> np.ndarray[Any, Any]:
    result = np.zeros(len(flags), dtype=bool)
    for offset in range(horizon + 1):
        valid = len(flags) - offset
        if valid <= 0:
            break
        result[:valid] |= flags[offset : offset + valid]
    return result


def _rmse(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))
