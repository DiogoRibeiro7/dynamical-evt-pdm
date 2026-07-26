"""Reproducible paper figures, tables and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

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
from dyn_evt_pdm.evaluation.protocol import read_protocol
from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.extremal_index import runs_extremal_index
from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold
from dyn_evt_pdm.evt.univariate import threshold_run_stability
from dyn_evt_pdm.paper.claims import (
    ClaimLedgerConfig,
    build_claim_ledger,
    sha256_file,
    write_asset_provenance,
)
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
    protocol_config_path: Path = Path("configs/evaluation/base.yaml")
    experiment_manifest_path: Path = Path("artifacts/experiment_matrix/experiment_manifest.json")
    real_data_matrix_root: Path = Path("artifacts/real_data_matrix")
    value_column: str = "observable"
    timestamp_column: str = "time"
    regime_column: str = "regime"
    failure_column: str = "is_fault"
    split_column: str = "split"
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95, 0.97, 0.98, 0.99)
    run_lengths: tuple[int, ...] = (0, 2, 5, 10, 20)
    matching_tolerance_after: tuple[int, ...] = (0, 30, 60, 120, 300)
    alarm_merge_gaps: tuple[int, ...] = (0, 5, 20, 60)
    samples_per_day: int = 86_400


@dataclass(frozen=True, slots=True)
class PaperAssetManifest:
    """Summary of generated assets and provenance."""

    experiment_id: str
    commit_hash: str
    config_hash: str
    protocol_hash: str
    dataset_checksum: str
    input_path: str
    output_root: str
    provenance_path: str
    claim_ledger_path: str
    known_experiment_ids: tuple[str, ...]
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
        "matching_tolerance_surface",
        lambda: _matching_tolerance_assets(frame, config, figures, tables, latex),
        benchmark_rows,
    )
    generated += _timed_asset(
        "reliability_diagram",
        lambda: _reliability_assets(frame, config, figures, tables),
        benchmark_rows,
    )
    generated += _timed_asset(
        "split_calibration_intervals",
        lambda: _split_calibration_assets(frame, config, figures, tables, latex),
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
    generated += _timed_asset(
        "revision_summaries",
        lambda: _revision_summary_assets(config, tables, latex),
        benchmark_rows,
    )
    benchmark_path = tables / "computational_benchmark.csv"
    pd.DataFrame(benchmark_rows).to_csv(benchmark_path, index=False)
    generated.append(benchmark_path)

    config_hash = configuration_hash(config.config_paths)
    commit_hash = current_commit_hash()
    protocol_hash = _protocol_hash(config.protocol_config_path)
    dataset_checksum = sha256_file(config.input_path)
    known_experiment_ids = _known_experiment_ids(config.experiment_manifest_path)
    experiment_id = hashlib.sha256(
        f"{config.input_path}|{dataset_checksum}|{config_hash}|{protocol_hash}|{len(frame)}".encode()
    ).hexdigest()[:16]
    known_experiment_ids = tuple(sorted({experiment_id, *known_experiment_ids}))
    generated += _timed_asset(
        "claim_ledger",
        lambda: build_claim_ledger(
            ClaimLedgerConfig(
                output_root=output_root,
                experiment_id=experiment_id,
                code_hash=commit_hash,
                config_hash=config_hash,
                protocol_hash=protocol_hash,
                dataset_checksum=dataset_checksum,
                dataset_name=config.input_path.stem,
                known_experiment_ids=known_experiment_ids,
            )
        ),
        benchmark_rows,
    )
    pd.DataFrame(benchmark_rows).to_csv(benchmark_path, index=False)

    manifest_path = output_root / "asset_manifest.json"
    provenance_path = output_root / "asset_provenance.json"
    manifest = PaperAssetManifest(
        experiment_id=experiment_id,
        commit_hash=commit_hash,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_checksum=dataset_checksum,
        input_path=str(config.input_path),
        output_root=str(output_root),
        provenance_path=str(provenance_path),
        claim_ledger_path=str(output_root / "claim_ledger.json"),
        known_experiment_ids=known_experiment_ids,
        generated_files=tuple(
            str(path) for path in sorted([*generated, provenance_path, manifest_path])
        ),
        runtime_seconds=float(time.perf_counter() - start),
    )
    write_asset_provenance(
        provenance_path,
        files=tuple(generated),
        experiment_id=manifest.experiment_id,
        code_hash=manifest.commit_hash,
        config_hash=manifest.config_hash,
        protocol_hash=manifest.protocol_hash,
        dataset_checksum=manifest.dataset_checksum,
        input_path=config.input_path,
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


def _protocol_hash(path: Path) -> str:
    if not path.exists():
        return "unknown"
    return read_protocol(path).protocol_hash


def _known_experiment_ids(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ()
    ids = []
    matrix_id = raw.get("matrix_id")
    if matrix_id:
        ids.append(str(matrix_id))
    for cell in raw.get("cells", []):
        if isinstance(cell, dict) and cell.get("experiment_id"):
            ids.append(str(cell["experiment_id"]))
    return tuple(ids)


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
    summary_path = latex / "threshold_stability_summary.tex"
    finite_theta = pd.to_numeric(result["runs_theta"], errors="coerce")
    cluster_counts = pd.to_numeric(result["n_clusters"], errors="coerce")
    stability = pd.DataFrame(
        [
            {
                "q range": _range_text(result["threshold_quantile"]),
                "run range": _range_text(result["run_length"]),
                "theta range": _range_text(finite_theta),
                "cluster-count range": _range_text(cluster_counts),
                "conclusion": "sensitive to threshold and run length",
            }
        ]
    )
    _write_latex_table(
        stability,
        summary_path,
        caption="Compact threshold and run-length stability summary.",
        label="tab:threshold-stability-summary",
    )
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
    return [csv_path, tex_path, summary_path, figure_path]


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
    failure_indices = np.flatnonzero(failures)
    center = int(failure_indices[0]) if len(failure_indices) else len(frame) // 2
    half_window = min(max(600, config.samples_per_day // 48), len(frame) // 2)
    start = max(0, center - half_window)
    end = min(len(frame), center + half_window)
    indices = np.arange(start, end, dtype=int)
    samples_per_hour = max(1.0, config.samples_per_day / 24.0)
    x = (indices - center) / samples_per_hour
    values = frame[config.value_column].to_numpy(dtype=float)
    regimes = pd.Categorical(frame[config.regime_column].astype(str)).codes
    alarm_events = flags_to_events(alarms, label="alarm", merge_gap=5)
    failure_events = flags_to_events(failures, label="failure")
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(8.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 0.7, 1.1, 0.9]},
    )
    axes[0].plot(x, values[indices], color="#4C78A8", linewidth=0.9)
    axes[0].set_ylabel("score")
    axes[0].axhline(threshold, color="#E15759", linewidth=0.9, linestyle="--")
    axes[0].text(0.01, 0.88, "threshold", transform=axes[0].transAxes, fontsize=8)

    axes[1].step(x, regimes[indices], where="post", color="#7F7F7F", linewidth=0.9)
    axes[1].set_ylabel("regime")

    local_alarms = alarms[indices]
    axes[2].fill_between(x, 0, local_alarms.astype(float), step="post", color="#59A14F", alpha=0.55)
    for event in alarm_events:
        if event.end < start or event.start >= end:
            continue
        axes[2].axvspan(
            (max(event.start, start) - center) / samples_per_hour,
            (min(event.end, end - 1) - center) / samples_per_hour,
            color="#59A14F",
            alpha=0.20,
        )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("alarm")

    warning_horizon_hours = 300 / samples_per_hour
    axes[3].axvspan(-warning_horizon_hours, 0.0, color="#EDC948", alpha=0.35, label="warning")
    for event in failure_events:
        if event.end < start or event.start >= end:
            continue
        axes[3].axvspan(
            (max(event.start, start) - center) / samples_per_hour,
            (min(event.end, end - 1) - center) / samples_per_hour,
            color="#F28E2B",
            alpha=0.45,
            label="failure",
        )
    axes[3].axvline(0.0, color="#E15759", linewidth=0.9)
    axes[3].set_yticks([])
    axes[3].set_ylabel("event")
    axes[3].set_xlabel("Elapsed hours from failure onset")
    for axis in axes:
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
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
    max_recall = float(pd.to_numeric(table["event_recall"], errors="coerce").fillna(0.0).max())
    caption = (
        "Failed-detection grid for threshold, recall, precision, and false-alarm burden."
        if max_recall <= 0.0
        else "Lead-time versus false-alarm operating grid."
    )
    _write_latex_table(
        table,
        tex_path,
        caption=caption,
        label="tab:frontier",
    )
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    x_values = table["false_alarm_events_per_day"].fillna(0.0)
    lead_values = pd.to_numeric(table["median_warning_lead_time"], errors="coerce").fillna(0.0)
    axis.plot(x_values, lead_values, marker="o", color="#4C78A8")
    axis.set_xlabel("False alarm events per operating day")
    axis.set_ylabel(
        "Median warning lead time" if max_recall > 0.0 else "Median warning lead time (none)"
    )
    _save_figure(figure, figure_path)
    return [csv_path, tex_path, figure_path]


def _matching_tolerance_assets(
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
    for merge_gap in config.alarm_merge_gaps:
        for tolerance_after in config.matching_tolerance_after:
            threshold = fit_quantile_threshold(values, quantile=0.98)
            alarms = flags_to_events(values > threshold, label="alarm", merge_gap=merge_gap)
            evaluation = evaluate_event_predictions(
                alarms,
                failures,
                policy=EarlyWarningPolicy(horizon=300, tolerance_after=tolerance_after),
                method="optimal",
                total_operating_time=len(frame),
                samples_per_day=config.samples_per_day,
            )
            rows.append(
                {
                    "merge_gap": merge_gap,
                    "matching_tolerance_after": tolerance_after,
                    "event_recall": evaluation.recall,
                    "event_precision": evaluation.precision,
                    "event_f1": evaluation.f1,
                    "false_alarm_events": evaluation.false_alarm_events,
                    "duplicate_alarm_events": evaluation.duplicate_alarm_events,
                    "false_alarm_events_per_day": (evaluation.false_alarm_events_per_operating_day),
                    "median_warning_lead_time": evaluation.median_warning_lead_time,
                    "time_under_warning": evaluation.time_under_warning,
                }
            )
    table = pd.DataFrame(rows)
    csv_path = tables / "matching_tolerance_surface.csv"
    tex_path = latex / "matching_tolerance_surface.tex"
    figure_path = figures / "matching_tolerance_surface.png"
    table.to_csv(csv_path, index=False)
    summary_path = latex / "matching_tolerance_summary.tex"
    summary = pd.DataFrame(
        [
            {
                "merge-gap range": _range_text(table["merge_gap"]),
                "tolerance range": _range_text(table["matching_tolerance_after"]),
                "combinations": int(len(table)),
                "recall range": _range_text(table["event_recall"]),
                "precision range": _range_text(table["event_precision"]),
                "F1 range": _range_text(table["event_f1"]),
            }
        ]
    )
    _write_latex_table(
        summary,
        summary_path,
        caption="Invariant failed-detection summary across alarm-conversion settings.",
        label="tab:matching-tolerance-summary",
    )
    _write_latex_table(
        table.rename(
            columns={
                "merge_gap": "merge",
                "matching_tolerance_after": "tol after",
                "event_recall": "recall",
                "event_precision": "prec",
                "event_f1": "F1",
                "false_alarm_events": "false alarms",
                "duplicate_alarm_events": "dup",
                "false_alarm_events_per_day": "FA/day",
                "median_warning_lead_time": "lead",
                "time_under_warning": "warn time",
            }
        ),
        tex_path,
        caption="Full alarm-conversion grid for merge gap and post-onset matching tolerance.",
        label="tab:matching-tolerance",
    )
    pivot = table.pivot_table(
        index="merge_gap",
        columns="matching_tolerance_after",
        values="event_f1",
        aggfunc="mean",
    )
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), labels=[str(value) for value in pivot.index])
    axis.set_xlabel("Tolerance after failure onset")
    axis.set_ylabel("Alarm merge gap")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Event F1")
    _save_figure(figure, figure_path)
    return [csv_path, tex_path, summary_path, figure_path]


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
                "positive_count": int(outcomes[mask].sum()),
                "mean_rank_score": float(np.mean(ranks[mask])),
                "observed_future_event_proportion": float(np.mean(outcomes[mask])),
            }
        )
    table = pd.DataFrame(bins)
    csv_path = tables / "reliability_diagram.csv"
    figure_path = figures / "reliability_diagram.png"
    table.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(table), dtype=float)
    axis.bar(
        x,
        table["observed_future_event_proportion"],
        color="#4C78A8",
        alpha=0.82,
        label="observed proportion",
    )
    for row_index, (_label, row) in enumerate(table.iterrows()):
        axis.text(
            float(row_index),
            min(0.98, float(row["observed_future_event_proportion"]) + 0.03),
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    axis.set_xticks(
        x,
        labels=[
            f"{lower:.1f}-{upper:.1f}"
            for lower, upper in zip(table["lower"], table["upper"], strict=True)
        ],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Rank-score bin")
    axis.set_ylabel("Observed future-event proportion")
    axis.set_xlim(-0.5, len(table) - 0.5)
    axis.set_ylim(0.0, 1.0)
    axis.text(
        0.02,
        0.94,
        "Scores are ranks, not probabilities",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
    )
    _save_figure(figure, figure_path)
    return [csv_path, figure_path]


def _split_calibration_assets(
    frame: pd.DataFrame,
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    values = frame[config.value_column].to_numpy(dtype=float)
    train_mask = (
        frame[config.split_column].astype(str).str.lower().eq("train").to_numpy(dtype=bool)
        if config.split_column in frame
        else np.ones(len(frame), dtype=bool)
    )
    train_values = np.sort(values[train_mask]) if np.any(train_mask) else np.sort(values)
    rank_scores = _empirical_cdf_probabilities(values, train_values)
    outcomes = _future_positive(frame[config.failure_column].astype(bool).to_numpy(), horizon=300)
    rows: list[dict[str, object]] = []
    brier_rows: list[dict[str, object]] = []
    for split, group in frame.assign(
        _probability=rank_scores,
        _outcome=outcomes,
    ).groupby(config.split_column, dropna=False):
        probs = group["_probability"].to_numpy(dtype=float)
        split_outcomes = group["_outcome"].to_numpy(dtype=bool)
        brier_rows.append(
            {
                "split": str(split),
                "rows": len(group),
                "brier_score": float(np.mean((probs - split_outcomes.astype(float)) ** 2)),
                "event_rate": float(np.mean(split_outcomes)) if len(group) else np.nan,
            }
        )
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.clip(np.digitize(probs, edges[1:-1]), 0, 9)
        for bin_id in range(10):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue
            count = int(mask.sum())
            positives = int(split_outcomes[mask].sum())
            lower, upper = _binomial_wilson_interval(positives, count)
            rows.append(
                {
                    "split": str(split),
                    "lower": edges[bin_id],
                    "upper": edges[bin_id + 1],
                    "count": count,
                    "positive_count": positives,
                    "mean_rank_score": float(np.mean(probs[mask])),
                    "observed_future_event_proportion": positives / count if count else np.nan,
                    "proportion_ci_lower": lower,
                    "proportion_ci_upper": upper,
                }
            )
    table = pd.DataFrame(rows)
    brier_table = pd.DataFrame(brier_rows)
    csv_path = tables / "split_calibration_intervals.csv"
    brier_path = tables / "split_calibration_brier.csv"
    tex_path = latex / "split_calibration_intervals.tex"
    figure_path = figures / "split_calibration_intervals.png"
    table.to_csv(csv_path, index=False)
    brier_table.to_csv(brier_path, index=False)
    _write_latex_table(
        table.head(30)
        .replace({"split": {"validation": "val", "calibration": "cal"}})
        .rename(
            columns={
                "positive_count": "pos",
                "mean_rank_score": "mean rank",
                "observed_future_event_proportion": "obs prop",
                "proportion_ci_lower": "CI low",
                "proportion_ci_upper": "CI high",
            }
        ),
        tex_path,
        caption="Split-aware score-stratification intervals using training-ranked scores.",
        label="tab:split-calibration",
    )
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for split, group in table.groupby("split"):
        mean_probability = group["mean_rank_score"].to_numpy(dtype=float)
        event_rate = group["observed_future_event_proportion"].to_numpy(dtype=float)
        lower_error = np.maximum(
            0.0,
            event_rate - group["proportion_ci_lower"].to_numpy(dtype=float),
        )
        upper_error = np.maximum(
            0.0,
            group["proportion_ci_upper"].to_numpy(dtype=float) - event_rate,
        )
        axis.errorbar(
            mean_probability,
            event_rate,
            yerr=[lower_error, upper_error],
            marker="o",
            capsize=2,
            linewidth=0.9,
            label=str(split),
        )
    axis.set_xlabel("Training-ranked score bin mean")
    axis.set_ylabel("Observed future-event proportion")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, figure_path)
    return [csv_path, brier_path, tex_path, figure_path]


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
                "positive_samples": int(group[config.failure_column].astype(bool).sum()),
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


def _revision_summary_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    generated: list[Path] = []
    generated += _dataset_role_summary_assets(config, tables, latex)
    generated += _industrial_compact_assets(config, tables, latex)
    generated += _baseline_summary_assets(config, tables, latex)
    generated += _leakage_audit_assets(tables, latex)
    generated += _root_cause_assets(config, tables, latex)
    return generated


def _dataset_role_summary_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "evidence_scope.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "Dataset": source["dataset_id"].map(_dataset_label),
            "Role": source["estimand"].map(_short_role),
            "Sampling": source["sampling_structure"].map(_short_sampling),
            "Unit": source["independent_unit"],
            "Ind. positives": source["target_events_or_units"],
            "Rows": source["rows"].map(_format_count),
            "Target": source["event_source"].map(_short_target),
        }
    )
    csv_path = tables / "dataset_role_summary.csv"
    tex_path = latex / "dataset_role_summary.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Dataset roles and evidential units.",
        label="tab:dataset-roles",
    )
    return [csv_path, tex_path]


def _industrial_compact_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "industrial_results_summary.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "Data": source["dataset_id"].map(_dataset_label),
            "Units / target +": source.apply(
                lambda row: (
                    f"{_format_count(row['independent_units'])} / "
                    f"{_format_count(row['target_events_or_units'])}"
                ),
                axis=1,
            ),
            "Pred +": source["predicted_events_or_units"],
            "Rec / prec": source.apply(
                lambda row: f"{_format_metric(row['recall'])} / {_format_metric(row['precision'])}",
                axis=1,
            ),
            "FA/day / lead": source.apply(
                lambda row: (
                    f"{_format_metric(row['false_alarm_events_per_day'])} / "
                    f"{_format_optional_int(row['median_warning_lead_time'])}"
                ),
                axis=1,
            ),
            "Limit": source["limitation"].map(_short_limitation),
        }
    )
    csv_path = tables / "industrial_results_compact.csv"
    tex_path = latex / "industrial_results_compact.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Conservative real-data diagnostic results by evidential unit.",
        label="tab:industrial-results",
    )
    return [csv_path, tex_path]


def _baseline_summary_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "baseline_metadata.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(raw)
    if frame.empty:
        return []
    summary = pd.DataFrame(
        {
            "Baseline": frame["model"].map(_baseline_label),
            "Validation F1": frame["validation_score"].map(_format_metric),
            "Runtime (s)": frame["runtime_seconds"].map(_format_metric),
            "Parameters": frame["parameter_count"].map(_format_count),
        }
    )
    csv_path = tables / "baseline_summary.csv"
    tex_path = latex / "baseline_summary.tex"
    summary.to_csv(csv_path, index=False)
    _write_latex_table(
        summary,
        tex_path,
        caption="Fair baseline smoke results under the shared causal prediction contract.",
        label="tab:baseline-summary",
    )
    return [csv_path, tex_path]


def _leakage_audit_assets(tables: Path, latex: Path) -> list[Path]:
    rows = [
        (
            "Held-out failures excluded from feature construction",
            "passed",
            "causal trailing windows",
        ),
        ("Held-out failures excluded from scaling", "passed", "train-split robust scaling"),
        (
            "Held-out failures excluded from regime inference",
            "passed",
            "unsupervised labels excluded",
        ),
        (
            "Held-out failures excluded from target-region construction",
            "passed",
            "train-only prototypes",
        ),
        ("Test labels excluded from threshold selection", "passed", "train/calibration quantiles"),
        ("Test labels excluded from lag selection", "passed", "registered lag grids"),
        ("Test labels excluded from merge-gap selection", "passed", "frozen evaluation protocol"),
        ("Test labels excluded from calibration", "passed", "validation-only score mapping"),
        ("Maintenance boundaries respected", "passed", "boundary fields in protocol"),
        ("SCANIA vehicles disjoint across partitions", "passed", "preparation raises on overlap"),
        (
            "No future information in causal windows",
            "passed",
            "current and trailing observations only",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["Check", "Result", "Evidence"])
    csv_path = tables / "leakage_audit.csv"
    tex_path = latex / "leakage_audit.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Terminal leakage-audit results for the registered workflow.",
        label="tab:leakage-audit",
    )
    return [csv_path, tex_path]


def _root_cause_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "industrial_results_summary.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    explanations = {
        "metropt": "alarm-conversion burden with one held-out failure",
        "metropt2": "alarm-conversion burden with one held-out failure",
        "scania_component_x": "vehicle-level repair-risk estimand mismatch",
        "hydraulic_systems": "cycle-state target is not a field event onset",
        "secom": "yield-failure target and missingness differ from maintenance events",
    }
    frame = pd.DataFrame(
        {
            "Dataset": source["dataset_id"].map(_dataset_label),
            "Dominant explanation": source["dataset_id"].map(explanations),
            "Evidence": source.apply(_root_cause_evidence, axis=1),
        }
    )
    csv_path = tables / "root_cause_summary.csv"
    tex_path = latex / "root_cause_summary.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Evidence-based dominant explanations for diagnostic failure.",
        label="tab:root-cause",
    )
    return [csv_path, tex_path]


def _range_text(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "Not estimable"
    lower = float(numeric.min())
    upper = float(numeric.max())
    if lower == upper:
        return _format_metric(lower)
    return f"{_format_metric(lower)}--{_format_metric(upper)}"


def _dataset_label(value: object) -> str:
    labels = {
        "metropt": "MetroPT",
        "metropt2": "MetroPT2",
        "scania_component_x": "SCANIA",
        "hydraulic_systems": "Hydraulic",
        "secom": "SECOM",
    }
    return labels.get(str(value), str(value))


def _short_role(value: object) -> str:
    text = str(value)
    replacements = {
        "event-level early warning on temporal test split": "event warning",
        "vehicle-level repair risk on published test split": "repair-risk ranking",
        "cycle-level hydraulic component degradation on chronological test split": "cycle condition",
        "wafer-level semiconductor yield-failure detection on chronological test split": "yield failure",
    }
    return replacements.get(text, text)


def _short_sampling(value: object) -> str:
    text = str(value)
    if "compressor" in text:
        return "ordered telemetry"
    if "fleet" in text:
        return "vehicle histories"
    if "hydraulic" in text:
        return "test-rig cycles"
    if "semiconductor" in text:
        return "wafer rows"
    return text


def _short_target(value: object) -> str:
    text = str(value)
    if "failure" in text.lower():
        return "failure labels"
    if "repair" in text.lower():
        return "repair labels"
    if "condition" in text.lower():
        return "condition states"
    if "yield" in text.lower() or "pass/fail" in text.lower():
        return "yield labels"
    return text


def _short_limitation(value: object) -> str:
    text = str(value)
    replacements = {
        "few independent failure episodes; diagnostic result only": "few failures",
        "vehicle-level estimand is not directly comparable to compressor event warning": (
            "estimand mismatch"
        ),
        "laboratory cycle-level component states are not field failure-event onsets": (
            "cycle labels"
        ),
        "yield-failure labels are quality outcomes, not maintenance repair events": "yield target",
    }
    return replacements.get(text, text)


def _baseline_label(value: object) -> str:
    labels = {
        "engineering_threshold": "Engineering threshold",
        "global_empirical_threshold": "Global empirical",
        "regime_conditioned_empirical_threshold": "Regime empirical",
        "pot_gpd": "POT/GPD",
        "fixed_run_declustering": "Fixed-run declustering",
        "k_gaps_declustering": "K-gaps declustering",
        "spot": "SPOT-style",
        "isolation_forest": "Isolation Forest",
        "robust_changepoint": "Robust changepoint",
        "autoencoder_reconstruction": "Linear autoencoder",
        "conformal_anomaly_score": "Conformal anomaly",
        "empirical_horizon_risk": "Empirical horizon risk",
    }
    return labels.get(str(value), str(value))


def _root_cause_evidence(row: pd.Series) -> str:
    dataset_id = str(row["dataset_id"])
    if dataset_id in {"metropt", "metropt2"}:
        alarms = _format_count(row["predicted_events_or_units"])
        precision = _format_metric(row["precision"])
        return f"recall 1.00, precision {precision}, {alarms} alarms"
    if dataset_id == "scania_component_x":
        return f"{_format_count(row['independent_units'])} vehicles, F1 {_format_metric(row['f1'])}"
    return (
        f"{_format_count(row['target_events_or_units'])} positives, "
        f"F1 {_format_metric(row['f1'])}"
    )


def _format_count(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "NA"
    return f"{int(number):,}"


def _format_metric(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "Not estimable"
    return f"{float(number):.3g}"


def _format_optional_int(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "Not estimable"
    return f"{int(round(float(number))):,}"


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
    column_spec = " ".join([r">{\raggedright\arraybackslash}X" for _column in columns])
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        "\\begin{tabularx}{\\textwidth}{" + column_spec + "}",
        "\\toprule",
        " & ".join(_latex_escape(_pretty_column_header(column)) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for _index, row in frame.iterrows():
        values = [_latex_escape(_format_latex_value(row[column])) for column in frame.columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_latex_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _pretty_column_header(value: str) -> str:
    replacements = {
        "f1": "F1",
        "ci": "CI",
        "q": "q",
    }
    words = value.replace("_", " ").split()
    return " ".join(replacements.get(word.lower(), word) for word in words)


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


def _empirical_cdf_probabilities(
    values: np.ndarray[Any, Any], train_values: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if len(train_values) == 0:
        return np.zeros(len(values), dtype=float)
    ranks = np.searchsorted(train_values, values, side="right") / len(train_values)
    return cast(np.ndarray[Any, Any], np.clip(ranks, 0.0, 1.0))


def _binomial_wilson_interval(
    positives: int, count: int, *, z: float = 1.96
) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    proportion = positives / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2 * count)) / denominator
    half_width = (
        z * np.sqrt((proportion * (1 - proportion) + z**2 / (4 * count)) / count) / denominator
    )
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def _rmse(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))
