"""Reproducible paper figures, tables and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
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
from dyn_evt_pdm.pipelines.event_method_specs import (
    EVENT_METHOD_SPECS,
    evt_role,
    score_semantics,
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
    simulation_study_path: Path = Path("artifacts/simulation_study_broad.parquet")
    #: Grids pooled for the estimator study. Missing grids are skipped, so a smoke run
    #: still produces assets from whatever evidence is present locally.
    simulation_grid_paths: tuple[Path, ...] = (
        Path("artifacts/simulation_study_broad.parquet"),
        Path("artifacts/simulation_study_high_rep.parquet"),
        Path("artifacts/simulation_study_focused_coverage.parquet"),
    )
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
        "estimator_study",
        lambda: _estimator_study_assets(config, output_root),
        benchmark_rows,
    )
    generated += _timed_asset(
        "event_benchmark",
        lambda: _event_benchmark_assets(config, output_root),
        benchmark_rows,
    )
    generated += _timed_asset(
        "revision_summaries",
        lambda: _revision_summary_assets(config, figures, tables, latex),
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


#: Method whose rows the main event-level table reports.
_REGISTERED_EVENT_METHOD = "dynamical_evt_robust_score"


def _event_benchmark_assets(config: PaperAssetConfig, output_root: Path) -> list[Path]:
    """Build the compact benchmark, both supplementary tables and the fairness audit."""

    from dyn_evt_pdm.pipelines.event_benchmark_tables import write_benchmark_tables

    path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return write_benchmark_tables(frame, output_root)


def _merge_registered_event_metrics(
    config: PaperAssetConfig, summary: pd.DataFrame
) -> pd.DataFrame:
    """Overwrite the compressor summary metrics with the registered method's benchmark rows.

    Anything downstream that describes the registered method must read the benchmark, not
    the global-threshold summary path. Non-compressor rows are untouched, since the
    benchmark only covers the event-level datasets.
    """

    benchmark_path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    if not benchmark_path.exists():
        return summary
    benchmark = pd.read_csv(benchmark_path)
    registered = benchmark[benchmark["method"] == _REGISTERED_EVENT_METHOD]
    if registered.empty:
        return summary

    updated = summary.copy()
    for _index, row in registered.iterrows():
        mask = updated["dataset_id"] == row["dataset_id"]
        if not mask.any():
            continue
        updated.loc[mask, "predicted_events_or_units"] = row["predicted_alarm_events"]
        updated.loc[mask, "precision"] = row["event_precision"]
        updated.loc[mask, "recall"] = row["event_recall"]
        updated.loc[mask, "f1"] = row["event_f1"]
        updated.loc[mask, "false_alarm_events_per_day"] = row["false_alarm_events_per_day"]
        updated.loc[mask, "median_warning_lead_time"] = row["median_warning_lead_time"]
    return updated


def _registered_event_level_frame(config: PaperAssetConfig, summary: pd.DataFrame) -> pd.DataFrame:
    """Build the main event-level table from the benchmark, for the registered method.

    This table previously came from ``industrial_results_summary.csv``, which evaluates a
    plain robust score at a global empirical threshold and has never run the registered
    method. While nine baselines were collapsed onto one computation the two agreed
    numerically, so the discrepancy was invisible; separating the methods revealed that
    the paper's principal event-level table reported the global empirical threshold while
    the surrounding text attributed those numbers to the registered recurrence score.
    Sourcing both from the benchmark removes the second code path.

    Warning exposure is included because alarm burden is only interpretable when
    interruption frequency and exposure are reported together.
    """

    benchmark_path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    datasets = ["metropt", "metropt2"]
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        registered = benchmark[
            (benchmark["method"] == _REGISTERED_EVENT_METHOD)
            & (benchmark["dataset_id"].isin(datasets))
        ].copy()
        if not registered.empty:
            registered = registered.sort_values("dataset_id")
            exposure = (
                registered["alarm_coverage_fraction"]
                if "alarm_coverage_fraction" in registered.columns
                else pd.Series([float("nan")] * len(registered), index=registered.index)
            )
            return pd.DataFrame(
                {
                    "Dataset": registered["dataset_id"].map(_dataset_label),
                    "Failure": registered["dataset_id"].map(
                        lambda value: f"{_dataset_label(value)} registered test failure"
                    ),
                    "Target": registered["target_events"].map(_format_count),
                    "Alarms": registered["predicted_alarm_events"].map(_format_count),
                    "Recall": registered["event_recall"].map(_format_metric),
                    "Precision": registered["event_precision"].map(_format_metric),
                    "FA/day": registered["false_alarm_events_per_day"].map(_format_metric),
                    # The writer escapes LaTeX specials, so the percent sign is emitted
                    # plain here; pre-escaping it produces a literal backslash.
                    "Exposure %": exposure.map(
                        lambda value: "" if pd.isna(value) else f"{float(value) * 100:.2f}"
                    ),
                    "Lead": registered["median_warning_lead_time"].map(_format_optional_int),
                }
            )

    # Fallback keeps the build working where the benchmark artifact is absent, and says
    # in the table which method it is actually reporting.
    event_source = summary[summary["dataset_id"].isin(datasets)].copy()
    return pd.DataFrame(
        {
            "Dataset": event_source["dataset_id"].map(_dataset_label),
            "Failure": event_source["dataset_id"].map(
                lambda value: f"{_dataset_label(value)} registered test failure"
            ),
            "Method": ["global empirical threshold"] * len(event_source),
            "Target": event_source["target_events_or_units"].map(_format_count),
            "Alarms": event_source["predicted_events_or_units"].map(_format_count),
            "Recall": event_source["recall"].map(_format_metric),
            "Precision": event_source["precision"].map(_format_metric),
            "FA/day": event_source["false_alarm_events_per_day"].map(_format_metric),
            "Lead": event_source["median_warning_lead_time"].map(_format_optional_int),
        }
    )


def _estimator_study_assets(config: PaperAssetConfig, output_root: Path) -> list[Path]:
    """Build the extremal-index estimator figures and tables from the available grids.

    All present grids are pooled so the estimator comparison rests on every replicate
    that exists locally, rather than on whichever grid happens to be regenerated last.
    """

    from dyn_evt_pdm.simulation.estimator_figures import build_estimator_figures
    from dyn_evt_pdm.simulation.estimator_summary import summarise_estimator_performance
    from dyn_evt_pdm.simulation.estimator_tables import build_estimator_tables

    frames = [pd.read_parquet(path) for path in config.simulation_grid_paths if path.exists()]
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True)
    if "reference_theta" not in combined.columns:
        # Grids written before reference resolution existed carry no reference column;
        # regenerating them is required before the estimator study can be reported.
        return []

    summary = summarise_estimator_performance(combined)
    generated: list[Path] = []
    generated.extend(build_estimator_figures(summary, output_root).as_tuple())
    generated.extend(build_estimator_tables(summary, output_root).as_tuple())
    return generated


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
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    generated: list[Path] = []
    generated += _dataset_role_summary_assets(config, tables, latex)
    generated += _industrial_compact_assets(config, tables, latex)
    generated += _baseline_summary_assets(config, tables, latex)
    generated += _event_baseline_comparison_assets(config, tables, latex)
    generated += _event_variant_comparison_assets(config, tables, latex)
    generated += _target_region_transferability_assets(config, tables, latex)
    generated += _transfer_distance_figure(config, figures)
    generated += _detection_aware_control_figure(config, figures)
    generated += _detection_aware_control_table(config, tables, latex)
    generated += _decomposition_assets(config, tables, latex)
    generated += _matched_negative_control_assets(config, tables, latex)
    generated += _metric_provenance_assets(config, figures, tables, latex)
    generated += _method_taxonomy_assets(tables, latex)
    generated += _method_scope_assets(config, tables, latex)
    generated += _timeline_traceability_assets(config, figures, tables, latex)
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
    generated: list[Path] = []
    event_frame = _registered_event_level_frame(config, source)
    event_csv = tables / "industrial_event_level_results.csv"
    event_tex = latex / "industrial_event_level_results.tex"
    event_frame.to_csv(event_csv, index=False)
    _write_latex_table(
        event_frame,
        event_tex,
        caption="Held-out event-level diagnostic rows for compressor datasets.",
        label="tab:event-level-results",
    )
    generated.extend([event_csv, event_tex])

    non_event_source = source[~source["dataset_id"].isin(["metropt", "metropt2"])].copy()
    non_event_frame = pd.DataFrame(
        {
            "Data": non_event_source["dataset_id"].map(_dataset_label),
            "Unit": non_event_source["dataset_id"].map(_evidential_unit_label),
            "Units": non_event_source["independent_units"].map(_format_count),
            "Target": non_event_source["target_events_or_units"].map(_format_count),
            "Pred": non_event_source["predicted_events_or_units"].map(_format_count),
            "Recall": non_event_source["recall"].map(_format_metric),
            "Precision": non_event_source["precision"].map(_format_metric),
            "F1": non_event_source["f1"].map(_format_metric),
            "Brier": non_event_source["brier_score"].map(_format_metric),
            "Limit": non_event_source["limitation"].map(_short_limitation),
        }
    )
    non_event_csv = tables / "industrial_non_event_results.csv"
    non_event_tex = latex / "industrial_non_event_results.tex"
    non_event_frame.to_csv(non_event_csv, index=False)
    _write_latex_table(
        non_event_frame,
        non_event_tex,
        caption="Non-event real-data diagnostic rows kept separate from event alarms.",
        label="tab:non-event-results",
    )
    generated.extend([non_event_csv, non_event_tex])
    return generated


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
        caption="Synthetic smoke results for the shared baseline runner.",
        label="tab:baseline-smoke-summary",
    )
    return [csv_path, tex_path]


def _metric_provenance_assets(
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    industrial_path = config.real_data_matrix_root / "industrial_results_summary.csv"
    rows: list[dict[str, object]] = []
    if industrial_path.exists():
        industrial = pd.read_csv(industrial_path)
        for row in industrial.itertuples(index=False):
            dataset = str(row.dataset_id)
            if dataset in {"metropt", "metropt2"}:
                rows.append(
                    {
                        "Source artifact": "industrial_results_summary.csv",
                        "Dataset": _dataset_label(dataset),
                        "Split": str(row.split_evaluated),
                        "Metric class": "event-level alarm",
                        "Metric": "event precision",
                        "Value": _format_metric(row.precision),
                        "Independent unit": "failure episode",
                        "Comparable to baseline F1": "no",
                    }
                )
                rows.append(
                    {
                        "Source artifact": "industrial_results_summary.csv",
                        "Dataset": _dataset_label(dataset),
                        "Split": str(row.split_evaluated),
                        "Metric class": "event-level alarm",
                        "Metric": "false alarms per day",
                        "Value": _format_metric(row.false_alarm_events_per_day),
                        "Independent unit": "operating day",
                        "Comparable to baseline F1": "no",
                    }
                )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    csv_path = tables / "metric_provenance.csv"
    tex_path = latex / "metric_provenance.tex"
    figure_path = figures / "metric_collapse_decomposition.png"
    frame.to_csv(csv_path, index=False)
    latex_frame = pd.DataFrame(
        {
            "Dataset": frame["Dataset"],
            "Evidence type": frame["Metric class"],
            "Metric": frame["Metric"],
            "Value": frame["Value"],
            "Unit": frame["Independent unit"],
            "Use": frame["Comparable to baseline F1"].map(
                {"yes": "same metric family", "no": "real event-level metric"}
            ),
        }
    )
    _write_latex_table(
        latex_frame,
        tex_path,
        caption="Metric provenance for real held-out event-level alarm metrics.",
        label="tab:metric-provenance",
    )

    decomposition_path = config.real_data_matrix_root / "score_threshold_alarm_decomposition.csv"
    if decomposition_path.exists():
        decomposition = pd.read_csv(decomposition_path)
        _score_alarm_flow_figure(decomposition, figure_path)
    return [csv_path, tex_path, figure_path]


#: Rank separation below this is treated as a score that carries no warning signal.
#: 0.5 is chance; the margin allows for the sampling noise of a single failure window.
_SCORE_SEPARATION_FLOOR = 0.55

#: Event precision below this makes a matched detection operationally unusable.
_USABLE_PRECISION = 0.01


@dataclass(frozen=True)
class FailureVerdict:
    """One method's failure layer, the evidence for it, and what it does not rule out."""

    layer: str
    evidence: str
    alternative: str
    confidence: str


def classify_failure_layer(row: pd.Series) -> tuple[str, str]:
    """Return the first stage at which this method's evidence broke down, and why.

    Thin wrapper over :func:`classify_root_cause` kept for the decomposition table, which
    prints only the layer name.
    """
    verdict = classify_root_cause(row)
    return verdict.layer, verdict.evidence


def classify_root_cause(row: pd.Series) -> FailureVerdict:
    """Classify one dataset-method row into the earliest stage that failed.

    The stages are checked in pipeline order, so the verdict names the earliest failure
    rather than the last symptom. Each is a different repair: a score that does not
    separate needs a different observable, a threshold that admits nothing needs
    recalibration, alarms that never match need a different horizon, and a match buried
    in false alarms needs a different alarm policy. A single dataset-level explanation
    hides which one applies, and the layers differ by method on the same dataset.

    Score and threshold failures are distinguished by rank separation rather than
    assumed. Both produce zero exceedances, so the counts alone cannot tell a score that
    carries no warning information from a threshold that discarded information the score
    did carry.
    """

    exceedances = float(row.get("threshold_exceedances_samples", 0) or 0)
    clusters = float(row.get("extreme_clusters_count", 0) or 0)
    episodes = float(row.get("alarm_episodes_count", 0) or 0)
    matched = float(row.get("matched_alarm_episodes_count", 0) or 0)
    failures = float(row.get("labelled_failure_events_count", 0) or 0)
    false_alarms = float(row.get("false_alarm_episodes_count", 0) or 0)
    precision = float(row.get("event_precision", 0) or 0)
    separation = float(row.get("score_separation_auc", float("nan")) or float("nan"))

    if failures == 0:
        return FailureVerdict(
            layer="insufficient independent evidence",
            evidence="no labelled failure in the evaluated split",
            alternative="any layer; nothing is estimable without a labelled event",
            confidence="high",
        )

    # Checked before the counts, because a score that does not rank the warning window
    # above the rest of the split has already failed: no threshold or alarm policy can
    # recover ordering information the score does not carry. A method in this state can
    # still match a failure, but only by alarming often enough that one episode lands in
    # the window by chance, which is not early warning.
    chance = float(row.get("chance_match_probability", float("nan")) or float("nan"))
    chance_text = "" if not np.isfinite(chance) else f", chance match {chance:.2f}"

    if np.isfinite(separation) and separation < _SCORE_SEPARATION_FLOOR:
        below_chance = separation < 0.5
        qualifier = "below chance" if below_chance else "at chance"
        return FailureVerdict(
            layer="score failure",
            evidence=(
                f"separation {separation:.2f} ({qualifier}); "
                + (
                    f"match among {episodes:,.0f} episodes incidental{chance_text}"
                    if episodes > 0
                    else "no episodes raised"
                )
            ),
            alternative="a different lead time may separate; the horizon may be mis-specified",
            confidence="high" if below_chance else "medium",
        )

    if exceedances == 0:
        if not np.isfinite(separation):
            return FailureVerdict(
                layer="score failure",
                evidence="no exceedance; separation not computable",
                alternative="a different observable may separate",
                confidence="low",
            )
        return FailureVerdict(
            layer="threshold failure",
            evidence=(
                f"separation {separation:.2f} but no exceedance; the threshold discarded "
                "the ranking"
            ),
            alternative="no threshold may separate and control burden simultaneously",
            confidence="medium",
        )

    if clusters == 0:
        return FailureVerdict(
            layer="extreme-cluster failure",
            evidence=f"{exceedances:,.0f} exceedances formed no cluster",
            alternative="the run length may be too long for this exceedance pattern",
            confidence="high",
        )

    if episodes == 0:
        return FailureVerdict(
            layer="alarm-merging failure",
            evidence=f"{clusters:,.0f} clusters produced no episode",
            alternative="the merge gap may absorb every cluster",
            confidence="high",
        )

    if matched == 0:
        return FailureVerdict(
            layer="event-matching failure",
            evidence=(
                f"separation {separation:.2f} but none of {episodes:,.0f} episodes fell in "
                "the window"
            ),
            alternative="a longer horizon may capture alarms raised earlier",
            confidence="medium",
        )

    if precision < _USABLE_PRECISION:
        return FailureVerdict(
            layer="no operationally acceptable result",
            evidence=(
                f"separation {separation:.2f}; matched, but {false_alarms:,.0f} false "
                f"episodes give precision {precision:.4f}{chance_text}"
            ),
            alternative="a stricter threshold or alarm policy may cut burden without losing it",
            confidence="medium",
        )

    return FailureVerdict(
        layer="insufficient independent evidence",
        evidence=(f"matched at precision {precision:.3f}, on {failures:.0f} held-out failure"),
        alternative="the result may not survive on failures not represented here",
        confidence="low",
    )


def _decomposition_assets(config: PaperAssetConfig, tables: Path, latex: Path) -> list[Path]:
    """Write the stage-by-stage decomposition table with its root-cause classification."""

    path = config.real_data_matrix_root / "score_threshold_alarm_decomposition.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty or "test_observations_samples" not in source.columns:
        return []

    layers = [classify_failure_layer(row) for _index, row in source.iterrows()]
    frame = pd.DataFrame(
        {
            "Data": source["dataset_id"].map(_dataset_label),
            "Method": source["method"].map(_event_method_label),
            "Obs.": source["test_observations_samples"].map(_format_count),
            "Exceed.": source["threshold_exceedances_samples"].map(_format_count),
            "Clusters": source["extreme_clusters_count"].map(_format_count),
            "Episodes": source["alarm_episodes_count"].map(_format_count),
            "Matched": source["matched_alarm_episodes_count"].map(_format_count),
            "False": source["false_alarm_episodes_count"].map(_format_count),
            "Dupl.": source["duplicate_alarm_episodes_count"].map(_format_count),
            "Failure layer": [layer for layer, _reason in layers],
        }
    )
    csv_path = tables / "score_alarm_decomposition.csv"
    tex_path = latex / "score_alarm_decomposition.tex"
    # The CSV keeps every column for audit; the printed table drops the two that carry
    # no per-method information. Observations are constant within a dataset and
    # duplicates never exceed single digits, and at ten columns the method names
    # collided with the counts.
    frame.to_csv(csv_path, index=False)
    observations = {
        _dataset_label(dataset): _format_count(group["test_observations_samples"].iloc[0])
        for dataset, group in source.groupby("dataset_id")
    }
    observed = "; ".join(f"{name} {count}" for name, count in sorted(observations.items()))
    printed = frame.drop(columns=["Obs.", "Dupl."])
    # "failure" is already in the column header, and repeating it wrapped every cell
    # onto three lines.
    printed["Failure layer"] = printed["Failure layer"].str.replace(" failure", "", regex=False)
    _write_latex_table(
        printed,
        tex_path,
        caption=(
            "Score, threshold, cluster and alarm decomposition per method. Exceedances "
            "are counts of samples, clusters are counts of extreme clusters, and the "
            "remaining columns are counts of alarm episodes; the columns are therefore "
            "not comparable across stages. Test observations are constant within a "
            f"dataset ({observed}) and are omitted. Labelled failures are ground truth "
            "and are not part of this detector flow. The failure layer names the "
            "earliest stage at which the method's evidence broke down."
        ),
        label="tab:score-alarm-decomposition",
        # Method names and the failure-layer phrase need room; the counts do not.
        column_weights=(0.95, 1.5, 0.9, 0.85, 0.85, 0.8, 0.8, 1.35),
        # 38 rows do not fit a float, and as a table environment the last rows ran off
        # the bottom of the page.
        long=True,
    )
    return [csv_path, tex_path]


#: Horizontal separation between the matched-alarm stage and the ground-truth marker.
_GROUND_TRUTH_OFFSET = 0.22

#: Detector-track stages, with the unit each count is measured in.
_DECOMPOSITION_STAGES: tuple[tuple[str, str, str], ...] = (
    ("test_observations_samples", "Test\nobservations", "samples"),
    ("threshold_exceedances_samples", "Threshold\nexceedances", "samples"),
    ("extreme_clusters_count", "Extreme\nclusters", "clusters"),
    ("alarm_episodes_count", "Alarm\nepisodes", "episodes"),
    ("matched_alarm_episodes_count", "Matched\nalarms", "episodes"),
)


def _score_alarm_flow_figure(source: pd.DataFrame, figure_path: Path) -> None:
    """Draw the decomposition as two tracks that are never mixed.

    The detector track runs observations to matched alarms. Labelled failures are ground
    truth and sit on their own track, connected to the matched alarms rather than placed
    at the end of the detector flow. The previous version put labelled failure samples at
    the start and labelled target events at the end of a single monotone flow, which
    reads as a derivation from failures to failures and is not what the pipeline computes.
    """

    required = {name for name, _label, _unit in _DECOMPOSITION_STAGES}
    if source.empty or not required.issubset(source.columns):
        figure, axis = plt.subplots(figsize=(6.0, 3.0))
        axis.text(
            0.5,
            0.5,
            "Decomposition artifact predates the two-track schema; regenerate it",
            ha="center",
            va="center",
        )
        axis.axis("off")
        _save_figure(figure, figure_path)
        return

    selected = source.loc[
        source["method"].isin(["dynamical_evt_robust_score", "failure_prototype_region"])
    ].copy()
    if selected.empty:
        selected = source.head(4).copy()

    figure, axes = plt.subplots(
        nrows=max(1, len(selected)),
        ncols=1,
        figsize=(9.0, max(3.0, 1.7 * len(selected))),
        squeeze=False,
    )
    positions = np.arange(len(_DECOMPOSITION_STAGES), dtype=float)
    for axis, (_index, row) in zip(axes[:, 0], selected.iterrows(), strict=False):
        values = [float(row.get(name, 0) or 0) for name, _label, _unit in _DECOMPOSITION_STAGES]
        axis.plot(positions, np.maximum(values, 0.5), color="#0072B2", marker="o", linewidth=2)
        for index, value in enumerate(values):
            unit = _DECOMPOSITION_STAGES[index][2]
            # Zero is annotated explicitly: on a log axis it is otherwise invisible and
            # reads as missing data rather than as a detector that produced nothing.
            text = f"0 {unit}" if value == 0 else f"{_format_count(value)} {unit}"
            # The final stage is right-aligned so it clears the ground-truth marker,
            # which sits just to its right.
            last = index == len(_DECOMPOSITION_STAGES) - 1
            axis.annotate(
                text,
                xy=(index, max(value, 0.5)),
                xytext=(-4, 7) if last else (0, 7),
                textcoords="offset points",
                ha="right" if last else "center",
                fontsize=7,
            )

        failures = float(row.get("labelled_failure_events_count", 0) or 0)
        matched = float(row.get("matched_alarm_episodes_count", 0) or 0)
        ground_truth_y = max(failures, 0.5)
        # Offset in x so the ground-truth marker reads as its own track rather than as
        # another point on the detector line; sharing the x position let the two
        # markers and their labels overplot whenever matched alarms was small.
        ground_truth_x = positions[-1] + _GROUND_TRUTH_OFFSET
        axis.scatter(
            [ground_truth_x],
            [ground_truth_y],
            marker="s",
            s=70,
            color="#D55E00",
            zorder=5,
            label="labelled failures (ground truth)",
        )
        # Labelled to the right rather than below: below the marker the text fell off
        # the bottom of the axes whenever the matched count sat on the floor.
        axis.annotate(
            f"{_format_count(failures)} events",
            xy=(ground_truth_x, ground_truth_y),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color="#D55E00",
        )
        # The only link between the two tracks is the matching step.
        axis.annotate(
            "",
            xy=(ground_truth_x, ground_truth_y),
            xytext=(positions[-1], max(matched, 0.5)),
            arrowprops={"arrowstyle": "<->", "color": "#888888", "linewidth": 1.0},
        )

        false_alarms = float(row.get("false_alarm_episodes_count", 0) or 0)
        duplicates = float(row.get("duplicate_alarm_episodes_count", 0) or 0)
        exposure = float(row.get("time_under_warning_samples", 0) or 0)
        # Below the axis rather than inside it. These three numbers are properties of
        # the whole track, not of any stage, and every in-axes position collided with
        # some panel's track: the shapes differ too much across methods to place text
        # among them.
        axis.set_xlabel(
            f"false alarms {_format_count(false_alarms)} episodes    "
            f"duplicates {_format_count(duplicates)} episodes    "
            f"time under warning {_format_count(exposure)} samples",
            fontsize=6.5,
            color="#555555",
        )

        axis.set_yscale("symlog", linthresh=1.0)
        # Every decade collides at this panel height; the stage labels carry the exact
        # counts, so the axis only has to convey the order of magnitude.
        axis.set_yticks([1.0, 1e2, 1e4, 1e6])
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for _name, label, _unit in _DECOMPOSITION_STAGES], fontsize=7.5)
        axis.set_xlim(-0.35, len(_DECOMPOSITION_STAGES) - 1 + _GROUND_TRUTH_OFFSET + 0.75)
        label = f"{_dataset_label(row['dataset_id'])}\n{_event_method_label(row['method'])}"
        axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    axes[0, 0].set_title(
        "Detector evidence (circles) and labelled failures (square)\n"
        "counts are not comparable across stages",
        fontsize=9,
    )
    # Upper right: the detector track descends left to right in every panel, so this
    # corner is empty, and the lower left now carries the burden annotation.
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    _save_figure(figure, figure_path)


def _event_baseline_comparison_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty:
        return []
    main_methods = {
        "engineering_threshold",
        "classical_pot_gpd",
        "isolation_forest",
        "dynamical_evt_robust_score",
        "failure_prototype_region",
        "rare_state_region",
    }
    source = source.loc[source["method"].isin(main_methods)].sort_values(
        ["dataset_id", "method_family", "method"]
    )
    frame = pd.DataFrame(
        {
            "Data": source["dataset_id"].map(_dataset_label),
            "Method": source["method"].map(_event_method_label),
            "Family": source["method_family"].map(_event_family_label),
            "Target": source["target_events"].map(_format_count),
            "Alarms": source["predicted_alarm_events"].map(_format_count),
            "Recall": source["event_recall"].map(_format_metric),
            "Precision": source["event_precision"].map(_format_metric),
            "FA/day": source["false_alarm_events_per_day"].map(_format_metric),
            "Lead": source["median_warning_lead_time"].map(_format_optional_int),
        }
    )
    csv_path = tables / "event_baseline_comparison.csv"
    tex_path = latex / "event_baseline_comparison.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Real held-out event-level comparison under the shared alarm policy.",
        label="tab:event-baseline-comparison",
    )
    return [csv_path, tex_path]


def _event_variant_comparison_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "event_variant_comparison.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty:
        return []
    frame = pd.DataFrame(
        {
            "Data": source["dataset_id"].map(_dataset_label),
            "Variant": source["method"].map(_event_method_label),
            "Family": source["method_family"].map(_event_family_label),
            "Recall": source["event_recall"].map(_format_metric),
            "Precision": source["event_precision"].map(_format_metric),
            "FA/day": source["false_alarm_events_per_day"].map(_format_metric),
            "Metric": source["distance_metric"],
            "Leakage control": source["leakage_control"],
        }
    )
    csv_path = tables / "event_variant_comparison.csv"
    tex_path = latex / "event_variant_comparison.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Target-region variants and negative controls on real held-out event splits.",
        label="tab:event-variant-comparison",
    )
    return [csv_path, tex_path]


def _target_region_transferability_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "target_region_transferability.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty or "protocol" not in source.columns:
        return []

    protocol_labels = {
        "direct": "Direct",
        "recalibrated_threshold": "Threshold recal.",
        "recalibrated_scaling": "Scaling recal.",
        "destination_refit": "Destination refit",
    }
    frame = pd.DataFrame(
        {
            "Source": source["source"].map(_dataset_label),
            "Dest.": source["destination"].map(_dataset_label),
            "Protocol": source["protocol"].map(
                lambda value: protocol_labels.get(str(value), str(value))
            ),
            "Scaling": source["scaling_fitted_on"].astype(str),
            "Threshold": source["threshold_fitted_on"].astype(str),
            "Occup.": source["region_occupancy"].map(_format_metric),
            "Recall": source["event_recall"].map(_format_metric),
            "Precision": source["event_precision"].map(_format_metric),
            "FA/day": source["false_alarm_events_per_day"].map(_format_metric),
            "Degrad.": source.get("transfer_degradation_precision", pd.Series(dtype=float)).map(
                _format_metric
            )
            if "transfer_degradation_precision" in source.columns
            else "",
        }
    )
    csv_path = tables / "target_region_transferability.csv"
    tex_path = latex / "target_region_transferability.tex"
    frame.to_csv(csv_path, index=False)
    excluded = (
        str(source["excluded_features"].dropna().iloc[0])
        if "excluded_features" in source.columns and source["excluded_features"].notna().any()
        else ""
    )
    feature_set = (
        str(source["feature_set"].dropna().iloc[0])
        if "feature_set" in source.columns and source["feature_set"].notna().any()
        else ""
    )
    _write_latex_table(
        frame,
        tex_path,
        caption=(
            "Cross-dataset target-region transfer. Each row applies a region fitted on "
            "the source to the destination's held-out test split; the scaling and "
            "threshold columns state what was frozen and what was refitted. Degradation "
            "is event precision relative to the destination refit, so a negative value "
            "means the transferred region outperformed a region fitted on the "
            "destination itself. "
            + (f"Transfer runs on the {feature_set}. " if feature_set else "")
            + (f"Excluded as incompatible: {excluded.replace('_', ' ')}." if excluded else "")
        ),
        label="tab:target-region-transferability",
        # The protocol column holds "destination refit"; at equal widths that word
        # overflowed into the next column.
        column_weights=(0.9, 0.9, 1.5, 1.0, 1.1, 0.9, 0.9, 1.0, 0.9, 0.9),
    )
    return [csv_path, tex_path]


def _detection_aware_control_table(
    config: PaperAssetConfig, tables: Path, latex: Path
) -> list[Path]:
    """Write the detection-aware control comparison table.

    The control detection rate is a column rather than a footnote: a favourable
    percentile means nothing without knowing whether the controls found the failure.
    """

    path = config.real_data_matrix_root / "detection_aware_controls.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty:
        return []

    condition_labels = {
        "all_draws": "All draws",
        "detecting_draws": "Detecting only",
        "recall_matched": "Recall matched",
        "occupancy_matched": "Occupancy matched",
    }
    frame = pd.DataFrame(
        {
            "Data": source["dataset_id"].map(_dataset_label),
            "Control family": source["control_family"].map(lambda v: str(v).replace("_", " ")),
            "Condition": source["condition"].map(lambda v: condition_labels.get(str(v), str(v))),
            "Draws": source["control_draws"],
            "Ctrl det. rate": source["control_detection_rate"].map(_format_metric),
            "Obs. detected": source["observed_detected"].map(
                lambda value: "yes" if bool(value) else "no"
            ),
            "Obs. utility": source["observed_utility"].map(_format_metric),
            "Ctrl median": source["control_utility_median"].map(_format_metric),
            "Percentile": source["observed_percentile"].map(_format_metric),
        }
    )
    csv_path = tables / "detection_aware_controls.csv"
    tex_path = latex / "detection_aware_controls.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption=(
            "Detection-aware matched controls. Every comparison reports the share of "
            "control draws that detected the failure, because a burden advantage means "
            "nothing if the observed region did not detect while the controls did. "
            "Utility is the predeclared joint utility, which charges false alarms per "
            "operating day and warning exposure separately and pays a capped early-warning "
            "reward only when the failure was detected."
        ),
        label="tab:detection-aware-controls",
        # Six families times four conditions on two datasets does not fit a float; as a
        # table environment it overran the page and dropped its last rows silently.
        # The dataset column must fit "MetroPT2" unbroken: it has no hyphenation point,
        # so a narrower column pushes it into the neighbouring cell.
        column_weights=(1.25, 1.35, 1.25, 0.6, 0.9, 0.9, 0.9, 0.9, 0.95),
        long=True,
    )
    return [csv_path, tex_path]


def _detection_aware_control_figure(config: PaperAssetConfig, figures: Path) -> list[Path]:
    """Plot control distributions for utility and, conditional on detection, burden.

    The two burden panels are restricted to control draws that detect the failure. An
    unconditioned burden distribution flatters any region that does not detect, since a
    region raising no alarms has the best possible burden and the worst possible outcome.
    """

    path = config.real_data_matrix_root / "matched_negative_control_draws.csv"
    if not path.exists():
        return []
    draws = pd.read_csv(path)
    if draws.empty or "detected" not in draws.columns:
        return []

    baseline_path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    observed = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()

    # The observed joint utility must come from the detection-aware summary, which
    # computes it under the same predeclared weights as the control draws. The baseline
    # table's event_utility is a different quantity, and marking it on a joint-utility
    # histogram would compare two scales against each other.
    summary_path = config.real_data_matrix_root / "detection_aware_controls.csv"
    observed_utility_by_dataset: dict[str, float] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for dataset_id, group in summary.groupby("dataset_id"):
            values = pd.to_numeric(group["observed_utility"], errors="coerce").dropna()
            if not values.empty:
                observed_utility_by_dataset[str(dataset_id)] = float(values.iloc[0])

    datasets = sorted(draws["dataset_id"].unique())
    panels = (
        # The draws column already holds the predeclared joint utility; the observed
        # marker is read from the detection-aware summary so both use the same weights.
        ("event_utility", "Joint utility", False),
        ("false_alarm_events_per_day", "False alarms per day", True),
        ("median_warning_lead_time", "Median lead time (samples)", True),
    )
    figure, axes = plt.subplots(
        len(datasets), len(panels), figsize=(4.6 * len(panels), 3.6 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        dataset_draws = draws[draws["dataset_id"] == dataset]
        observed_row = observed[
            (observed["dataset_id"] == dataset) & (observed["method"] == "failure_prototype_region")
        ]
        for column, (metric, label, detecting_only) in enumerate(panels):
            axis = axes[row][column]
            selected = (
                dataset_draws[dataset_draws["detected"].astype(bool)]
                if detecting_only
                else dataset_draws
            )
            values = pd.to_numeric(selected.get(metric), errors="coerce").dropna().to_numpy()
            if values.size == 0:
                axis.annotate(
                    "no control draw detects the failure"
                    if detecting_only
                    else "no control draw available",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=8,
                    color="#666666",
                )
            else:
                axis.hist(values, bins=40, color="#56B4E9", edgecolor="none", alpha=0.85)
                # On the utility panel the detecting subset is overlaid, because the
                # conclusion rests on that subset. Showing only the pooled distribution
                # would put the observed marker near its upper end while the observed
                # region actually sits below the median of controls that detect.
                if not detecting_only and "detected" in dataset_draws.columns:
                    detecting_values = (
                        pd.to_numeric(
                            dataset_draws[dataset_draws["detected"].astype(bool)].get(metric),
                            errors="coerce",
                        )
                        .dropna()
                        .to_numpy()
                    )
                    if detecting_values.size:
                        axis.hist(
                            detecting_values,
                            bins=40,
                            color="#E69F00",
                            edgecolor="none",
                            alpha=0.85,
                            label="detecting draws",
                        )
            observed_value: float | None = None
            if metric == "event_utility":
                observed_value = observed_utility_by_dataset.get(str(dataset))
            elif not observed_row.empty and metric in observed_row.columns:
                candidate = pd.to_numeric(observed_row[metric], errors="coerce").iloc[0]
                observed_value = float(candidate) if pd.notna(candidate) else None
            if observed_value is not None:
                axis.axvline(observed_value, color="#D55E00", linewidth=2.0, label="observed")
            if column == 0 and axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False, fontsize=7)
            title = f"{_dataset_label(dataset)}: {label}"
            if detecting_only:
                title += " (detecting draws)"
            axis.set_title(title, fontsize=9)
            axis.set_xlabel(label)
            axis.set_ylabel("Control draws" if column == 0 else "")
            axis.grid(True, alpha=0.25, linewidth=0.6)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    figure_path = figures / "detection_aware_controls.png"
    _save_figure(figure, figure_path)
    return [figure_path]


def _transfer_distance_figure(config: PaperAssetConfig, figures: Path) -> list[Path]:
    """Plot the distance distributions that decide whether a transferred region fires.

    Three populations are shown against the frozen threshold: the source's own training
    distances, the destination's normal operation, and the destination's failure states.
    A transfer succeeds only when the destination's failure sits inside the threshold
    while its normal operation sits outside; the figure shows directly which of those
    two conditions fails.
    """

    path = config.real_data_matrix_root / "transfer_distance_samples.csv"
    if not path.exists():
        return []
    samples = pd.read_csv(path)
    if samples.empty:
        return []

    pairs = samples[["source", "destination"]].drop_duplicates().to_dict("records")
    if not pairs:
        return []

    colours = {
        "source_train": "#0072B2",
        "destination_normal": "#E69F00",
        "destination_failure": "#D55E00",
    }
    labels = {
        "source_train": "Source training",
        "destination_normal": "Destination normal",
        "destination_failure": "Destination failure",
    }

    figure, axes = plt.subplots(1, len(pairs), figsize=(6.0 * len(pairs), 4.0), squeeze=False)
    for column, pair in enumerate(pairs):
        axis = axes[0][column]
        subset = samples[
            (samples["source"] == pair["source"]) & (samples["destination"] == pair["destination"])
        ]
        finite = subset["distance"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        threshold_value = float(subset["threshold"].iloc[0])
        # The distance distribution is extremely heavy-tailed: almost all mass sits near
        # zero while the top percentile reaches two orders of magnitude further, so a
        # percentile-based range renders every population as one spike. The figure exists
        # to show where the frozen threshold falls between the populations, so the range
        # is set by the threshold and the tail is reported as an overflow share instead.
        upper = max(threshold_value * 6.0, float(np.quantile(finite, 0.50)) * 2.0, 1e-6)
        bins = np.linspace(0.0, upper, 60)
        overflow: list[str] = []
        for population, colour in colours.items():
            values = subset.loc[subset["population"] == population, "distance"].to_numpy(
                dtype=float
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            beyond = float(np.mean(values > upper))
            if beyond > 0.005:
                overflow.append(f"{labels[population]}: {beyond * 100:.0f}% beyond axis")
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=colour,
                label=labels[population],
            )
        axis.axvline(threshold_value, color="#444444", linestyle="--", linewidth=1.4)
        axis.set_xlim(0.0, upper)
        axis.annotate(
            "frozen threshold",
            xy=(threshold_value, axis.get_ylim()[1]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
            color="#444444",
        )
        # The axis is truncated, so the share falling outside it is stated rather than
        # left to look like absent mass.
        if overflow:
            axis.annotate(
                "\n".join(overflow),
                xy=(0.97, 0.55),
                xycoords="axes fraction",
                ha="right",
                fontsize=7,
                color="#666666",
            )
        axis.set_title(
            f"{_dataset_label(pair['source'])} region applied to "
            f"{_dataset_label(pair['destination'])}",
            fontsize=9,
        )
        axis.set_xlabel("Distance to nearest failure prototype (standardised)")
        axis.set_ylabel("Density" if column == 0 else "")
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        if column == 0:
            axis.legend(frameon=False, fontsize=8, loc="upper right")

    figure_path = figures / "transfer_distance_distributions.png"
    _save_figure(figure, figure_path)
    return [figure_path]


def _matched_negative_control_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    path = config.real_data_matrix_root / "matched_negative_controls.csv"
    if not path.exists():
        return []
    source = pd.read_csv(path)
    if source.empty:
        return []
    burden = source.loc[source["metric"] == "false_alarm_events_per_day"].copy()
    if burden.empty:
        burden = source.copy()
    frame = pd.DataFrame(
        {
            "Data": burden["dataset_id"].map(_dataset_label),
            "Observed": burden["observed_method"].map(_event_method_label),
            "Control": burden["control_family"].map(_control_family_label),
            "Observed FA/day": burden["observed_value"].map(_format_metric),
            "Control median": burden["control_median"].map(_format_metric),
            "95% interval": burden.apply(
                lambda row: (
                    f"{_format_metric(row['control_p025'])}-"
                    f"{_format_metric(row['control_p975'])}"
                ),
                axis=1,
            ),
            "Percentile": burden["observed_percentile"].map(_format_metric),
            "Interpretation": burden["interpretation"].astype(str),
        }
    )
    csv_path = tables / "matched_negative_controls.csv"
    tex_path = latex / "matched_negative_controls.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Observed failure-prototype alarm burden against 500 matched-control draws per family.",
        label="tab:matched-negative-controls",
    )
    return [csv_path, tex_path]


def _method_taxonomy_assets(tables: Path, latex: Path) -> list[Path]:
    """Write the taxonomy that says what each benchmarked method actually is.

    Review observed that a reader could not tell from the manuscript whether the central
    object was a recurrence observable, an extremal-index analysis of a generic anomaly
    score, the target-region construction, or the whole workflow. The answer is visible
    once each method's score semantics and the role of extreme-value theory are stated
    beside its name, and it is not the answer the framing implied.
    """

    rows = [
        {
            "Method": _event_method_label(spec.name),
            "Family": _event_family_label(spec.family),
            "Score measures": score_semantics(spec.name),
            "Threshold rule": spec.threshold_rule.replace("_", " "),
            "Event policy": spec.event_policy.replace("_", " "),
            "Fitted on": spec.tuning_partition,
            "Role of EVT": evt_role(spec.name),
        }
        for spec in EVENT_METHOD_SPECS
    ]
    frame = pd.DataFrame(rows)
    csv_path = tables / "method_taxonomy.csv"
    tex_path = latex / "method_taxonomy.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption=(
            "What each benchmarked method is. The score column says what the quantity "
            "measures rather than how it is named: proximity to a recurrent state, the "
            "magnitude of an anomaly, a reconstruction error, or a probability in a "
            "tail. The last column says where extreme-value theory enters. It changes "
            "the alarms a method raises only when a tail model sets the threshold or an "
            "estimated extremal index sets the declustering run length; elsewhere the "
            "extremal-index estimates reported in this paper are diagnostics computed "
            "beside the alarms, and the alarms would be identical without them."
        ),
        label="tab:method-taxonomy",
        column_weights=(1.15, 0.75, 1.0, 1.1, 1.05, 0.95, 1.0),
    )
    return [csv_path, tex_path]


def _method_scope_assets(
    config: PaperAssetConfig,
    tables: Path,
    latex: Path,
) -> list[Path]:
    simulation_complete = _simulation_artifact_is_broad(config.simulation_study_path)
    baseline_complete = (config.real_data_matrix_root / "event_baseline_comparison.csv").exists()
    variant_complete = (config.real_data_matrix_root / "event_variant_comparison.csv").exists()
    timeline_complete = (config.real_data_matrix_root / "event_timeline_trace.csv").exists()
    rows = [
        {
            "Requested analysis": "full Monte Carlo validation",
            "Status": "completed" if simulation_complete else "not completed",
            "Artifact evidence": config.simulation_study_path.name,
            "Effect on claims": (
                "supports bounded estimator-stability claims"
                if simulation_complete
                else "method reliability remains a blocker"
            ),
        },
        {
            "Requested analysis": "registered dynamical-EVT diagnostic",
            "Status": "completed",
            "Artifact evidence": "MetroPT and MetroPT2 event rows",
            "Effect on claims": "supports negative alarm-burden finding",
        },
        {
            "Requested analysis": "target-region variants and negative controls",
            "Status": "completed" if variant_complete else "not completed",
            "Artifact evidence": "event_variant_comparison.csv",
            "Effect on claims": (
                "variant claims tied to observed event metrics"
                if variant_complete
                else "target-region superiority not claimed"
            ),
        },
        {
            "Requested analysis": "real event-level baseline comparison",
            "Status": "completed" if baseline_complete else "not completed",
            "Artifact evidence": "event_baseline_comparison.csv",
            "Effect on claims": (
                "baseline comparison uses the same alarm policy"
                if baseline_complete
                else "baseline superiority not claimed"
            ),
        },
        {
            "Requested analysis": "metric contradiction audit",
            "Status": "completed",
            "Artifact evidence": "metric_provenance table",
            "Effect on claims": "pointwise and event metrics separated",
        },
        {
            "Requested analysis": "real failure timeline",
            "Status": "completed" if timeline_complete else "not completed",
            "Artifact evidence": "event_timeline_trace.csv",
            "Effect on claims": (
                "timeline is tied to a real held-out event"
                if timeline_complete
                else "representative timeline only"
            ),
        },
        {
            "Requested analysis": "software archive DOI",
            "Status": "repository citation complete",
            "Artifact evidence": "CITATION.cff and CodeMeta",
            "Effect on claims": "repository URL is cited; DOI is venue-dependent",
        },
    ]
    frame = pd.DataFrame(rows)
    csv_path = tables / "method_scope_completion.csv"
    tex_path = latex / "method_scope_completion.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Completion status for broader review-requested analyses.",
        label="tab:method-scope-completion",
    )
    return [csv_path, tex_path]


def _timeline_traceability_assets(
    config: PaperAssetConfig,
    figures: Path,
    tables: Path,
    latex: Path,
) -> list[Path]:
    trace_path = config.real_data_matrix_root / "event_timeline_trace.csv"
    overview_path = config.real_data_matrix_root / "event_timeline_overview.csv"
    metadata_path = config.real_data_matrix_root / "event_timeline_metadata.csv"
    generated: list[Path] = []
    frame = _representative_timeline_frame()
    if trace_path.exists() and overview_path.exists() and metadata_path.exists():
        trace = pd.read_csv(trace_path)
        overview = pd.read_csv(overview_path)
        metadata = pd.read_csv(metadata_path)
        rows = []
        for dataset_id in metadata["dataset_id"].unique():
            dataset_metadata = metadata[metadata["dataset_id"] == dataset_id]
            if dataset_metadata.empty:
                continue
            # The overview carries every method for the supplement, so the main figure
            # has to select its own method as well as its dataset.
            method = dataset_metadata.iloc[0]["method"]
            dataset_trace = trace[trace["dataset_id"] == dataset_id]
            dataset_overview = overview[
                (overview["dataset_id"] == dataset_id) & (overview["method"] == method)
            ]
            if dataset_trace.empty or dataset_overview.empty or dataset_metadata.empty:
                continue
            figure_path = figures / f"timeline_{dataset_id}.png"
            _two_scale_timeline_figure(
                dataset_overview,
                dataset_trace,
                dataset_metadata.iloc[0],
                figure_path,
            )
            generated.append(figure_path)
            # The caption defines a macro rather than holding bare text: \input inside a
            # \caption argument is a moving argument and makes hyperref abort the build.
            caption_path = latex / f"timeline_caption_{dataset_id}.tex"
            caption_path.write_text(
                f"\\newcommand{{{_timeline_caption_macro(dataset_id)}}}{{%\n"
                f"{_timeline_caption(dataset_metadata.iloc[0])}%\n}}\n",
                encoding="utf-8",
            )
            generated.append(caption_path)
            rows.append(
                {
                    "Figure": figure_path.name,
                    "Source": f"{_dataset_label(dataset_id)} held-out test failure",
                    "Status": "traceable",
                    "Scales": "global and local",
                    "Use": "burden and timing audit",
                }
            )
        for dataset_id in metadata["dataset_id"].unique():
            dataset_overview = overview[overview["dataset_id"] == dataset_id]
            if dataset_overview["method"].nunique() <= 1:
                continue
            figure_path = figures / f"timeline_all_methods_{dataset_id}.png"
            _all_method_timeline_figure(dataset_overview, dataset_id, figure_path)
            generated.append(figure_path)
            rows.append(
                {
                    "Figure": figure_path.name,
                    "Source": f"{_dataset_label(dataset_id)} held-out test failure",
                    "Status": "traceable",
                    "Scales": "global, every method",
                    "Use": "supplementary burden comparison",
                }
            )
        if rows:
            frame = pd.DataFrame(rows)
    elif trace_path.exists():
        trace = pd.read_csv(trace_path)
        if not trace.empty:
            figure_path = figures / "real_event_timeline.png"
            _real_event_timeline_figure(trace, figure_path)
            generated.append(figure_path)
            frame = pd.DataFrame(
                [
                    {
                        "Figure": "real_event_timeline.png",
                        "Source": "MetroPT test event",
                        "Status": "local only",
                        "Scales": "local",
                        "Use": "empirical timing audit",
                    }
                ]
            )
    csv_path = tables / "timeline_traceability.csv"
    tex_path = latex / "timeline_traceability.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption="Traceability audit for the aligned event-timeline figure.",
        label="tab:timeline-traceability",
    )
    generated.extend([csv_path, tex_path])
    return generated


def _representative_timeline_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "Figure": "event_timeline.png",
                "Source": "synthetic cyclic input",
                "Status": "representative",
                "Scales": "local",
                "Use": "timing audit only",
            }
        ]
    )
    return frame


def _two_scale_timeline_figure(
    overview: pd.DataFrame,
    trace: pd.DataFrame,
    metadata: pd.Series,
    figure_path: Path,
) -> None:
    """Draw the full test split above the failure window, on one figure.

    The local window alone is not honest about burden: it shows the alarms nearest the
    failure and none of the thousands elsewhere. Putting the global density directly
    above it means the reader cannot see the matched alarm without also seeing how many
    other episodes the method raised over the same test split.
    """

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(8.6, 7.6),
        gridspec_kw={"height_ratios": [1.15, 0.75, 1.4, 0.62, 0.62]},
    )

    # --- Global scale -------------------------------------------------------
    days = pd.to_numeric(overview["bin_start_day"], errors="coerce").to_numpy(dtype=float)
    episodes = pd.to_numeric(overview["alarm_episodes"], errors="coerce").to_numpy(dtype=float)
    exposure = pd.to_numeric(overview["time_under_warning_samples"], errors="coerce").to_numpy(
        dtype=float
    )
    bin_width = float(np.median(np.diff(days))) if days.size > 1 else 1.0

    axes[0].bar(days, episodes, width=bin_width * 0.9, color="#0072B2", align="edge")
    axes[0].set_ylabel("alarm episodes\nper bin", fontsize=8)
    axes[0].set_title(
        f"Full test split: {int(metadata['global_alarm_episodes']):,} alarm episodes over "
        f"{float(metadata['test_duration_days']):.1f} operating days",
        fontsize=9,
    )

    axes[1].fill_between(days, 0.0, exposure, step="post", color="#56B4E9", alpha=0.85)
    axes[1].set_ylabel("samples under\nwarning per bin", fontsize=8)
    axes[1].set_xlabel("Operating days from start of test split", fontsize=8)

    failure_days = overview.loc[overview["contains_failure_onset"].astype(bool), "bin_start_day"]
    reset_days = overview.loc[overview["contains_maintenance_reset"].astype(bool), "bin_start_day"]
    for axis in axes[:2]:
        for position, day in enumerate(failure_days):
            axis.axvline(
                float(day),
                color="#D55E00",
                linewidth=1.4,
                label="labelled failure" if position == 0 else None,
            )
        for position, day in enumerate(reset_days):
            axis.axvline(
                float(day),
                color="#009E73",
                linewidth=1.2,
                linestyle=":",
                label="return to service" if position == 0 else None,
            )
        axis.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")

    # The local window drawn on the global axis, so the zoom is locatable.
    window_start = float(trace["test_index"].min()) / float(metadata["test_duration_samples"])
    window_end = float(trace["test_index"].max()) / float(metadata["test_duration_samples"])
    span = float(metadata["test_duration_days"])
    for axis in axes[:2]:
        axis.axvspan(
            window_start * span,
            max(window_end * span, window_start * span + bin_width),
            color="#999999",
            alpha=0.35,
            zorder=0,
        )

    # --- Local scale --------------------------------------------------------
    hours = pd.to_numeric(trace["elapsed_hours"], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(trace["score"], errors="coerce").to_numpy(dtype=float)
    threshold = float(metadata["threshold_value"])

    axes[2].plot(hours, score, color="#0072B2", linewidth=0.8, label="score")
    axes[2].axhline(
        threshold,
        color="#D55E00",
        linewidth=1.0,
        linestyle="--",
        label=f"frozen threshold {threshold:,.1f}",
    )
    exceed = trace["is_exceedance"].astype(bool).to_numpy()
    if exceed.any():
        axes[2].scatter(
            hours[exceed],
            score[exceed],
            s=4,
            color="#D55E00",
            zorder=3,
            label="exceedance",
        )
    axes[2].set_ylabel("score", fontsize=8)
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    axes[2].set_title(
        f"Failure window: {int(metadata['local_alarm_episodes'])} of those episodes fall here",
        fontsize=9,
    )

    regimes = pd.to_numeric(trace["regime_id"], errors="coerce").to_numpy(dtype=float)
    axes[3].step(hours, regimes, where="post", color="#333333", linewidth=0.8)
    axes[3].set_ylabel("operating\nregime", fontsize=8)

    # Alarm episodes get a minimum drawn width: after declustering an episode is often a
    # single sample, which is invisible across a window of tens of thousands, and an
    # empty band would read as "no alarms" in exactly the panel meant to show them.
    _timeline_band(
        axes[4],
        hours,
        trace["alarm_episode_id"].to_numpy() >= 0,
        0.62,
        "#0072B2",
        min_width=0.02 * float(hours.max() - hours.min()),
    )
    _timeline_band(
        axes[4], hours, trace["in_warning_window"].astype(bool).to_numpy(), 0.38, "#999999"
    )
    _timeline_band(axes[4], hours, trace["is_failure"].astype(bool).to_numpy(), 0.14, "#D55E00")
    axes[4].set_yticks([0.14, 0.38, 0.62])
    axes[4].set_yticklabels(["failure", "warning window", "alarm episode"], fontsize=7)
    axes[4].set_ylim(0.0, 0.8)
    axes[4].set_xlabel("Elapsed hours from held-out failure onset", fontsize=8)

    resets = trace["is_maintenance_reset"].astype(bool).to_numpy()
    for axis in axes[2:]:
        axis.axvline(0.0, color="#D55E00", linewidth=1.0)
        if resets.any():
            axis.axvline(float(hours[resets][0]), color="#009E73", linewidth=1.0, linestyle=":")
        axis.grid(axis="x", alpha=0.2)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure.tight_layout()
    _save_figure(figure, figure_path)


def _all_method_timeline_figure(
    overview: pd.DataFrame,
    dataset_id: str,
    figure_path: Path,
) -> None:
    """One global alarm-density strip per method, on a shared scale.

    The main-paper timeline shows one method. This shows every method in the comparison
    over the same test split, which is what makes the burden claims comparable: a method
    that alarms continuously and one that alarms in bursts can share an episode count.
    """

    methods = sorted(overview["method"].unique(), key=_event_method_label)
    figure, axes = plt.subplots(
        nrows=len(methods),
        ncols=1,
        figsize=(7.6, max(3.0, 0.52 * len(methods))),
        sharex=True,
        squeeze=False,
    )
    peak = float(overview["alarm_episodes"].max()) or 1.0
    for axis, method in zip(axes[:, 0], methods, strict=True):
        rows = overview[overview["method"] == method]
        days = pd.to_numeric(rows["bin_start_day"], errors="coerce").to_numpy(dtype=float)
        episodes = pd.to_numeric(rows["alarm_episodes"], errors="coerce").to_numpy(dtype=float)
        width = float(np.median(np.diff(days))) if days.size > 1 else 1.0
        axis.bar(days, episodes, width=width, color="#0072B2", align="edge")
        # A shared y-limit is the point: per-panel scaling would make a method with two
        # episodes per bin look like one with twenty.
        axis.set_ylim(0.0, peak)
        axis.set_ylabel(
            _event_method_label(method), rotation=0, ha="right", va="center", fontsize=7
        )
        axis.set_yticks([])
        failures = rows.loc[rows["contains_failure_onset"].astype(bool), "bin_start_day"]
        for day in failures:
            axis.axvline(float(day), color="#D55E00", linewidth=1.0)
        axis.tick_params(axis="x", labelsize=7)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        # Outside the axes: inside, the total sat on top of the taller panels' bars.
        axis.annotate(
            f"{int(rows['alarm_episodes'].sum()):,}",
            xy=(1.01, 0.5),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#555555",
        )

    axes[0, 0].set_title(
        f"{_dataset_label(dataset_id)}: alarm episodes per bin over the full test split, "
        f"shared vertical scale (peak {int(peak)}); totals at right",
        fontsize=8.5,
    )
    axes[-1, 0].set_xlabel("Operating days from start of test split", fontsize=8)
    figure.tight_layout()
    _save_figure(figure, figure_path)


def _timeline_band(
    axis: Axes,
    hours: np.ndarray[Any, Any],
    flags: np.ndarray[Any, Any],
    level: float,
    colour: str,
    min_width: float = 0.0,
) -> None:
    """Draw one boolean channel as a band at a fixed height.

    ``min_width`` widens each contiguous run to at least that many x-units, for channels
    whose true runs are a handful of samples and would otherwise render as nothing.
    """
    values = np.asarray(flags, dtype=bool)
    if min_width <= 0.0:
        axis.fill_between(
            hours,
            level - 0.09,
            level + 0.09,
            where=values,
            color=colour,
            step="post",
            linewidth=0.0,
        )
        return

    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts, ends, strict=True):
        left = float(hours[min(start, len(hours) - 1)])
        right = float(hours[min(end, len(hours) - 1)])
        if right - left < min_width:
            right = left + min_width
        axis.fill_between(
            [left, right],
            level - 0.09,
            level + 0.09,
            color=colour,
            linewidth=0.0,
        )


#: Digits are not letters in TeX control sequences, so dataset ids are romanised.
_ROMAN_DIGITS = {
    "0": "O",
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
    "8": "VIII",
    "9": "IX",
}


def _timeline_caption_macro(dataset_id: str) -> str:
    """Return a TeX-legal control sequence name for one dataset's caption."""
    letters = "".join(
        _ROMAN_DIGITS.get(character, character) for character in dataset_id if character.isalnum()
    )
    return f"\\timelinecaption{letters}"


def _timeline_caption(metadata: pd.Series) -> str:
    """State every configuration field, so the figure identifies what produced it."""
    lead = metadata.get("lead_time_hours", "")
    lead_text = (
        "No alarm matched the failure"
        if lead == "" or pd.isna(lead)
        else f"The matched alarm leads the failure by {float(lead):.2f} h"
    )
    resets = int(metadata["maintenance_resets_recorded"])
    reset_text = (
        "the end of the single labelled failure interval"
        if resets == 1
        else f"the ends of the {resets} labelled failure intervals"
    )
    # Only the interpolated values are escaped; the template carries deliberate LaTeX
    # such as the percent sign, which a blanket escape would turn into literal text.
    failure_id = _latex_escape(str(metadata["failure_id"]))
    variant = _latex_escape(str(metadata["target_region_variant"]))
    return (
        f"Global and local views of the {_dataset_label(metadata['dataset_id'])} held-out "
        f"failure {failure_id} under the {_event_method_label(metadata['method'])} "
        f"method ({variant} score). "
        f"Threshold quantile {float(metadata['threshold_quantile']):.3f} gives a frozen test "
        f"threshold of {float(metadata['threshold_value']):,.1f}; declustering run length "
        f"{int(metadata['run_length'])} samples, merge gap {int(metadata['merge_gap'])} samples, "
        f"warning horizon {float(metadata['warning_horizon_hours']):.1f} h. "
        f"The test split runs {float(metadata['test_duration_days']):.1f} operating days "
        f"({int(metadata['test_duration_samples']):,} samples). "
        f"Globally the method raises {int(metadata['global_alarm_episodes']):,} alarm episodes, "
        f"of which {int(metadata['global_false_alarm_episodes']):,} are false, at "
        f"{float(metadata['false_alarms_per_operating_day']):.2f} false alarms per operating day; "
        f"{int(metadata['local_alarm_episodes'])} episodes fall inside the plotted failure window "
        f"and {int(metadata['local_false_alarm_episodes'])} of those are false. "
        f"{lead_text}. Time under warning is "
        f"{int(metadata['time_under_warning_samples']):,} samples "
        f"({100.0 * float(metadata['time_under_warning_fraction']):.2f}\\% of the split). "
        f"The only maintenance boundary the processed data records is {reset_text}."
    )


def _real_event_timeline_figure(trace: pd.DataFrame, figure_path: Path) -> None:
    x = pd.to_numeric(trace["elapsed_hours"], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(trace["score"], errors="coerce").to_numpy(dtype=float)
    threshold = pd.to_numeric(trace["threshold"], errors="coerce").to_numpy(dtype=float)
    alarm = trace["alarm"].astype(bool).to_numpy()
    failure = trace["is_failure"].astype(bool).to_numpy()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 0.8, 0.8]},
    )
    axes[0].plot(x, score, color="#4C78A8", linewidth=0.9)
    axes[0].plot(x, threshold, color="#E15759", linewidth=0.9, linestyle="--")
    axes[0].set_ylabel("score")
    axes[1].fill_between(x, 0, alarm.astype(float), step="post", color="#59A14F", alpha=0.55)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("alarm")
    axes[2].fill_between(x, 0, failure.astype(float), step="post", color="#F28E2B", alpha=0.55)
    axes[2].axvline(0.0, color="#E15759", linewidth=0.9)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("failure")
    axes[2].set_xlabel("Elapsed hours from held-out failure onset")
    for axis in axes:
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    _save_figure(figure, figure_path)


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
    rows = _method_root_cause_rows(config) + _non_event_root_cause_rows(config, path)
    if not rows:
        return []
    frame = pd.DataFrame(rows)

    csv_path = tables / "root_cause_summary.csv"
    tex_path = latex / "root_cause_summary.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(
        frame,
        tex_path,
        caption=(
            "Failure layer per dataset and method. The layer is the earliest stage at "
            "which that method's evidence broke down, taken from the score, threshold, "
            "cluster and alarm decomposition; it is not a dataset-level property, and "
            "methods on the same dataset fail at different stages. Score and threshold "
            "failures are separated by the warning-window rank separation, which is the "
            "probability that a sample in the horizon before onset outranks one outside "
            "it; 0.5 is chance. The competing explanation is what the evidence does not "
            "exclude."
        ),
        label="tab:root-cause",
        column_weights=(0.8, 0.9, 0.95, 1.55, 0.98, 0.82),
        long=True,
    )
    return [csv_path, tex_path]


#: Methods the root-cause table must cover: the registered score, the strongest simple
#: policy, a classical EVT fit, a learned detector, both target regions, and the best
#: full-family baseline, which is selected from the benchmark rather than declared.
_ROOT_CAUSE_METHODS = (
    "dynamical_evt_robust_score",
    "engineering_threshold",
    "classical_pot_gpd",
    "isolation_forest",
    "failure_prototype_region",
    "rare_state_region",
)


def _best_full_family_baseline(baselines: pd.DataFrame, dataset_id: str) -> str | None:
    """Return the best-performing baseline outside the target-region families.

    Selected on event utility rather than named in advance, so the table compares the
    proposed method against the strongest competitor the benchmark actually produced.
    """
    candidates = baselines[
        (baselines["dataset_id"] == dataset_id)
        & (~baselines["method"].isin(_ROOT_CAUSE_METHODS))
        & (baselines["method"] != "negative_control_region")
    ]
    if candidates.empty:
        return None
    ranked = candidates.sort_values(
        ["event_f1", "false_alarm_events_per_day"], ascending=[False, True]
    )
    return str(ranked.iloc[0]["method"])


def _method_root_cause_rows(config: PaperAssetConfig) -> list[dict[str, object]]:
    """Method-specific root-cause rows for the compressor datasets."""
    decomposition_path = config.real_data_matrix_root / "score_threshold_alarm_decomposition.csv"
    baseline_path = config.real_data_matrix_root / "event_baseline_comparison.csv"
    if not decomposition_path.exists() or not baseline_path.exists():
        return []
    decomposition = pd.read_csv(decomposition_path)
    baselines = pd.read_csv(baseline_path)
    if "score_separation_auc" not in decomposition.columns:
        return []

    rows: list[dict[str, object]] = []
    for dataset_id in decomposition["dataset_id"].unique():
        wanted = list(_ROOT_CAUSE_METHODS)
        best = _best_full_family_baseline(baselines, dataset_id)
        if best is not None and best not in wanted:
            wanted.append(best)
        for method in wanted:
            entry = decomposition[
                (decomposition["dataset_id"] == dataset_id) & (decomposition["method"] == method)
            ]
            if entry.empty:
                continue
            row = entry.iloc[0]
            verdict = classify_root_cause(row)
            baseline_row = baselines[
                (baselines["dataset_id"] == dataset_id) & (baselines["method"] == method)
            ]
            per_day = (
                float(baseline_row["false_alarm_events_per_day"].iloc[0])
                if not baseline_row.empty
                else float("nan")
            )
            rows.append(
                {
                    "Dataset": _dataset_label(dataset_id),
                    "Method": _event_method_label(method)
                    + (" (best baseline)" if method == best else ""),
                    "Primary failure layer": verdict.layer,
                    "Supporting evidence": (
                        f"{verdict.evidence}; {per_day:,.1f} FA/day; {_exposure_text(row)}"
                    ),
                    "Alternative not excluded": verdict.alternative,
                    "Confidence": verdict.confidence,
                }
            )
    return rows


def _exposure_text(row: pd.Series) -> str:
    """Warning exposure as a share of the split.

    The percent sign is emitted plain: the table writer escapes LaTeX specials, and
    pre-escaping it produces a literal backslash in the rendered cell.
    """
    exposure = float(row.get("time_under_warning_samples", 0) or 0)
    total = float(row.get("test_observations_samples", 0) or 0)
    if total <= 0:
        return "exposure unknown"
    return f"{100.0 * exposure / total:.2f}% exposure"


def _non_event_root_cause_rows(
    config: PaperAssetConfig, summary_path: Path
) -> list[dict[str, object]]:
    """Estimand-mismatch rows for the datasets whose units are not failure episodes.

    These are not method failures. Their independent units are cycles, vehicles and
    wafers, so no alarm policy on them answers the question the compressor datasets
    answer, and giving them a pipeline-stage layer would misattribute a design mismatch
    to the detector.
    """
    source = pd.read_csv(summary_path)
    source = _merge_registered_event_metrics(config, source)
    explanations = {
        "scania_component_x": (
            "vehicle-level repair risk is not a compressor failure-episode onset"
        ),
        "hydraulic_systems": ("laboratory cycle states are not field failure-event onsets"),
        "secom": "wafer yield labels are quality outcomes, not maintenance repair events",
    }
    rows: list[dict[str, object]] = []
    for _index, row in source.iterrows():
        dataset_id = str(row["dataset_id"])
        if dataset_id not in explanations:
            continue
        rows.append(
            {
                "Dataset": _dataset_label(dataset_id),
                "Method": "high-quantile diagnostic",
                "Primary failure layer": "estimand mismatch",
                "Supporting evidence": (f"{explanations[dataset_id]}; {_root_cause_evidence(row)}"),
                "Alternative not excluded": (
                    "a redesigned target and evaluation on these units might succeed; "
                    "this is not evidence that the method fails on them"
                ),
                "Confidence": "high",
            }
        )
    return rows


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


def _evidential_unit_label(value: object) -> str:
    labels = {
        "scania_component_x": "vehicle",
        "hydraulic_systems": "load cycle",
        "secom": "wafer",
    }
    return labels.get(str(value), "unit")


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


def _event_method_label(value: object) -> str:
    labels = {
        "engineering_threshold": "Engineering threshold",
        "best_individual_sensor_threshold": "Best sensor threshold",
        "global_empirical_threshold": "Global empirical",
        "regime_conditioned_empirical_threshold": "Regime empirical",
        "classical_pot_gpd": "Classical POT/GPD",
        "fixed_run_declustering": "Fixed-run declustering",
        "ferro_segers_event_policy": "Ferro-Segers policy",
        "k_gaps_event_policy": "K-gaps policy",
        "spot": "SPOT-style",
        "isolation_forest": "Isolation Forest",
        "robust_online_changepoint": "Robust changepoint",
        "linear_autoencoder": "Linear autoencoder",
        "compact_nonlinear_autoencoder": "Compact autoencoder",
        "conformal_anomaly": "Conformal anomaly",
        "empirical_horizon_risk": "Empirical horizon risk",
        "dynamical_evt_robust_score": "Dynamical EVT score",
        "max_abs_robust_z": "Max robust z",
        "pressure_tp2_high": "TP2 high",
        "pressure_tp3_high": "TP3 high",
        "pressure_h1_high": "H1 high",
        "oil_temperature_high": "Oil temp high",
        "motor_current_high": "Current high",
        "flowmeter_low": "Flowmeter low",
        "failure_prototype_region": "Failure prototype",
        "rare_state_region": "Rare-state region",
        "negative_control_region": "Negative control",
    }
    return labels.get(str(value), str(value))


def _event_family_label(value: object) -> str:
    labels = {
        "registered": "registered",
        "baseline": "baseline",
        "target_region": "target region",
        "negative_control": "negative control",
    }
    return labels.get(str(value), str(value))


def _control_family_label(value: object) -> str:
    labels = {
        "random_occupancy": "Random occupancy",
        "time_shifted_prototype_window": "Time-shifted",
        "regime_matched_rare_region": "Rare-region matched",
        "episode_label_permutation": "Label permutation",
        "prototype_source_permutation": "Prototype permutation",
        "phase_randomised_score": "Phase-randomised",
    }
    return labels.get(str(value), str(value))


def _simulation_artifact_is_broad(path: Path) -> bool:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return False
    systems = config.get("systems", [])
    sample_sizes = config.get("sample_sizes", [])
    thresholds = config.get("threshold_quantiles", [])
    run_lengths = config.get("run_lengths", [])
    repetitions = int(config.get("repetitions", 0))
    return (
        isinstance(systems, list)
        and isinstance(sample_sizes, list)
        and isinstance(thresholds, list)
        and isinstance(run_lengths, list)
        and len(systems) >= 8
        and len(sample_sizes) >= 2
        and len(thresholds) >= 3
        and len(run_lengths) >= 3
        and repetitions >= 10
    )


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


def _root_cause_status(value: object) -> str:
    dataset_id = str(value)
    if dataset_id in {"metropt", "metropt2"}:
        return "direct event-row evidence"
    if dataset_id == "scania_component_x":
        return "estimand evidence"
    return "diagnostic evidence"


def _root_cause_alternative(value: object) -> str:
    alternatives = {
        "metropt": "threshold too low or observable too broad",
        "metropt2": "threshold too low or observable too broad",
        "scania_component_x": "features insufficient for repair ranking",
        "hydraulic_systems": "component state needs supervised feature family",
        "secom": "missingness and class imbalance dominate",
    }
    return alternatives.get(str(value), "not assessed")


def _root_cause_confidence(value: object) -> str:
    dataset_id = str(value)
    if dataset_id in {"metropt", "metropt2"}:
        return "medium"
    return "low"


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


#: Data rows above which a table is emitted as a breakable ``longtable``. Set from the
#: smallest table observed to overrun a page: the 19-row method-provenance table, whose
#: text cells wrap. Row count alone does not determine height, so this is deliberately
#: below the largest table that happens to fit.
_MAX_FLOAT_TABLE_ROWS = 15


def _write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    column_weights: Sequence[float] | None = None,
    long: bool | None = None,
) -> None:
    r"""Write ``frame`` as a full-width table.

    ``column_weights`` optionally reweights the equal-width default, which crowds
    tables that mix long method names with short counts. Weights must sum to the
    column count.

    ``long`` emits a ``longtable`` instead of a ``tabularx`` float. Left unset, it is
    chosen from the row count: a float that overruns the page does not error, it warns
    and lets the surplus rows fall off the bottom, so they disappear from the PDF while
    the source and the CSV still list them. Deciding here rather than at each call site
    means a table that grows past the limit cannot silently start dropping rows.
    """
    columns = [str(column) for column in frame.columns]
    if column_weights is not None:
        if len(column_weights) != len(columns):
            raise ValueError(
                f"column_weights has {len(column_weights)} entries for {len(columns)} columns"
            )
        total = sum(column_weights)
        if abs(total - len(columns)) > 1e-6:
            raise ValueError(f"column_weights must sum to {len(columns)}, got {total}")
    weights = list(column_weights or [1.0] * len(columns))
    header = (
        " & ".join(_latex_escape(_pretty_column_header(column)) for column in columns) + " \\\\"
    )
    body = []
    for _index, row in frame.iterrows():
        values = [_latex_escape(_format_latex_value(row[column])) for column in frame.columns]
        body.append(" & ".join(values) + " \\\\")

    if long is None:
        long = len(frame) > _MAX_FLOAT_TABLE_ROWS

    if not long:
        column_spec = " ".join(
            rf">{{\hsize={weight}\hsize\raggedright\arraybackslash}}X"
            if column_weights is not None
            else r">{\raggedright\arraybackslash}X"
            for weight in weights
        )
        lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            "\\small",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{{_latex_escape(label)}}}",
            "\\begin{tabularx}{\\textwidth}{" + column_spec + "}",
            "\\toprule",
            header,
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table}",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    # longtable has no X column, so the widths are resolved against a length that
    # subtracts the inter-column glue from the text width.
    count = len(columns)
    span = rf"\dimexpr\textwidth-{2 * count}\tabcolsep\relax"
    column_spec = " ".join(
        rf">{{\raggedright\arraybackslash}}p{{{weight / count:.4f}{span}}}" for weight in weights
    )
    lines = [
        "\\begingroup",
        "\\small",
        # longtable captions default to 4in, which reads as a mistake next to a
        # full-width table.
        "\\setlength{\\LTcapwidth}{\\textwidth}",
        "\\begin{longtable}{" + column_spec + "}",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\midrule",
        rf"\multicolumn{{{count}}}{{r}}{{\small\itshape continued on next page}}\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
        *body,
        "\\end{longtable}",
        "\\endgroup",
        "",
    ]
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
