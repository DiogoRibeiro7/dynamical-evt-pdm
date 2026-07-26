"""Conservative industrial result artifacts from prepared real datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, flags_to_events
from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions

IndustrialResultStatus = Literal["succeeded", "not_estimable", "failed"]

REQUIRED_DATASETS = (
    "hydraulic_systems",
    "metropt",
    "metropt2",
    "scania_component_x",
    "secom",
)
METROPT_FEATURE_PRIORITY = (
    "pressure_tp2",
    "pressure_tp3",
    "pressure_h1",
    "reservoir_pressure",
    "oil_temperature",
    "motor_current",
    "flowmeter",
)


@dataclass(frozen=True, slots=True)
class IndustrialResultsConfig:
    """Input and output locations for real-data result artifacts."""

    processed_root: Path = Path("data/processed")
    output_root: Path = Path("artifacts/real_data_matrix")
    threshold_quantile: float = 0.98
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    horizon: int = 3_600
    matching_tolerance_after: int = 300
    merge_gap: int = 60
    samples_per_day: int = 86_400


@dataclass(frozen=True, slots=True)
class IndustrialDatasetResult:
    """One dataset-level industrial diagnostic result."""

    dataset_id: str
    status: IndustrialResultStatus
    estimand: str
    rows: int
    independent_units: int
    split_evaluated: str
    feature_columns: tuple[str, ...]
    threshold_quantile: float
    threshold: float | None
    target_events_or_units: int
    predicted_events_or_units: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_alarm_events_per_day: float | None
    median_warning_lead_time: float | None
    brier_score: float | None
    limitation: str


@dataclass(frozen=True, slots=True)
class IndustrialResultsManifest:
    """Manifest for generated industrial result artifacts."""

    output_root: str
    datasets: tuple[str, ...]
    summary_csv: str
    details_json: str
    latex_table: str
    results: tuple[IndustrialDatasetResult, ...]
    event_baseline_csv: str | None = None
    event_variant_csv: str | None = None
    event_timeline_csv: str | None = None


def run_industrial_results(config: IndustrialResultsConfig) -> IndustrialResultsManifest:
    """Generate conservative real-data result artifacts from processed parquet files."""

    if not 0.0 < config.threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be in (0, 1)")
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a test split")

    config.output_root.mkdir(parents=True, exist_ok=True)
    results = tuple(_run_dataset(dataset_id, config) for dataset_id in REQUIRED_DATASETS)
    summary_csv = config.output_root / "industrial_results_summary.csv"
    details_json = config.output_root / "industrial_results.json"
    latex_table = config.output_root / "industrial_results.tex"
    summary = pd.DataFrame([_result_row(result) for result in results])
    summary.to_csv(summary_csv, index=False)
    details_json.write_text(
        json.dumps({"results": [asdict(result) for result in results]}, indent=2, default=str),
        encoding="utf-8",
    )
    latex_table.write_text(_render_latex_table(results), encoding="utf-8")
    event_artifacts = _run_event_level_comparison_artifacts(config)
    return IndustrialResultsManifest(
        output_root=str(config.output_root),
        datasets=REQUIRED_DATASETS,
        summary_csv=str(summary_csv),
        details_json=str(details_json),
        latex_table=str(latex_table),
        results=results,
        event_baseline_csv=event_artifacts.get("event_baseline_csv"),
        event_variant_csv=event_artifacts.get("event_variant_csv"),
        event_timeline_csv=event_artifacts.get("event_timeline_csv"),
    )


def _run_dataset(dataset_id: str, config: IndustrialResultsConfig) -> IndustrialDatasetResult:
    manifest_path = config.processed_root / dataset_id / "manifest.json"
    if not manifest_path.exists():
        return _failed_result(dataset_id, f"missing processed manifest: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    try:
        if dataset_id in {"metropt", "metropt2"}:
            return _run_metropt_family(dataset_id, manifest, config)
        if dataset_id == "scania_component_x":
            return _run_scania(manifest, config)
        if dataset_id == "hydraulic_systems":
            return _run_tabular_condition_dataset(
                dataset_id=dataset_id,
                manifest=manifest,
                config=config,
                feature_prefixes=("",),
                excluded_columns={
                    "cycle_id",
                    "cooler_condition",
                    "valve_condition",
                    "internal_pump_leakage",
                    "accumulator_pressure",
                    "stable_flag",
                    "is_degraded",
                    "split",
                },
                target_column="is_degraded",
                entity_column="cycle_id",
                estimand="cycle-level hydraulic component degradation on chronological test split",
                limitation=(
                    "laboratory cycle-level component states are not field failure-event onsets"
                ),
            )
        if dataset_id == "secom":
            return _run_tabular_condition_dataset(
                dataset_id=dataset_id,
                manifest=manifest,
                config=config,
                feature_prefixes=("sensor_",),
                excluded_columns={
                    "wafer_id",
                    "label_raw",
                    "is_failure",
                    "source_timestamp",
                    "split",
                },
                target_column="is_failure",
                entity_column="wafer_id",
                estimand="wafer-level semiconductor yield-failure detection on chronological test split",
                limitation=(
                    "yield-failure labels are quality outcomes, not maintenance repair events"
                ),
            )
    except Exception as exc:
        return _failed_result(dataset_id, f"{type(exc).__name__}: {exc}")
    return _failed_result(dataset_id, "unsupported dataset")


def _run_metropt_family(
    dataset_id: str,
    manifest: dict[str, Any],
    config: IndustrialResultsConfig,
) -> IndustrialDatasetResult:
    files = _manifest_files(manifest)
    rows = int(manifest.get("rows", 0))
    feature_columns = _available_columns(manifest, METROPT_FEATURE_PRIORITY)
    if not files or rows <= 0 or not feature_columns:
        return _failed_result(dataset_id, "missing parquet files, rows, or feature columns")
    threshold, center, scale = _fit_temporal_threshold(files, rows, feature_columns, config)
    test_flags: list[np.ndarray[Any, Any]] = []
    test_targets: list[np.ndarray[Any, Any]] = []
    offset = 0
    for path in files:
        frame = pd.read_parquet(path, columns=[*feature_columns, "is_failure"])
        split = _temporal_split(offset, len(frame), rows, config)
        scores = _robust_scores(frame, feature_columns, center=center, scale=scale)
        test_mask = split == "test"
        test_flags.append(scores[test_mask] > threshold)
        test_targets.append(frame["is_failure"].astype(bool).to_numpy()[test_mask])
        offset += len(frame)
    alarm_flags = np.concatenate(test_flags) if test_flags else np.array([], dtype=bool)
    target_flags = np.concatenate(test_targets) if test_targets else np.array([], dtype=bool)
    failures = flags_to_events(target_flags, label="failure")
    alarms = flags_to_events(alarm_flags, label="alarm", merge_gap=config.merge_gap)
    if not failures:
        return IndustrialDatasetResult(
            dataset_id=dataset_id,
            status="not_estimable",
            estimand="event-level early warning on temporal test split",
            rows=rows,
            independent_units=0,
            split_evaluated="test",
            feature_columns=tuple(feature_columns),
            threshold_quantile=config.threshold_quantile,
            threshold=threshold,
            target_events_or_units=0,
            predicted_events_or_units=len(alarms),
            precision=None,
            recall=None,
            f1=None,
            false_alarm_events_per_day=None,
            median_warning_lead_time=None,
            brier_score=None,
            limitation="no labelled failure event appears in the temporal test split",
        )
    evaluation = evaluate_event_predictions(
        alarms,
        failures,
        policy=EarlyWarningPolicy(
            horizon=config.horizon,
            tolerance_after=config.matching_tolerance_after,
        ),
        method="optimal",
        total_operating_time=len(target_flags),
        samples_per_day=config.samples_per_day,
    )
    return IndustrialDatasetResult(
        dataset_id=dataset_id,
        status="succeeded",
        estimand="event-level early warning on temporal test split",
        rows=rows,
        independent_units=len(failures),
        split_evaluated="test",
        feature_columns=tuple(feature_columns),
        threshold_quantile=config.threshold_quantile,
        threshold=threshold,
        target_events_or_units=len(failures),
        predicted_events_or_units=len(alarms),
        precision=evaluation.precision,
        recall=evaluation.recall,
        f1=evaluation.f1,
        false_alarm_events_per_day=evaluation.false_alarm_events_per_operating_day,
        median_warning_lead_time=evaluation.median_warning_lead_time,
        brier_score=None,
        limitation="few independent failure episodes; diagnostic result only",
    )


def _run_event_level_comparison_artifacts(config: IndustrialResultsConfig) -> dict[str, str]:
    baseline_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    timeline_frames: list[pd.DataFrame] = []
    for dataset_id in ("metropt", "metropt2"):
        manifest_path = config.processed_root / dataset_id / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json_object(manifest_path)
        rows, variants, timeline = _run_metropt_event_comparison(dataset_id, manifest, config)
        baseline_rows.extend(rows)
        variant_rows.extend(variants)
        if timeline is not None:
            timeline_frames.append(timeline)

    artifacts: dict[str, str] = {}
    if baseline_rows:
        path = config.output_root / "event_baseline_comparison.csv"
        pd.DataFrame(baseline_rows).to_csv(path, index=False)
        artifacts["event_baseline_csv"] = str(path)
    if variant_rows:
        path = config.output_root / "event_variant_comparison.csv"
        pd.DataFrame(variant_rows).to_csv(path, index=False)
        artifacts["event_variant_csv"] = str(path)
    if timeline_frames:
        path = config.output_root / "event_timeline_trace.csv"
        pd.concat(timeline_frames, ignore_index=True).to_csv(path, index=False)
        artifacts["event_timeline_csv"] = str(path)
    manifest_path = config.output_root / "event_level_comparison_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


def _run_metropt_event_comparison(
    dataset_id: str,
    manifest: dict[str, Any],
    config: IndustrialResultsConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame | None]:
    files = _manifest_files(manifest)
    rows = int(manifest.get("rows", 0))
    feature_columns = _available_columns(manifest, METROPT_FEATURE_PRIORITY)
    if not files or rows <= 0 or not feature_columns:
        return [], [], None

    train = _collect_metropt_split(files, rows, feature_columns, config, split_name="train")
    if train.empty:
        return [], [], None
    center, scale = _robust_center_scale(train, feature_columns)
    train_refs = _event_method_references(train, feature_columns, center=center, scale=scale)
    train_scores = _event_method_scores(
        train, feature_columns, center=center, scale=scale, refs=train_refs
    )
    thresholds = {
        method: float(np.nanquantile(scores[np.isfinite(scores)], config.threshold_quantile))
        for method, scores in train_scores.items()
        if np.isfinite(scores).any()
    }
    if not thresholds:
        return [], [], None

    flags_by_method: dict[str, list[np.ndarray[Any, Any]]] = {method: [] for method in thresholds}
    target_parts: list[np.ndarray[Any, Any]] = []
    timestamp_parts: list[np.ndarray[Any, Any]] = []
    robust_score_parts: list[np.ndarray[Any, Any]] = []
    offset = 0
    columns = [*feature_columns, "timestamp", "is_failure"]
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        split = _temporal_split(offset, len(frame), rows, config)
        test = frame.loc[split == "test", columns]
        if not test.empty:
            scores = _event_method_scores(
                test, feature_columns, center=center, scale=scale, refs=train_refs
            )
            for method, threshold in thresholds.items():
                flags_by_method[method].append(scores[method] > threshold)
            target_parts.append(test["is_failure"].astype(bool).to_numpy())
            timestamp_parts.append(test["timestamp"].to_numpy())
            robust_score_parts.append(scores["dynamical_evt_robust_score"])
        offset += len(frame)

    target_flags = np.concatenate(target_parts) if target_parts else np.array([], dtype=bool)
    failures = flags_to_events(target_flags, label="failure")
    if not failures:
        return [], [], None
    baseline_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for method, parts in flags_by_method.items():
        alarm_flags = np.concatenate(parts) if parts else np.array([], dtype=bool)
        alarms = flags_to_events(alarm_flags, label="alarm", merge_gap=config.merge_gap)
        evaluation = evaluate_event_predictions(
            alarms,
            failures,
            policy=EarlyWarningPolicy(
                horizon=config.horizon,
                tolerance_after=config.matching_tolerance_after,
            ),
            method="optimal",
            total_operating_time=len(target_flags),
            samples_per_day=config.samples_per_day,
        )
        row = {
            "dataset_id": dataset_id,
            "failure_id": f"{dataset_id}_test_failure_001",
            "method": method,
            "method_family": _event_method_family(method),
            "target_events": len(failures),
            "predicted_alarm_events": len(alarms),
            "event_recall": evaluation.recall,
            "event_precision": evaluation.precision,
            "event_f1": evaluation.f1,
            "false_alarm_events_per_day": evaluation.false_alarm_events_per_operating_day,
            "duplicate_alarm_events": evaluation.duplicate_alarm_events,
            "median_warning_lead_time": evaluation.median_warning_lead_time,
            "time_under_warning": evaluation.time_under_warning,
            "threshold": thresholds[method],
            "leakage_control": "train thresholds and references only",
        }
        baseline_rows.append(row)
        if row["method_family"] in {"target_region", "negative_control", "registered"}:
            variant_rows.append(
                {
                    **row,
                    "distance_metric": "robust euclidean" if "region" in method else "robust max-z",
                    "variant_status": "completed",
                }
            )

    timeline = None
    if dataset_id == "metropt" and timestamp_parts and robust_score_parts:
        timestamps = np.concatenate(timestamp_parts)
        robust_scores = np.concatenate(robust_score_parts)
        robust_flags = (
            np.concatenate(flags_by_method["dynamical_evt_robust_score"])
            if "dynamical_evt_robust_score" in flags_by_method
            else np.zeros(len(target_flags), dtype=bool)
        )
        timeline = _event_timeline_trace(
            dataset_id,
            timestamps=timestamps,
            scores=robust_scores,
            alarm_flags=robust_flags,
            target_flags=target_flags,
            threshold=thresholds["dynamical_evt_robust_score"],
            samples_per_day=config.samples_per_day,
        )
    return baseline_rows, variant_rows, timeline


def _collect_metropt_split(
    files: tuple[Path, ...],
    rows: int,
    feature_columns: tuple[str, ...],
    config: IndustrialResultsConfig,
    *,
    split_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    columns = [*feature_columns, "is_failure"]
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        split = _temporal_split(offset, len(frame), rows, config)
        selected = frame.loc[split == split_name, columns]
        if not selected.empty:
            frames.append(selected)
        offset += len(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _robust_center_scale(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    center = numeric.median(numeric_only=True)
    q75 = numeric.quantile(0.75, numeric_only=True)
    q25 = numeric.quantile(0.25, numeric_only=True)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(1.0)
    return center, scale


def _event_method_references(
    train: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    center: pd.Series,
    scale: pd.Series,
) -> dict[str, np.ndarray[Any, Any]]:
    scaled = _scaled_matrix(train, feature_columns, center=center, scale=scale)
    robust = _nanmax_abs(pd.DataFrame(scaled, columns=list(feature_columns)), feature_columns)
    targets = train["is_failure"].astype(bool).to_numpy()
    references: dict[str, np.ndarray[Any, Any]] = {}
    references["failure_prototype_region"] = _reference_subset(scaled[targets], limit=16)
    rare_count = min(16, len(scaled))
    if rare_count:
        rare_indices = np.argsort(robust)[-rare_count:]
        references["rare_state_region"] = scaled[rare_indices]
    references["negative_control_region"] = _reference_subset(scaled[~targets], limit=16)
    return references


def _event_method_scores(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    center: pd.Series,
    scale: pd.Series,
    refs: dict[str, np.ndarray[Any, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    scaled = _scaled_matrix(frame, feature_columns, center=center, scale=scale)
    robust = np.nanmax(np.abs(np.where(np.isfinite(scaled), scaled, np.nan)), axis=1)
    robust = np.nan_to_num(robust, nan=0.0, posinf=0.0, neginf=0.0)
    scores: dict[str, np.ndarray[Any, Any]] = {
        "dynamical_evt_robust_score": robust,
        "max_abs_robust_z": robust,
    }
    for column in (
        "pressure_tp2",
        "pressure_tp3",
        "pressure_h1",
        "oil_temperature",
        "motor_current",
    ):
        if column in feature_columns:
            index = feature_columns.index(column)
            scores[f"{column}_high"] = np.nan_to_num(scaled[:, index], nan=0.0)
    if "flowmeter" in feature_columns:
        index = feature_columns.index("flowmeter")
        scores["flowmeter_low"] = -np.nan_to_num(scaled[:, index], nan=0.0)
    for method in ("failure_prototype_region", "rare_state_region", "negative_control_region"):
        reference = refs.get(method)
        if reference is not None and len(reference):
            scores[method] = -_minimum_distance(scaled, reference)
    return scores


def _scaled_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    center: pd.Series,
    scale: pd.Series,
) -> np.ndarray[Any, Any]:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    scaled = (numeric - center.loc[list(feature_columns)]) / scale.loc[list(feature_columns)]
    return cast(np.ndarray[Any, Any], scaled.to_numpy(dtype=float))


def _reference_subset(values: np.ndarray[Any, Any], *, limit: int) -> np.ndarray[Any, Any]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return cast(np.ndarray[Any, Any], values[indices])


def _minimum_distance(
    values: np.ndarray[Any, Any],
    references: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    clean_refs = np.nan_to_num(references, nan=0.0, posinf=0.0, neginf=0.0)
    best = np.full(len(clean_values), np.inf, dtype=float)
    for reference in clean_refs:
        distance = np.sqrt(np.sum((clean_values - reference) ** 2, axis=1))
        best = np.minimum(best, distance)
    return best


def _event_method_family(method: str) -> str:
    if method == "dynamical_evt_robust_score":
        return "registered"
    if method in {"failure_prototype_region", "rare_state_region"}:
        return "target_region"
    if method == "negative_control_region":
        return "negative_control"
    return "baseline"


def _event_timeline_trace(
    dataset_id: str,
    *,
    timestamps: np.ndarray[Any, Any],
    scores: np.ndarray[Any, Any],
    alarm_flags: np.ndarray[Any, Any],
    target_flags: np.ndarray[Any, Any],
    threshold: float,
    samples_per_day: int,
) -> pd.DataFrame:
    failure_indices = np.flatnonzero(target_flags)
    if not len(failure_indices):
        return pd.DataFrame()
    center = int(failure_indices[0])
    samples_per_hour = max(1.0, samples_per_day / 24.0)
    start = max(0, center - int(samples_per_hour))
    end = min(len(target_flags), center + int(2 * samples_per_hour))
    index = np.arange(start, end, dtype=int)
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "failure_id": f"{dataset_id}_test_failure_001",
            "test_index": index,
            "timestamp": timestamps[index].astype(str),
            "elapsed_hours": (index - center) / samples_per_hour,
            "score": scores[index],
            "threshold": threshold,
            "alarm": alarm_flags[index],
            "is_failure": target_flags[index],
        }
    )


def _run_scania(
    manifest: dict[str, Any],
    config: IndustrialResultsConfig,
) -> IndustrialDatasetResult:
    rows = int(manifest.get("rows", 0))
    feature_columns = _scania_feature_columns(manifest)
    if rows <= 0 or not feature_columns:
        return _failed_result("scania_component_x", "missing rows or increment feature columns")
    vehicle_scores: dict[str, dict[int, float]] = {"train": {}, "validation": {}, "test": {}}
    vehicle_targets: dict[str, dict[int, bool]] = {"train": {}, "validation": {}, "test": {}}
    split_manifests = cast(dict[str, Any], manifest.get("split_manifests", {}))
    for split in ("train", "validation", "test"):
        split_manifest = cast(dict[str, Any], split_manifests.get(split, {}))
        split_schema = cast(dict[str, Any], split_manifest.get("schema", {}))
        target_column = _scania_target_column(split_schema)
        for path in _manifest_files(split_manifest):
            columns = ["vehicle_id", *feature_columns]
            if target_column is not None:
                columns.append(target_column)
            frame = pd.read_parquet(
                path,
                columns=columns,
            )
            scores = _nanmax_abs(frame, feature_columns)
            vehicle_ids = pd.to_numeric(frame["vehicle_id"], errors="coerce").to_numpy(dtype=float)
            row_targets = (
                frame[target_column].astype(bool).to_numpy()
                if target_column is not None
                else np.zeros(len(frame), dtype=bool)
            )
            for vehicle_id, score, target in zip(vehicle_ids, scores, row_targets, strict=True):
                if not np.isfinite(vehicle_id):
                    continue
                key = int(vehicle_id)
                current = vehicle_scores[split].get(key, float("-inf"))
                vehicle_scores[split][key] = max(current, float(score))
                vehicle_targets[split][key] = bool(vehicle_targets[split].get(key, False) or target)
    train_scores = np.array(list(vehicle_scores["train"].values()), dtype=float)
    train_scores = train_scores[np.isfinite(train_scores)]
    if len(train_scores) == 0:
        return _failed_result("scania_component_x", "train split has no finite vehicle scores")
    threshold = float(np.quantile(train_scores, config.threshold_quantile))
    test_items = sorted(vehicle_scores["test"])
    predictions = np.array([vehicle_scores["test"][item] > threshold for item in test_items])
    test_targets = np.array([vehicle_targets["test"].get(item, False) for item in test_items])
    precision, recall, f1 = _binary_metrics(predictions, test_targets)
    return IndustrialDatasetResult(
        dataset_id="scania_component_x",
        status="succeeded",
        estimand="vehicle-level repair risk on published test split",
        rows=rows,
        independent_units=len(test_items),
        split_evaluated="test",
        feature_columns=tuple(feature_columns),
        threshold_quantile=config.threshold_quantile,
        threshold=threshold,
        target_events_or_units=int(test_targets.sum()),
        predicted_events_or_units=int(predictions.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_events_per_day=None,
        median_warning_lead_time=None,
        brier_score=float(np.mean((predictions.astype(float) - test_targets.astype(float)) ** 2))
        if len(test_targets)
        else None,
        limitation="vehicle-level estimand is not directly comparable to compressor event warning",
    )


def _run_tabular_condition_dataset(
    *,
    dataset_id: str,
    manifest: dict[str, Any],
    config: IndustrialResultsConfig,
    feature_prefixes: tuple[str, ...],
    excluded_columns: set[str],
    target_column: str,
    entity_column: str,
    estimand: str,
    limitation: str,
) -> IndustrialDatasetResult:
    files = _manifest_files(manifest)
    rows = int(manifest.get("rows", 0))
    schema = cast(dict[str, Any], manifest.get("schema", {}))
    feature_columns = tuple(
        column
        for column in sorted(schema)
        if column not in excluded_columns
        and any(column.startswith(prefix) for prefix in feature_prefixes)
        and _is_numeric_schema(schema[column])
    )
    if not files or rows <= 0 or not feature_columns or target_column not in schema:
        return _failed_result(dataset_id, "missing parquet files, rows, features, or target column")

    frame = pd.concat(
        [
            pd.read_parquet(path, columns=[entity_column, *feature_columns, target_column, "split"])
            for path in files
        ],
        ignore_index=True,
    )
    train = frame.loc[frame["split"].astype(str) == "train", list(feature_columns)]
    if train.empty:
        return _failed_result(dataset_id, "train split has no rows")
    center = train.median(numeric_only=True)
    q75 = train.quantile(0.75, numeric_only=True)
    q25 = train.quantile(0.25, numeric_only=True)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(1.0)
    train_scores = _robust_scores(train, feature_columns, center=center, scale=scale)
    train_scores = train_scores[np.isfinite(train_scores)]
    if len(train_scores) == 0:
        return _failed_result(dataset_id, "train split has no finite scores")
    threshold = float(np.quantile(train_scores, config.threshold_quantile))

    test = frame.loc[frame["split"].astype(str) == "test"].copy()
    if test.empty:
        return _failed_result(dataset_id, "test split has no rows")
    test_scores = _robust_scores(test, feature_columns, center=center, scale=scale)
    predictions = np.asarray(test_scores > threshold, dtype=bool)
    targets = test[target_column].astype(bool).to_numpy()
    precision, recall, f1 = _binary_metrics(predictions, targets)
    return IndustrialDatasetResult(
        dataset_id=dataset_id,
        status="succeeded",
        estimand=estimand,
        rows=rows,
        independent_units=int(test[entity_column].nunique(dropna=True)),
        split_evaluated="test",
        feature_columns=feature_columns,
        threshold_quantile=config.threshold_quantile,
        threshold=threshold,
        target_events_or_units=int(targets.sum()),
        predicted_events_or_units=int(predictions.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_events_per_day=None,
        median_warning_lead_time=None,
        brier_score=float(np.mean((predictions.astype(float) - targets.astype(float)) ** 2))
        if len(targets)
        else None,
        limitation=limitation,
    )


def _fit_temporal_threshold(
    files: tuple[Path, ...],
    rows: int,
    feature_columns: tuple[str, ...],
    config: IndustrialResultsConfig,
) -> tuple[float, pd.Series, pd.Series]:
    train_frames: list[pd.DataFrame] = []
    offset = 0
    for path in files:
        frame = pd.read_parquet(path, columns=list(feature_columns))
        split = _temporal_split(offset, len(frame), rows, config)
        train = frame.loc[split == "train", list(feature_columns)]
        if not train.empty:
            train_frames.append(train)
        offset += len(frame)
    if not train_frames:
        raise ValueError("temporal train split has no rows")
    train_frame = pd.concat(train_frames, ignore_index=True)
    center = train_frame.median(numeric_only=True)
    q75 = train_frame.quantile(0.75, numeric_only=True)
    q25 = train_frame.quantile(0.25, numeric_only=True)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(1.0)
    scores = _robust_scores(train_frame, feature_columns, center=center, scale=scale)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise ValueError("temporal train split has no finite scores")
    return float(np.quantile(finite, config.threshold_quantile)), center, scale


def _robust_scores(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    center: pd.Series,
    scale: pd.Series,
) -> np.ndarray[Any, Any]:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    scaled = (numeric - center.loc[list(feature_columns)]) / scale.loc[list(feature_columns)]
    return _nanmax_abs(scaled, feature_columns)


def _nanmax_abs(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray[Any, Any]:
    values = (
        frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    if values.size == 0:
        return np.array([], dtype=float)
    finite = np.where(np.isfinite(values), np.abs(values), np.nan)
    all_nan = np.isnan(finite).all(axis=1)
    finite[all_nan, :] = 0.0
    result = np.nanmax(finite, axis=1)
    return cast(np.ndarray[Any, Any], result)


def _temporal_split(
    offset: int,
    length: int,
    rows: int,
    config: IndustrialResultsConfig,
) -> np.ndarray[Any, Any]:
    indices = np.arange(offset, offset + length)
    train_end = int(rows * config.train_fraction)
    validation_end = int(rows * (config.train_fraction + config.validation_fraction))
    result = np.full(length, "test", dtype=object)
    result[indices < validation_end] = "validation"
    result[indices < train_end] = "train"
    return cast(np.ndarray[Any, Any], result)


def _binary_metrics(
    predictions: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
) -> tuple[float, float, float]:
    predicted = np.asarray(predictions, dtype=bool)
    actual = np.asarray(targets, dtype=bool)
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _manifest_files(manifest: dict[str, Any]) -> tuple[Path, ...]:
    return tuple(Path(str(path)) for path in manifest.get("parquet_files", []))


def _available_columns(
    manifest: dict[str, Any],
    priority: tuple[str, ...],
) -> tuple[str, ...]:
    schema = cast(dict[str, Any], manifest.get("schema", {}))
    return tuple(column for column in priority if column in schema)


def _scania_feature_columns(manifest: dict[str, Any]) -> tuple[str, ...]:
    schema = cast(dict[str, Any], manifest.get("schema", {}))
    candidates = sorted(column for column in schema if column.endswith("__increment"))
    if candidates:
        return tuple(candidates[:12])
    split_manifests = cast(dict[str, Any], manifest.get("split_manifests", {}))
    for split_manifest in split_manifests.values():
        split_schema = cast(dict[str, Any], split_manifest.get("schema", {}))
        candidates = sorted(column for column in split_schema if column.endswith("__increment"))
        if candidates:
            return tuple(candidates[:12])
    return ()


def _scania_target_column(schema: dict[str, Any]) -> str | None:
    if "in_study_repair" in schema:
        return "in_study_repair"
    if "class_label" in schema:
        return "class_label"
    return None


def _is_numeric_schema(schema_entry: object) -> bool:
    if not isinstance(schema_entry, dict):
        return False
    dtype = str(schema_entry.get("dtype", "")).lower()
    return any(token in dtype for token in ("int", "float", "bool"))


def _failed_result(dataset_id: str, limitation: str) -> IndustrialDatasetResult:
    return IndustrialDatasetResult(
        dataset_id=dataset_id,
        status="failed",
        estimand="not available",
        rows=0,
        independent_units=0,
        split_evaluated="none",
        feature_columns=(),
        threshold_quantile=float("nan"),
        threshold=None,
        target_events_or_units=0,
        predicted_events_or_units=0,
        precision=None,
        recall=None,
        f1=None,
        false_alarm_events_per_day=None,
        median_warning_lead_time=None,
        brier_score=None,
        limitation=limitation,
    )


def _result_row(result: IndustrialDatasetResult) -> dict[str, object]:
    row = asdict(result)
    row["feature_columns"] = ",".join(result.feature_columns)
    return row


def _render_latex_table(results: tuple[IndustrialDatasetResult, ...]) -> str:
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Status & Estimand & Units & Targets & Predictions & F1 \\\\",
        "\\midrule",
    ]
    for result in results:
        lines.append(
            " & ".join(
                [
                    _latex_escape(result.dataset_id),
                    _latex_escape(result.status),
                    _latex_escape(result.estimand),
                    str(result.independent_units),
                    str(result.target_events_or_units),
                    str(result.predicted_events_or_units),
                    _format_optional(result.f1),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], raw)
