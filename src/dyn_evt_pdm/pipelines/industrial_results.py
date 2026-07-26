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

REQUIRED_DATASETS = ("metropt", "metropt2", "scania_component_x")
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
    return IndustrialResultsManifest(
        output_root=str(config.output_root),
        datasets=REQUIRED_DATASETS,
        summary_csv=str(summary_csv),
        details_json=str(details_json),
        latex_table=str(latex_table),
        results=results,
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
