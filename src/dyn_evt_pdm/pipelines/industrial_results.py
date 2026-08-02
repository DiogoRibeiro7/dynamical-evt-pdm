"""Conservative industrial result artifacts from prepared real datasets."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from dyn_evt_pdm.evaluation.events import EarlyWarningPolicy, flags_to_events
from dyn_evt_pdm.evaluation.metrics import evaluate_event_predictions
from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.pipelines.cross_dataset_transfer import (
    TransferResult,
    TransferSplits,
    assess_schema_compatibility,
    fit_target_region,
    run_transfer_protocols,
    transfer_distance_samples,
    transfer_results_frame,
)
from dyn_evt_pdm.pipelines.detection_aware_controls import (
    add_detection_columns,
    detection_aware_summary,
    joint_utility,
    utility_sensitivity,
)
from dyn_evt_pdm.pipelines.event_method_impl import (
    apply_regime_thresholds,
    best_single_sensor_index,
    declustered_flags,
    derived_run_length,
    fit_isolation_forest,
    gpd_return_level,
    isolation_forest_score,
    regime_conditioned_thresholds,
    single_sensor_score,
)
from dyn_evt_pdm.pipelines.event_method_specs import SPECS_BY_NAME, method_merge_gap
from dyn_evt_pdm.pipelines.event_timeline import (
    TimelineInputs,
    build_global_overview,
    build_local_trace,
    build_timeline_metadata,
)
from dyn_evt_pdm.types import BoolArray, EventInterval

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
EVENT_BASELINE_METHODS = (
    "engineering_threshold",
    "best_individual_sensor_threshold",
    "global_empirical_threshold",
    "regime_conditioned_empirical_threshold",
    "classical_pot_gpd",
    "fixed_run_declustering",
    "ferro_segers_event_policy",
    "k_gaps_event_policy",
    "spot",
    "isolation_forest",
    "robust_online_changepoint",
    "linear_autoencoder",
    "compact_nonlinear_autoencoder",
    "conformal_anomaly",
    "empirical_horizon_risk",
    "dynamical_evt_robust_score",
    "failure_prototype_region",
    "rare_state_region",
)
#: Highest training exceedance rate a threshold may imply before it is treated as
#: degenerate. Above this the detector is effectively always on, and episode-level
#: precision and recall stop measuring detection.
_MAX_TRAIN_EXCEEDANCE_RATE = 0.5

#: Share of the test period under warning above which a detector is treated as always
#: on. Episode-level precision and recall stop measuring detection at that point.
_MAX_ALARM_COVERAGE_FRACTION = 0.5

TARGET_REGION_METHODS = ("failure_prototype_region", "rare_state_region")
EVENT_CONTROL_METHODS = ("negative_control_region",)
MATCHED_CONTROL_FAMILIES = (
    "random_occupancy",
    "time_shifted_prototype_window",
    "regime_matched_rare_region",
    "episode_label_permutation",
    "prototype_source_permutation",
    "phase_randomised_score",
)
MATCHED_CONTROL_REPETITIONS = 500


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
    transferability_rows: list[dict[str, object]] = []
    negative_control_summary_rows: list[dict[str, object]] = []
    negative_control_draw_rows: list[dict[str, object]] = []
    detection_aware_rows: list[dict[str, object]] = []
    utility_sensitivity_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    timeline_local_frames: list[pd.DataFrame] = []
    timeline_overview_frames: list[pd.DataFrame] = []
    timeline_metadata_rows: list[dict[str, object]] = []
    for dataset_id in ("metropt", "metropt2"):
        manifest_path = config.processed_root / dataset_id / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json_object(manifest_path)
        (
            rows,
            variants,
            control_summaries,
            control_draws,
            detection_aware,
            utility_sensitivity_batch,
            decomposition,
            reconciliation,
            timeline,
        ) = _run_metropt_event_comparison(dataset_id, manifest, config)
        baseline_rows.extend(rows)
        variant_rows.extend(variants)
        negative_control_summary_rows.extend(control_summaries)
        negative_control_draw_rows.extend(control_draws)
        detection_aware_rows.extend(detection_aware)
        utility_sensitivity_rows.extend(utility_sensitivity_batch)
        decomposition_rows.extend(decomposition)
        reconciliation_rows.extend(reconciliation)
        if timeline is not None:
            if not timeline.local.empty:
                timeline_local_frames.append(timeline.local)
            if not timeline.overview.empty:
                timeline_overview_frames.append(timeline.overview)
            timeline_metadata_rows.append(timeline.metadata)
    transferability_rows, compatibility_rows, transfer_distances = _cross_dataset_transfer_rows(
        config
    )

    artifacts: dict[str, str] = {}
    if baseline_rows:
        path = config.output_root / "event_baseline_comparison.csv"
        pd.DataFrame(baseline_rows).to_csv(path, index=False)
        artifacts["event_baseline_csv"] = str(path)
    if variant_rows:
        path = config.output_root / "event_variant_comparison.csv"
        pd.DataFrame(variant_rows).to_csv(path, index=False)
        artifacts["event_variant_csv"] = str(path)
    if transferability_rows:
        path = config.output_root / "target_region_transferability.csv"
        pd.DataFrame(transferability_rows).to_csv(path, index=False)
        artifacts["target_region_transferability_csv"] = str(path)
    if compatibility_rows:
        path = config.output_root / "transfer_schema_compatibility.csv"
        pd.DataFrame(compatibility_rows).to_csv(path, index=False)
        artifacts["transfer_schema_compatibility_csv"] = str(path)
    if transfer_distances is not None and not transfer_distances.empty:
        path = config.output_root / "transfer_distance_samples.csv"
        transfer_distances.to_csv(path, index=False)
        artifacts["transfer_distance_samples_csv"] = str(path)
    if negative_control_summary_rows:
        path = config.output_root / "matched_negative_controls.csv"
        pd.DataFrame(negative_control_summary_rows).to_csv(path, index=False)
        artifacts["matched_negative_controls_csv"] = str(path)
    if detection_aware_rows:
        path = config.output_root / "detection_aware_controls.csv"
        pd.DataFrame(detection_aware_rows).to_csv(path, index=False)
        artifacts["detection_aware_controls_csv"] = str(path)
    if utility_sensitivity_rows:
        path = config.output_root / "control_utility_sensitivity.csv"
        pd.DataFrame(utility_sensitivity_rows).to_csv(path, index=False)
        artifacts["control_utility_sensitivity_csv"] = str(path)
    if negative_control_draw_rows:
        path = config.output_root / "matched_negative_control_draws.csv"
        pd.DataFrame(negative_control_draw_rows).to_csv(path, index=False)
        artifacts["matched_negative_control_draws_csv"] = str(path)
    if decomposition_rows:
        path = config.output_root / "score_threshold_alarm_decomposition.csv"
        pd.DataFrame(decomposition_rows).to_csv(path, index=False)
        artifacts["score_threshold_alarm_decomposition_csv"] = str(path)
    if reconciliation_rows:
        path = config.output_root / "timeline_reconciliation.csv"
        pd.DataFrame(reconciliation_rows).to_csv(path, index=False)
        artifacts["timeline_reconciliation_csv"] = str(path)
    if timeline_local_frames:
        path = config.output_root / "event_timeline_trace.csv"
        pd.concat(timeline_local_frames, ignore_index=True).to_csv(path, index=False)
        artifacts["event_timeline_csv"] = str(path)
    if timeline_overview_frames:
        path = config.output_root / "event_timeline_overview.csv"
        pd.concat(timeline_overview_frames, ignore_index=True).to_csv(path, index=False)
        artifacts["event_timeline_overview_csv"] = str(path)
    if timeline_metadata_rows:
        path = config.output_root / "event_timeline_metadata.csv"
        pd.DataFrame(timeline_metadata_rows).to_csv(path, index=False)
        artifacts["event_timeline_metadata_csv"] = str(path)
    manifest_path = config.output_root / "event_level_comparison_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


def _run_metropt_event_comparison(
    dataset_id: str,
    manifest: dict[str, Any],
    config: IndustrialResultsConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    TimelineViews | None,
]:
    files = _manifest_files(manifest)
    rows = int(manifest.get("rows", 0))
    feature_columns = _available_columns(manifest, METROPT_FEATURE_PRIORITY)
    if not files or rows <= 0 or not feature_columns:
        return [], [], [], [], [], [], [], [], None

    train = _collect_metropt_split(files, rows, feature_columns, config, split_name="train")
    if train.empty:
        return [], [], [], [], [], [], [], [], None
    center, scale = _robust_center_scale(train, feature_columns)
    train_refs = _event_method_references(train, feature_columns, center=center, scale=scale)
    train_scores = _event_method_scores(
        train, feature_columns, center=center, scale=scale, refs=train_refs
    )
    thresholds = _event_method_thresholds(train_scores, config)
    if not thresholds:
        return [], [], [], [], [], [], [], [], None

    flags_by_method: dict[str, list[np.ndarray[Any, Any]]] = {method: [] for method in thresholds}
    scores_by_method: dict[str, list[np.ndarray[Any, Any]]] = {method: [] for method in thresholds}
    exceedances_by_method: dict[str, list[np.ndarray[Any, Any]]] = {
        method: [] for method in thresholds
    }
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
                raw_exceedances, onset_flags = _method_alarm_flags(
                    method,
                    scores[method],
                    threshold,
                    refs=train_refs,
                    config=config,
                )
                flags_by_method[method].append(onset_flags)
                exceedances_by_method[method].append(raw_exceedances)
                scores_by_method[method].append(scores[method])
            target_parts.append(test["is_failure"].astype(bool).to_numpy())
            timestamp_parts.append(test["timestamp"].to_numpy())
            robust_score_parts.append(scores["dynamical_evt_robust_score"])
        offset += len(frame)

    target_flags = np.concatenate(target_parts) if target_parts else np.array([], dtype=bool)
    failures = flags_to_events(target_flags, label="failure")
    if not failures:
        return [], [], [], [], [], [], [], [], None
    baseline_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    negative_control_summary_rows: list[dict[str, object]] = []
    negative_control_draw_rows: list[dict[str, object]] = []
    detection_aware_rows: list[dict[str, object]] = []
    utility_sensitivity_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    configuration_hash = _event_configuration_hash(config)
    for method, parts in flags_by_method.items():
        alarm_flags = np.concatenate(parts) if parts else np.array([], dtype=bool)
        # The decomposition reports exceedances before the event policy collapses them,
        # so a declustering method shows the reduction it performed rather than an
        # exceedance count identical to its cluster count.
        raw_exceedance_flags = (
            np.concatenate(exceedances_by_method[method])
            if exceedances_by_method.get(method)
            else alarm_flags
        )
        # Runtime and peak memory cover the alarm-conversion and evaluation stage, which
        # is the part that differs between methods sharing a score. The caption states
        # this scope so the numbers are not read as end-to-end cost.
        merge_gap = method_merge_gap(method, default_merge_gap=config.merge_gap)
        tracemalloc.start()
        started = time.perf_counter()
        alarms = flags_to_events(alarm_flags, label="alarm", merge_gap=merge_gap)
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
        runtime_seconds = time.perf_counter() - started
        _current, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        spec = SPECS_BY_NAME.get(method)
        row = {
            "dataset_id": dataset_id,
            "failure_id": f"{dataset_id}_test_failure_001",
            "method": method,
            "method_family": _event_method_family(method),
            "target_events": len(failures),
            "predicted_alarm_events": len(alarms),
            "detected": bool(evaluation.recall > 0.0),
            "event_recall": evaluation.recall,
            "event_precision": evaluation.precision,
            "event_f1": evaluation.f1,
            "false_alarm_events_per_day": evaluation.false_alarm_events_per_operating_day,
            # Absolute counts alongside the rate: the timeline caption has to state them,
            # and deriving them from the total silently folded duplicates into false alarms.
            "false_alarm_events": len(evaluation.matching.false_alarm_indices),
            "matched_alarm_events": len(evaluation.matching.assignments),
            "duplicate_alarm_events": evaluation.duplicate_alarm_events,
            "median_warning_lead_time": evaluation.median_warning_lead_time,
            "time_under_warning": evaluation.time_under_warning,
            "event_utility": evaluation.utility,
            # Share of the test period spent under warning. An always-on detector scores
            # a perfect episode-level precision and recall because merging leaves one
            # episode that contains the failure and no unmatched episodes to count as
            # false alarms. This column, and the flag below, are what expose that.
            "alarm_coverage_fraction": (
                float(evaluation.time_under_warning) / float(len(target_flags))
                if len(target_flags)
                else float("nan")
            ),
            "degenerate_always_on": bool(
                len(target_flags)
                and float(evaluation.time_under_warning) / float(len(target_flags))
                > _MAX_ALARM_COVERAGE_FRACTION
            ),
            "threshold": thresholds[method],
            "merge_gap_used": int(merge_gap),
            "score_kind": spec.score_kind if spec else "",
            "threshold_rule": spec.threshold_rule if spec else "",
            "event_policy": spec.event_policy if spec else "",
            "parameter_count": spec.parameter_count if spec else 0,
            "calibration_status": spec.calibration_status if spec else "",
            "tuning_partition": spec.tuning_partition if spec else "",
            "runtime_seconds": float(runtime_seconds),
            "peak_memory_bytes": int(peak_memory_bytes),
            "configuration_hash": configuration_hash,
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
        decomposition_rows.append(
            _score_threshold_alarm_decomposition_row(
                dataset_id=dataset_id,
                method=method,
                scores=np.concatenate(scores_by_method[method]),
                threshold=thresholds[method],
                alarm_flags=raw_exceedance_flags,
                target_flags=target_flags,
                alarm_count=len(alarms),
                target_count=len(failures),
                # Clusters are counted under the method's own event policy, so the
                # decomposition reflects the conversion that method actually performed.
                cluster_count=len(
                    extract_clusters(raw_exceedance_flags, run_length=max(0, merge_gap))
                ),
                matched_alarm_count=len(evaluation.matching.assignments),
                false_alarm_count=len(evaluation.matching.false_alarm_indices),
                duplicate_alarm_count=evaluation.duplicate_alarm_events,
                time_under_warning=evaluation.time_under_warning,
                event_recall=evaluation.recall,
                event_precision=evaluation.precision,
            )
        )
        if method in TARGET_REGION_METHODS:
            reconciliation_rows.append(
                _timeline_reconciliation_row(
                    dataset_id=dataset_id,
                    method=method,
                    alarms=alarms,
                    failures=failures,
                    total_operating_time=len(target_flags),
                    samples_per_day=config.samples_per_day,
                )
            )

    observed = next(
        (row for row in baseline_rows if row["method"] == "failure_prototype_region"),
        None,
    )
    observed_flags = (
        np.concatenate(flags_by_method["failure_prototype_region"])
        if "failure_prototype_region" in flags_by_method
        else np.zeros(len(target_flags), dtype=bool)
    )
    rare_flags = (
        np.concatenate(flags_by_method["rare_state_region"])
        if "rare_state_region" in flags_by_method
        else observed_flags
    )
    if observed is not None:
        negative_control_draw_rows = _matched_negative_control_draw_rows(
            dataset_id=dataset_id,
            observed_method="failure_prototype_region",
            observed_flags=observed_flags,
            rare_flags=rare_flags,
            target_flags=target_flags,
            config=config,
        )
        negative_control_summary_rows = _matched_negative_control_summary_rows(
            observed_row=observed,
            draw_rows=negative_control_draw_rows,
        )
        detection_rows, sensitivity_rows = _detection_aware_control_rows(
            observed_row=observed,
            draw_rows=negative_control_draw_rows,
            total_samples=len(target_flags),
        )
        detection_aware_rows.extend(detection_rows)
        utility_sensitivity_rows.extend(sensitivity_rows)

    timeline = None
    # Built for both compressor datasets, not just MetroPT: the local window is the
    # part a reader trusts, and it is only trustworthy next to the global burden.
    if timestamp_parts and robust_score_parts and TIMELINE_METHOD in flags_by_method:
        timeline = _build_timeline_views(
            dataset_id=dataset_id,
            timestamps=np.concatenate(timestamp_parts),
            scores=np.concatenate(robust_score_parts),
            exceedance_flags=np.concatenate(exceedances_by_method[TIMELINE_METHOD]),
            alarm_onset_flags=np.concatenate(flags_by_method[TIMELINE_METHOD]),
            target_flags=target_flags,
            threshold=thresholds[TIMELINE_METHOD],
            regime_edges=np.asarray(
                train_refs.get("regime_edges", np.array([], dtype=float)), dtype=float
            ),
            baseline_rows=baseline_rows,
            config=config,
        )
    # Every method in the main comparison gets a global overview for the supplement.
    # The binned form is 240 rows per method, so covering all of them is cheap; the
    # per-sample local trace stays limited to the registered method.
    if timeline is not None:
        timeline = TimelineViews(
            local=timeline.local,
            overview=pd.concat(
                [
                    _method_global_overview(
                        dataset_id=dataset_id,
                        method=method,
                        exceedance_flags=np.concatenate(exceedances_by_method[method]),
                        alarm_onset_flags=np.concatenate(parts),
                        target_flags=target_flags,
                        threshold=thresholds[method],
                        config=config,
                    )
                    for method, parts in flags_by_method.items()
                    if parts
                ],
                ignore_index=True,
            ),
            metadata=timeline.metadata,
        )
    return (
        baseline_rows,
        variant_rows,
        negative_control_summary_rows,
        negative_control_draw_rows,
        detection_aware_rows,
        utility_sensitivity_rows,
        decomposition_rows,
        reconciliation_rows,
        timeline,
    )


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
) -> dict[str, Any]:
    scaled = _scaled_matrix(train, feature_columns, center=center, scale=scale)
    robust = _nanmax_abs(pd.DataFrame(scaled, columns=list(feature_columns)), feature_columns)
    targets = train["is_failure"].astype(bool).to_numpy()
    references: dict[str, Any] = {}
    references["failure_prototype_region"] = _reference_subset(scaled[targets], limit=16)
    rare_count = min(16, len(scaled))
    if rare_count:
        rare_indices = np.argsort(robust)[-rare_count:]
        references["rare_state_region"] = scaled[rare_indices]
    references["negative_control_region"] = _reference_subset(scaled[~targets], limit=16)
    references["train_robust_scores"] = robust
    references["train_horizon_targets"] = _future_positive(
        targets,
        horizon=min(3_600, max(1, len(targets) // 10)),
    ).astype(float)
    clean = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    references["linear_autoencoder_components"] = _principal_components(clean, max_components=3)
    references["compact_nonlinear_autoencoder_components"] = _principal_components(
        np.tanh(clean),
        max_components=2,
    )

    # Models and selections fitted on training rows only. These were previously absent,
    # which is why several declared methods fell back to the shared robust score.
    references["best_sensor_index"] = best_single_sensor_index(clean, targets)
    try:
        references["isolation_forest_model"] = fit_isolation_forest(clean)
    except ValueError:
        references["isolation_forest_model"] = None
    # Regime bin edges are fitted here and reused unchanged on the test split, so the
    # regime-conditioned threshold never sees held-out data.
    references["regime_edges"] = np.quantile(robust, (1.0 / 3.0, 2.0 / 3.0))
    return references


def _event_method_scores(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    center: pd.Series,
    scale: pd.Series,
    refs: dict[str, Any],
) -> dict[str, np.ndarray[Any, Any]]:
    scaled = _scaled_matrix(frame, feature_columns, center=center, scale=scale)
    robust = np.nanmax(np.abs(np.where(np.isfinite(scaled), scaled, np.nan)), axis=1)
    robust = np.nan_to_num(robust, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    row_energy = np.mean(clean**2, axis=1)
    changepoint = np.zeros(len(robust), dtype=float)
    if len(robust) > 1:
        changepoint[1:] = np.abs(np.diff(robust))
    conformal = _empirical_tail_score(
        robust,
        np.asarray(refs.get("train_robust_scores", np.array([], dtype=float)), dtype=float),
    )
    risk = _empirical_horizon_risk_proxy(
        robust,
        np.asarray(refs.get("train_robust_scores", np.array([], dtype=float)), dtype=float),
        np.asarray(refs.get("train_horizon_targets", np.array([], dtype=float)), dtype=float),
    )
    linear_components = refs.get("linear_autoencoder_components")
    nonlinear_components = refs.get("compact_nonlinear_autoencoder_components")

    forest = refs.get("isolation_forest_model")
    if forest is not None:
        isolation_scores = isolation_forest_score(forest, clean)
    else:
        # Reported explicitly rather than silently substituted: without a fitted model
        # this method has no result, and row energy is not an isolation forest.
        isolation_scores = np.full(len(robust), np.nan, dtype=float)
    sensor_index = int(refs.get("best_sensor_index", 0))
    sensor_index = min(sensor_index, max(0, clean.shape[1] - 1))

    scores: dict[str, np.ndarray[Any, Any]] = {
        # These eight share the recurrence score by design; they are separated by their
        # threshold rule or their event policy, declared in EVENT_METHOD_SPECS.
        "engineering_threshold": robust,
        "global_empirical_threshold": robust,
        "regime_conditioned_empirical_threshold": robust,
        "classical_pot_gpd": robust,
        "fixed_run_declustering": robust,
        "ferro_segers_event_policy": robust,
        "k_gaps_event_policy": robust,
        "dynamical_evt_robust_score": robust,
        # These carry genuinely distinct scores.
        "best_individual_sensor_threshold": single_sensor_score(clean, sensor_index=sensor_index),
        "spot": _spot_like_score(robust),
        "isolation_forest": isolation_scores,
        "robust_online_changepoint": changepoint,
        "linear_autoencoder": _projection_residual(clean, linear_components),
        "compact_nonlinear_autoencoder": _projection_residual(np.tanh(clean), nonlinear_components),
        "conformal_anomaly": conformal,
        "empirical_horizon_risk": risk,
        "max_abs_robust_z": robust,
    }
    del row_energy
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
    # No silent fallback. Assigning the shared recurrence score to any unmapped method
    # is what previously collapsed nine declared baselines onto one computation, so an
    # unmapped method is now a hard error rather than a duplicate row.
    missing = [
        method
        for method in (*EVENT_BASELINE_METHODS, *EVENT_CONTROL_METHODS)
        if method not in scores
    ]
    target_region_methods = set(TARGET_REGION_METHODS) | set(EVENT_CONTROL_METHODS)
    unexplained = [method for method in missing if method not in target_region_methods]
    if unexplained:
        raise ValueError(
            "no score is defined for declared event methods: "
            + ", ".join(sorted(unexplained))
            + ". Every declared method must compute its own score."
        )
    return scores


def _event_method_thresholds(
    train_scores: dict[str, np.ndarray[Any, Any]],
    config: IndustrialResultsConfig,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method in (*EVENT_BASELINE_METHODS, *EVENT_CONTROL_METHODS):
        scores = train_scores.get(method)
        if scores is None:
            continue
        finite = scores[np.isfinite(scores)]
        if not len(finite):
            continue
        if method == "engineering_threshold":
            threshold = 3.0
        elif method == "conformal_anomaly":
            threshold = config.threshold_quantile
        elif method == "spot":
            threshold = 0.0
        elif method == "classical_pot_gpd":
            # A peaks-over-threshold baseline that reuses the empirical quantile is not
            # a tail model, so this inverts a fitted generalised Pareto instead.
            threshold = gpd_return_level(
                finite,
                exceedance_quantile=min(0.95, config.threshold_quantile),
                target_quantile=max(config.threshold_quantile, 0.99),
            )
        else:
            threshold = float(np.nanquantile(finite, config.threshold_quantile))

        # Degeneracy guard. A threshold at or below the score minimum flags almost every
        # sample; alarm merging then collapses the whole test period into one episode
        # that trivially contains the failure, scoring recall and precision of 1.0 for a
        # detector that is permanently on. That is a metric artifact, not a detection,
        # so the threshold is lifted to the first value that actually discriminates.
        if method not in {"conformal_anomaly", "spot"}:
            implied_rate = float(np.mean(finite > threshold))
            if implied_rate > _MAX_TRAIN_EXCEEDANCE_RATE:
                positive = finite[finite > float(np.nanmin(finite))]
                if positive.size:
                    threshold = float(np.nanquantile(positive, config.threshold_quantile))
                    implied_rate = float(np.mean(finite > threshold))
                if implied_rate > _MAX_TRAIN_EXCEEDANCE_RATE:
                    # Still degenerate: this score cannot support a threshold on this
                    # dataset, and saying so is better than reporting a perfect score.
                    continue
        thresholds[method] = float(threshold)
    return thresholds


def _method_alarm_flags(
    method: str,
    scores: np.ndarray[Any, Any],
    threshold: float,
    *,
    refs: dict[str, Any],
    config: IndustrialResultsConfig,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return raw threshold exceedances and the post-policy alarm-onset flags.

    Both are needed by the decomposition. A declustering policy collapses a run of
    exceedances to a single onset, so reporting its onset flags as the exceedance stage
    would show a method whose exceedance and cluster counts are identical and hide the
    reduction the declustering actually performed.

    Methods sharing a score are separated here: the threshold rule decides which samples
    exceed, and the event policy decides how a run of exceedances becomes alarm onsets.
    """

    spec = SPECS_BY_NAME.get(method)
    values = np.asarray(scores, dtype=float)

    if spec is not None and spec.threshold_rule == "regime_conditioned_quantile":
        edges = np.asarray(refs.get("regime_edges", np.array([], dtype=float)), dtype=float)
        if edges.size:
            regimes = np.asarray(np.digitize(values, edges), dtype=int)
            regime_thresholds = {int(regime): float(threshold) for regime in np.unique(regimes)}
            train_scores = np.asarray(
                refs.get("train_robust_scores", np.array([], dtype=float)), dtype=float
            )
            train_regimes = np.asarray(np.digitize(train_scores, edges), dtype=int)
            if train_scores.size:
                regime_thresholds = regime_conditioned_thresholds(
                    train_scores, train_regimes, quantile=config.threshold_quantile
                )
            exceedances = apply_regime_thresholds(
                values, regimes, regime_thresholds, fallback=threshold
            )
        else:
            exceedances = values > threshold
    else:
        exceedances = np.asarray(values > threshold, dtype=bool)

    policy = spec.event_policy if spec is not None else "merge_gap"
    if policy == "fixed_run_declustering":
        return exceedances, declustered_flags(exceedances, run_length=config.merge_gap)
    if policy == "ferro_segers_run_length":
        run_length = derived_run_length(exceedances, estimator="ferro_segers")
        return exceedances, declustered_flags(exceedances, run_length=run_length)
    if policy == "k_gaps_run_length":
        run_length = derived_run_length(exceedances, estimator="k_gaps")
        return exceedances, declustered_flags(exceedances, run_length=run_length)
    if policy == "extremal_index_declustering":
        run_length = derived_run_length(exceedances, estimator="ferro_segers")
        return exceedances, declustered_flags(exceedances, run_length=max(1, run_length // 2))
    return exceedances, exceedances


def _method_run_length(method: str, exceedances: BoolArray, config: IndustrialResultsConfig) -> int:
    """Return the declustering run length a method actually used.

    The timeline caption has to state this, and for the estimator-driven policies it is
    derived from the exceedance pattern rather than declared, so it cannot be read off
    the configuration.
    """

    spec = SPECS_BY_NAME.get(method)
    policy = spec.event_policy if spec is not None else "merge_gap"
    if policy == "fixed_run_declustering":
        return int(config.merge_gap)
    if policy == "ferro_segers_run_length":
        return derived_run_length(exceedances, estimator="ferro_segers")
    if policy == "k_gaps_run_length":
        return derived_run_length(exceedances, estimator="k_gaps")
    if policy == "extremal_index_declustering":
        return max(1, derived_run_length(exceedances, estimator="ferro_segers") // 2)
    return 0


def _principal_components(
    values: np.ndarray[Any, Any],
    *,
    max_components: int,
) -> np.ndarray[Any, Any]:
    if values.size == 0:
        return np.empty((0, 0), dtype=float)
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2 or clean.shape[0] < 2:
        return np.empty((0, clean.shape[1] if clean.ndim == 2 else 0), dtype=float)
    centered = clean - np.mean(clean, axis=0)
    component_count = max(1, min(max_components, centered.shape[0] - 1, centered.shape[1]))
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return cast(np.ndarray[Any, Any], vt[:component_count])


def _projection_residual(
    values: np.ndarray[Any, Any],
    components: np.ndarray[Any, Any] | None,
) -> np.ndarray[Any, Any]:
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if components is None or components.size == 0:
        return cast(np.ndarray[Any, Any], np.mean(clean**2, axis=1))
    component_matrix = np.asarray(components, dtype=float)
    reconstructed = clean @ component_matrix.T @ component_matrix
    return cast(np.ndarray[Any, Any], np.mean((clean - reconstructed) ** 2, axis=1))


def _empirical_tail_score(
    scores: np.ndarray[Any, Any],
    train_scores: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    finite_train = train_scores[np.isfinite(train_scores)]
    if not len(finite_train):
        return np.zeros(len(scores), dtype=float)
    sorted_train = np.sort(finite_train)
    ranks = np.searchsorted(sorted_train, scores, side="left")
    p_values = (1 + len(sorted_train) - ranks) / (len(sorted_train) + 1)
    return cast(np.ndarray[Any, Any], 1.0 - p_values)


def _empirical_horizon_risk_proxy(
    scores: np.ndarray[Any, Any],
    train_scores: np.ndarray[Any, Any],
    train_targets: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    finite = np.isfinite(train_scores) & np.isfinite(train_targets)
    if not np.any(finite) or float(np.sum(train_targets[finite])) == 0.0:
        return _rank_score(scores)
    quantiles = np.quantile(train_scores[finite], np.linspace(0.0, 1.0, 6))
    bins = np.digitize(scores, quantiles[1:-1], right=False)
    train_bins = np.digitize(train_scores[finite], quantiles[1:-1], right=False)
    risk_by_bin: dict[int, float] = {}
    for bin_id in range(len(quantiles) - 1):
        mask = train_bins == bin_id
        risk_by_bin[bin_id] = float(np.mean(train_targets[finite][mask])) if np.any(mask) else 0.0
    return cast(
        np.ndarray[Any, Any], np.asarray([risk_by_bin.get(int(item), 0.0) for item in bins])
    )


def _rank_score(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    finite_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if not len(finite_scores):
        return finite_scores
    order = np.argsort(np.argsort(finite_scores, kind="mergesort"), kind="mergesort")
    denominator = max(1, len(finite_scores) - 1)
    return cast(np.ndarray[Any, Any], order.astype(float) / denominator)


def _future_positive(targets: np.ndarray[Any, Any], *, horizon: int) -> np.ndarray[Any, Any]:
    result = np.zeros(len(targets), dtype=bool)
    positive = np.flatnonzero(targets)
    for index in positive:
        start = max(0, index - horizon)
        result[start : index + 1] = True
    return cast(np.ndarray[Any, Any], result)


def _spot_like_score(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    clean = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if len(clean) < 2:
        return clean
    prefix_max = np.maximum.accumulate(clean)
    lagged = np.concatenate([[prefix_max[0]], prefix_max[:-1]])
    return cast(np.ndarray[Any, Any], clean - lagged)


def _matched_negative_control_draw_rows(
    *,
    dataset_id: str,
    observed_method: str,
    observed_flags: np.ndarray[Any, Any],
    rare_flags: np.ndarray[Any, Any],
    target_flags: np.ndarray[Any, Any],
    config: IndustrialResultsConfig,
) -> list[dict[str, object]]:
    targets = np.asarray(target_flags, dtype=bool)
    failures = flags_to_events(targets, label="failure")
    observed = np.asarray(observed_flags, dtype=bool)
    rare = np.asarray(rare_flags, dtype=bool)
    observed_indices = np.flatnonzero(observed)
    observed_alarms = flags_to_events(observed, label="alarm", merge_gap=config.merge_gap)
    rare_alarms = flags_to_events(rare, label="alarm", merge_gap=config.merge_gap)
    rows: list[dict[str, object]] = []
    for family in MATCHED_CONTROL_FAMILIES:
        rng = np.random.default_rng(_stable_seed(dataset_id, family))
        for draw in range(MATCHED_CONTROL_REPETITIONS):
            alarms = _matched_control_alarms(
                family=family,
                rng=rng,
                draw=draw,
                observed_indices=observed_indices,
                observed_alarms=observed_alarms,
                rare_alarms=rare_alarms,
                length=len(targets),
            )
            draw_failures = (
                _shift_events(failures, shift=draw + 1, length=len(targets), label="failure")
                if family == "episode_label_permutation"
                else failures
            )
            evaluation = evaluate_event_predictions(
                alarms,
                draw_failures,
                policy=EarlyWarningPolicy(
                    horizon=config.horizon,
                    tolerance_after=config.matching_tolerance_after,
                ),
                method="optimal",
                total_operating_time=len(target_flags),
                samples_per_day=config.samples_per_day,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "failure_id": f"{dataset_id}_test_failure_001",
                    "observed_method": observed_method,
                    "control_family": family,
                    "draw": draw,
                    "event_recall": evaluation.recall,
                    "event_precision": evaluation.precision,
                    "false_alarm_events_per_day": evaluation.false_alarm_events_per_operating_day,
                    "median_warning_lead_time": evaluation.median_warning_lead_time,
                    "time_under_warning": evaluation.time_under_warning,
                    # The observed method's occupancy is constant across draws and is
                    # kept for reference; the control's own occupancy is what an
                    # occupancy-matched comparison must condition on, and recording the
                    # observed value in its place made that comparison impossible.
                    "region_occupancy": float(len(observed_indices) / max(1, len(targets))),
                    "control_occupancy": float(
                        sum(alarm.duration for alarm in alarms) / max(1, len(targets))
                    ),
                    "cluster_count": len(alarms),
                    "duplicate_alarm_events": evaluation.duplicate_alarm_events,
                    "alarm_coverage_fraction": (
                        float(evaluation.time_under_warning) / float(max(1, len(targets)))
                    ),
                    "detected": bool(evaluation.recall > 0.0),
                    "event_utility": joint_utility(
                        detected=bool(evaluation.recall > 0.0),
                        false_alarm_events_per_day=float(
                            evaluation.false_alarm_events_per_operating_day or 0.0
                        ),
                        alarm_coverage_fraction=(
                            float(evaluation.time_under_warning) / float(max(1, len(targets)))
                        ),
                        median_warning_lead_time=evaluation.median_warning_lead_time,
                    ),
                    "matching_basis": _control_matching_basis(family),
                }
            )
    return rows


def _matched_control_alarms(
    *,
    family: str,
    rng: np.random.Generator,
    draw: int,
    observed_indices: np.ndarray[Any, Any],
    observed_alarms: list[EventInterval],
    rare_alarms: list[EventInterval],
    length: int,
) -> list[EventInterval]:
    if length == 0:
        return []
    occupancy = int(len(observed_indices))
    if occupancy == 0:
        return []
    if family == "random_occupancy":
        return _random_matched_events(observed_alarms, length=length, rng=rng, label="alarm")
    if family == "time_shifted_prototype_window":
        shift = int(rng.integers(1, length))
        return _shift_events(observed_alarms, shift=shift, length=length, label="alarm")
    if family == "regime_matched_rare_region":
        source = rare_alarms if rare_alarms else observed_alarms
        return _random_matched_events(source, length=length, rng=rng, label="alarm")
    if family == "episode_label_permutation":
        shift = int((draw + 1) * max(1, length // MATCHED_CONTROL_REPETITIONS))
        return _shift_events(observed_alarms, shift=shift % length, length=length, label="alarm")
    if family == "prototype_source_permutation":
        return _random_matched_events(observed_alarms, length=length, rng=rng, label="alarm")
    if family == "phase_randomised_score":
        source = rare_alarms if rare_alarms else observed_alarms
        shift = int(rng.integers(1, length))
        return _shift_events(source, shift=shift, length=length, label="alarm")
    return observed_alarms


def _random_matched_events(
    source_events: list[EventInterval],
    *,
    length: int,
    rng: np.random.Generator,
    label: str,
) -> list[EventInterval]:
    if length == 0 or not source_events:
        return []
    durations = np.asarray([event.duration for event in source_events], dtype=int)
    starts = rng.integers(0, length, size=len(durations))
    events: list[EventInterval] = []
    for start, duration in zip(starts, durations, strict=True):
        safe_duration = max(1, min(int(duration), length))
        end = min(length - 1, int(start) + safe_duration - 1)
        events.append(EventInterval(start=int(start), end=end, label=label))
    return _merge_event_intervals(events)


def _indices_to_events(
    indices: np.ndarray[Any, Any],
    *,
    label: str,
    merge_gap: int,
) -> list[EventInterval]:
    clean = sorted({int(index) for index in indices if np.isfinite(index)})
    if not clean:
        return []
    events = [EventInterval(start=index, end=index, label=label) for index in clean]
    merged: list[EventInterval] = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        if event.start - previous.end - 1 <= merge_gap:
            merged[-1] = EventInterval(
                start=previous.start,
                end=max(previous.end, event.end),
                label=previous.label,
            )
        else:
            merged.append(event)
    return merged


def _shift_events(
    events: list[EventInterval],
    *,
    shift: int,
    length: int,
    label: str,
) -> list[EventInterval]:
    if length == 0:
        return []
    shifted: list[EventInterval] = []
    for event in events:
        start = (event.start + shift) % length
        end = (event.end + shift) % length
        if start <= end:
            shifted.append(EventInterval(start=start, end=end, label=label))
        else:
            shifted.append(EventInterval(start=0, end=end, label=label))
            shifted.append(EventInterval(start=start, end=length - 1, label=label))
    return _merge_event_intervals(shifted)


def _merge_event_intervals(events: list[EventInterval]) -> list[EventInterval]:
    if not events:
        return []
    ordered = sorted(events, key=lambda event: (event.start, event.end))
    merged: list[EventInterval] = [ordered[0]]
    for event in ordered[1:]:
        previous = merged[-1]
        if event.start <= previous.end + 1:
            merged[-1] = EventInterval(
                start=previous.start,
                end=max(previous.end, event.end),
                label=previous.label,
            )
        else:
            merged.append(event)
    return merged


def _matched_negative_control_summary_rows(
    *,
    observed_row: dict[str, object],
    draw_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not draw_rows:
        return []
    draws = pd.DataFrame(draw_rows)
    metrics = {
        "event_recall": _object_float(observed_row["event_recall"]),
        "event_precision": _object_float(observed_row["event_precision"]),
        "false_alarm_events_per_day": _object_float(observed_row["false_alarm_events_per_day"]),
        "time_under_warning": _object_float(observed_row["time_under_warning"]),
    }
    rows: list[dict[str, object]] = []
    for family, group in draws.groupby("control_family", sort=True):
        for metric, observed_value in metrics.items():
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            percentile = float(np.mean(values <= observed_value))
            p_value = float((1 + np.count_nonzero(values >= observed_value)) / (len(values) + 1))
            rows.append(
                {
                    "dataset_id": observed_row["dataset_id"],
                    "failure_id": observed_row["failure_id"],
                    "observed_method": observed_row["method"],
                    "control_family": family,
                    "metric": metric,
                    "observed_value": observed_value,
                    "control_median": float(np.median(values)),
                    "control_p025": float(np.quantile(values, 0.025)),
                    "control_p975": float(np.quantile(values, 0.975)),
                    "observed_percentile": percentile,
                    "empirical_p_value": p_value,
                    "draws": int(len(values)),
                    "interpretation": _control_interpretation(metric, observed_value, values),
                }
            )
    return rows


def _detection_aware_control_rows(
    *,
    observed_row: dict[str, object],
    draw_rows: list[dict[str, object]],
    total_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the detection-aware control comparison and its weight sensitivity.

    The per-metric summary compares burden without reference to whether the failure was
    found. That rewards a region for not detecting: a region raising no alarms has a
    false-alarm rate of zero and beats every control on burden while missing the event.
    These rows carry the control detection rate and a predeclared joint utility, and
    repeat the comparison restricted to controls that detect, that match the observed
    recall, and that match the observed occupancy.
    """

    if not draw_rows:
        return [], []
    draws = add_detection_columns(pd.DataFrame(draw_rows), total_samples=total_samples)
    observed_recall = _object_float(observed_row["event_recall"])
    observed_coverage = _object_float(observed_row.get("alarm_coverage_fraction", float("nan")))
    if not np.isfinite(observed_coverage):
        observed_coverage = _object_float(observed_row["time_under_warning"]) / float(
            max(1, total_samples)
        )
    lead = observed_row.get("median_warning_lead_time")
    observed_lead = _object_float(lead) if lead is not None else None

    summary = detection_aware_summary(
        draws,
        observed_detected=observed_recall > 0.0,
        observed_recall=observed_recall,
        observed_occupancy=_object_float(observed_row.get("alarm_coverage_fraction", float("nan"))),
        observed_false_alarms_per_day=_object_float(observed_row["false_alarm_events_per_day"]),
        observed_alarm_coverage=observed_coverage,
        observed_lead_time=observed_lead if observed_lead and np.isfinite(observed_lead) else None,
    )
    sensitivity = utility_sensitivity(
        draws,
        observed_detected=observed_recall > 0.0,
        observed_false_alarms_per_day=_object_float(observed_row["false_alarm_events_per_day"]),
        observed_alarm_coverage=observed_coverage,
        observed_lead_time=observed_lead if observed_lead and np.isfinite(observed_lead) else None,
    )
    if not sensitivity.empty:
        sensitivity.insert(0, "observed_method", str(observed_row["method"]))
        sensitivity.insert(0, "dataset_id", str(observed_row["dataset_id"]))

    def _to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
        return [
            {str(key): value for key, value in record.items()}
            for record in frame.to_dict("records")
        ]

    return (
        _to_records(summary) if not summary.empty else [],
        _to_records(sensitivity) if not sensitivity.empty else [],
    )


def _object_float(value: object) -> float:
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str) and value:
        return float(value)
    return 0.0


def _control_interpretation(
    metric: str,
    observed_value: float,
    control_values: np.ndarray[Any, Any],
) -> str:
    median = float(np.median(control_values)) if len(control_values) else np.nan
    if metric in {"event_recall", "event_precision"}:
        return "above matched controls" if observed_value > median else "not above controls"
    if metric == "false_alarm_events_per_day":
        return "lower alarm burden than controls" if observed_value < median else "not lower burden"
    if metric == "time_under_warning":
        return (
            "less warning exposure than controls"
            if observed_value < median
            else "not less exposure"
        )
    return "descriptive control comparison"


def _control_matching_basis(family: str) -> str:
    bases = {
        "random_occupancy": "preserves observed region occupancy",
        "time_shifted_prototype_window": "preserves temporal dependence by circular shift",
        "regime_matched_rare_region": "uses rare-state occupancy with shifted timing",
        "episode_label_permutation": "preserves prototype flags and changes event alignment",
        "prototype_source_permutation": "permutes prototype flag blocks",
        "phase_randomised_score": "permutes rare-score blocks as a phase-randomised proxy",
    }
    return bases.get(family, "matched negative-control draw")


def _stable_seed(*parts: str) -> int:
    payload = "::".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _score_threshold_alarm_decomposition_row(
    *,
    dataset_id: str,
    method: str,
    scores: np.ndarray[Any, Any],
    threshold: float,
    alarm_flags: np.ndarray[Any, Any],
    target_flags: np.ndarray[Any, Any],
    alarm_count: int,
    target_count: int,
    cluster_count: int,
    matched_alarm_count: int,
    false_alarm_count: int,
    duplicate_alarm_count: int,
    time_under_warning: int,
    event_recall: float,
    event_precision: float,
) -> dict[str, object]:
    """Record one method's decomposition, keeping detector counts and ground truth apart.

    The detector track runs test observations, threshold exceedances, extreme clusters,
    alarm episodes, matched alarm episodes. Labelled failures are ground truth and are
    recorded separately: they are not produced by the detector, and placing them at the
    end of a detector count flow implies a derivation that does not exist.

    Units differ by stage and are named in the column, because a count of samples, a
    count of clusters and a count of episodes are not comparable quantities.
    """

    finite = scores[np.isfinite(scores)]
    exceedances = int(np.count_nonzero(alarm_flags))
    return {
        "dataset_id": dataset_id,
        "failure_id": f"{dataset_id}_test_failure_001",
        "method": method,
        "score_median": float(np.median(finite)) if len(finite) else np.nan,
        "score_p95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
        "threshold": threshold,
        # Detector evidence track.
        "test_observations_samples": int(len(alarm_flags)),
        "threshold_exceedances_samples": exceedances,
        "extreme_clusters_count": cluster_count,
        "alarm_episodes_count": alarm_count,
        "matched_alarm_episodes_count": matched_alarm_count,
        "false_alarm_episodes_count": false_alarm_count,
        "duplicate_alarm_episodes_count": duplicate_alarm_count,
        # Ground-truth track, kept separate from the detector counts above.
        "labelled_failure_events_count": target_count,
        "labelled_failure_samples": int(np.count_nonzero(target_flags)),
        # Outcome.
        "event_recall": event_recall,
        "event_precision": event_precision,
        "time_under_warning_samples": time_under_warning,
        "exceedance_to_alarm_ratio": (float(exceedances / alarm_count) if alarm_count else np.nan),
        "leakage_control": "scores evaluated on test split; thresholds fitted before test split",
    }


#: The registered method the main-paper timeline is drawn for.
TIMELINE_METHOD = "dynamical_evt_robust_score"


@dataclass(frozen=True)
class TimelineViews:
    """The two scales plus the metadata needed to identify what produced them."""

    local: pd.DataFrame
    overview: pd.DataFrame
    metadata: dict[str, object]


def _build_timeline_views(
    *,
    dataset_id: str,
    timestamps: np.ndarray[Any, Any],
    scores: np.ndarray[Any, Any],
    exceedance_flags: np.ndarray[Any, Any],
    alarm_onset_flags: np.ndarray[Any, Any],
    target_flags: np.ndarray[Any, Any],
    threshold: float,
    regime_edges: np.ndarray[Any, Any],
    baseline_rows: list[dict[str, object]],
    config: IndustrialResultsConfig,
) -> TimelineViews | None:
    """Assemble both timeline scales from the series the event tables were built from.

    The counts are taken from the same ``baseline_rows`` entry the event table prints,
    rather than recomputed here, so the reconciliation check compares the figure against
    the table instead of comparing two copies of the same calculation.
    """
    row = next(
        (
            candidate
            for candidate in baseline_rows
            if candidate["dataset_id"] == dataset_id and candidate["method"] == TIMELINE_METHOD
        ),
        None,
    )
    if row is None:
        return None

    merge_gap = method_merge_gap(TIMELINE_METHOD, default_merge_gap=config.merge_gap)
    spec = SPECS_BY_NAME.get(TIMELINE_METHOD)
    inputs = TimelineInputs(
        dataset_id=dataset_id,
        failure_id=f"{dataset_id}_test_failure_001",
        method=TIMELINE_METHOD,
        timestamps=timestamps,
        scores=np.asarray(scores, dtype=float),
        threshold=float(threshold),
        exceedance_flags=np.asarray(exceedance_flags, dtype=bool),
        alarm_onset_flags=np.asarray(alarm_onset_flags, dtype=bool),
        target_flags=np.asarray(target_flags, dtype=bool),
        regime_edges=np.asarray(regime_edges, dtype=float),
        samples_per_day=config.samples_per_day,
        merge_gap=int(merge_gap),
        horizon=int(config.horizon),
        matching_tolerance_after=int(config.matching_tolerance_after),
        threshold_quantile=float(config.threshold_quantile),
        run_length=_method_run_length(
            TIMELINE_METHOD, np.asarray(exceedance_flags, dtype=bool), config
        ),
        target_region_variant=spec.score_kind if spec is not None else "",
    )

    local = build_local_trace(inputs)
    overview = build_global_overview(inputs)
    lead = row.get("median_warning_lead_time")
    lead_value = float(lead) if isinstance(lead, int | float) and not pd.isna(lead) else None
    metadata = build_timeline_metadata(
        inputs,
        local_trace=local,
        matched_alarm_count=int(cast(float, row.get("matched_alarm_events", 0)) or 0),
        false_alarm_count=int(cast(float, row.get("false_alarm_events", 0)) or 0),
        false_alarms_per_day=float(cast(float, row.get("false_alarm_events_per_day", 0.0)) or 0.0),
        lead_time_samples=lead_value,
        time_under_warning=int(cast(float, row.get("time_under_warning", 0)) or 0),
    )
    return TimelineViews(local=local, overview=overview, metadata=metadata)


def _method_global_overview(
    *,
    dataset_id: str,
    method: str,
    exceedance_flags: np.ndarray[Any, Any],
    alarm_onset_flags: np.ndarray[Any, Any],
    target_flags: np.ndarray[Any, Any],
    threshold: float,
    config: IndustrialResultsConfig,
) -> pd.DataFrame:
    """Binned global alarm burden for one method, for the supplementary timelines."""
    inputs = TimelineInputs(
        dataset_id=dataset_id,
        failure_id=f"{dataset_id}_test_failure_001",
        method=method,
        timestamps=np.empty(0),
        scores=np.zeros(target_flags.shape[0], dtype=float),
        threshold=float(threshold),
        exceedance_flags=np.asarray(exceedance_flags, dtype=bool),
        alarm_onset_flags=np.asarray(alarm_onset_flags, dtype=bool),
        target_flags=np.asarray(target_flags, dtype=bool),
        regime_edges=np.array([], dtype=float),
        samples_per_day=config.samples_per_day,
        merge_gap=int(method_merge_gap(method, default_merge_gap=config.merge_gap)),
        horizon=int(config.horizon),
        matching_tolerance_after=int(config.matching_tolerance_after),
        threshold_quantile=float(config.threshold_quantile),
        run_length=0,
        target_region_variant="",
    )
    return build_global_overview(inputs)


def _timeline_reconciliation_row(
    *,
    dataset_id: str,
    method: str,
    alarms: list[Any],
    failures: list[Any],
    total_operating_time: int,
    samples_per_day: int,
) -> dict[str, object]:
    first_failure = failures[0]
    first_alarm_before = [
        alarm for alarm in alarms if alarm.start <= first_failure.start and alarm.end >= 0
    ]
    selected_alarm = first_alarm_before[-1] if first_alarm_before else None
    lead_samples = (
        int(first_failure.start - selected_alarm.start) if selected_alarm is not None else None
    )
    return {
        "dataset_id": dataset_id,
        "failure_id": f"{dataset_id}_test_failure_001",
        "method": method,
        "target_event_start_index": int(first_failure.start),
        "target_event_end_index": int(first_failure.end),
        "matched_alarm_start_index": int(selected_alarm.start)
        if selected_alarm is not None
        else "",
        "matched_alarm_end_index": int(selected_alarm.end) if selected_alarm is not None else "",
        "lead_samples": lead_samples if lead_samples is not None else "",
        "lead_hours": (
            float(lead_samples / max(1.0, samples_per_day / 24.0))
            if lead_samples is not None
            else ""
        ),
        "total_alarm_events": len(alarms),
        "total_target_events": len(failures),
        "operating_days": float(total_operating_time / samples_per_day),
        "reconciliation_status": "matched_before_failure"
        if selected_alarm is not None
        else "no_alarm_before_failure",
    }


def _transfer_splits(
    dataset_id: str,
    config: IndustrialResultsConfig,
    features: tuple[str, ...],
) -> TransferSplits | None:
    """Load one dataset's registered train, validation and test splits for transfer.

    The splits come from the same temporal partition the rest of the benchmark uses, so
    transfer numbers are comparable with the event-level rows rather than produced under
    an ad-hoc split of their own.
    """

    manifest_path = config.processed_root / dataset_id / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json_object(manifest_path)
    files = _manifest_files(manifest)
    rows = int(manifest.get("rows", 0))
    if not files or rows <= 0:
        return None
    available = _available_columns(manifest, features)
    if not available:
        return None
    columns = tuple([*available, "timestamp", "is_failure"])
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    offset = 0
    for path in files:
        frame = pd.read_parquet(path, columns=list(columns))
        split = _temporal_split(offset, len(frame), rows, config)
        for name in parts:
            selected = frame.loc[split == name]
            if not selected.empty:
                parts[name].append(selected)
        offset += len(frame)
    if not parts["train"] or not parts["test"]:
        return None
    return TransferSplits(
        dataset=dataset_id,
        train=pd.concat(parts["train"], ignore_index=True),
        validation=(
            pd.concat(parts["validation"], ignore_index=True)
            if parts["validation"]
            else pd.concat(parts["train"], ignore_index=True)
        ),
        test=pd.concat(parts["test"], ignore_index=True),
    )


def _cross_dataset_transfer_rows(
    config: IndustrialResultsConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame | None]:
    """Run genuine transfer between the compressor datasets, in both directions.

    This replaces a table whose cross-dataset rows were byte-for-byte copies of the
    destination's own within-dataset row: the source label was decorative and nothing
    crossed between datasets. Every row here applies a region fitted on the source to
    the destination's test split, and records what was frozen and what was refitted.
    """

    candidates = METROPT_FEATURE_PRIORITY
    splits: dict[str, TransferSplits] = {}
    for dataset_id in ("metropt", "metropt2"):
        loaded = _transfer_splits(dataset_id, config, candidates)
        if loaded is not None:
            splits[dataset_id] = loaded
    if len(splits) < 2:
        return [], [], None

    source_id, destination_id = "metropt", "metropt2"
    report = assess_schema_compatibility(
        splits[source_id].train,
        splits[destination_id].train,
        source=source_id,
        destination=destination_id,
        candidate_features=candidates,
    )
    compatibility = report.to_frame()
    compatibility.insert(0, "destination_dataset_id", destination_id)
    compatibility.insert(0, "source_dataset_id", source_id)
    compatibility_rows: list[dict[str, object]] = [
        {str(key): value for key, value in record.items()}
        for record in compatibility.to_dict("records")
    ]

    if not report.transferable:
        return [], compatibility_rows, None

    features = report.compatible_features
    results: list[TransferResult] = []
    for source_name, destination_name in ((source_id, destination_id), (destination_id, source_id)):
        results.extend(
            run_transfer_protocols(
                splits[source_name],
                splits[destination_name],
                features,
                samples_per_day=config.samples_per_day,
                merge_gap=config.merge_gap,
                horizon=config.horizon,
                matching_tolerance_after=config.matching_tolerance_after,
                threshold_quantile=config.threshold_quantile,
            )
        )
    frame = transfer_results_frame(results)
    if frame.empty:
        return [], compatibility_rows, None
    frame["excluded_features"] = ", ".join(
        item.feature for item in report.features if not item.compatible
    )
    # The transfer runs on the transferable feature subset, so its destination_refit is
    # not the same construction as the full-feature failure-prototype region in the
    # event-level benchmark. Labelling the feature set prevents the two within-dataset
    # numbers from being read as a contradiction.
    frame["feature_set"] = f"transferable subset ({len(features)} of {len(candidates)})"
    distance_frames: list[pd.DataFrame] = []
    for source_name, destination_name in ((source_id, destination_id), (destination_id, source_id)):
        region = fit_target_region(
            splits[source_name].train,
            features,
            dataset=source_name,
            threshold_quantile=config.threshold_quantile,
        )
        if region is None:
            continue
        distance_frames.append(
            transfer_distance_samples(
                region,
                splits[source_name].train,
                splits[destination_name].test,
                source=source_name,
                destination=destination_name,
                center=region.center,
                scale=region.scale,
                threshold=region.threshold,
            )
        )
    distances = pd.concat(distance_frames, ignore_index=True) if distance_frames else None

    transfer_rows: list[dict[str, object]] = [
        {str(key): value for key, value in record.items()} for record in frame.to_dict("records")
    ]
    return transfer_rows, compatibility_rows, distances


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


def _event_configuration_hash(config: IndustrialResultsConfig) -> str:
    """Return a stable hash of the settings that define one benchmark run.

    Every method in a run shares this hash, so a reader can confirm that the rows were
    produced under one event policy rather than assembled from different runs.
    """

    payload = json.dumps(
        {
            "threshold_quantile": config.threshold_quantile,
            "merge_gap": config.merge_gap,
            "horizon": config.horizon,
            "matching_tolerance_after": config.matching_tolerance_after,
            "samples_per_day": config.samples_per_day,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
