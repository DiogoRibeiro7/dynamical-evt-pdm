"""Preparation pipelines from immutable raw files to processed Parquet parts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from dyn_evt_pdm.data.io import iter_csv_chunks, write_parquet
from dyn_evt_pdm.data.metropt import METROPT2_FAILURES, TimestampedFailure, annotate_failures
from dyn_evt_pdm.data.scania import validate_operational_readouts

METROPT_ALIASES = {
    "TP2": "pressure_tp2",
    "TP3": "pressure_tp3",
    "H1": "pressure_h1",
    "DV_pressure": "diverter_valve_pressure",
    "Reservoirs": "reservoir_pressure",
    "Oil_temperature": "oil_temperature",
    "Flowmeter": "flowmeter",
    "Motor_current": "motor_current",
    "COMP": "compressor_active",
    "DV_eletric": "diverter_valve_electric",
    "Towers": "towers_active",
    "MPG": "mpg_active",
    "LPS": "low_pressure_switch",
    "Pressure_switch": "pressure_switch",
    "Oil_level": "oil_level",
    "Caudal_impulses": "caudal_impulses",
    "gpsLong": "gps_longitude",
    "gpsLat": "gps_latitude",
    "gpsSpeed": "gps_speed",
    "gpsQuality": "gps_quality",
}


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Summary of one dataset preparation run."""

    dataset: str
    raw_path: str
    output_root: str
    manifest_path: str
    rows: int
    chunks: int
    parquet_files: list[str]


@dataclass(slots=True)
class ColumnStats:
    """Streaming schema statistics for one column."""

    dtype: str | None = None
    missing: int = 0
    numeric_min: float | None = None
    numeric_max: float | None = None
    unique_values: set[str] | None = None
    unique_overflow: bool = False

    def update(self, series: pd.Series, *, unique_limit: int) -> None:
        """Accumulate dtype, missingness, finite range and bounded uniqueness."""

        self.dtype = str(series.dtype)
        self.missing += int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if len(finite):
            current_min = float(finite.min())
            current_max = float(finite.max())
            self.numeric_min = (
                current_min if self.numeric_min is None else min(self.numeric_min, current_min)
            )
            self.numeric_max = (
                current_max if self.numeric_max is None else max(self.numeric_max, current_max)
            )
        if self.unique_overflow:
            return
        values = set(series.dropna().astype(str).unique())
        if self.unique_values is None:
            self.unique_values = set()
        self.unique_values.update(values)
        if len(self.unique_values) > unique_limit:
            self.unique_values = None
            self.unique_overflow = True

    def to_manifest(self, *, rows: int) -> dict[str, Any]:
        """Serialize statistics to JSON-compatible manifest fields."""

        unique_count = (
            None if self.unique_overflow or self.unique_values is None else len(self.unique_values)
        )
        return {
            "dtype": self.dtype,
            "missing_count": self.missing,
            "missing_rate": self.missing / rows if rows else None,
            "min": self.numeric_min,
            "max": self.numeric_max,
            "unique_count": unique_count,
            "unique_count_exact": not self.unique_overflow,
        }


def prepare_metropt(
    *,
    dataset: str,
    raw_path: Path,
    output_root: Path,
    timestamp_column: str = "timestamp",
    failure_yaml: Path | None = None,
    chunk_size: int = 500_000,
) -> PreparationResult:
    """Prepare MetroPT-family telemetry into Parquet parts with labels and manifest."""

    if dataset not in {"metropt", "metropt2"}:
        raise ValueError("dataset must be 'metropt' or 'metropt2'")
    failures = _load_failures(dataset=dataset, failure_yaml=failure_yaml)
    output_root.mkdir(parents=True, exist_ok=True)
    _clear_parquet_parts(output_root)

    stats: dict[str, ColumnStats] = {}
    parquet_files: list[str] = []
    rows = 0
    duplicate_timestamps = 0
    non_monotonic_timestamps = 0
    sampling_gap_count = 0
    previous_timestamp: pd.Timestamp | None = None

    for chunk_index, chunk in enumerate(iter_csv_chunks(raw_path, chunk_size=chunk_size)):
        if timestamp_column not in chunk:
            raise ValueError(f"timestamp column {timestamp_column!r} not found in {raw_path}")
        frame = chunk.copy()
        frame[timestamp_column] = pd.to_datetime(
            frame[timestamp_column], errors="raise", format="mixed"
        )
        diffs = frame[timestamp_column].diff()
        if previous_timestamp is not None and not frame.empty:
            first_diff = frame[timestamp_column].iloc[0] - previous_timestamp
            diffs.iloc[0] = first_diff
        duplicate_timestamps += int((diffs == pd.Timedelta(0)).sum())
        non_monotonic_timestamps += int((diffs < pd.Timedelta(0)).sum())
        sampling_gap_count += int((diffs > pd.Timedelta(seconds=1.5)).sum())
        previous_timestamp = pd.Timestamp(frame[timestamp_column].iloc[-1])

        _add_metropt_aliases(frame)
        labels = annotate_failures(frame[timestamp_column], failures)
        frame["is_failure"] = labels["is_failure"]
        frame["failure_type"] = labels["failure_type"]
        frame["exclusion_mask"] = False

        _update_stats(stats, frame)
        part_path = output_root / f"part-{chunk_index:05d}.parquet"
        write_parquet(frame, part_path)
        parquet_files.append(str(part_path))
        rows += len(frame)

    manifest = {
        "dataset": dataset,
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "output_root": str(output_root),
        "timestamp_column": timestamp_column,
        "timestamp_mapping": {"source": timestamp_column, "canonical": "timestamp"},
        "canonical_aliases": METROPT_ALIASES,
        "failure_labels": [asdict(failure) for failure in failures],
        "rows": rows,
        "chunks": len(parquet_files),
        "parquet_files": parquet_files,
        "quality_checks": {
            "duplicate_timestamps": duplicate_timestamps,
            "non_monotonic_timestamps": non_monotonic_timestamps,
            "sampling_gaps_gt_1_5s": sampling_gap_count,
        },
        "schema": _schema_manifest(stats, rows=rows),
    }
    manifest_path = output_root / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return PreparationResult(
        dataset=dataset,
        raw_path=str(raw_path),
        output_root=str(output_root),
        manifest_path=str(manifest_path),
        rows=rows,
        chunks=len(parquet_files),
        parquet_files=parquet_files,
    )


def prepare_scania(
    *,
    raw_root: Path,
    output_root: Path,
    chunk_size: int = 100_000,
) -> PreparationResult:
    """Prepare SCANIA Component X splits with entity metadata and counter increments."""

    output_root.mkdir(parents=True, exist_ok=True)
    _clear_parquet_parts(output_root)

    all_parquet_files: list[str] = []
    split_manifests: dict[str, Any] = {}
    split_entities: dict[str, set[int]] = {}
    total_rows = 0

    for split in ("train", "validation", "test"):
        split_output = output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        _clear_parquet_parts(split_output)
        split_manifest = _prepare_scania_split(
            raw_root=raw_root,
            output_root=split_output,
            split=split,
            chunk_size=chunk_size,
        )
        split_manifests[split] = split_manifest
        split_entities[split] = set(split_manifest["vehicle_ids"])
        all_parquet_files.extend(split_manifest["parquet_files"])
        total_rows += int(split_manifest["rows"])

    leakage = {
        f"{left}_vs_{right}": sorted(split_entities[left].intersection(split_entities[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(leakage.values()):
        raise ValueError(f"vehicle leakage across published Scania splits: {leakage}")

    manifest = {
        "dataset": "scania_component_x",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "rows": total_rows,
        "chunks": len(all_parquet_files),
        "parquet_files": all_parquet_files,
        "split_manifests": split_manifests,
        "entity_leakage": leakage,
    }
    manifest_path = output_root / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return PreparationResult(
        dataset="scania_component_x",
        raw_path=str(raw_root),
        output_root=str(output_root),
        manifest_path=str(manifest_path),
        rows=total_rows,
        chunks=len(all_parquet_files),
        parquet_files=all_parquet_files,
    )


def sha256_file(path: Path) -> str:
    """Hash a local raw file for manifest provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_scania_split(
    *,
    raw_root: Path,
    output_root: Path,
    split: str,
    chunk_size: int,
) -> dict[str, Any]:
    operational_path = raw_root / f"{split}_operational_readouts.csv"
    specifications_path = raw_root / f"{split}_specifications.csv"
    labels_path = raw_root / f"{split}_labels.csv"
    tte_path = raw_root / f"{split}_tte.csv"
    if not operational_path.exists():
        raise FileNotFoundError(operational_path)
    metadata = _load_scania_metadata(
        specifications_path=specifications_path,
        labels_path=labels_path if labels_path.exists() else None,
        tte_path=tte_path if tte_path.exists() else None,
    )

    stats: dict[str, ColumnStats] = {}
    parquet_files: list[str] = []
    rows = 0
    duplicate_keys = 0
    non_monotonic_time = 0
    reset_rows = 0
    feature_families: dict[str, list[str]] = {}
    vehicle_ids: set[int] = set()
    last_by_vehicle: dict[int, tuple[float, pd.Series]] = {}

    for chunk_index, chunk in enumerate(iter_csv_chunks(operational_path, chunk_size=chunk_size)):
        validate_operational_readouts(chunk)
        frame = chunk.sort_values(["vehicle_id", "time_step"], kind="stable").copy()
        counter_columns = [
            column
            for column in frame.columns
            if column not in {"vehicle_id", "time_step"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        for column in counter_columns:
            feature_families.setdefault(column.split("_", 1)[0], []).append(column)

        duplicate_keys += int(frame.duplicated(["vehicle_id", "time_step"]).sum())
        non_monotonic_time += _count_scania_non_monotonic(frame, last_by_vehicle)
        increments, reset_mask = _scania_counter_increments(frame, counter_columns, last_by_vehicle)
        reset_rows += int(reset_mask.sum())
        frame = pd.concat([frame.reset_index(drop=True), increments.reset_index(drop=True)], axis=1)
        frame["counter_reset_detected"] = reset_mask.reset_index(drop=True)
        frame = frame.merge(metadata, on="vehicle_id", how="left", validate="many_to_one")

        _update_last_by_vehicle(frame, counter_columns, last_by_vehicle)
        vehicle_ids.update(int(value) for value in frame["vehicle_id"].dropna().unique())
        _update_stats(stats, frame)
        part_path = output_root / f"part-{chunk_index:05d}.parquet"
        write_parquet(frame, part_path)
        parquet_files.append(str(part_path))
        rows += len(frame)

    return {
        "split": split,
        "raw_operational_path": str(operational_path),
        "raw_operational_sha256": sha256_file(operational_path),
        "rows": rows,
        "chunks": len(parquet_files),
        "parquet_files": parquet_files,
        "vehicle_ids": sorted(vehicle_ids),
        "feature_families": {key: sorted(set(value)) for key, value in feature_families.items()},
        "quality_checks": {
            "duplicate_vehicle_time_rows": duplicate_keys,
            "non_monotonic_time_rows": non_monotonic_time,
            "counter_reset_rows": reset_rows,
        },
        "schema": _schema_manifest(stats, rows=rows),
    }


def _scania_counter_increments(
    frame: pd.DataFrame,
    counter_columns: list[str],
    last_by_vehicle: dict[int, tuple[float, pd.Series]],
) -> tuple[pd.DataFrame, pd.Series]:
    grouped = frame.groupby("vehicle_id", sort=False)[counter_columns]
    increments = grouped.diff()
    first_indices = grouped.head(1).index
    for index in first_indices:
        index_key = cast(Any, index)
        vehicle_id = _as_int(frame.at[index_key, "vehicle_id"])
        if vehicle_id not in last_by_vehicle:
            continue
        _last_time, last_values = last_by_vehicle[vehicle_id]
        increments.loc[index_key, counter_columns] = (
            frame.loc[index_key, counter_columns] - last_values
        )
    reset_mask = (increments < 0).any(axis=1)
    increments = increments.mask(increments < 0)
    increments.columns = [f"{column}__increment" for column in counter_columns]
    return increments, reset_mask


def _count_scania_non_monotonic(
    frame: pd.DataFrame,
    last_by_vehicle: dict[int, tuple[float, pd.Series]],
) -> int:
    count = int((frame.groupby("vehicle_id", sort=False)["time_step"].diff() < 0).sum())
    first_rows = frame.groupby("vehicle_id", sort=False).head(1)
    for row in first_rows.itertuples(index=False):
        vehicle_id = _as_int(row.vehicle_id)
        if (
            vehicle_id in last_by_vehicle
            and _as_float(row.time_step) < last_by_vehicle[vehicle_id][0]
        ):
            count += 1
    return count


def _update_last_by_vehicle(
    frame: pd.DataFrame,
    counter_columns: list[str],
    last_by_vehicle: dict[int, tuple[float, pd.Series]],
) -> None:
    last_rows = frame.groupby("vehicle_id", sort=False).tail(1)
    for row_index, row in last_rows.iterrows():
        row_index_key = cast(Any, row_index)
        vehicle_id = _as_int(row["vehicle_id"])
        last_by_vehicle[vehicle_id] = (
            _as_float(row["time_step"]),
            cast(pd.Series, frame.loc[row_index_key, counter_columns]),
        )


def _load_scania_metadata(
    *,
    specifications_path: Path,
    labels_path: Path | None,
    tte_path: Path | None,
) -> pd.DataFrame:
    if not specifications_path.exists():
        raise FileNotFoundError(specifications_path)
    metadata = pd.read_csv(specifications_path)
    for path in (labels_path, tte_path):
        if path is None:
            continue
        table = pd.read_csv(path)
        metadata = metadata.merge(table, on="vehicle_id", how="left", validate="one_to_one")
    return metadata


def _load_failures(
    *,
    dataset: str,
    failure_yaml: Path | None,
) -> tuple[TimestampedFailure, ...]:
    if dataset == "metropt2":
        return METROPT2_FAILURES
    if failure_yaml is None or not failure_yaml.exists():
        return ()
    raw = yaml.safe_load(failure_yaml.read_text(encoding="utf-8"))
    failures: list[TimestampedFailure] = []
    for item in raw.get("failures", []):
        if item.get("start") is None or item.get("end") is None:
            continue
        failures.append(
            TimestampedFailure(
                start=pd.Timestamp(item["start"]),
                end=pd.Timestamp(item["end"]),
                failure_type=str(item["type"]),
            )
        )
    return tuple(failures)


def _add_metropt_aliases(frame: pd.DataFrame) -> None:
    for source, canonical in METROPT_ALIASES.items():
        if source in frame and canonical not in frame:
            frame[canonical] = frame[source]


def _update_stats(stats: dict[str, ColumnStats], frame: pd.DataFrame) -> None:
    for column in frame.columns:
        stats.setdefault(column, ColumnStats()).update(frame[column], unique_limit=1_000)


def _schema_manifest(stats: dict[str, ColumnStats], *, rows: int) -> dict[str, dict[str, Any]]:
    return {column: stat.to_manifest(rows=rows) for column, stat in sorted(stats.items())}


def _clear_parquet_parts(output_root: Path) -> None:
    for path in output_root.glob("part-*.parquet"):
        path.unlink()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    serializable = _json_safe(manifest)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def validate_no_duplicate_columns(columns: Iterable[str]) -> None:
    """Reject duplicate source columns before writing irreversible artifacts."""

    seen: set[str] = set()
    duplicates: set[str] = set()
    for column in columns:
        if column in seen:
            duplicates.add(column)
        seen.add(column)
    if duplicates:
        raise ValueError(f"duplicate columns: {sorted(duplicates)}")
