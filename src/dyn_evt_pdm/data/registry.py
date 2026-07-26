"""Typed dataset registry, verification, and dataset-report rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class DatasetRegistryEntry:
    """Static provenance contract for one supported public dataset."""

    dataset_id: str
    version: str
    official_landing_page: str
    primary_publication_citation: str
    license: str
    access_conditions: str
    timestamp_timezone: str
    timestamp_resolution: str
    entity_identifier: str
    sampling_structure: str
    failure_or_repair_event_source: str
    statistical_independence_unit: str
    known_limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileVerification:
    """Observed integrity status for one local raw or derived file."""

    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    expected_size_bytes: int | None = None
    expected_sha256: str | None = None
    checksum: str | None = None
    status: str = "unchecked"
    message: str = ""


@dataclass(frozen=True, slots=True)
class DatasetVerification:
    """Verification summary for one dataset."""

    dataset_id: str
    raw_manifest_path: str
    processed_manifest_path: str
    raw_files: tuple[FileVerification, ...]
    derived_files: tuple[FileVerification, ...]
    processed_rows: int | None
    processed_chunks: int | None
    status: str


REGISTRY: dict[str, DatasetRegistryEntry] = {
    "metropt": DatasetRegistryEntry(
        dataset_id="metropt",
        version="Zenodo record 6854240",
        official_landing_page="https://zenodo.org/records/6854240",
        primary_publication_citation=(
            "Veloso et al., Scientific Data 9, 764 (2022), " "doi:10.1038/s41597-022-01877-3"
        ),
        license="CC-BY-4.0",
        access_conditions="Open public Zenodo record; raw data are fetched locally and not vendored.",
        timestamp_timezone="source local time; treated as timezone-naive unless configured",
        timestamp_resolution="nominal one second",
        entity_identifier="single compressor",
        sampling_structure="ordered compressor telemetry",
        failure_or_repair_event_source="version-controlled failure metadata YAML",
        statistical_independence_unit="failure episode / operating day, not timestamp",
        known_limitations=(
            "three source-verified catastrophic failure episodes limit inferential power",
            "small number of independent failure episodes",
        ),
    ),
    "metropt2": DatasetRegistryEntry(
        dataset_id="metropt2",
        version="Zenodo record 7766691",
        official_landing_page="https://zenodo.org/records/7766691",
        primary_publication_citation=(
            "Veloso et al., MetroPT2: A Benchmark dataset for predictive maintenance, "
            "Zenodo (2023), doi:10.5281/zenodo.7766691"
        ),
        license="CC-BY-4.0",
        access_conditions="Open public Zenodo record; raw data are fetched locally and not vendored.",
        timestamp_timezone="source local time; treated as timezone-naive unless configured",
        timestamp_resolution="nominal one second",
        entity_identifier="single compressor",
        sampling_structure="ordered compressor telemetry",
        failure_or_repair_event_source="dataset-specific timestamped failure metadata in code",
        statistical_independence_unit="failure episode / operating day, not timestamp",
        known_limitations=(
            "schema and event timing are distinct from MetroPT",
            "small number of independent failure episodes",
        ),
    ),
    "scania_component_x": DatasetRegistryEntry(
        dataset_id="scania_component_x",
        version="Researchdata.se dataset 2024-34",
        official_landing_page="https://researchdata.se/en/catalogue/dataset/2024-34",
        primary_publication_citation=(
            "Kharazian et al., Scientific Data 12, 493 (2025), " "doi:10.1038/s41597-025-04802-6"
        ),
        license="CC-BY-4.0",
        access_conditions=(
            "Open public Researchdata.se record; raw data are fetched locally and not vendored."
        ),
        timestamp_timezone="not timestamped; ordered by vehicle time_step",
        timestamp_resolution="irregular counter/time_step observations",
        entity_identifier="vehicle_id",
        sampling_structure="fleet-level irregular vehicle histories",
        failure_or_repair_event_source="published labels and time-to-event files",
        statistical_independence_unit="vehicle",
        known_limitations=(
            "not physically identical to MetroPT compressor episodes",
            "censoring and repair labels require estimand-specific handling",
        ),
    ),
}

ALIASES = {
    "all": tuple(REGISTRY),
    "metropt": ("metropt",),
    "metropt2": ("metropt2",),
    "scania": ("scania_component_x",),
    "scania_component_x": ("scania_component_x",),
    "scania-component-x": ("scania_component_x",),
}


def normalize_dataset_ids(dataset: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize user-facing dataset names to canonical registry IDs."""

    names = [dataset] if isinstance(dataset, str) else list(dataset)
    selected: list[str] = []
    for name in names:
        key = name.strip().lower().replace("-", "_")
        alias_key = "scania-component-x" if name.strip().lower() == "scania-component-x" else key
        if alias_key not in ALIASES:
            valid = ", ".join(sorted(ALIASES))
            raise ValueError(f"unknown dataset {name!r}; expected one of: {valid}")
        selected.extend(ALIASES[alias_key])
    if not selected:
        selected.extend(ALIASES["all"])
    return tuple(dict.fromkeys(selected))


def registry_records() -> list[dict[str, Any]]:
    """Return registry entries as JSON-compatible dictionaries."""

    return [asdict(REGISTRY[key]) for key in sorted(REGISTRY)]


def verify_dataset(
    dataset_id: str,
    *,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
) -> DatasetVerification:
    """Verify raw and processed manifests for one dataset."""

    canonical = normalize_dataset_ids(dataset_id)[0]
    raw_manifest_path = raw_root / canonical / "manifest.json"
    processed_manifest_path = processed_root / canonical / "manifest.json"
    raw_files = _verify_raw_manifest(raw_manifest_path)
    derived_files, rows, chunks = _verify_processed_manifest(processed_manifest_path)
    statuses = [item.status for item in (*raw_files, *derived_files)]
    status = "verified" if statuses and all(item == "verified" for item in statuses) else "blocked"
    return DatasetVerification(
        dataset_id=canonical,
        raw_manifest_path=str(raw_manifest_path),
        processed_manifest_path=str(processed_manifest_path),
        raw_files=tuple(raw_files),
        derived_files=tuple(derived_files),
        processed_rows=rows,
        processed_chunks=chunks,
        status=status,
    )


def build_data_report(
    *,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
) -> dict[str, Any]:
    """Build a machine-readable dataset report from registry and manifests."""

    datasets: list[dict[str, Any]] = []
    for dataset_id in sorted(REGISTRY):
        verification = verify_dataset(dataset_id, raw_root=raw_root, processed_root=processed_root)
        registry = asdict(REGISTRY[dataset_id])
        processed_manifest = _read_json_object(Path(verification.processed_manifest_path))
        quality = processed_manifest.get("quality_checks", {})
        schema_columns = len(processed_manifest.get("schema", {}))
        if dataset_id == "scania_component_x":
            split_manifests = cast(dict[str, Any], processed_manifest.get("split_manifests", {}))
            quality = {
                split: manifest.get("quality_checks", {})
                for split, manifest in split_manifests.items()
            }
            schema_columns = max(
                (
                    len(cast(dict[str, Any], manifest).get("schema", {}))
                    for manifest in split_manifests.values()
                ),
                default=schema_columns,
            )
        datasets.append(
            {
                "registry": registry,
                "verification": asdict(verification),
                "rows": verification.processed_rows,
                "chunks": verification.processed_chunks,
                "quality_checks": quality,
                "schema_columns": schema_columns,
            }
        )
    return {"datasets": datasets}


def render_dataset_latex_table(report: dict[str, Any]) -> str:
    """Render a concise LaTeX dataset-characteristics table."""

    rows = [
        "\\begin{tabular}{llllr}",
        "\\toprule",
        "Dataset & Version & Unit & Event source & Rows \\\\",
        "\\midrule",
    ]
    for item in report["datasets"]:
        registry = item["registry"]
        rows.append(
            " & ".join(
                [
                    _latex_escape(str(registry["dataset_id"])),
                    _latex_escape(str(registry["version"])),
                    _latex_escape(str(registry["statistical_independence_unit"])),
                    _latex_escape(str(registry["failure_or_repair_event_source"])),
                    str(item.get("rows") or "NA"),
                ]
            )
            + " \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(rows) + "\n"


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_raw_manifest(manifest_path: Path) -> list[FileVerification]:
    if not manifest_path.exists():
        return [
            FileVerification(
                path=str(manifest_path),
                exists=False,
                size_bytes=None,
                sha256=None,
                status="missing_manifest",
                message="raw manifest is missing",
            )
        ]
    try:
        manifest = _read_json_list(manifest_path)
    except ValueError as exc:
        return [
            FileVerification(
                path=str(manifest_path),
                exists=True,
                size_bytes=manifest_path.stat().st_size,
                sha256=sha256_file(manifest_path),
                status="invalid_manifest",
                message=str(exc),
            )
        ]
    return [_verify_manifest_file(item, expected_sha_key="sha256") for item in manifest]


def _verify_processed_manifest(
    manifest_path: Path,
) -> tuple[list[FileVerification], int | None, int | None]:
    if not manifest_path.exists():
        return (
            [
                FileVerification(
                    path=str(manifest_path),
                    exists=False,
                    size_bytes=None,
                    sha256=None,
                    status="missing_manifest",
                    message="processed manifest is missing",
                )
            ],
            None,
            None,
        )
    manifest = _read_json_object(manifest_path)
    files = [manifest_path, *(Path(path) for path in manifest.get("parquet_files", []))]
    verifications = [
        _verify_manifest_file(
            {"path": str(path), "sha256": sha256_file(path) if path.exists() else None}
        )
        for path in files
    ]
    return verifications, _optional_int(manifest.get("rows")), _optional_int(manifest.get("chunks"))


def _verify_manifest_file(
    item: dict[str, Any],
    *,
    expected_sha_key: str = "sha256",
) -> FileVerification:
    path = Path(str(item.get("path", "")))
    expected_size = _optional_int(item.get("size_bytes"))
    expected_sha = str(item[expected_sha_key]) if item.get(expected_sha_key) else None
    checksum = str(item["checksum"]) if item.get("checksum") else None
    if not path.exists():
        return FileVerification(
            path=str(path),
            exists=False,
            size_bytes=None,
            sha256=None,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha,
            checksum=checksum,
            status="missing_file",
            message="file is missing",
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    errors: list[str] = []
    if expected_size is not None and size != expected_size:
        errors.append(f"size {size} != expected {expected_size}")
    if expected_sha is not None and digest != expected_sha:
        errors.append("sha256 mismatch")
    return FileVerification(
        path=str(path),
        exists=True,
        size_bytes=size,
        sha256=digest,
        expected_size_bytes=expected_size,
        expected_sha256=expected_sha,
        checksum=checksum,
        status="failed" if errors else "verified",
        message="; ".join(errors),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], raw)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{path} must contain a list of JSON objects")
    return cast(list[dict[str, Any]], raw)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
