"""Raw dataset acquisition from public research-data records."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrap, wrapping

from dyn_evt_pdm.data.io import ensure_parent, write_text

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
SCANIA_DATASET_PAGE = "https://researchdata.se/en/catalogue/dataset/2024-34"
UCI_STATIC_PUBLIC = "https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One public source file to download into the raw-data tree."""

    dataset: str
    filename: str
    url: str
    destination: Path
    size_bytes: int | None = None
    checksum: str | None = None
    source_record: str | None = None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Local acquisition result with integrity metadata."""

    dataset: str
    filename: str
    path: str
    url: str
    size_bytes: int
    sha256: str
    checksum: str | None
    status: str
    acquisition_timestamp_utc: str


def planned_files(
    dataset_names: list[str], *, raw_root: Path = Path("data/raw")
) -> list[RemoteFile]:
    """Resolve dataset names to concrete public files."""

    selected = _normalize_dataset_names(dataset_names)
    plans: list[RemoteFile] = []
    if "metropt" in selected:
        plans.extend(
            _zenodo_files("metropt", "6854240", raw_root / "metropt", only={"dataset_train.csv"})
        )
    if "metropt2" in selected:
        plans.extend(
            _zenodo_files("metropt2", "7766691", raw_root / "metropt2", only={"MetroPT2.csv"})
        )
    if "scania_component_x" in selected:
        plans.extend(_scania_files(raw_root / "scania_component_x"))
    if "hydraulic_systems" in selected:
        plans.append(
            _uci_zip_file(
                dataset="hydraulic_systems",
                dataset_id=447,
                slug="condition+monitoring+of+hydraulic+systems",
                destination_root=raw_root / "hydraulic_systems",
            )
        )
    if "secom" in selected:
        plans.append(
            _uci_zip_file(
                dataset="secom",
                dataset_id=179,
                slug="secom",
                destination_root=raw_root / "secom",
            )
        )
    return plans


def fetch_datasets(
    dataset_names: list[str],
    *,
    raw_root: Path = Path("data/raw"),
    overwrite: bool = False,
    timeout_seconds: int = 60,
    retries: int = 2,
) -> list[DownloadResult]:
    """Download selected public datasets and write per-dataset manifests."""

    files = planned_files(dataset_names, raw_root=raw_root)
    results: list[DownloadResult] = []
    for remote in files:
        results.append(
            download_file(
                remote,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
        )

    by_dataset: dict[str, list[DownloadResult]] = {}
    for result in results:
        by_dataset.setdefault(result.dataset, []).append(result)
    for dataset, dataset_results in by_dataset.items():
        manifest_path = raw_root / dataset / "manifest.json"
        ensure_parent(manifest_path)
        write_text(
            manifest_path, json.dumps([asdict(result) for result in dataset_results], indent=2)
        )
    return results


def download_file(
    remote: RemoteFile,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    retries: int = 2,
) -> DownloadResult:
    """Download one file, reusing a valid existing local copy when possible."""

    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    ensure_parent(remote.destination)
    if remote.destination.exists() and not overwrite:
        result = _local_result(remote, status="exists")
        _validate_result(result, remote)
        return result

    temporary = remote.destination.with_suffix(remote.destination.suffix + ".part")
    if overwrite and temporary.exists():
        with wrapping(OSError, FileWriteError, path=str(temporary)):
            temporary.unlink()

    request = urllib.request.Request(remote.url, headers={"User-Agent": "dyn-evt-pdm/0.1"})
    last_error: OSError | DataLoadingError | None = None
    for attempt in range(retries + 1):
        try:
            with (
                urllib.request.urlopen(request, timeout=timeout_seconds) as response,
                wrapping(OSError, FileWriteError, path=str(temporary)),
                temporary.open("wb") as handle,
            ):
                while True:
                    with wrapping(OSError, DataLoadingError, source=remote.url):
                        chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            last_error = None
            break
        except FileWriteError:
            if temporary.exists():
                with wrapping(OSError, FileWriteError, path=str(temporary)):
                    temporary.unlink()
            raise
        except (OSError, DataLoadingError) as exc:
            last_error = exc
            if temporary.exists():
                with wrapping(OSError, FileWriteError, path=str(temporary)):
                    temporary.unlink()
            if attempt >= retries:
                break
            time.sleep(min(2.0**attempt, 8.0))
    if last_error is not None:
        if isinstance(last_error, DataLoadingError):
            raise last_error
        raise wrap(last_error, DataLoadingError, source=remote.url) from last_error

    with wrapping(OSError, FileWriteError, path=str(remote.destination)):
        temporary.replace(remote.destination)
    result = _local_result(remote, status="downloaded")
    _validate_result(result, remote)
    return result


def _normalize_dataset_names(dataset_names: list[str]) -> set[str]:
    aliases = {
        "all": {"metropt", "metropt2", "scania_component_x", "hydraulic_systems", "secom"},
        "scania": {"scania_component_x"},
        "scania_component_x": {"scania_component_x"},
        "hydraulic": {"hydraulic_systems"},
        "hydraulic_systems": {"hydraulic_systems"},
        "hydraulic-systems": {"hydraulic_systems"},
        "metropt": {"metropt"},
        "metropt2": {"metropt2"},
        "secom": {"secom"},
    }
    selected: set[str] = set()
    for name in dataset_names:
        key = name.strip().lower().replace("-", "_")
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"unknown dataset {name!r}; expected one of: {valid}")
        selected.update(aliases[key])
    return selected or aliases["all"]


def _zenodo_files(
    dataset: str,
    record_id: str,
    destination_root: Path,
    *,
    only: set[str],
) -> list[RemoteFile]:
    url = ZENODO_API.format(record_id=record_id)
    with (
        wrapping(OSError, DataLoadingError, source=url),
        wrapping(json.JSONDecodeError, DataLoadingError, source=url),
        urllib.request.urlopen(url, timeout=30) as response,
    ):
        record: dict[str, Any] = json.load(response)

    files: list[RemoteFile] = []
    for entry in record.get("files", []):
        filename = str(entry["key"])
        if filename not in only:
            continue
        files.append(
            RemoteFile(
                dataset=dataset,
                filename=filename,
                url=str(entry["links"]["self"]),
                destination=destination_root / filename,
                size_bytes=int(entry["size"]),
                checksum=str(entry.get("checksum")) if entry.get("checksum") else None,
                source_record=f"https://zenodo.org/records/{record_id}",
            )
        )
    if len(files) != len(only):
        found = {file.filename for file in files}
        missing = ", ".join(sorted(only - found))
        raise RuntimeError(f"Zenodo record {record_id} did not expose expected file(s): {missing}")
    return files


def _scania_files(destination_root: Path) -> list[RemoteFile]:
    with wrapping(OSError, DataLoadingError, source=SCANIA_DATASET_PAGE):
        html = (
            urllib.request.urlopen(SCANIA_DATASET_PAGE, timeout=30)
            .read()
            .decode("utf-8", "replace")
        )
    urls = sorted(set(re.findall(r'https://api\.researchdata\.se/[^"]+', html)))
    selected: list[RemoteFile] = []
    for url in urls:
        clean_url = url.replace("&amp;", "&")
        if "filePath=" not in clean_url:
            continue
        filename = clean_url.split("filePath=", 1)[1]
        if "/" in filename or "\\" in filename:
            raise RuntimeError(f"unexpected Scania filename from source page: {filename}")
        selected.append(
            RemoteFile(
                dataset="scania_component_x",
                filename=filename,
                url=clean_url,
                destination=destination_root / filename,
                source_record=SCANIA_DATASET_PAGE,
            )
        )
    if not selected:
        raise RuntimeError("Scania dataset page did not expose any downloadable files")
    return selected


def _uci_zip_file(
    *,
    dataset: str,
    dataset_id: int,
    slug: str,
    destination_root: Path,
) -> RemoteFile:
    return RemoteFile(
        dataset=dataset,
        filename=f"{slug}.zip",
        url=UCI_STATIC_PUBLIC.format(dataset_id=dataset_id, slug=slug),
        destination=destination_root / f"{slug}.zip",
        source_record=f"https://archive.ics.uci.edu/dataset/{dataset_id}/{slug}",
    )


def _local_result(remote: RemoteFile, *, status: str) -> DownloadResult:
    digest = hashlib.sha256()
    size = 0
    with (
        wrapping(OSError, FileReadError, path=str(remote.destination)),
        remote.destination.open("rb") as handle,
    ):
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return DownloadResult(
        dataset=remote.dataset,
        filename=remote.filename,
        path=str(remote.destination),
        url=remote.url,
        size_bytes=size,
        sha256=digest.hexdigest(),
        checksum=remote.checksum,
        status=status,
        acquisition_timestamp_utc=datetime.now(UTC).isoformat(),
    )


def _validate_result(result: DownloadResult, remote: RemoteFile) -> None:
    if remote.size_bytes is not None and result.size_bytes != remote.size_bytes:
        raise RuntimeError(
            f"{remote.filename} has {result.size_bytes} bytes; expected {remote.size_bytes}"
        )
    if not remote.checksum:
        return
    algorithm, _, expected = remote.checksum.partition(":")
    if algorithm.lower() != "md5" or not expected:
        raise RuntimeError(f"unsupported checksum format for {remote.filename}: {remote.checksum}")
    digest = hashlib.md5(usedforsecurity=False)
    with (
        wrapping(OSError, FileReadError, path=str(remote.destination)),
        remote.destination.open("rb") as handle,
    ):
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"{remote.filename} md5 mismatch: expected {expected}, got {actual}")
