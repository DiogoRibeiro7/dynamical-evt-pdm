"""Memory-aware input/output helpers for large industrial datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrapping


def ensure_parent(path: Path) -> None:
    """Create the parent directory of an artifact with its path on failure."""

    with wrapping(OSError, FileWriteError, path=str(path.parent)):
        path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 metadata and retain the underlying filesystem error."""

    with wrapping((OSError, UnicodeError), FileWriteError, path=str(path)):
        path.write_text(content, encoding="utf-8")


def iter_csv_chunks(
    path: Path,
    *,
    chunk_size: int = 500_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks without loading the complete dataset into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.exists():
        raise FileNotFoundError(path)
    # A chunk may fail after iteration has started, so keep the wrapper around
    # the entire generator rather than only around the reader's construction.
    with (
        wrapping((OSError, UnicodeError), FileReadError, path=str(path)),
        wrapping(pd.errors.ParserError, DataLoadingError, source=str(path)),
    ):
        yield from pd.read_csv(path, chunksize=chunk_size, usecols=usecols)


def read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet table based on its suffix."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with (
            wrapping((OSError, UnicodeError), FileReadError, path=str(path)),
            wrapping(pd.errors.ParserError, DataLoadingError, source=str(path)),
        ):
            return pd.read_csv(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        with wrapping(OSError, FileReadError, path=str(path)):
            return pd.read_parquet(path, columns=columns)
    raise ValueError(f"unsupported table format: {suffix}")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame atomically to a compressed Parquet file."""

    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wrapping(OSError, FileWriteError, path=str(temporary)):
        frame.to_parquet(temporary, index=False, compression="zstd")
    with wrapping(OSError, FileWriteError, path=str(path)):
        temporary.replace(path)
