"""Memory-aware input/output helpers for large industrial datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


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
    yield from pd.read_csv(path, chunksize=chunk_size, usecols=usecols)


def read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet table based on its suffix."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"unsupported table format: {suffix}")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame atomically to a compressed Parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
