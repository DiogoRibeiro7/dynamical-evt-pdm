from pathlib import Path

import pandas as pd
import pytest

from dyn_evt_pdm.data.contracts import TimeSeriesContract, require_columns
from dyn_evt_pdm.data.io import iter_csv_chunks, read_table, write_parquet
from dyn_evt_pdm.data.metropt import METROPT2_FAILURES, annotate_failures, load_metropt
from dyn_evt_pdm.data.scania import compute_counter_increments, validate_operational_readouts
from dyn_evt_pdm.data.splits import sort_by_time


def test_contract_and_require_columns() -> None:
    frame = pd.DataFrame({"ts": [1], "x": [2.0], "regime": ["a"]})
    TimeSeriesContract("ts", ("x",), ("regime",)).validate(frame)
    require_columns(frame, ["ts", "x"])
    with pytest.raises(ValueError):
        require_columns(frame, ["missing"])
    with pytest.raises(ValueError):
        TimeSeriesContract("ts", ("z",)).validate(frame)
    with pytest.raises(ValueError):
        TimeSeriesContract("ts", ("x",)).validate(frame.iloc[0:0])


def test_csv_and_parquet_io(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    csv = tmp_path / "x.csv"
    frame.to_csv(csv, index=False)
    chunks = list(iter_csv_chunks(csv, chunk_size=2, usecols=["a"]))
    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert read_table(csv, columns=["b"]).columns.tolist() == ["b"]

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass
    else:
        parquet = tmp_path / "x.parquet"
        write_parquet(frame, parquet)
        assert read_table(parquet).equals(frame)

    with pytest.raises(ValueError):
        list(iter_csv_chunks(csv, chunk_size=0))
    unknown = tmp_path / "x.txt"
    unknown.write_text("x")
    with pytest.raises(ValueError):
        read_table(unknown)


def test_metropt_loader_and_failure_annotation(tmp_path: Path) -> None:
    timestamps = [
        "2022-06-04 10:19:24.300",
        "2022-06-05 00:00:00",
        "2022-07-12 00:00:00",
    ]
    path = tmp_path / "metro.csv"
    pd.DataFrame({"timestamp": timestamps, "sensor": [1, 2, 3]}).to_csv(path, index=False)
    frame = load_metropt(path)
    labels = annotate_failures(frame["timestamp"], METROPT2_FAILURES)
    assert labels["is_failure"].tolist() == [True, False, True]
    with pytest.raises(ValueError):
        load_metropt(path, timestamp_column="missing")


def test_scania_counter_increments_and_validation() -> None:
    frame = pd.DataFrame(
        {
            "vehicle_id": ["a", "a", "a", "b"],
            "time_step": [0, 1, 2, 0],
            "counter": [5.0, 8.0, 2.0, 3.0],
        }
    )
    result = compute_counter_increments(frame, counter_columns=["counter"])
    assert result.loc[1, "counter__increment"] == 3.0
    assert pd.isna(result.loc[2, "counter__increment"])
    validate_operational_readouts(frame)
    with pytest.raises(ValueError):
        validate_operational_readouts(pd.DataFrame({"vehicle_id": ["a"]}))


def test_sort_by_time() -> None:
    frame = pd.DataFrame({"timestamp": ["2024-01-02", "2024-01-01"], "x": [2, 1]})
    result = sort_by_time(frame, "timestamp")
    assert result["x"].tolist() == [1, 2]
    with pytest.raises(ValueError):
        sort_by_time(frame, "missing")
