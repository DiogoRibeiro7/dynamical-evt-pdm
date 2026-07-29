"""Tests for the compact main table and supplementary estimator grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.evt.extremal_index import ESTIMATOR_NAMES
from dyn_evt_pdm.simulation.estimator_summary import summarise_estimator_performance
from dyn_evt_pdm.simulation.estimator_tables import (
    MAIN_TABLE_COLUMNS,
    build_estimator_tables,
    build_main_table,
    build_regime_table,
    select_representative_processes,
)
from dyn_evt_pdm.simulation.study import ESTIMATOR_COLUMN_STEMS


def _replicates(
    *,
    system: str,
    reference: float | None,
    reference_kind: str,
    threshold_quantile: float = 0.98,
    n_replicates: int = 20,
    seed: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    payload: dict[str, object] = {
        "system": [system] * n_replicates,
        "n_steps": [2000] * n_replicates,
        "threshold_quantile": [threshold_quantile] * n_replicates,
        "run_length": [5] * n_replicates,
        "noise_scale": [0.0] * n_replicates,
        "missing_rate": [0.0] * n_replicates,
        "reference_theta": [reference if reference is not None else np.nan] * n_replicates,
        "reference_kind": [reference_kind] * n_replicates,
    }
    centre = reference if reference is not None else 0.5
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        estimates = centre + rng.normal(0.0, 0.02, size=n_replicates)
        payload[f"{stem}_theta"] = estimates
        payload[f"{stem}_lower"] = estimates - 0.05
        payload[f"{stem}_upper"] = estimates + 0.05
        payload[f"{stem}_interval_width"] = np.full(n_replicates, 0.1)
        payload[f"{stem}_covered"] = pd.array([True] * n_replicates, dtype="boolean")
        payload[f"{stem}_tuning_parameter"] = np.full(n_replicates, 5.0)
        payload[f"{stem}_failed"] = [False] * n_replicates
    return pd.DataFrame(payload)


def _summary() -> pd.DataFrame:
    frames = [
        _replicates(system="iid_light_tail", reference=1.0, reference_kind="theoretical", seed=1),
        _replicates(system="downsampling", reference=0.98, reference_kind="numerical", seed=2),
        _replicates(
            system="logistic_periodic_target",
            reference=0.5,
            reference_kind="theoretical",
            seed=3,
        ),
        _replicates(system="smoothing", reference=0.27, reference_kind="numerical", seed=4),
        _replicates(
            system="regime_mixture", reference=None, reference_kind="not_established", seed=6
        ),
    ]
    return summarise_estimator_performance(pd.concat(frames, ignore_index=True))


def test_representative_selection_covers_each_regime_once() -> None:
    selected = select_representative_processes(_summary())
    systems = [system for system, _theta, _kind in selected]
    assert systems == ["iid_light_tail", "logistic_periodic_target", "smoothing"]


def test_representative_selection_prefers_theoretical_references() -> None:
    """A closed-form reference outranks a numerical one within the same regime."""

    selected = dict(
        (system, kind) for system, _theta, kind in select_representative_processes(_summary())
    )
    # iid_light_tail (theoretical) must win the near-independent slot over downsampling.
    assert selected["iid_light_tail"] == "theoretical"
    assert "downsampling" not in selected


def test_compact_main_table_is_one_process_per_regime() -> None:
    table = build_main_table(_summary())
    assert list(table.columns) == list(MAIN_TABLE_COLUMNS)
    assert len(table) == 3 * len(ESTIMATOR_NAMES)
    assert set(table["process"]) == {"iid light tail", "logistic periodic target", "smoothing"}


def test_full_table_keeps_every_referenced_process() -> None:
    table = build_main_table(_summary(), compact=False)
    assert set(table["process"]) == {
        "iid light tail",
        "downsampling",
        "logistic periodic target",
        "smoothing",
    }
    # The unreferenced family never appears, since bias and coverage are undefined.
    assert "regime mixture" not in set(table["process"])


def test_threshold_range_reports_the_span_not_a_single_setting() -> None:
    frames = [
        _replicates(
            system="iid_light_tail",
            reference=1.0,
            reference_kind="theoretical",
            threshold_quantile=quantile,
            seed=index,
        )
        for index, quantile in enumerate((0.95, 0.98, 0.99))
    ]
    summary = summarise_estimator_performance(pd.concat(frames, ignore_index=True))
    table = build_main_table(summary)
    assert set(table["threshold_range"]) == {"0.95-0.99"}


def test_regime_table_records_unreferenced_families() -> None:
    regime = build_regime_table(_summary())
    row = regime[regime["process"] == "regime mixture"].iloc[0]
    assert int(row["with_reference"]) == 0
    assert row["reference_kind"] == "none established"
    assert row["regime"] == "not established"


def test_regime_table_covers_every_family() -> None:
    regime = build_regime_table(_summary())
    assert len(regime) == 5
    assert int(regime["configurations"].sum()) == 5


def test_empty_summary_yields_empty_tables() -> None:
    empty = pd.DataFrame(
        columns=[
            "system",
            "n_steps",
            "threshold_quantile",
            "run_length",
            "noise_scale",
            "missing_rate",
            "reference_theta",
            "reference_kind",
        ]
    )
    assert build_main_table(empty).empty
    assert select_representative_processes(empty) == []


def test_table_builder_writes_every_artifact(tmp_path: Path) -> None:
    paths = build_estimator_tables(_summary(), tmp_path)
    for path in paths.as_tuple():
        assert path.exists()
        assert path.stat().st_size > 0
    main_tex = paths.main_tex.read_text(encoding="utf-8")
    assert "\\begin{table}" in main_tex
    assert "tab:estimator-main-summary" in main_tex


def test_supplement_csv_carries_the_full_grid(tmp_path: Path) -> None:
    summary = _summary()
    paths = build_estimator_tables(summary, tmp_path)
    supplement = pd.read_csv(paths.supplement_csv)
    assert len(supplement) == len(summary)
    main = pd.read_csv(paths.main_csv)
    assert len(main) < len(supplement)


def test_main_table_values_reproduce_from_the_summary() -> None:
    """The compact table must be derivable from the artifact it claims to summarise."""

    summary = _summary()
    table = build_main_table(summary)
    row = table[(table["process"] == "smoothing") & (table["estimator"] == "Runs")].iloc[0]
    source = summary[(summary["system"] == "smoothing") & (summary["estimator"] == "runs")]
    assert row["bias"] == pytest.approx(round(float(source["bias"].mean()), 3))
    assert row["rmse"] == pytest.approx(round(float(source["rmse"].mean()), 3))
    assert row["coverage"] == pytest.approx(round(float(source["coverage"].mean()), 3))
