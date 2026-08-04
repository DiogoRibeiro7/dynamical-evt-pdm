"""Exercise the manuscript asset builders on synthetic inputs.

These builders read the real-data artifact directory, which is not committed, so
without this module they are only ever exercised on a machine that has already run the
pipeline. That is the wrong place for the coverage to live: a figure or table builder
that raises on a shape it has not seen should fail in CI, not during a release build.

Every fixture here is synthetic and small. The tests assert that each builder produces
its declared outputs and that the content a reader depends on reaches the file, not that
the numbers are correct; correctness of the numbers is covered by the reconciliation and
decomposition tests against the real artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dyn_evt_pdm.paper.assets import (
    PaperAssetConfig,
    _all_method_timeline_figure,
    _best_full_family_baseline,
    _decomposition_assets,
    _exposure_text,
    _method_root_cause_rows,
    _non_event_root_cause_rows,
    _root_cause_assets,
    _score_alarm_flow_figure,
    _timeline_caption,
    _timeline_caption_macro,
    _timeline_traceability_assets,
    _two_scale_timeline_figure,
)

DATASETS = ("metropt", "metropt2")
METHODS = (
    "dynamical_evt_robust_score",
    "engineering_threshold",
    "classical_pot_gpd",
    "isolation_forest",
    "failure_prototype_region",
    "rare_state_region",
    "spot",
)


def _decomposition_frame() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for index, method in enumerate(METHODS):
            # Vary the stages so every classifier branch is reachable across the frame.
            exceedances = 0 if method == "failure_prototype_region" else 5_000 + index
            matched = 0 if method == "spot" else 1
            rows.append(
                {
                    "dataset_id": dataset,
                    "failure_id": f"{dataset}_test_failure_001",
                    "method": method,
                    "score_median": 1.0,
                    "score_p95": 2.0,
                    "score_separation_auc": 0.46 if index % 2 else 0.86,
                    "chance_match_probability": 0.99,
                    "threshold": 10.0,
                    "test_observations_samples": 100_000,
                    "threshold_exceedances_samples": exceedances,
                    "extreme_clusters_count": exceedances // 10,
                    "alarm_episodes_count": exceedances // 20,
                    "matched_alarm_episodes_count": matched,
                    "false_alarm_episodes_count": max(0, exceedances // 20 - matched),
                    "duplicate_alarm_episodes_count": 3,
                    "labelled_failure_events_count": 1,
                    "labelled_failure_samples": 500,
                    "event_recall": float(matched),
                    "event_precision": 0.0004,
                    "time_under_warning_samples": exceedances,
                    "exceedance_to_alarm_ratio": 20.0,
                    "leakage_control": "test",
                }
            )
    return pd.DataFrame(rows)


def _baseline_frame() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in (*METHODS, "best_individual_sensor_threshold"):
            # The single-sensor threshold is given a clearly better F1 so the
            # best-baseline selection is unambiguous rather than decided by row order.
            best = method == "best_individual_sensor_threshold"
            rows.append(
                {
                    "dataset_id": dataset,
                    "method": method,
                    "predicted_alarm_events": 100,
                    "event_f1": 0.010 if best else 0.002,
                    "event_recall": 1.0,
                    "event_precision": 0.005 if best else 0.001,
                    "false_alarm_events_per_day": 4.0 if best else 12.5,
                    "median_warning_lead_time": 3_000.0,
                    "time_under_warning": 5_000,
                }
            )
    return pd.DataFrame(rows)


def _overview_frame() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            for index in range(12):
                rows.append(
                    {
                        "dataset_id": dataset,
                        "failure_id": f"{dataset}_test_failure_001",
                        "method": method,
                        "bin_index": index,
                        "bin_start_index": index * 100,
                        "bin_end_index": (index + 1) * 100 - 1,
                        "bin_start_day": index * 0.5,
                        "bin_end_day": (index + 1) * 0.5,
                        "bin_start_hours": index * 12.0,
                        "alarm_episodes": index,
                        "time_under_warning_samples": index * 10,
                        "exceedance_samples": index * 20,
                        "contains_failure_onset": index == 8,
                        "contains_maintenance_reset": index == 10,
                        "failure_samples": 5 if index == 8 else 0,
                    }
                )
    return pd.DataFrame(rows)


def _trace_frame() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for index in range(60):
            rows.append(
                {
                    "dataset_id": dataset,
                    "failure_id": f"{dataset}_test_failure_001",
                    "method": "dynamical_evt_robust_score",
                    "test_index": 800 + index,
                    "timestamp": str(index),
                    "elapsed_hours": (index - 40) / 10.0,
                    "score": 1.0 + (index % 7),
                    "threshold": 5.0,
                    "regime_id": index % 3,
                    "is_exceedance": index % 7 == 6,
                    "cluster_id": index // 7,
                    "alarm_episode_id": index // 7 if index % 7 == 6 else -1,
                    "in_warning_window": 30 <= index <= 40,
                    "is_failure": index >= 40,
                    "is_maintenance_reset": index == 55,
                }
            )
    return pd.DataFrame(rows)


def _metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "failure_id": f"{dataset}_test_failure_001",
                "method": "dynamical_evt_robust_score",
                "target_region_variant": "robust_max_abs_z",
                "threshold_quantile": 0.98,
                "threshold_value": 5.0,
                "run_length": 6,
                "merge_gap": 0,
                "warning_horizon_samples": 3_600,
                "warning_horizon_hours": 1.0,
                "test_duration_samples": 100_000,
                "test_duration_days": 1.16,
                "global_alarm_episodes": 66,
                "local_alarm_episodes": 8,
                "global_false_alarm_episodes": 62,
                "local_false_alarm_episodes": 6,
                "matched_alarm_episodes": 1,
                "false_alarms_per_operating_day": 53.4,
                "lead_time_samples": 3_346.0,
                "lead_time_hours": 0.93,
                "time_under_warning_samples": 2_631,
                "time_under_warning_fraction": 0.0263,
                "labelled_failure_events": 1,
                "maintenance_resets_recorded": 1,
                "local_window_before_hours": 6.0,
                "local_window_after_hours": 2.0,
            }
            for dataset in DATASETS
        ]
    )


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    """A synthetic stand-in for the real-data artifact directory."""
    root = tmp_path / "real_data_matrix"
    root.mkdir()
    _decomposition_frame().to_csv(root / "score_threshold_alarm_decomposition.csv", index=False)
    _baseline_frame().to_csv(root / "event_baseline_comparison.csv", index=False)
    _overview_frame().to_csv(root / "event_timeline_overview.csv", index=False)
    _trace_frame().to_csv(root / "event_timeline_trace.csv", index=False)
    _metadata_frame().to_csv(root / "event_timeline_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "estimand": "event onset",
                "independent_units": 1,
                "target_events_or_units": 1,
                "predicted_events_or_units": 100,
                "precision": 0.001,
                "recall": 1.0,
                "f1": 0.002,
                "false_alarm_events_per_day": 12.5,
                "median_warning_lead_time": 3_000.0,
                "threshold_quantile": 0.98,
                "rows": 100_000,
            }
            for dataset in (*DATASETS, "scania_component_x", "hydraulic_systems", "secom")
        ]
    ).to_csv(root / "industrial_results_summary.csv", index=False)
    return root


def test_decomposition_figure_renders_every_selected_row(tmp_path: Path) -> None:
    path = tmp_path / "decomposition.png"
    _score_alarm_flow_figure(_decomposition_frame(), path)
    assert path.exists() and path.stat().st_size > 0


def test_decomposition_figure_handles_a_pre_schema_artifact(tmp_path: Path) -> None:
    """An artifact predating the two-track schema must produce a legible placeholder."""
    path = tmp_path / "old.png"
    _score_alarm_flow_figure(pd.DataFrame({"dataset_id": ["metropt"]}), path)
    assert path.exists()


def test_decomposition_assets_write_csv_and_latex(artifact_root: Path, tmp_path: Path) -> None:
    tables, latex = tmp_path / "tables", tmp_path / "latex"
    tables.mkdir()
    latex.mkdir()
    config = PaperAssetConfig(real_data_matrix_root=artifact_root)
    generated = _decomposition_assets(config, tables, latex)
    assert len(generated) == 2
    text = (latex / "score_alarm_decomposition.tex").read_text(encoding="utf-8")
    assert "Failure layer" in text
    # Observations are constant per dataset and are stated in the caption instead.
    assert "Obs." not in text


def test_two_scale_timeline_figure_renders(tmp_path: Path) -> None:
    path = tmp_path / "timeline.png"
    overview = _overview_frame()
    overview = overview[
        (overview["dataset_id"] == "metropt") & (overview["method"] == "dynamical_evt_robust_score")
    ]
    trace = _trace_frame()
    trace = trace[trace["dataset_id"] == "metropt"]
    _two_scale_timeline_figure(overview, trace, _metadata_frame().iloc[0], path)
    assert path.exists() and path.stat().st_size > 0


def test_all_method_timeline_figure_renders_one_strip_per_method(tmp_path: Path) -> None:
    path = tmp_path / "all_methods.png"
    overview = _overview_frame()
    _all_method_timeline_figure(overview[overview["dataset_id"] == "metropt"], "metropt", path)
    assert path.exists() and path.stat().st_size > 0


def test_timeline_caption_states_every_configuration_field() -> None:
    caption = _timeline_caption(_metadata_frame().iloc[0])
    for fragment in (
        "threshold",
        "run length",
        "merge gap",
        "warning horizon",
        "operating days",
        "alarm episodes",
        "false alarms per operating day",
        "Time under warning",
    ):
        assert fragment in caption, fragment
    # The percent sign is escaped exactly once for LaTeX.
    assert "\\%" in caption
    assert "\\\\%" not in caption


def test_timeline_caption_reports_a_missing_match() -> None:
    metadata = _metadata_frame().iloc[0].copy()
    metadata["lead_time_hours"] = ""
    assert "No alarm matched the failure" in _timeline_caption(metadata)


def test_caption_macro_names_are_tex_legal() -> None:
    """Control sequences cannot contain digits, so dataset ids are romanised."""
    assert _timeline_caption_macro("metropt") == "\\timelinecaptionmetropt"
    assert _timeline_caption_macro("metropt2") == "\\timelinecaptionmetroptII"
    assert not any(character.isdigit() for character in _timeline_caption_macro("secom2"))


def test_timeline_traceability_assets_build_both_datasets(
    artifact_root: Path, tmp_path: Path
) -> None:
    figures, tables, latex = tmp_path / "f", tmp_path / "t", tmp_path / "l"
    for directory in (figures, tables, latex):
        directory.mkdir()
    config = PaperAssetConfig(real_data_matrix_root=artifact_root)
    generated = _timeline_traceability_assets(config, figures, tables, latex)
    names = {path.name for path in generated}
    assert "timeline_metropt.png" in names
    assert "timeline_metropt2.png" in names
    assert "timeline_all_methods_metropt.png" in names
    assert (latex / "timeline_caption_metropt.tex").exists()


def test_best_full_family_baseline_excludes_the_named_methods() -> None:
    baselines = _baseline_frame()
    best = _best_full_family_baseline(baselines, "metropt")
    assert best == "best_individual_sensor_threshold"
    assert _best_full_family_baseline(baselines, "absent") is None


def test_exposure_text_emits_a_plain_percent() -> None:
    row = pd.Series({"time_under_warning_samples": 2_500, "test_observations_samples": 100_000})
    assert _exposure_text(row) == "2.50% exposure"
    assert _exposure_text(pd.Series({"test_observations_samples": 0})) == "exposure unknown"


def test_root_cause_rows_are_method_specific(artifact_root: Path) -> None:
    config = PaperAssetConfig(real_data_matrix_root=artifact_root)
    rows = _method_root_cause_rows(config)
    assert rows
    layers = {row["Primary failure layer"] for row in rows}
    # More than one layer across methods is the point of the table.
    assert len(layers) > 1
    for row in rows:
        assert row["Alternative not excluded"]
        assert row["Confidence"] in {"high", "medium", "low"}


def test_non_event_rows_carry_an_estimand_mismatch(artifact_root: Path) -> None:
    config = PaperAssetConfig(real_data_matrix_root=artifact_root)
    rows = _non_event_root_cause_rows(config, artifact_root / "industrial_results_summary.csv")
    assert {row["Dataset"] for row in rows}
    for row in rows:
        assert row["Primary failure layer"] == "estimand mismatch"
        assert "not evidence that the method fails" in row["Alternative not excluded"]


def test_root_cause_assets_combine_both_row_sources(artifact_root: Path, tmp_path: Path) -> None:
    tables, latex = tmp_path / "t", tmp_path / "l"
    tables.mkdir()
    latex.mkdir()
    config = PaperAssetConfig(real_data_matrix_root=artifact_root)
    generated = _root_cause_assets(config, tables, latex)
    assert len(generated) == 2
    frame = pd.read_csv(tables / "root_cause_summary.csv")
    assert "estimand mismatch" in set(frame["Primary failure layer"])
    assert len(set(frame["Primary failure layer"])) > 1
