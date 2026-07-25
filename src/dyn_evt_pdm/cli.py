"""Command-line entry points for reproducible research workflows."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer
import yaml

from dyn_evt_pdm.data.acquisition import fetch_datasets, planned_files
from dyn_evt_pdm.data.prepare import prepare_metropt, prepare_scania
from dyn_evt_pdm.data.registry import (
    build_data_report,
    normalize_dataset_ids,
    registry_records,
    render_dataset_latex_table,
    verify_dataset,
)
from dyn_evt_pdm.evaluation.protocol import read_protocol, write_protocol
from dyn_evt_pdm.evt.dangerous_region import (
    StateProvenance,
    dangerous_region_from_training_failures,
    estimate_horizon_risk,
    score_dangerous_region,
)
from dyn_evt_pdm.evt.multivariate import multivariate_evt_report
from dyn_evt_pdm.evt.univariate import fit_univariate_evt
from dyn_evt_pdm.features.regimes import (
    ChangePointRegimeConfig,
    CompressorRegimeRules,
    HiddenStateRegimeModel,
    infer_changepoint_regime,
    infer_compressor_regime,
    regime_report,
)
from dyn_evt_pdm.models.baseline_runner import (
    BaselineRunConfig,
    baseline_metadata_to_frame,
    run_baseline_experiment,
)
from dyn_evt_pdm.paper.assets import PaperAssetConfig, build_paper_assets
from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.study import (
    simulation_study_config_from_mapping,
    write_simulation_study,
)
from dyn_evt_pdm.types import BoolArray

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Real-data acquisition and provenance.")
app.add_typer(data_app, name="data")


@data_app.command("list")
def data_list_command(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """List registered real datasets and access metadata."""

    records = registry_records()
    if json_output:
        typer.echo(json.dumps(records, indent=2))
        return
    for record in records:
        typer.echo(
            f"{record['dataset_id']}: {record['version']} " f"({record['official_landing_page']})"
        )


@data_app.command("fetch")
def data_fetch_command(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset to fetch: all, metropt, metropt2, scania, scania-component-x.",
        ),
    ] = "all",
    raw_root: Annotated[Path, typer.Option(help="Raw-data root directory.")] = Path("data/raw"),
    dry_run: Annotated[bool, typer.Option(help="Resolve files without downloading.")] = False,
    force: Annotated[bool, typer.Option(help="Replace existing local raw files.")] = False,
    timeout_seconds: Annotated[int, typer.Option(min=1, help="Per-request timeout.")] = 60,
    retries: Annotated[int, typer.Option(min=0, help="Bounded retry count per file.")] = 2,
) -> None:
    """Fetch registered raw datasets, or show the acquisition plan."""

    dataset_ids = list(normalize_dataset_ids(dataset))
    files = planned_files(dataset_ids, raw_root=raw_root)
    if dry_run:
        for remote in files:
            size = (
                f"{remote.size_bytes:,} bytes" if remote.size_bytes is not None else "size unknown"
            )
            typer.echo(f"{remote.dataset}: {remote.filename} -> {remote.destination} ({size})")
        return
    results = fetch_datasets(
        dataset_ids,
        raw_root=raw_root,
        overwrite=force,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    for result in results:
        typer.echo(
            f"{result.status}: {result.dataset}/{result.filename} "
            f"({result.size_bytes:,} bytes) -> {result.path}"
        )


@data_app.command("verify")
def data_verify_command(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to verify: all, metropt, metropt2, scania."),
    ] = "all",
    raw_root: Annotated[Path, typer.Option(help="Raw-data root directory.")] = Path("data/raw"),
    processed_root: Annotated[Path, typer.Option(help="Processed-data root directory.")] = Path(
        "data/processed"
    ),
    output: Annotated[Path | None, typer.Option(help="Optional JSON verification output.")] = None,
) -> None:
    """Verify raw and processed dataset manifests and file checksums."""

    payload = [
        verify_dataset(dataset_id, raw_root=raw_root, processed_root=processed_root)
        for dataset_id in normalize_dataset_ids(dataset)
    ]
    text = json.dumps([asdict(item) for item in payload], indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    for item in payload:
        typer.echo(f"{item.dataset_id}: {item.status}")


@data_app.command("prepare")
def data_prepare_command(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", help="Dataset to prepare: metropt, metropt2, scania-component-x."
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional YAML preparation config."
        ),
    ] = None,
    raw_root: Annotated[Path, typer.Option(help="Raw-data root directory.")] = Path("data/raw"),
    processed_root: Annotated[Path, typer.Option(help="Processed-data root directory.")] = Path(
        "data/processed"
    ),
    chunk_size: Annotated[int, typer.Option(min=1_000)] = 500_000,
) -> None:
    """Prepare one registered raw dataset into deterministic Parquet artifacts."""

    dataset_id = normalize_dataset_ids(dataset)[0]
    config = _load_optional_yaml_mapping(config_path)
    configured_chunk_size = _config_int(config, "chunk_size", chunk_size)
    output_root = _config_path(config, "output_root", processed_root / dataset_id)
    if dataset_id == "metropt":
        raw_path = _config_path(config, "raw_path", raw_root / "metropt" / "dataset_train.csv")
        failure_yaml = _config_path(
            config, "failure_yaml", Path("docs/datasets/metropt_failures.yaml")
        )
        result = prepare_metropt(
            dataset="metropt",
            raw_path=raw_path,
            output_root=output_root,
            failure_yaml=failure_yaml,
            chunk_size=configured_chunk_size,
        )
    elif dataset_id == "metropt2":
        raw_path = _config_path(config, "raw_path", raw_root / "metropt2" / "MetroPT2.csv")
        result = prepare_metropt(
            dataset="metropt2",
            raw_path=raw_path,
            output_root=output_root,
            chunk_size=configured_chunk_size,
        )
    elif dataset_id == "scania_component_x":
        configured_scania_chunk_size = _config_int(config, "chunk_size", min(chunk_size, 100_000))
        result = prepare_scania(
            raw_root=_config_path(config, "raw_root", raw_root / "scania_component_x"),
            output_root=output_root,
            chunk_size=configured_scania_chunk_size,
        )
    else:
        raise typer.BadParameter(f"unsupported dataset for preparation: {dataset_id}")
    typer.echo(
        f"prepared {result.dataset}: {result.rows:,} rows across {result.chunks} parts; "
        f"manifest {result.manifest_path}"
    )


@data_app.command("report")
def data_report_command(
    output: Annotated[Path, typer.Option(help="JSON dataset report output.")] = Path(
        "artifacts/data/dataset_report.json"
    ),
    latex_output: Annotated[Path, typer.Option(help="LaTeX dataset table output.")] = Path(
        "reports/paper/tables/dataset_characteristics.tex"
    ),
    raw_root: Annotated[Path, typer.Option(help="Raw-data root directory.")] = Path("data/raw"),
    processed_root: Annotated[Path, typer.Option(help="Processed-data root directory.")] = Path(
        "data/processed"
    ),
) -> None:
    """Generate machine-readable and LaTeX dataset provenance reports."""

    report = build_data_report(raw_root=raw_root, processed_root=processed_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latex_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text(render_dataset_latex_table(report), encoding="utf-8")
    typer.echo(f"wrote dataset report to {output} and table to {latex_output}")


@app.command("simulate")
def simulate_command(
    output: Annotated[Path, typer.Option(help="Output CSV or Parquet path.")],
    n_steps: Annotated[int, typer.Option(min=1_000)] = 20_000,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Generate the cyclic degradation benchmark."""

    frame = simulate_cyclic_machine(CyclicSimulationConfig(n_steps=n_steps, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    elif output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    else:
        raise typer.BadParameter("output suffix must be .csv, .parquet or .pq")
    typer.echo(f"wrote {len(frame):,} rows to {output}")


@app.command("analyse-series")
def analyse_series_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(help="JSON summary output.")],
    value_column: Annotated[str, typer.Option()] = "observable",
    regime_column: Annotated[str, typer.Option()] = "regime",
    quantile: Annotated[float, typer.Option(min=0.5, max=0.9999)] = 0.98,
    run_length: Annotated[int, typer.Option(min=0)] = 10,
) -> None:
    """Analyse a scalar observable with regime-conditioned EVT."""

    frame = _read_cli_table(input_path)

    summary = analyse_series(
        frame,
        value_column=value_column,
        regime_column=regime_column,
        quantile=quantile,
        run_length=run_length,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"wrote analysis to {output}")


@app.command("analyse-univariate")
def analyse_univariate_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    value_column: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(help="JSON EVT result output.")],
    quantile: Annotated[float, typer.Option(min=0.5, max=0.9999)] = 0.98,
    run_length: Annotated[int, typer.Option(min=0)] = 10,
    min_exceedances: Annotated[int, typer.Option(min=3)] = 30,
    bootstrap_repetitions: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Fit structured univariate EVT diagnostics for one scalar column."""

    frame = _read_cli_table(input_path)
    if value_column not in frame:
        raise typer.BadParameter(f"value column {value_column!r} not found")
    result = fit_univariate_evt(
        frame[value_column].to_numpy(dtype=float),
        threshold_quantile=quantile,
        run_length=run_length,
        min_exceedances=min_exceedances,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    typer.echo(f"wrote univariate EVT analysis to {output}")


@app.command("analyse-regimes")
def analyse_regimes_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    value_column: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(help="JSON regime report output.")],
    method: Annotated[str, typer.Option(help="rules, hidden-state, or changepoint.")] = "rules",
    current_column: Annotated[
        str, typer.Option(help="Current column for rule regimes.")
    ] = "current",
    pressure_column: Annotated[
        str, typer.Option(help="Pressure column for rule regimes.")
    ] = "pressure",
    feature_columns: Annotated[
        str, typer.Option(help="Comma-separated feature columns for hidden-state regimes.")
    ] = "",
    quantile: Annotated[float, typer.Option(min=0.5, max=0.9999)] = 0.98,
    n_states: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    """Compare regime diagnostics and thresholds for one regime method."""

    frame = _read_cli_table(input_path)
    if value_column not in frame:
        raise typer.BadParameter(f"value column {value_column!r} not found")
    if method == "rules":
        for column in (current_column, pressure_column):
            if column not in frame:
                raise typer.BadParameter(f"rule column {column!r} not found")
        regimes = infer_compressor_regime(
            frame[current_column],
            frame[pressure_column],
            CompressorRegimeRules(current_on_threshold=1.0, pressure_recovery_derivative=0.05),
        )
    elif method == "hidden-state":
        columns = [column.strip() for column in feature_columns.split(",") if column.strip()]
        if not columns:
            raise typer.BadParameter("--feature-columns is required for hidden-state")
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise typer.BadParameter(f"missing feature columns: {missing}")
        model = HiddenStateRegimeModel(n_states=n_states).fit(frame[columns])
        regimes = model.predict(frame[columns])
    elif method == "changepoint":
        regimes = infer_changepoint_regime(
            frame, ChangePointRegimeConfig(value_column=value_column)
        )
    else:
        raise typer.BadParameter("method must be rules, hidden-state, or changepoint")
    report = regime_report(frame[value_column], regimes, quantile=quantile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"wrote regime report to {output}")


@app.command("analyse-dangerous-region")
def analyse_dangerous_region_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    state_columns: Annotated[str, typer.Option(help="Comma-separated numeric state columns.")],
    target_column: Annotated[str, typer.Option(help="Boolean training target-state column.")],
    output: Annotated[Path, typer.Option(help="Parquet score output.")],
    report_output: Annotated[Path, typer.Option(help="JSON report output.")],
    split_column: Annotated[str, typer.Option(help="Optional split column.")] = "",
    failure_id_column: Annotated[
        str, typer.Option(help="Optional failure/provenance column.")
    ] = "",
    train_split: Annotated[
        str, typer.Option(help="Split value allowed for target states.")
    ] = "train",
    hit_radius: Annotated[float, typer.Option(min=0.0)] = 0.1,
    horizons: Annotated[
        str, typer.Option(help="Comma-separated positive integer horizons.")
    ] = "900,1800,3600,7200",
) -> None:
    """Score recurrence to a dangerous region built from training target states."""

    frame = _read_cli_table(input_path)
    columns = [column.strip() for column in state_columns.split(",") if column.strip()]
    if not columns:
        raise typer.BadParameter("--state-columns must name at least one column")
    missing = sorted(set([*columns, target_column]).difference(frame.columns))
    if missing:
        raise typer.BadParameter(f"missing columns: {missing}")
    state_frame = (
        frame.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if split_column and split_column in frame:
        train_mask = frame[split_column].astype("string").eq(train_split).fillna(False)
    else:
        train_mask = pd.Series(True, index=frame.index)
    training_medians = state_frame.loc[train_mask].median(numeric_only=True)
    global_medians = state_frame.median(numeric_only=True)
    medians = training_medians.fillna(global_medians).fillna(0.0)
    imputed_state_values = int(state_frame.isna().sum().sum())
    states = state_frame.fillna(medians).to_numpy(dtype=float)
    target_flags = frame[target_column].astype(bool).to_numpy()
    provenance = _build_cli_provenance(
        frame,
        target_flags=target_flags,
        split_column=split_column,
        failure_id_column=failure_id_column,
        train_split=train_split,
    )
    allowed_ids = {
        item.failure_id
        for item, flag in zip(provenance, target_flags, strict=True)
        if flag and item.split == "train" and item.failure_id is not None
    }
    if not allowed_ids:
        allowed_ids = {"training_target"}
        provenance = tuple(
            StateProvenance(
                index=item.index,
                source=item.source,
                failure_id="training_target" if target_flags[item.index] else item.failure_id,
                split=item.split,
            )
            for item in provenance
        )
    region = dangerous_region_from_training_failures(
        states,
        provenance,
        allowed_failure_ids=set(allowed_ids),
        max_references=250,
    )
    scores = score_dangerous_region(states, region)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output, index=False)
    horizon_values = tuple(int(value.strip()) for value in horizons.split(",") if value.strip())
    risk = estimate_horizon_risk(
        scores["distance_to_dangerous_region"].to_numpy(dtype=float),
        hit_radius=hit_radius,
        horizons=horizon_values,
    )
    report = {
        "region_method": region.method,
        "n_references": len(region.references),
        "metadata": region.metadata,
        "imputed_state_values": imputed_state_values,
        "score_output": str(output),
        "horizon_risk": {
            "horizons": risk.horizons,
            "probabilities": risk.probabilities,
            "brier_scores": risk.brier_scores,
            "calibration_bins": risk.calibration_bins,
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"wrote dangerous-region scores to {output} and report to {report_output}")


@app.command("analyse-multivariate-extremes")
def analyse_multivariate_extremes_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    component_columns: Annotated[
        str, typer.Option(help="Comma-separated component columns to threshold.")
    ],
    regime_column: Annotated[str, typer.Option(help="Operating-regime column.")],
    split_column: Annotated[str, typer.Option(help="Train/validation/heldout split column.")],
    output: Annotated[Path, typer.Option(help="JSON multivariate EVT report output.")],
    train_split: Annotated[str, typer.Option()] = "train",
    validation_split: Annotated[str, typer.Option()] = "validation",
    heldout_split: Annotated[str, typer.Option()] = "test",
    quantile: Annotated[float, typer.Option(min=0.5, max=0.9999)] = 0.98,
    max_lag: Annotated[int, typer.Option(min=0)] = 10,
    n_null: Annotated[int, typer.Option(min=0)] = 199,
    null_model: Annotated[
        str, typer.Option(help="time_shift or cluster_permutation.")
    ] = "time_shift",
    alpha: Annotated[float, typer.Option(min=0.001, max=1.0)] = 0.1,
    min_regime_samples: Annotated[int, typer.Option(min=1)] = 100,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Analyse simultaneous and lagged multivariate extreme patterns."""

    frame = _read_cli_table(input_path)
    columns = tuple(column.strip() for column in component_columns.split(",") if column.strip())
    if len(columns) < 2:
        raise typer.BadParameter("--component-columns must name at least two columns")
    missing = sorted(set([*columns, regime_column, split_column]).difference(frame.columns))
    if missing:
        raise typer.BadParameter(f"missing columns: {missing}")
    splits = frame[split_column].astype("string")
    train_mask = (splits == train_split).fillna(False).to_numpy(dtype=bool)
    validation_mask = (splits == validation_split).fillna(False).to_numpy(dtype=bool)
    heldout_mask = (splits == heldout_split).fillna(False).to_numpy(dtype=bool)
    if not train_mask.any():
        raise typer.BadParameter("train split has no rows")
    if not validation_mask.any():
        raise typer.BadParameter("validation split has no rows")
    report = multivariate_evt_report(
        frame,
        component_columns=columns,
        regime_column=regime_column,
        train_mask=train_mask,
        validation_mask=validation_mask,
        heldout_mask=heldout_mask if heldout_mask.any() else None,
        quantile=quantile,
        max_lag=max_lag,
        n_null=n_null,
        null_model=null_model,
        alpha=alpha,
        min_regime_samples=min_regime_samples,
        random_state=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"wrote multivariate EVT report to {output}")


@app.command("run-baselines")
def run_baselines_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    feature_columns: Annotated[
        str, typer.Option(help="Comma-separated feature columns shared by all baselines.")
    ],
    output: Annotated[Path, typer.Option(help="Parquet standardized prediction table output.")],
    metadata_output: Annotated[Path, typer.Option(help="JSON metadata output.")],
    dataset_id: Annotated[
        str, typer.Option(help="Dataset identifier for prediction rows.")
    ] = "unknown",
    entity_id_column: Annotated[str, typer.Option(help="Optional entity identifier column.")] = "",
    timestamp_column: Annotated[str, typer.Option(help="Optional timestamp column.")] = "",
    regime_column: Annotated[str, typer.Option(help="Optional regime column.")] = "regime",
    split_column: Annotated[str, typer.Option(help="Split column.")] = "split",
    target_column: Annotated[str, typer.Option(help="Optional boolean target column.")] = "",
    exclusion_column: Annotated[str, typer.Option(help="Optional exclusion mask column.")] = "",
    train_split: Annotated[str, typer.Option()] = "train",
    validation_split: Annotated[str, typer.Option()] = "validation",
    test_split: Annotated[str, typer.Option()] = "test",
    window_size: Annotated[int, typer.Option(min=1)] = 1,
    horizon: Annotated[int, typer.Option(min=1)] = 60,
    isolation_estimators: Annotated[int, typer.Option(min=1)] = 100,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Run fair baselines and write standardized prediction rows."""

    frame = _read_cli_table(input_path)
    columns = tuple(column.strip() for column in feature_columns.split(",") if column.strip())
    if not columns:
        raise typer.BadParameter("--feature-columns must name at least one column")
    config = BaselineRunConfig(
        feature_columns=columns,
        dataset_id=dataset_id,
        entity_id_column=entity_id_column or None,
        timestamp_column=timestamp_column or None,
        regime_column=regime_column or None,
        split_column=split_column,
        target_column=target_column or None,
        exclusion_column=exclusion_column or None,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        window_size=window_size,
        horizon=horizon,
        isolation_n_estimators=isolation_estimators,
        random_state=seed,
    )
    try:
        result = run_baseline_experiment(frame, config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(output, index=False)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame = baseline_metadata_to_frame(result.metadata)
    metadata_output.write_text(
        json.dumps(metadata_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    typer.echo(f"wrote baseline predictions to {output} and metadata to {metadata_output}")


@app.command("freeze-evaluation-protocol")
def freeze_evaluation_protocol_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/evaluation/base.yaml"
    ),
    output: Annotated[Path, typer.Option(help="Hashed JSON protocol artifact output.")] = Path(
        "artifacts/protocols/evaluation_protocol.json"
    ),
) -> None:
    """Validate and freeze a versioned event-evaluation protocol."""

    try:
        protocol = read_protocol(config_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    write_protocol(protocol, output)
    typer.echo(f"wrote protocol {protocol.protocol_id} hash={protocol.protocol_hash} to {output}")


@app.command("build-paper-assets")
def build_paper_assets_command(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, help="Cached processed table.")
    ] = Path("data/processed/synthetic_cyclic.csv"),
    output_root: Annotated[
        Path, typer.Option(help="Output root for figures, tables and manifest.")
    ] = Path("reports/paper"),
    simulation_study_path: Annotated[
        Path, typer.Option(help="Cached or generated simulation-study Parquet.")
    ] = Path("artifacts/simulation_study_smoke.parquet"),
    value_column: Annotated[str, typer.Option()] = "observable",
    timestamp_column: Annotated[str, typer.Option()] = "time",
    regime_column: Annotated[str, typer.Option()] = "regime",
    failure_column: Annotated[str, typer.Option()] = "is_fault",
    split_column: Annotated[str, typer.Option()] = "split",
) -> None:
    """Build publication figures, tables and provenance manifest from cached data."""

    try:
        manifest = build_paper_assets(
            PaperAssetConfig(
                input_path=input_path,
                output_root=output_root,
                simulation_study_path=simulation_study_path,
                value_column=value_column,
                timestamp_column=timestamp_column,
                regime_column=regime_column,
                failure_column=failure_column,
                split_column=split_column,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"wrote {len(manifest.generated_files)} paper assets; "
        f"manifest {output_root / 'asset_manifest.json'}"
    )


@app.command("fetch-data")
def fetch_data_command(
    dataset: Annotated[
        str,
        typer.Option(
            help="Dataset to fetch: all, metropt, metropt2, scania or scania_component_x."
        ),
    ] = "all",
    raw_root: Annotated[Path, typer.Option(help="Raw-data root directory.")] = Path("data/raw"),
    overwrite: Annotated[bool, typer.Option(help="Replace existing local raw files.")] = False,
    list_only: Annotated[bool, typer.Option(help="List source files without downloading.")] = False,
    timeout_seconds: Annotated[int, typer.Option(min=1, help="Per-request timeout.")] = 60,
    retries: Annotated[int, typer.Option(min=0, help="Bounded retry count per file.")] = 2,
) -> None:
    """Fetch public real datasets into the local raw-data directory."""

    dataset_names = [name.strip() for name in dataset.split(",") if name.strip()]
    files = planned_files(dataset_names or ["all"], raw_root=raw_root)
    if list_only:
        for remote in files:
            size = (
                f"{remote.size_bytes:,} bytes" if remote.size_bytes is not None else "size unknown"
            )
            typer.echo(f"{remote.dataset}: {remote.filename} -> {remote.destination} ({size})")
        return

    results = fetch_datasets(
        dataset_names or ["all"],
        raw_root=raw_root,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    for result in results:
        typer.echo(
            f"{result.status}: {result.dataset}/{result.filename} "
            f"({result.size_bytes:,} bytes) -> {result.path}"
        )


@app.command("prepare-metropt")
def prepare_metropt_command(
    raw_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/raw/metropt/dataset_train.csv"
    ),
    output_root: Annotated[Path, typer.Option(help="Processed output directory.")] = Path(
        "data/processed/metropt"
    ),
    failure_yaml: Annotated[Path, typer.Option(help="MetroPT failure-label YAML.")] = Path(
        "docs/datasets/metropt_failures.yaml"
    ),
    chunk_size: Annotated[int, typer.Option(min=1_000)] = 500_000,
) -> None:
    """Prepare the real MetroPT raw CSV into Parquet parts."""

    result = prepare_metropt(
        dataset="metropt",
        raw_path=raw_path,
        output_root=output_root,
        failure_yaml=failure_yaml,
        chunk_size=chunk_size,
    )
    typer.echo(
        f"prepared {result.dataset}: {result.rows:,} rows across {result.chunks} parts; "
        f"manifest {result.manifest_path}"
    )


@app.command("prepare-metropt2")
def prepare_metropt2_command(
    raw_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/raw/metropt2/MetroPT2.csv"
    ),
    output_root: Annotated[Path, typer.Option(help="Processed output directory.")] = Path(
        "data/processed/metropt2"
    ),
    chunk_size: Annotated[int, typer.Option(min=1_000)] = 500_000,
) -> None:
    """Prepare the real MetroPT2 raw CSV into Parquet parts."""

    result = prepare_metropt(
        dataset="metropt2",
        raw_path=raw_path,
        output_root=output_root,
        chunk_size=chunk_size,
    )
    typer.echo(
        f"prepared {result.dataset}: {result.rows:,} rows across {result.chunks} parts; "
        f"manifest {result.manifest_path}"
    )


@app.command("prepare-scania")
def prepare_scania_command(
    raw_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/raw/scania_component_x"
    ),
    output_root: Annotated[Path, typer.Option(help="Processed output directory.")] = Path(
        "data/processed/scania_component_x"
    ),
    chunk_size: Annotated[int, typer.Option(min=1_000)] = 100_000,
) -> None:
    """Prepare the real SCANIA Component X raw CSV files into Parquet parts."""

    result = prepare_scania(raw_root=raw_root, output_root=output_root, chunk_size=chunk_size)
    typer.echo(
        f"prepared {result.dataset}: {result.rows:,} rows across {result.chunks} parts; "
        f"manifest {result.manifest_path}"
    )


@app.command("run-simulation-study")
def run_simulation_study_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/simulation/cyclic_degradation.yaml"
    ),
    output: Annotated[Path, typer.Option(help="Tidy Parquet result path.")] = Path(
        "artifacts/simulation_study.parquet"
    ),
    smoke: Annotated[bool, typer.Option(help="Use a small CI-friendly grid.")] = False,
    n_jobs: Annotated[int, typer.Option(help="Parallel jobs for repetitions.")] = 1,
) -> None:
    """Run a deterministic finite-sample simulation study."""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("simulation config root must be a mapping")
    study_config = simulation_study_config_from_mapping(raw, smoke=smoke, n_jobs=n_jobs)
    result = write_simulation_study(study_config, output)
    typer.echo(f"wrote {len(result):,} simulation rows to {output}")


def _read_cli_table(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise typer.BadParameter("input suffix must be .csv, .parquet or .pq")


def _load_optional_yaml_mapping(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise typer.BadParameter("preparation config root must be a mapping")
    return cast(dict[str, object], raw)


def _config_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise typer.BadParameter(f"config value {key!r} must be an integer")


def _config_path(config: dict[str, object], key: str, default: Path) -> Path:
    value = config.get(key, default)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise typer.BadParameter(f"config value {key!r} must be a path string")


def _build_cli_provenance(
    frame: pd.DataFrame,
    *,
    target_flags: BoolArray,
    split_column: str,
    failure_id_column: str,
    train_split: str,
) -> tuple[StateProvenance, ...]:
    if split_column and split_column not in frame:
        raise typer.BadParameter(f"split column {split_column!r} not found")
    if failure_id_column and failure_id_column not in frame:
        raise typer.BadParameter(f"failure id column {failure_id_column!r} not found")
    provenance: list[StateProvenance] = []
    for position, (_index, row) in enumerate(frame.iterrows()):
        split = str(row[split_column]) if split_column else train_split
        failure_id = (
            str(row[failure_id_column])
            if target_flags[position] and failure_id_column and pd.notna(row[failure_id_column])
            else None
        )
        provenance.append(
            StateProvenance(
                index=position,
                source="cli_input",
                failure_id=failure_id,
                split="train" if split == train_split else split,
            )
        )
    return tuple(provenance)


if __name__ == "__main__":
    app()
