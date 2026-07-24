"""Command-line entry points for reproducible research workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml

from dyn_evt_pdm.data.acquisition import fetch_datasets, planned_files
from dyn_evt_pdm.data.prepare import prepare_metropt, prepare_scania
from dyn_evt_pdm.evt.dangerous_region import (
    StateProvenance,
    dangerous_region_from_training_failures,
    estimate_horizon_risk,
    score_dangerous_region,
)
from dyn_evt_pdm.evt.univariate import fit_univariate_evt
from dyn_evt_pdm.features.regimes import (
    ChangePointRegimeConfig,
    CompressorRegimeRules,
    HiddenStateRegimeModel,
    infer_changepoint_regime,
    infer_compressor_regime,
    regime_report,
)
from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.study import (
    simulation_study_config_from_mapping,
    write_simulation_study,
)
from dyn_evt_pdm.types import BoolArray

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
    states = frame[columns].to_numpy(dtype=float)
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

    results = fetch_datasets(dataset_names or ["all"], raw_root=raw_root, overwrite=overwrite)
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
