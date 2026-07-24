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
from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.study import (
    simulation_study_config_from_mapping,
    write_simulation_study,
)

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

    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(input_path)
    else:
        raise typer.BadParameter("input suffix must be .csv, .parquet or .pq")

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


if __name__ == "__main__":
    app()
