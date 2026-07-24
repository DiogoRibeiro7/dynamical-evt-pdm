"""Command-line entry points for reproducible research workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from dyn_evt_pdm.data.acquisition import fetch_datasets, planned_files
from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine

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


if __name__ == "__main__":
    app()
