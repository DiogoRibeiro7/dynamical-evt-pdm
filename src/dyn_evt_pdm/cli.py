"""Command-line entry points for reproducible research workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("simulate")
def simulate_command(
    output: Path = typer.Option(..., help="Output CSV or Parquet path."),
    n_steps: int = typer.Option(20_000, min=1_000),
    seed: int = typer.Option(42),
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
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    value_column: str = typer.Option("observable"),
    regime_column: str = typer.Option("regime"),
    quantile: float = typer.Option(0.98, min=0.5, max=0.9999),
    run_length: int = typer.Option(10, min=0),
    output: Path = typer.Option(..., help="JSON summary output."),
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


if __name__ == "__main__":
    app()
