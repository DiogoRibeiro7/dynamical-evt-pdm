"""Plotting helpers for simulation-study outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_theta_bias_by_quantile(results: pd.DataFrame, output: Path) -> None:
    """Save a compact estimator-bias figure from tidy simulation results."""

    required = {"threshold_quantile", "runs_bias", "intervals_bias"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"missing required result columns: {missing}")
    summary = (
        results.groupby("threshold_quantile", as_index=False)[["runs_bias", "intervals_bias"]]
        .mean(numeric_only=True)
        .sort_values("threshold_quantile")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(summary["threshold_quantile"], summary["runs_bias"], marker="o", label="runs")
    axis.plot(
        summary["threshold_quantile"],
        summary["intervals_bias"],
        marker="s",
        label="intervals",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Threshold quantile")
    axis.set_ylabel("Mean theta bias")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
