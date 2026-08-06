"""Main-paper figures for the extremal-index estimator study.

Every figure is faceted by reference regime rather than pooled. The grid establishes
finite-sample behaviour for near-independent processes and a single strongly clustered
process, and the estimator ranking inverts between the two, so a pooled curve would
average across that inversion and report a ranking that holds in neither regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dyn_evt_pdm.evt.extremal_index import ESTIMATOR_NAMES

#: Colourblind-safe categorical assignment, fixed per estimator and never cycled.
#:
#: Validated in light mode for the CVD adjacent-pair floor, the normal-vision floor,
#: the lightness band and the chroma floor. Three hues sit below the 3:1 contrast
#: ratio against white, so every series also carries a distinct marker as secondary
#: encoding and the same numbers ship as CSV and LaTeX tables.
ESTIMATOR_COLOURS: dict[str, str] = {
    "runs": "#0072B2",
    "ferro_segers_intervals": "#D55E00",
    "k_gaps": "#009E73",
    "reciprocal_mean_cluster": "#E69F00",
    "block": "#CC79A7",
    "no_declustering": "#56B4E9",
}

#: Secondary encoding so identity never rests on colour alone.
ESTIMATOR_MARKERS: dict[str, str] = {
    "runs": "o",
    "ferro_segers_intervals": "s",
    "k_gaps": "^",
    "reciprocal_mean_cluster": "D",
    "block": "v",
    "no_declustering": "X",
}

#: Human-readable estimator labels for legends and axis ticks.
ESTIMATOR_LABELS: dict[str, str] = {
    "runs": "Runs",
    "ferro_segers_intervals": "Ferro-Segers",
    "k_gaps": "K-gaps",
    "reciprocal_mean_cluster": "Reciprocal block cluster",
    "block": "Disjoint blocks",
    "no_declustering": "No declustering",
}

REGIME_LABELS: dict[str, str] = {
    "near_independent": r"Near-independent ($\theta \geq 0.9$)",
    "moderately_clustered": r"Moderately clustered ($0.5 \leq \theta < 0.9$)",
    "strongly_clustered": r"Strongly clustered ($\theta < 0.5$)",
}

NOMINAL_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class FigureSet:
    """Paths written by the estimator figure builder."""

    rmse_by_sample_size: Path
    coverage_by_family: Path
    failure_by_threshold: Path
    sensitivity: Path

    def as_tuple(self) -> tuple[Path, ...]:
        """Return every generated path."""

        return (
            self.rmse_by_sample_size,
            self.coverage_by_family,
            self.failure_by_threshold,
            self.sensitivity,
        )


def assign_regime(theta: float) -> str:
    """Classify a reference extremal index into a clustering regime."""

    if not np.isfinite(theta):
        return "unreferenced"
    if theta >= 0.9:
        return "near_independent"
    if theta >= 0.5:
        return "moderately_clustered"
    return "strongly_clustered"


def _referenced(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[summary["reference_theta"].notna()].copy()
    frame["regime"] = [assign_regime(float(value)) for value in frame["reference_theta"]]
    return frame


def _style_axis(axis: Axes, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title, fontsize=9)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _save(figure: Figure, path: Path, *, legend_rows: int = 0) -> Path:
    """Save a figure, reserving space below the axes for a figure-level legend.

    ``tight_layout`` recomputes the axes rectangle and would otherwise overwrite any
    prior ``subplots_adjust``, dropping the legend on top of the x-axis labels.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.0 if legend_rows == 0 else min(0.30, 0.035 + 0.045 * legend_rows)
    figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    figure.savefig(path, dpi=180, metadata={"Creator": "dyn-evt-pdm estimator figures"})
    plt.close(figure)
    return path


def _estimator_legend(figure: Figure, axis: Axes, *, ncol: int = 3) -> int:
    """Attach a figure-level legend and return the number of rows it occupies."""

    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return 0
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    return int(np.ceil(len(handles) / ncol))


def plot_rmse_by_sample_size(summary: pd.DataFrame, path: Path) -> Path:
    """Plot RMSE against sample size, faceted by clustering regime."""

    frame = _referenced(summary)
    regimes = [name for name in REGIME_LABELS if name in set(frame["regime"])]
    if not regimes:
        raise ValueError("no referenced configurations available for the RMSE figure")

    # Independent y-scales: the no-declustering reference has RMSE |1 - theta| by
    # construction, so a shared scale lets it compress the five real estimators into
    # an unreadable band in the strongly clustered panel.
    figure, axes = plt.subplots(
        1, len(regimes), figsize=(4.2 * len(regimes), 3.8), squeeze=False, sharey=False
    )
    sample_sizes = sorted(set(frame["n_steps"]))
    for column, regime in enumerate(regimes):
        axis = axes[0][column]
        subset = frame[frame["regime"] == regime]
        for name in ESTIMATOR_NAMES:
            rows = subset[subset["estimator"] == name].sort_values("n_steps")
            if rows.empty:
                continue
            grouped = rows.groupby("n_steps", as_index=False).agg(
                rmse=("rmse", "mean"), error=("rmse_standard_error", "mean")
            )
            axis.errorbar(
                grouped["n_steps"],
                grouped["rmse"],
                yerr=1.96 * grouped["error"].fillna(0.0),
                marker=ESTIMATOR_MARKERS[name],
                markersize=6,
                linewidth=2.0,
                capsize=3,
                color=ESTIMATOR_COLOURS[name],
                label=ESTIMATOR_LABELS[name] if column == 0 else None,
            )
        _style_axis(
            axis,
            xlabel="Series length (samples)",
            ylabel=r"RMSE of $\hat{\theta}$" if column == 0 else "",
            title=REGIME_LABELS[regime],
        )
        # Only a handful of series lengths are simulated; a continuous axis would
        # imply intermediate points that were never run.
        axis.set_xticks(sample_sizes)
        axis.set_xticklabels([f"{int(value):,}" for value in sample_sizes])
        axis.set_xlim(
            min(sample_sizes) - 0.08 * (max(sample_sizes) - min(sample_sizes) or 1),
            max(sample_sizes) + 0.08 * (max(sample_sizes) - min(sample_sizes) or 1),
        )
        axis.set_ylim(bottom=0.0)
    rows_used = _estimator_legend(figure, axes[0][0])
    return _save(figure, path, legend_rows=rows_used)


def plot_coverage_by_family(summary: pd.DataFrame, path: Path) -> Path:
    """Plot empirical interval coverage per process family against the nominal level."""

    frame = _referenced(summary)
    families = sorted(set(frame["system"]))
    if not families:
        raise ValueError("no referenced configurations available for the coverage figure")

    # Horizontal bars: with eleven families and six estimators the vertical form needed
    # rotated tick labels that were unreadable at print size, and family names are long.
    figure, axis = plt.subplots(figsize=(8.0, max(5.0, 0.62 * len(families) + 1.6)))
    positions = np.arange(len(families), dtype=float)
    height_step = 0.13
    for index, name in enumerate(ESTIMATOR_NAMES):
        rows = frame[frame["estimator"] == name]
        values: list[float] = []
        errors: list[float] = []
        for family in families:
            family_rows = rows[rows["system"] == family]
            values.append(float(family_rows["coverage"].mean()) if len(family_rows) else np.nan)
            errors.append(
                float(family_rows["coverage_standard_error"].mean()) if len(family_rows) else 0.0
            )
        offset = (index - (len(ESTIMATOR_NAMES) - 1) / 2.0) * height_step
        lengths = np.asarray(values, dtype=float)
        axis.barh(
            positions + offset,
            np.nan_to_num(lengths),
            height=height_step * 0.88,
            xerr=1.96 * np.nan_to_num(np.asarray(errors, dtype=float)),
            capsize=2,
            color=ESTIMATOR_COLOURS[name],
            label=ESTIMATOR_LABELS[name],
            linewidth=0.0,
        )
        # A coverage of exactly zero and an unestimable coverage both draw no bar, so
        # they are marked apart explicitly rather than left to look like the same gap.
        for position, length in zip(positions + offset, lengths, strict=True):
            if np.isnan(length):
                axis.annotate(
                    "n/a",
                    xy=(0.012, position),
                    ha="left",
                    va="center",
                    fontsize=5.5,
                    color="#777777",
                )
            elif length <= 0.005:
                axis.plot(
                    [0.006, 0.006],
                    [position - height_step * 0.3, position + height_step * 0.3],
                    color=ESTIMATOR_COLOURS[name],
                    linewidth=2.0,
                    solid_capstyle="butt",
                )
    axis.axvline(NOMINAL_LEVEL, color="#444444", linestyle="--", linewidth=1.2, zorder=5)
    axis.annotate(
        f"Nominal {NOMINAL_LEVEL:.2f}",
        xy=(NOMINAL_LEVEL, len(families) - 0.45),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color="#444444",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels([family.replace("_", " ") for family in families], fontsize=8.5)
    axis.set_ylim(-0.6, len(families) - 0.4)
    axis.set_xlim(0.0, 1.10)
    _style_axis(axis, xlabel="Empirical coverage", ylabel="Process family")
    rows_used = _estimator_legend(figure, axis)
    return _save(figure, path, legend_rows=rows_used)


def plot_failure_by_threshold(summary: pd.DataFrame, path: Path) -> Path:
    """Plot non-estimability against threshold quantile for each estimator."""

    frame = _referenced(summary)
    quantiles = sorted(set(frame["threshold_quantile"]))
    if not quantiles:
        raise ValueError("no referenced configurations available for the failure figure")

    panels = (
        ("invalid_probability", "Non-estimable share"),
        ("clipping_probability", "Boundary-clipped share"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(9.0, 3.8), squeeze=False, sharey=True)
    for column, (metric, label) in enumerate(panels):
        axis = axes[0][column]
        for name in ESTIMATOR_NAMES:
            rows = frame[frame["estimator"] == name]
            if rows.empty:
                continue
            means = [
                float(rows[rows["threshold_quantile"] == value][metric].mean())
                for value in quantiles
            ]
            axis.plot(
                quantiles,
                means,
                marker=ESTIMATOR_MARKERS[name],
                markersize=5,
                linewidth=2.0,
                color=ESTIMATOR_COLOURS[name],
                label=ESTIMATOR_LABELS[name] if column == 0 else None,
            )
        _style_axis(
            axis,
            xlabel="Threshold quantile",
            ylabel=label if column == 0 else "",
            title=label,
        )
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(quantiles)
        axis.set_xticklabels([f"{value:.2f}" for value in quantiles])
        span = (max(quantiles) - min(quantiles)) or 0.01
        axis.set_xlim(min(quantiles) - 0.12 * span, max(quantiles) + 0.12 * span)
        # A panel that is flat at zero is a result, not a rendering fault; say so.
        panel_values = frame.groupby("threshold_quantile")[metric].mean()
        if float(panel_values.max()) <= 0.005:
            axis.annotate(
                "No estimator failed to return\nan estimate at any threshold",
                xy=(0.5, 0.55),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="#666666",
            )
    rows_used = _estimator_legend(figure, axes[0][0])
    return _save(figure, path, legend_rows=rows_used)


def plot_sensitivity(summary: pd.DataFrame, path: Path) -> Path:
    """Plot estimate sensitivity to noise, missingness and regime mixture.

    Systems without an established reference still carry information about estimator
    dispersion, so they appear here as estimates rather than as bias.
    """

    frame = summary.copy()
    frame["regime"] = [
        assign_regime(float(value)) if pd.notna(value) else "unreferenced"
        for value in frame["reference_theta"]
    ]
    conditions = [
        ("noise_scale", "Noise scale"),
        ("missing_rate", "Missing rate"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), squeeze=False, sharey=True)
    for column, (field, label) in enumerate(conditions):
        axis = axes[0][column]
        levels = sorted(set(frame[field]))
        for name in ESTIMATOR_NAMES:
            rows = frame[frame["estimator"] == name]
            means = [float(rows[rows[field] == level]["mean_estimate"].mean()) for level in levels]
            axis.plot(
                levels,
                means,
                marker=ESTIMATOR_MARKERS[name],
                markersize=6,
                linewidth=2.0,
                color=ESTIMATOR_COLOURS[name],
                label=ESTIMATOR_LABELS[name] if column == 0 else None,
            )
        _style_axis(
            axis,
            xlabel=label,
            ylabel=r"Mean $\hat{\theta}$" if column == 0 else "",
            title=f"Sensitivity to {label.lower()}",
        )
        axis.set_ylim(0.0, 1.08)
        axis.set_xticks(levels)
        axis.set_xticklabels([f"{value:g}" for value in levels])
        span = (max(levels) - min(levels)) or 1.0
        axis.set_xlim(min(levels) - 0.15 * span, max(levels) + 0.15 * span)

    # Regime mixture is a process family rather than a swept parameter, so it is shown
    # against the pooled remainder instead of along an axis.
    axis = axes[0][2]
    mixture_mask = frame["system"] == "regime_mixture"
    groups = [("Regime mixture", frame[mixture_mask]), ("All other families", frame[~mixture_mask])]
    positions = np.arange(len(groups), dtype=float)
    width = 0.13
    for index, name in enumerate(ESTIMATOR_NAMES):
        offset = (index - (len(ESTIMATOR_NAMES) - 1) / 2.0) * width
        heights = [
            float(subset[subset["estimator"] == name]["mean_estimate"].mean())
            for _, subset in groups
        ]
        axis.bar(
            positions + offset,
            np.nan_to_num(np.asarray(heights, dtype=float)),
            width=width * 0.88,
            color=ESTIMATOR_COLOURS[name],
            linewidth=0.0,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([label for label, _ in groups])
    axis.set_ylim(0.0, 1.08)
    _style_axis(axis, xlabel="Process family", ylabel="", title="Sensitivity to regime mixture")

    rows_used = _estimator_legend(figure, axes[0][0], ncol=6)
    return _save(figure, path, legend_rows=rows_used)


def build_estimator_figures(summary: pd.DataFrame, output_root: Path) -> FigureSet:
    """Build every main-paper estimator figure from one summary frame."""

    figures = output_root / "figures"
    return FigureSet(
        rmse_by_sample_size=plot_rmse_by_sample_size(
            summary, figures / "estimator_rmse_by_sample_size.png"
        ),
        coverage_by_family=plot_coverage_by_family(
            summary, figures / "estimator_coverage_by_family.png"
        ),
        failure_by_threshold=plot_failure_by_threshold(
            summary, figures / "estimator_failure_by_threshold.png"
        ),
        sensitivity=plot_sensitivity(summary, figures / "estimator_sensitivity.png"),
    )
