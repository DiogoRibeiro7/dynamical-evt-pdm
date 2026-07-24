"""Small end-to-end analysis pipeline used by the command-line interface."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.extremal_index import intervals_extremal_index, runs_extremal_index
from dyn_evt_pdm.evt.thresholds import apply_regime_thresholds, fit_regime_thresholds


def analyse_series(
    frame: pd.DataFrame,
    *,
    value_column: str,
    regime_column: str,
    quantile: float,
    run_length: int,
) -> dict[str, object]:
    """Fit per-regime thresholds and summarize extreme clusters."""

    for column in (value_column, regime_column):
        if column not in frame:
            raise ValueError(f"missing required column: {column}")

    thresholds = fit_regime_thresholds(
        frame[value_column],
        frame[regime_column],
        quantile=quantile,
        min_regime_samples=20,
    )
    exceedances = apply_regime_thresholds(
        frame[value_column], frame[regime_column], thresholds
    ).to_numpy(dtype=bool)
    indices = np.flatnonzero(exceedances)
    clusters = extract_clusters(exceedances, run_length=run_length)

    intervals_theta = intervals_extremal_index(indices) if len(indices) >= 2 else None
    runs_theta = runs_extremal_index(exceedances, run_length=run_length) if len(indices) else None

    return {
        "n_samples": int(len(frame)),
        "n_exceedances": int(len(indices)),
        "n_clusters": int(len(clusters)),
        "thresholds": asdict(thresholds),
        "extremal_index": {
            "intervals": intervals_theta,
            "runs": runs_theta,
        },
        "clusters": [
            {
                "start": cluster.interval.start,
                "end": cluster.interval.end,
                "duration": cluster.interval.duration,
                "size": cluster.size,
            }
            for cluster in clusters
        ],
    }
