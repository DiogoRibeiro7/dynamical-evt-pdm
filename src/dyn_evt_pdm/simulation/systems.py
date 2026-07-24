"""Simulation systems for finite-sample EVT studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.maps import logistic_map
from dyn_evt_pdm.types import FloatArray


@dataclass(frozen=True, slots=True)
class SimulatedSeries:
    """One scalar series with optional ground-truth metadata."""

    system: str
    values: FloatArray
    true_theta: float | None
    frame: pd.DataFrame


def simulate_iid_pareto(
    n_steps: int,
    *,
    rng: np.random.Generator,
    shape: float = 2.0,
    missing_rate: float = 0.0,
) -> SimulatedSeries:
    """Generate an IID Pareto reference process with extremal index one."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if shape <= 0.0:
        raise ValueError("shape must be positive")
    values = (rng.pareto(shape, size=n_steps) + 1.0).astype(np.float64)
    values = _apply_missingness(values, rng=rng, missing_rate=missing_rate)
    frame = pd.DataFrame({"time": np.arange(n_steps, dtype=np.int64), "observable": values})
    return SimulatedSeries(system="iid_pareto", values=values, true_theta=1.0, frame=frame)


def simulate_logistic_target_observable(
    n_steps: int,
    *,
    rng: np.random.Generator,
    target: float = 0.75,
    parameter: float = 4.0,
    noise_scale: float = 0.0,
    missing_rate: float = 0.0,
) -> SimulatedSeries:
    """Generate a ``-log(distance)`` observable from a chaotic logistic map."""

    x0 = float(rng.uniform(0.05, 0.95))
    states = logistic_map(n_steps, x0=x0, parameter=parameter)
    distances = np.abs(states - target)
    if noise_scale > 0.0:
        distances = np.abs(distances + rng.normal(0.0, noise_scale, n_steps))
    values = -np.log(np.maximum(distances, np.finfo(float).eps)).astype(np.float64)
    values = _apply_missingness(values, rng=rng, missing_rate=missing_rate)
    true_theta = 0.5 if parameter == 4.0 and target == 0.75 and noise_scale == 0.0 else None
    frame = pd.DataFrame(
        {
            "time": np.arange(n_steps, dtype=np.int64),
            "state": states,
            "observable": values,
            "target": target,
        }
    )
    return SimulatedSeries(
        system="logistic_periodic_target",
        values=values,
        true_theta=true_theta,
        frame=frame,
    )


def simulate_cyclic_degradation_series(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_scale: float = 0.08,
    missing_rate: float = 0.0,
    fault_duration_fraction: float = 0.12,
) -> SimulatedSeries:
    """Generate the cyclic industrial surrogate with a persistent fault episode."""

    seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    config = CyclicSimulationConfig(
        n_steps=n_steps,
        noise_scale=noise_scale,
        fault_duration_fraction=fault_duration_fraction,
        seed=seed,
    )
    frame = simulate_cyclic_machine(config)
    values: FloatArray = frame["observable"].to_numpy(dtype=np.float64)
    values = _apply_missingness(values, rng=rng, missing_rate=missing_rate)
    frame = frame.copy()
    frame["observable"] = values
    return SimulatedSeries(
        system="cyclic_degradation",
        values=values,
        true_theta=None,
        frame=frame,
    )


def simulate_lagged_multivariate_extremes(
    n_steps: int,
    *,
    rng: np.random.Generator,
    lags: tuple[int, ...] = (0, 3, 7),
    missing_rate: float = 0.0,
) -> SimulatedSeries:
    """Generate lagged component extremes with known ordered pattern times."""

    if n_steps < max(lags) + 10:
        raise ValueError("n_steps is too short for requested lags")
    baseline = rng.normal(0.0, 1.0, size=(n_steps, len(lags)))
    starts = np.arange(20, n_steps - max(lags), 97, dtype=np.int64)
    for start in starts:
        for component, lag in enumerate(lags):
            baseline[start + lag, component] += 6.0
    values = baseline.max(axis=1).astype(np.float64)
    values = _apply_missingness(values, rng=rng, missing_rate=missing_rate)
    frame = pd.DataFrame(
        baseline,
        columns=[f"component_{index}" for index in range(len(lags))],
    )
    frame.insert(0, "time", np.arange(n_steps, dtype=np.int64))
    frame["observable"] = values
    frame["pattern_start"] = np.isin(np.arange(n_steps, dtype=np.int64), starts)
    return SimulatedSeries(
        system="lagged_multivariate",
        values=values,
        true_theta=None,
        frame=frame,
    )


def _apply_missingness(
    values: FloatArray,
    *,
    rng: np.random.Generator,
    missing_rate: float,
) -> FloatArray:
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    result = np.asarray(values, dtype=np.float64).copy()
    if missing_rate > 0.0:
        mask = rng.random(len(result)) < missing_rate
        result[mask] = np.nan
    return result
