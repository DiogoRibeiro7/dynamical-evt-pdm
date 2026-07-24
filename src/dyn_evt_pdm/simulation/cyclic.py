"""Synthetic cyclic machine with degradation, regimes and known event labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CyclicSimulationConfig:
    """Parameters for a transparent cyclic degradation benchmark."""

    n_steps: int = 20_000
    period: int = 120
    fault_start_fraction: float = 0.70
    fault_duration_fraction: float = 0.12
    noise_scale: float = 0.08
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_steps < 1_000:
            raise ValueError("n_steps must be at least 1000")
        if self.period < 4:
            raise ValueError("period must be at least 4")
        if not 0.0 < self.fault_start_fraction < 1.0:
            raise ValueError("fault_start_fraction must be in (0, 1)")
        if not 0.0 < self.fault_duration_fraction < 1.0:
            raise ValueError("fault_duration_fraction must be in (0, 1)")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")


def simulate_cyclic_machine(config: CyclicSimulationConfig) -> pd.DataFrame:
    """Generate sensor states, operating regimes and a dangerous-region observable.

    The degradation episode increases oscillation amplitude, persistence and lagged
    cross-sensor dependence. This gives simulations a known event interval while
    retaining periodic normal extremes.
    """

    rng = np.random.default_rng(config.seed)
    time = np.arange(config.n_steps, dtype=np.int64)
    phase = 2.0 * np.pi * time / float(config.period)

    phase_position = time % config.period
    regime = np.where(
        phase_position < config.period * 0.20,
        "off",
        np.where(phase_position < config.period * 0.65, "loaded", "recovery"),
    )

    fault_start = int(config.n_steps * config.fault_start_fraction)
    fault_duration = int(config.n_steps * config.fault_duration_fraction)
    fault_end = min(config.n_steps - 1, fault_start + fault_duration)
    is_fault = (time >= fault_start) & (time <= fault_end)

    progress = np.zeros(config.n_steps, dtype=np.float64)
    progress[is_fault] = np.linspace(0.0, 1.0, int(np.count_nonzero(is_fault)))
    progress[time > fault_end] = 1.0

    pressure = 6.0 + 1.2 * np.sin(phase) - 1.6 * progress
    current = 3.0 + 0.8 * np.cos(phase) + 1.4 * progress
    temperature = 50.0 + 2.5 * np.sin(phase - 0.35) + 8.0 * progress

    pressure += rng.normal(0.0, config.noise_scale, config.n_steps)
    current += rng.normal(0.0, config.noise_scale, config.n_steps)
    temperature += rng.normal(0.0, 3.0 * config.noise_scale, config.n_steps)

    # A scalar observable with normal periodic peaks and more persistent fault peaks.
    normalized_pressure_drop = np.maximum(0.0, (6.0 - pressure) / 1.5)
    normalized_current = np.maximum(0.0, (current - 3.0) / 1.2)
    normalized_temperature = np.maximum(0.0, (temperature - 50.0) / 7.0)
    observable = (
        0.45 * normalized_pressure_drop
        + 0.30 * normalized_current
        + 0.25 * normalized_temperature
    )

    return pd.DataFrame(
        {
            "time": time,
            "pressure": pressure,
            "current": current,
            "temperature": temperature,
            "regime": pd.Series(regime, dtype="string"),
            "degradation": progress,
            "observable": observable,
            "is_fault": is_fault,
        }
    )
