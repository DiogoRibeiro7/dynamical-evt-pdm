"""Operating-regime inference with transparent deterministic rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CompressorRegimeRules:
    """Simple rule thresholds used as an auditable first regime model."""

    current_on_threshold: float
    pressure_recovery_derivative: float
    off_label: str = "off"
    loaded_label: str = "loaded"
    recovery_label: str = "recovery"


def infer_compressor_regime(
    current: pd.Series,
    pressure: pd.Series,
    rules: CompressorRegimeRules,
) -> pd.Series:
    """Infer coarse regimes from current and pressure dynamics.

    This rule-based implementation is a baseline. A hidden-state or supervised
    regime model may replace it only after the baseline is reported.
    """

    if len(current) != len(pressure):
        raise ValueError("current and pressure series must have equal length")
    current_values = pd.to_numeric(current, errors="raise").to_numpy(dtype=float)
    pressure_values = pd.to_numeric(pressure, errors="raise").to_numpy(dtype=float)
    derivative = np.diff(pressure_values, prepend=pressure_values[0])

    labels = np.full(len(current_values), rules.loaded_label, dtype=object)
    labels[current_values <= rules.current_on_threshold] = rules.off_label
    labels[
        (current_values > rules.current_on_threshold)
        & (derivative >= rules.pressure_recovery_derivative)
    ] = rules.recovery_label
    return pd.Series(labels, index=current.index, dtype="string", name="regime")
