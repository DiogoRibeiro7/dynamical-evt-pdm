"""State-space construction and scaling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from dyn_evt_pdm.types import FloatArray


@dataclass(frozen=True, slots=True)
class StateSpaceConfig:
    """Columns and embedding settings for leakage-safe state construction."""

    continuous_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ()
    delay_lags: tuple[int, ...] = ()


class StateSpaceTransformer:
    """Fit a robust state-space representation without temporal leakage."""

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        config: StateSpaceConfig | None = None,
    ) -> None:
        if config is None:
            if not columns:
                raise ValueError("at least one state column is required")
            config = StateSpaceConfig(continuous_columns=tuple(columns))
        if not config.continuous_columns and not config.categorical_columns:
            raise ValueError("at least one state column is required")
        if any(lag < 1 for lag in config.delay_lags):
            raise ValueError("delay lags must be positive")
        self.config = config
        self.columns = list(config.continuous_columns)
        self.scaler = RobustScaler()
        self.categorical_levels: dict[str, list[str]] = {}
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame) -> StateSpaceTransformer:
        """Fit scaling statistics on training data only."""

        required = set(self.config.continuous_columns).union(self.config.categorical_columns)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing state columns: {missing}")
        values = frame[list(self.config.continuous_columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("state-space input contains non-finite values")
        if len(self.config.continuous_columns):
            self.scaler.fit(values)
        self.categorical_levels = {
            column: sorted(frame[column].astype("string").dropna().astype(str).unique().tolist())
            for column in self.config.categorical_columns
        }
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> FloatArray:
        """Transform data using previously fitted robust scaling."""

        if not self._is_fitted:
            raise RuntimeError("transformer must be fitted before transform")
        required = set(self.config.continuous_columns).union(self.config.categorical_columns)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing state columns: {missing}")
        parts: list[FloatArray] = []
        if self.config.continuous_columns:
            values = frame[list(self.config.continuous_columns)].to_numpy(dtype=float)
            parts.append(np.asarray(self.scaler.transform(values), dtype=np.float64))
        for column in self.config.categorical_columns:
            labels = frame[column].astype("string").astype(str)
            levels = self.categorical_levels[column]
            encoded = np.zeros((len(frame), len(levels)), dtype=np.float64)
            for level_index, level in enumerate(levels):
                encoded[:, level_index] = (labels == level).to_numpy(dtype=float)
            parts.append(encoded)
        matrix = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if self.config.delay_lags:
            matrix = add_causal_delay_embeddings(matrix, self.config.delay_lags)
        return np.asarray(matrix, dtype=np.float64)


def add_causal_delay_embeddings(states: FloatArray, lags: tuple[int, ...]) -> FloatArray:
    """Append past state vectors at positive lags without using future values."""

    matrix = np.asarray(states, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if any(lag < 1 for lag in lags):
        raise ValueError("lags must be positive")
    parts = [matrix]
    for lag in lags:
        lagged = np.empty_like(matrix)
        lagged[:lag, :] = np.nan
        lagged[lag:, :] = matrix[:-lag, :]
        parts.append(lagged)
    return np.concatenate(parts, axis=1)
