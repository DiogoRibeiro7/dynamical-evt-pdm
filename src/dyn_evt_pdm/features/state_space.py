"""State-space construction and scaling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from dyn_evt_pdm.types import FloatArray


class StateSpaceTransformer:
    """Fit a robust state-space representation without temporal leakage."""

    def __init__(self, columns: list[str]) -> None:
        if not columns:
            raise ValueError("at least one state column is required")
        self.columns = list(columns)
        self.scaler = RobustScaler()
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame) -> StateSpaceTransformer:
        """Fit scaling statistics on training data only."""

        missing = sorted(set(self.columns).difference(frame.columns))
        if missing:
            raise ValueError(f"missing state columns: {missing}")
        values = frame[self.columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("state-space input contains non-finite values")
        self.scaler.fit(values)
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> FloatArray:
        """Transform data using previously fitted robust scaling."""

        if not self._is_fitted:
            raise RuntimeError("transformer must be fitted before transform")
        values = frame[self.columns].to_numpy(dtype=float)
        return np.asarray(self.scaler.transform(values), dtype=np.float64)
