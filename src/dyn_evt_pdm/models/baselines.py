"""Required non-EVT anomaly-detection baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from dyn_evt_pdm.types import FloatArray


@dataclass(slots=True)
class IsolationForestBaseline:
    """Robust-scaled Isolation Forest with explicit training-only fitting."""

    contamination: float = 0.01
    random_state: int = 42
    scaler: RobustScaler = field(init=False, repr=False)
    model: IsolationForest = field(init=False, repr=False)
    _is_fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=300,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, features: FloatArray) -> IsolationForestBaseline:
        """Fit scaler and model on a normal training period."""

        values = np.asarray(features, dtype=float)
        if values.ndim != 2:
            raise ValueError("features must be two-dimensional")
        scaled = self.scaler.fit_transform(values)
        self.model.fit(scaled)
        self._is_fitted = True
        return self

    def score(self, features: FloatArray) -> FloatArray:
        """Return scores where larger values are more anomalous."""

        if not self._is_fitted:
            raise RuntimeError("baseline must be fitted before scoring")
        values = np.asarray(features, dtype=float)
        scaled = self.scaler.transform(values)
        return np.asarray(-self.model.score_samples(scaled), dtype=np.float64)
