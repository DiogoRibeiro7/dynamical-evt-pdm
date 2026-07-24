"""Training-only global and regime-conditioned threshold estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Quantile thresholds indexed by operating regime."""

    quantile: float
    values: dict[str, float]
    fallback: float


def fit_quantile_threshold(values: np.ndarray, *, quantile: float) -> float:
    """Fit a finite empirical quantile threshold."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("no finite values are available")
    return float(np.quantile(array, quantile))


def fit_regime_thresholds(
    values: pd.Series,
    regimes: pd.Series,
    *,
    quantile: float,
    min_regime_samples: int = 100,
) -> RegimeThresholds:
    """Fit one threshold per sufficiently represented training regime."""

    if len(values) != len(regimes):
        raise ValueError("values and regimes must have equal length")
    if min_regime_samples < 1:
        raise ValueError("min_regime_samples must be positive")

    numeric = pd.to_numeric(values, errors="coerce")
    labels = regimes.astype("string")
    fallback = fit_quantile_threshold(numeric.to_numpy(dtype=float), quantile=quantile)
    fitted: dict[str, float] = {}
    for regime, group in numeric.groupby(labels, dropna=False):
        clean = group.dropna().to_numpy(dtype=float)
        if len(clean) >= min_regime_samples:
            fitted[str(regime)] = fit_quantile_threshold(clean, quantile=quantile)
    return RegimeThresholds(quantile=quantile, values=fitted, fallback=fallback)


def apply_regime_thresholds(
    values: pd.Series,
    regimes: pd.Series,
    thresholds: RegimeThresholds,
) -> pd.Series:
    """Return exceedance indicators using each row's operating regime."""

    if len(values) != len(regimes):
        raise ValueError("values and regimes must have equal length")
    numeric = pd.to_numeric(values, errors="coerce")
    row_thresholds = regimes.astype("string").map(thresholds.values).fillna(thresholds.fallback)
    return (numeric > row_thresholds).fillna(False).astype(bool)
