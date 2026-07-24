"""Operating-regime inference and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold, fit_regime_thresholds
from dyn_evt_pdm.types import FloatArray


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


@dataclass(frozen=True, slots=True)
class RegimeDiagnostics:
    """Diagnostics for one regime assignment."""

    occupancy: dict[str, int]
    transition_matrix: dict[str, dict[str, int]]
    duration_samples: dict[str, list[int]]
    pooled_regimes: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(slots=True)
class HiddenStateRegimeModel:
    """Unsupervised hidden-state regime model fit only on training features."""

    n_states: int = 3
    random_state: int = 42
    min_regime_fraction: float = 0.02
    fallback_label: str = "unknown"
    scaler: RobustScaler = field(init=False, repr=False)
    model: GaussianMixture = field(init=False, repr=False)
    state_labels: dict[int, str] = field(init=False, default_factory=dict)
    pooled_labels: set[str] = field(init=False, default_factory=set)
    _is_fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("n_states must be positive")
        if not 0.0 <= self.min_regime_fraction < 1.0:
            raise ValueError("min_regime_fraction must lie in [0, 1)")
        self.scaler = RobustScaler()
        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
        )

    def fit(self, features: pd.DataFrame) -> HiddenStateRegimeModel:
        """Fit unsupervised states on training features only."""

        matrix = _finite_feature_matrix(features)
        scaled = self.scaler.fit_transform(matrix)
        self.model.fit(scaled)
        raw_labels = self.model.predict(scaled)
        centers = self.scaler.inverse_transform(self.model.means_)
        order = np.argsort(np.mean(centers, axis=1))
        self.state_labels = {
            int(component): f"state_{rank}" for rank, component in enumerate(order)
        }
        counts = pd.Series(raw_labels).map(self.state_labels).value_counts(normalize=True)
        self.pooled_labels = {
            str(label)
            for label, fraction in counts.items()
            if float(fraction) < self.min_regime_fraction
        }
        self._is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign causal neutral state labels to rows using fitted parameters."""

        if not self._is_fitted:
            raise RuntimeError("hidden-state regime model must be fitted before prediction")
        matrix = _finite_feature_matrix(features)
        raw_labels = self.model.predict(self.scaler.transform(matrix))
        labels = [self.state_labels.get(int(label), self.fallback_label) for label in raw_labels]
        pooled = [
            label if label not in self.pooled_labels else self.fallback_label for label in labels
        ]
        return pd.Series(pooled, index=features.index, dtype="string", name="regime")


@dataclass(frozen=True, slots=True)
class ChangePointRegimeConfig:
    """Causal rolling-change regime settings."""

    value_column: str
    window: int = 50
    z_threshold: float = 3.0
    stable_label: str = "stable"
    high_shift_label: str = "high_shift"
    low_shift_label: str = "low_shift"


def infer_changepoint_regime(
    frame: pd.DataFrame,
    config: ChangePointRegimeConfig,
) -> pd.Series:
    """Infer causal piecewise regimes from rolling one-step-ahead deviations."""

    if config.window < 2:
        raise ValueError("window must be at least two")
    if config.z_threshold <= 0.0:
        raise ValueError("z_threshold must be positive")
    if config.value_column not in frame:
        raise ValueError(f"missing value column: {config.value_column}")
    values = pd.to_numeric(frame[config.value_column], errors="raise")
    rolling_mean = values.shift(1).rolling(config.window, min_periods=config.window).mean()
    rolling_std = values.shift(1).rolling(config.window, min_periods=config.window).std(ddof=0)
    z_scores = (values - rolling_mean) / rolling_std.replace(0.0, np.nan)
    labels = np.full(len(frame), config.stable_label, dtype=object)
    labels[z_scores >= config.z_threshold] = config.high_shift_label
    labels[z_scores <= -config.z_threshold] = config.low_shift_label
    zero_std_shift = rolling_std.eq(0.0) & rolling_mean.notna() & values.ne(rolling_mean)
    labels[zero_std_shift & values.gt(rolling_mean)] = config.high_shift_label
    labels[zero_std_shift & values.lt(rolling_mean)] = config.low_shift_label
    return pd.Series(labels, index=frame.index, dtype="string", name="regime")


def regime_diagnostics(
    regimes: pd.Series,
    *,
    min_regime_fraction: float = 0.02,
    fallback_label: str = "pooled",
) -> RegimeDiagnostics:
    """Summarize occupancy, transitions, durations and tiny-regime pooling."""

    if not 0.0 <= min_regime_fraction < 1.0:
        raise ValueError("min_regime_fraction must lie in [0, 1)")
    labels = regimes.astype("string").fillna(fallback_label)
    occupancy = {
        str(label): int(count) for label, count in labels.value_counts().sort_index().items()
    }
    total = len(labels)
    pooled = tuple(
        sorted(
            label
            for label, count in occupancy.items()
            if total and count / total < min_regime_fraction
        )
    )
    transition_matrix: dict[str, dict[str, int]] = {}
    previous_values = labels.iloc[:-1].astype(str).to_numpy()
    current_values = labels.iloc[1:].astype(str).to_numpy()
    for previous, current in zip(previous_values, current_values, strict=True):
        transition_matrix.setdefault(previous, {})
        transition_matrix[previous][current] = transition_matrix[previous].get(current, 0) + 1
    durations: dict[str, list[int]] = {}
    if len(labels):
        current_label = str(labels.iloc[0])
        current_duration = 1
        for label in labels.iloc[1:].astype(str):
            if label == current_label:
                current_duration += 1
            else:
                durations.setdefault(current_label, []).append(current_duration)
                current_label = str(label)
                current_duration = 1
        durations.setdefault(current_label, []).append(current_duration)
    warnings = tuple(f"pooled tiny regime {label}" for label in pooled)
    return RegimeDiagnostics(
        occupancy=occupancy,
        transition_matrix=transition_matrix,
        duration_samples=durations,
        pooled_regimes=pooled,
        warnings=warnings,
    )


def pool_tiny_regimes(
    regimes: pd.Series,
    *,
    min_regime_fraction: float = 0.02,
    fallback_label: str = "pooled",
) -> pd.Series:
    """Pool regimes below a minimum occupancy into an explicit fallback label."""

    diagnostics = regime_diagnostics(
        regimes,
        min_regime_fraction=min_regime_fraction,
        fallback_label=fallback_label,
    )
    labels = regimes.astype("string").fillna(fallback_label)
    return labels.mask(labels.isin(diagnostics.pooled_regimes), fallback_label).rename("regime")


def regime_threshold_comparison(
    values: pd.Series,
    regimes: pd.Series,
    *,
    quantile: float,
    shrinkage: float = 0.25,
    min_regime_samples: int = 20,
) -> dict[str, object]:
    """Compare global, regime-specific and shrinkage thresholds."""

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    numeric = pd.to_numeric(values, errors="coerce")
    global_threshold = fit_quantile_threshold(numeric.to_numpy(dtype=float), quantile=quantile)
    regime_thresholds = fit_regime_thresholds(
        numeric,
        regimes,
        quantile=quantile,
        min_regime_samples=min_regime_samples,
    )
    shrinkage_thresholds = {
        label: float((1.0 - shrinkage) * threshold + shrinkage * global_threshold)
        for label, threshold in regime_thresholds.values.items()
    }
    mixture_gaps = {
        label: float(threshold - global_threshold)
        for label, threshold in regime_thresholds.values.items()
    }
    return {
        "global": global_threshold,
        "regime_specific": regime_thresholds.values,
        "fallback": regime_thresholds.fallback,
        "hierarchical_shrinkage": shrinkage_thresholds,
        "mixture_gaps": mixture_gaps,
    }


def regime_report(
    values: pd.Series,
    regimes: pd.Series,
    *,
    quantile: float,
    min_regime_fraction: float = 0.02,
    min_regime_samples: int = 20,
) -> dict[str, object]:
    """Build a serializable regime diagnostic and threshold-comparison report."""

    pooled = pool_tiny_regimes(regimes, min_regime_fraction=min_regime_fraction)
    diagnostics = regime_diagnostics(pooled, min_regime_fraction=min_regime_fraction)
    thresholds = regime_threshold_comparison(
        values,
        pooled,
        quantile=quantile,
        min_regime_samples=min_regime_samples,
    )
    return {
        "diagnostics": asdict(diagnostics),
        "thresholds": thresholds,
    }


def _finite_feature_matrix(features: pd.DataFrame) -> FloatArray:
    if features.empty:
        raise ValueError("features must not be empty")
    matrix = features.to_numpy(dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    return matrix
