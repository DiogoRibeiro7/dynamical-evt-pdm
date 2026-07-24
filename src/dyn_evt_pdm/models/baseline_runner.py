"""Fair baseline runner with standardized prediction-table output."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from dyn_evt_pdm.models.baselines import IsolationForestBaseline
from dyn_evt_pdm.types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class BaselineRunConfig:
    """Configuration shared by all baseline models in one fair run."""

    feature_columns: tuple[str, ...]
    timestamp_column: str | None = None
    regime_column: str | None = "regime"
    split_column: str = "split"
    target_column: str | None = None
    exclusion_column: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    window_size: int = 1
    threshold_quantiles: tuple[float, ...] = (0.95, 0.98, 0.99)
    engineering_z_thresholds: tuple[float, ...] = (3.0, 4.0, 5.0)
    isolation_contaminations: tuple[float, ...] = (0.01, 0.03, 0.05)
    isolation_n_estimators: int = 100
    changepoint_windows: tuple[int, ...] = (10, 25, 50)
    autoencoder_components: tuple[int, ...] = (1, 2, 4)
    fixed_run_lengths: tuple[int, ...] = (0, 5, 10)
    conformal_alpha: float = 0.1
    horizon: int = 60
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must lie in (0, 1)")
        if self.isolation_n_estimators < 1:
            raise ValueError("isolation_n_estimators must be positive")


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    """Selected validation-only settings for one baseline."""

    model_name: str
    threshold: float | FloatArray
    hyperparameters: dict[str, object]
    validation_score: float
    parameter_count: int


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Standardized predictions and per-model metadata."""

    predictions: pd.DataFrame
    metadata: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _FeatureBundle:
    matrix: FloatArray
    signal: FloatArray
    names: tuple[str, ...]
    train_mask: BoolArray
    validation_mask: BoolArray
    test_mask: BoolArray
    allowed_mask: BoolArray
    target: BoolArray | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    model_name: str
    score: FloatArray
    threshold: float | FloatArray
    hyperparameters: dict[str, object]
    parameter_count: int
    episode_run_length: int = 0


def run_baseline_experiment(
    frame: pd.DataFrame,
    config: BaselineRunConfig,
) -> BaselineRunResult:
    """Run all mandatory baselines and return one standardized prediction table."""

    bundle = _build_feature_bundle(frame, config)
    predictions: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    runners: tuple[Callable[[], _Candidate], ...] = (
        lambda: _run_engineering_threshold(bundle, config),
        lambda: _run_global_empirical_threshold(bundle, config),
        lambda: _run_pot_gpd(bundle, config),
        lambda: _run_fixed_run_declustering(bundle, config),
        lambda: _run_spot(bundle, config),
        lambda: _run_isolation_forest(bundle, config),
        lambda: _run_robust_changepoint(bundle, config),
        lambda: _run_autoencoder(bundle, config),
        lambda: _run_conformal(bundle, config),
    )

    for runner in runners:
        tracemalloc.start()
        start = time.perf_counter()
        candidate = runner()
        runtime = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        table = _standard_prediction_table(
            frame,
            config,
            bundle,
            candidate,
            runtime_seconds=runtime,
            peak_memory_bytes=int(peak),
        )
        predictions.append(table)
        metadata.append(
            {
                "model": candidate.model_name,
                "hyperparameters": candidate.hyperparameters,
                "validation_score": candidate.hyperparameters.get("validation_objective", np.nan),
                "runtime_seconds": runtime,
                "peak_memory_bytes": int(peak),
                "parameter_count": candidate.parameter_count,
            }
        )

    return BaselineRunResult(
        predictions=pd.concat(predictions, ignore_index=True),
        metadata=tuple(metadata),
    )


def _build_feature_bundle(frame: pd.DataFrame, config: BaselineRunConfig) -> _FeatureBundle:
    missing = sorted(set([*config.feature_columns, config.split_column]).difference(frame.columns))
    if config.regime_column is not None and config.regime_column not in frame:
        missing.append(config.regime_column)
    if config.timestamp_column is not None and config.timestamp_column not in frame:
        missing.append(config.timestamp_column)
    if config.target_column is not None and config.target_column not in frame:
        missing.append(config.target_column)
    if config.exclusion_column is not None and config.exclusion_column not in frame:
        missing.append(config.exclusion_column)
    if missing:
        raise ValueError(f"missing columns: {sorted(set(missing))}")

    splits = frame[config.split_column].astype("string")
    train_mask = (splits == config.train_split).fillna(False).to_numpy(dtype=bool)
    validation_mask = (splits == config.validation_split).fillna(False).to_numpy(dtype=bool)
    test_mask = (splits == config.test_split).fillna(False).to_numpy(dtype=bool)
    if not train_mask.any():
        raise ValueError("train split has no rows")
    if not validation_mask.any():
        raise ValueError("validation split has no rows")
    if config.exclusion_column is None:
        allowed_mask = np.ones(len(frame), dtype=bool)
    else:
        allowed_mask = ~frame[config.exclusion_column].astype(bool).to_numpy()

    matrix, names = _causal_feature_matrix(
        frame,
        feature_columns=config.feature_columns,
        window_size=config.window_size,
        train_mask=train_mask,
    )
    signal = _robust_max_z_signal(matrix, train_mask=train_mask)
    target = (
        frame[config.target_column].astype(bool).to_numpy()
        if config.target_column is not None
        else None
    )
    return _FeatureBundle(
        matrix=matrix,
        signal=signal,
        names=names,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        allowed_mask=allowed_mask,
        target=target,
    )


def _causal_feature_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    window_size: int,
    train_mask: BoolArray,
) -> tuple[FloatArray, tuple[str, ...]]:
    raw = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    train = raw.loc[train_mask]
    medians = train.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians)
    arrays: list[FloatArray] = []
    names: list[str] = []
    for lag in range(window_size):
        shifted = filled.shift(lag).fillna(medians)
        arrays.append(shifted.to_numpy(dtype=np.float64))
        suffix = "" if lag == 0 else f"_lag{lag}"
        names.extend(f"{column}{suffix}" for column in feature_columns)
    return np.hstack(arrays).astype(np.float64), tuple(names)


def _robust_max_z_signal(matrix: FloatArray, *, train_mask: BoolArray) -> FloatArray:
    train = matrix[train_mask]
    center = np.median(train, axis=0)
    scale = np.quantile(train, 0.75, axis=0) - np.quantile(train, 0.25, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return cast(FloatArray, np.max(np.abs((matrix - center) / scale), axis=1).astype(np.float64))


def _run_engineering_threshold(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    threshold = _select_scalar_threshold(
        bundle.signal,
        candidate_thresholds=config.engineering_z_thresholds,
        bundle=bundle,
    )
    return _Candidate(
        model_name="engineering_threshold",
        score=bundle.signal,
        threshold=threshold,
        hyperparameters={
            "z_threshold": float(threshold),
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(bundle.signal, threshold, bundle),
        },
        parameter_count=2 * bundle.matrix.shape[1],
    )


def _run_global_empirical_threshold(
    bundle: _FeatureBundle, config: BaselineRunConfig
) -> _Candidate:
    candidates = tuple(
        float(np.quantile(bundle.signal[bundle.train_mask], quantile))
        for quantile in config.threshold_quantiles
    )
    threshold = _select_scalar_threshold(
        bundle.signal, candidate_thresholds=candidates, bundle=bundle
    )
    quantile = config.threshold_quantiles[candidates.index(threshold)]
    return _Candidate(
        model_name="global_empirical_threshold",
        score=bundle.signal,
        threshold=threshold,
        hyperparameters={
            "train_quantile": quantile,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(bundle.signal, threshold, bundle),
        },
        parameter_count=1,
    )


def _run_pot_gpd(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    candidates: list[tuple[float, float, float, int]] = []
    train_signal = bundle.signal[bundle.train_mask]
    for quantile in config.threshold_quantiles:
        base_threshold = float(np.quantile(train_signal, quantile))
        excesses = train_signal[train_signal > base_threshold] - base_threshold
        if len(excesses) < 3:
            continue
        shape, _location, scale = genpareto.fit(excesses, floc=0.0)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
            continue
        alarm_threshold = _gpd_return_level(
            base_threshold=base_threshold,
            shape=float(shape),
            scale=float(scale),
            exceedance_probability=len(excesses) / len(train_signal),
            tail_probability=max(1e-4, 1.0 - quantile),
        )
        candidates.append(
            (float(alarm_threshold), float(quantile), float(shape), int(len(excesses)))
        )
    if not candidates:
        fallback = float(np.quantile(train_signal, max(config.threshold_quantiles)))
        candidates.append((fallback, max(config.threshold_quantiles), np.nan, 0))
    thresholds = tuple(item[0] for item in candidates)
    threshold = _select_scalar_threshold(
        bundle.signal, candidate_thresholds=thresholds, bundle=bundle
    )
    selected = candidates[thresholds.index(threshold)]
    return _Candidate(
        model_name="pot_gpd",
        score=bundle.signal,
        threshold=threshold,
        hyperparameters={
            "base_quantile": selected[1],
            "gpd_shape": selected[2],
            "n_excesses": selected[3],
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(bundle.signal, threshold, bundle),
        },
        parameter_count=3,
    )


def _run_fixed_run_declustering(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    base_threshold = float(np.quantile(bundle.signal[bundle.train_mask], 0.98))
    run_length = _select_run_length(
        bundle.signal > base_threshold, bundle, config.fixed_run_lengths
    )
    return _Candidate(
        model_name="fixed_run_declustering",
        score=bundle.signal,
        threshold=base_threshold,
        hyperparameters={
            "train_quantile": 0.98,
            "run_length": run_length,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(bundle.signal, base_threshold, bundle),
        },
        parameter_count=2,
        episode_run_length=run_length,
    )


def _run_spot(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    candidates: list[tuple[FloatArray, float, float, float]] = []
    for quantile in config.threshold_quantiles:
        thresholds = _spot_thresholds(
            bundle.signal,
            train_mask=bundle.train_mask,
            base_quantile=quantile,
        )
        validation_threshold = float(np.nanmedian(thresholds[bundle.validation_mask]))
        candidates.append((thresholds, validation_threshold, quantile, 1.0 - quantile))
    selected = max(
        candidates,
        key=lambda item: _validation_objective(bundle.signal, item[0], bundle),
    )
    return _Candidate(
        model_name="spot",
        score=bundle.signal,
        threshold=selected[0],
        hyperparameters={
            "base_quantile": selected[2],
            "risk_level": selected[3],
            "source": "Siffer et al. 2017, Anomaly Detection in Streams with Extreme Value Theory",
            "validation_objective": _validation_objective(bundle.signal, selected[0], bundle),
        },
        parameter_count=4,
    )


def _run_isolation_forest(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    candidates: list[tuple[float, FloatArray, float, IsolationForestBaseline]] = []
    for contamination in config.isolation_contaminations:
        model = IsolationForestBaseline(
            contamination=contamination,
            random_state=config.random_state,
            n_estimators=config.isolation_n_estimators,
        ).fit(bundle.matrix[bundle.train_mask])
        score = model.score(bundle.matrix)
        threshold = float(np.quantile(score[bundle.train_mask], 1.0 - contamination))
        candidates.append((threshold, score, contamination, model))
    selected = max(
        candidates,
        key=lambda item: _validation_objective(item[1], item[0], bundle),
    )
    threshold, score, contamination, _model = selected
    return _Candidate(
        model_name="isolation_forest",
        score=score,
        threshold=threshold,
        hyperparameters={
            "contamination": contamination,
            "n_estimators": config.isolation_n_estimators,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(score, threshold, bundle),
        },
        parameter_count=config.isolation_n_estimators * bundle.matrix.shape[1],
    )


def _run_robust_changepoint(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    candidates: list[tuple[FloatArray, float, int]] = []
    for window in config.changepoint_windows:
        score = _causal_changepoint_score(bundle.matrix, window=window)
        threshold = float(np.quantile(score[bundle.train_mask], 0.98))
        candidates.append((score, threshold, window))
    selected = max(
        candidates,
        key=lambda item: _validation_objective(item[0], item[1], bundle),
    )
    score, threshold, window = selected
    return _Candidate(
        model_name="robust_changepoint",
        score=score,
        threshold=threshold,
        hyperparameters={
            "window": window,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(score, threshold, bundle),
        },
        parameter_count=2 * bundle.matrix.shape[1],
    )


def _run_autoencoder(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(bundle.matrix[bundle.train_mask])
    all_scaled = scaler.transform(bundle.matrix)
    max_components = max(1, min(train_scaled.shape[0], train_scaled.shape[1]))
    candidates: list[tuple[FloatArray, float, int]] = []
    for requested in config.autoencoder_components:
        n_components = min(requested, max_components)
        pca = PCA(n_components=n_components, random_state=config.random_state)
        pca.fit(train_scaled)
        reconstructed = pca.inverse_transform(pca.transform(all_scaled))
        score = cast(
            FloatArray,
            np.mean((all_scaled - reconstructed) ** 2, axis=1).astype(np.float64),
        )
        threshold = float(np.quantile(score[bundle.train_mask], 0.98))
        candidates.append((score, threshold, n_components))
    selected = max(
        candidates,
        key=lambda item: _validation_objective(item[0], item[1], bundle),
    )
    score, threshold, n_components = selected
    parameter_count = bundle.matrix.shape[1] * n_components * 2 + bundle.matrix.shape[1]
    return _Candidate(
        model_name="autoencoder_reconstruction",
        score=score,
        threshold=threshold,
        hyperparameters={
            "architecture": "linear_pca_autoencoder",
            "latent_components": n_components,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(score, threshold, bundle),
        },
        parameter_count=parameter_count,
    )


def _run_conformal(bundle: _FeatureBundle, config: BaselineRunConfig) -> _Candidate:
    calibration_scores = bundle.signal[bundle.validation_mask & bundle.allowed_mask]
    if len(calibration_scores) == 0:
        calibration_scores = bundle.signal[bundle.train_mask]
    p_values = np.asarray(
        [
            (1 + np.count_nonzero(calibration_scores >= score)) / (len(calibration_scores) + 1)
            for score in bundle.signal
        ],
        dtype=np.float64,
    )
    nonconformity = cast(FloatArray, (1.0 - p_values).astype(np.float64))
    threshold = 1.0 - config.conformal_alpha
    return _Candidate(
        model_name="conformal_anomaly_score",
        score=nonconformity,
        threshold=threshold,
        hyperparameters={
            "alpha": config.conformal_alpha,
            "calibration_split": config.validation_split,
            "feature_window": config.window_size,
            "validation_objective": _validation_objective(nonconformity, threshold, bundle),
        },
        parameter_count=len(calibration_scores),
    )


def _standard_prediction_table(
    frame: pd.DataFrame,
    config: BaselineRunConfig,
    bundle: _FeatureBundle,
    candidate: _Candidate,
    *,
    runtime_seconds: float,
    peak_memory_bytes: int,
) -> pd.DataFrame:
    thresholds = _threshold_array(candidate.threshold, len(frame))
    raw_alarm = np.asarray(candidate.score >= thresholds, dtype=bool)
    alarm = raw_alarm & bundle.allowed_mask
    episode_ids = _episode_ids(alarm, run_length=candidate.episode_run_length)
    horizon_risk = _score_to_horizon_risk(candidate.score, bundle, config)
    timestamp = (
        frame[config.timestamp_column].to_numpy()
        if config.timestamp_column is not None
        else np.arange(len(frame), dtype=np.int64)
    )
    regime = (
        frame[config.regime_column].astype("string").fillna("unknown").to_numpy()
        if config.regime_column is not None
        else np.full(len(frame), "unknown", dtype=object)
    )
    provenance = (
        "train_only_fit;validation_only_selection;"
        f"train={config.train_split};validation={config.validation_split};"
        f"test={config.test_split};window={config.window_size};"
        f"excluded={config.exclusion_column or 'none'}"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "model": candidate.model_name,
            "score": candidate.score,
            "threshold": thresholds,
            "alarm_flag": alarm,
            "episode_id": episode_ids,
            "regime": regime,
            "horizon_risk": horizon_risk,
            "provenance": provenance,
            "split": frame[config.split_column].astype("string").fillna("unknown").to_numpy(),
            "runtime_seconds": runtime_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "parameter_count": candidate.parameter_count,
        }
    )


def _select_scalar_threshold(
    scores: FloatArray,
    *,
    candidate_thresholds: tuple[float, ...],
    bundle: _FeatureBundle,
) -> float:
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds must not be empty")
    finite = tuple(float(value) for value in candidate_thresholds if np.isfinite(value))
    if not finite:
        raise ValueError("candidate_thresholds contain no finite values")
    return max(finite, key=lambda threshold: _validation_objective(scores, threshold, bundle))


def _select_run_length(
    alarms: BoolArray,
    bundle: _FeatureBundle,
    candidates: tuple[int, ...],
) -> int:
    if not candidates:
        raise ValueError("fixed_run_lengths must not be empty")
    if bundle.target is None:
        return min(candidates)
    best = min(candidates)
    best_score = -np.inf
    for run_length in candidates:
        episode_flags = _episode_ids(alarms & bundle.allowed_mask, run_length=run_length) >= 0
        score = _f1_score(
            episode_flags[bundle.validation_mask], bundle.target[bundle.validation_mask]
        )
        if score > best_score:
            best = run_length
            best_score = score
    return int(best)


def _validation_objective(
    scores: FloatArray,
    threshold: float | FloatArray,
    bundle: _FeatureBundle,
) -> float:
    thresholds = _threshold_array(threshold, len(scores))
    validation = bundle.validation_mask & bundle.allowed_mask
    if not validation.any():
        return 0.0
    alarms = np.asarray(scores >= thresholds, dtype=bool)
    if bundle.target is not None:
        return _f1_score(alarms[validation], bundle.target[validation])
    alarm_rate = float(np.mean(alarms[validation]))
    return -abs(alarm_rate - 0.02)


def _f1_score(predicted: BoolArray, actual: BoolArray) -> float:
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(actual, dtype=bool)
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _threshold_array(threshold: float | FloatArray, length: int) -> FloatArray:
    if np.isscalar(threshold):
        scalar = cast(float, threshold)
        return np.full(length, float(scalar), dtype=np.float64)
    values = np.asarray(threshold, dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"threshold array must have length {length}")
    return values


def _episode_ids(alarms: BoolArray, *, run_length: int) -> IntArray:
    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    flags = np.asarray(alarms, dtype=bool)
    ids = np.full(len(flags), -1, dtype=np.int64)
    current_id = -1
    last_alarm = -run_length - 2
    for index, flag in enumerate(flags):
        if not flag:
            continue
        if current_id < 0 or index - last_alarm - 1 > run_length:
            current_id += 1
        ids[index] = current_id
        last_alarm = index
    return ids


def _score_to_horizon_risk(
    scores: FloatArray,
    bundle: _FeatureBundle,
    config: BaselineRunConfig,
) -> FloatArray:
    validation = bundle.validation_mask & bundle.allowed_mask
    validation_scores = scores[validation]
    if len(validation_scores) == 0:
        validation_scores = scores[bundle.train_mask]
    if bundle.target is None:
        ranks = np.searchsorted(np.sort(validation_scores), scores, side="right")
        return cast(FloatArray, (ranks / max(1, len(validation_scores))).astype(np.float64))
    outcomes = _future_positive(bundle.target, horizon=config.horizon)
    validation_outcomes = outcomes[validation]
    global_rate = float(np.mean(validation_outcomes)) if len(validation_outcomes) else 0.0
    order = np.argsort(validation_scores)
    sorted_scores = validation_scores[order]
    sorted_outcomes = validation_outcomes[order].astype(np.float64)
    suffix_counts = np.arange(len(sorted_scores), 0, -1, dtype=np.float64)
    suffix_events = np.cumsum(sorted_outcomes[::-1])[::-1]
    positions = np.searchsorted(sorted_scores, scores, side="left")
    risks = np.full(len(scores), global_rate, dtype=np.float64)
    valid = positions < len(sorted_scores)
    risks[valid] = suffix_events[positions[valid]] / suffix_counts[positions[valid]]
    return cast(FloatArray, np.clip(risks, 0.0, 1.0).astype(np.float64))


def _future_positive(target: BoolArray, *, horizon: int) -> BoolArray:
    flags = np.asarray(target, dtype=bool)
    result = np.zeros(len(flags), dtype=bool)
    for offset in range(horizon + 1):
        valid = len(flags) - offset
        if valid <= 0:
            break
        result[:valid] |= flags[offset : offset + valid]
    return result


def _gpd_return_level(
    *,
    base_threshold: float,
    shape: float,
    scale: float,
    exceedance_probability: float,
    tail_probability: float,
) -> float:
    ratio = max(exceedance_probability / max(tail_probability, 1e-12), 1e-12)
    if abs(shape) < 1e-8:
        return float(base_threshold + scale * np.log(ratio))
    return float(base_threshold + (scale / shape) * (ratio**shape - 1.0))


def _spot_thresholds(
    signal: FloatArray,
    *,
    train_mask: BoolArray,
    base_quantile: float,
) -> FloatArray:
    values = np.asarray(signal, dtype=np.float64)
    train = values[train_mask]
    base_threshold = float(np.quantile(train, base_quantile))
    excesses = [float(value - base_threshold) for value in train if value > base_threshold]
    thresholds = np.full(len(values), base_threshold, dtype=np.float64)
    current_threshold = base_threshold
    current_count = -1
    for index, value in enumerate(values):
        if len(excesses) >= 3 and len(excesses) != current_count:
            shape, _location, scale = genpareto.fit(np.asarray(excesses), floc=0.0)
            if np.isfinite(shape) and np.isfinite(scale) and scale > 0.0:
                current_threshold = _gpd_return_level(
                    base_threshold=base_threshold,
                    shape=float(shape),
                    scale=float(scale),
                    exceedance_probability=len(excesses) / max(1, index + 1),
                    tail_probability=max(1e-4, 1.0 - base_quantile),
                )
            current_count = len(excesses)
        thresholds[index] = current_threshold
        if value > base_threshold and value <= current_threshold:
            excesses.append(float(value - base_threshold))
    return thresholds


def _causal_changepoint_score(matrix: FloatArray, *, window: int) -> FloatArray:
    if window < 2:
        raise ValueError("changepoint window must be at least two")
    frame = pd.DataFrame(matrix)
    previous = frame.shift(window).rolling(window, min_periods=max(2, window // 2)).median()
    current = frame.rolling(window, min_periods=max(2, window // 2)).median()
    diff = (current - previous).abs()
    train_like = diff.fillna(0.0).to_numpy(dtype=np.float64)
    return cast(FloatArray, np.max(train_like, axis=1).astype(np.float64))


def baseline_metadata_to_frame(metadata: tuple[dict[str, object], ...]) -> pd.DataFrame:
    """Convert result metadata to a tabular form for reporting."""

    rows: list[dict[str, Any]] = []
    for item in metadata:
        row = {key: value for key, value in item.items() if key != "hyperparameters"}
        hyperparameters = cast(dict[str, object], item.get("hyperparameters", {}))
        for key, value in hyperparameters.items():
            row[f"hyperparameter_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
