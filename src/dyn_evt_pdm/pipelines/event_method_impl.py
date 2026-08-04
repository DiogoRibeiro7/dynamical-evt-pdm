"""Score, threshold and event-policy implementations for the baseline benchmark.

Each function here realises one axis of :mod:`dyn_evt_pdm.pipelines.event_method_specs`.
Methods that share a score must genuinely differ in threshold rule or event policy;
this module is where that difference is actually computed rather than merely declared.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.extremal_index import (
    intervals_extremal_index,
    k_gaps_extremal_index,
)
from dyn_evt_pdm.types import BoolArray, FloatArray, IntArray

#: Minimum training exceedances required to fit a generalised Pareto tail.
MIN_GPD_EXCEEDANCES = 50

#: Bounds on a run length derived from an estimated extremal index.
MIN_DERIVED_RUN_LENGTH = 1
MAX_DERIVED_RUN_LENGTH = 600


def best_single_sensor_index(
    train_matrix: FloatArray,
    train_targets: BoolArray | None = None,
    *,
    threshold_quantile: float = 0.98,
) -> int:
    """Return the single most diagnostic sensor, selected on training rows only.

    When training failure labels are available the sensor is chosen by its separation
    between failing and non-failing training rows, which is what an engineer choosing
    one channel would actually optimise. Selecting instead by raw dispersion picks the
    channel that already dominates the max-abs-z score, making this baseline a duplicate
    of the global empirical threshold rather than an independent comparison.

    The fallback to dispersion applies only when no training failure is present.
    """

    clean = np.nan_to_num(np.asarray(train_matrix, dtype=float), nan=0.0)
    if clean.ndim != 2 or clean.size == 0:
        raise ValueError("training matrix must be two-dimensional and non-empty")

    # A channel is only usable if its operating quantile actually separates from its own
    # bulk. When it does not -- a channel that is zero for almost all training rows, say
    # -- the threshold lands on the constant, nearly every test sample exceeds, and alarm
    # merging turns the whole period into one always-on episode that scores a spurious
    # precision and recall of one. Counting distinct values does not catch this: a
    # two-valued channel can look varied enough while still collapsing.
    magnitude = np.abs(clean)
    upper = np.nanquantile(magnitude, threshold_quantile, axis=0)
    bulk = np.nanmedian(magnitude, axis=0)
    eligible = np.asarray(upper > bulk, dtype=bool)
    if not eligible.any():
        eligible = np.ones(clean.shape[1], dtype=bool)

    if train_targets is not None:
        labels = np.asarray(train_targets, dtype=bool)
        if labels.shape[0] == clean.shape[0] and labels.any() and (~labels).any():
            failing = np.nanmedian(np.abs(clean[labels]), axis=0)
            healthy = np.nanmedian(np.abs(clean[~labels]), axis=0)
            spread = np.nanstd(np.abs(clean[~labels]), axis=0)
            spread = np.where(spread > 0.0, spread, 1.0)
            separation = (failing - healthy) / spread
            separation = np.where(eligible, separation, -np.inf)
            if np.any(np.isfinite(separation)):
                return int(np.nanargmax(np.where(np.isfinite(separation), separation, -np.inf)))

    upper = np.nanquantile(np.abs(clean), 0.999, axis=0)
    typical = np.nanquantile(np.abs(clean), 0.5, axis=0)
    separation = upper - typical
    if not np.any(np.isfinite(separation)):
        return 0
    return int(np.nanargmax(separation))


def single_sensor_score(matrix: FloatArray, *, sensor_index: int) -> FloatArray:
    """Return the absolute standardised value of one selected sensor."""

    clean = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if clean.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not 0 <= sensor_index < clean.shape[1]:
        raise ValueError("sensor_index out of range")
    return np.asarray(np.abs(clean[:, sensor_index]), dtype=float)


def fit_isolation_forest(train_matrix: FloatArray, *, seed: int = 42) -> IsolationForest:
    """Fit an isolation forest on training rows only."""

    clean = np.nan_to_num(np.asarray(train_matrix, dtype=float), nan=0.0)
    if clean.ndim != 2 or clean.shape[0] < 2:
        raise ValueError("training matrix must have at least two rows")
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(clean)
    return model


def isolation_forest_score(model: IsolationForest, matrix: FloatArray) -> FloatArray:
    """Return an anomaly score where larger values are more anomalous."""

    clean = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    # decision_function is higher for inliers, so it is negated to match the
    # convention that every score in this benchmark increases with abnormality.
    return np.asarray(-model.decision_function(clean), dtype=float)


def gpd_return_level(
    train_scores: FloatArray,
    *,
    exceedance_quantile: float,
    target_quantile: float,
) -> float:
    """Return a generalised Pareto return level for the requested tail probability.

    A peaks-over-threshold baseline is only meaningfully different from an empirical
    quantile when a tail model is actually fitted, so this fits a generalised Pareto to
    the training excesses and inverts it. It falls back to the empirical quantile when
    there are too few exceedances to support a fit, and reports that it did so by
    returning the empirical value rather than a fitted one.
    """

    if not 0.0 < exceedance_quantile < target_quantile < 1.0:
        raise ValueError("require 0 < exceedance_quantile < target_quantile < 1")
    finite = np.asarray(train_scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite training scores")

    base = float(np.quantile(finite, exceedance_quantile))
    excesses = finite[finite > base] - base
    if excesses.size < MIN_GPD_EXCEEDANCES:
        return float(np.quantile(finite, target_quantile))

    shape, _location, scale = genpareto.fit(excesses, floc=0.0)
    exceedance_rate = excesses.size / finite.size
    tail_probability = (1.0 - target_quantile) / exceedance_rate
    if not 0.0 < tail_probability < 1.0:
        return float(np.quantile(finite, target_quantile))
    return float(base + genpareto.ppf(1.0 - tail_probability, shape, loc=0.0, scale=scale))


def regime_conditioned_thresholds(
    train_scores: FloatArray,
    train_regimes: IntArray,
    *,
    quantile: float,
) -> dict[int, float]:
    """Return one training threshold per operating regime."""

    scores = np.asarray(train_scores, dtype=float)
    regimes = np.asarray(train_regimes)
    if scores.shape != regimes.shape:
        raise ValueError("scores and regimes must align")
    thresholds: dict[int, float] = {}
    for regime in np.unique(regimes):
        mask = regimes == regime
        subset = scores[mask]
        subset = subset[np.isfinite(subset)]
        if subset.size:
            thresholds[int(regime)] = float(np.quantile(subset, quantile))
    if not thresholds:
        raise ValueError("no regime carried finite training scores")
    return thresholds


def apply_regime_thresholds(
    scores: FloatArray,
    regimes: IntArray,
    thresholds: dict[int, float],
    *,
    fallback: float,
) -> BoolArray:
    """Flag exceedances against the threshold belonging to each sample's regime."""

    values = np.asarray(scores, dtype=float)
    labels = np.asarray(regimes)
    limits = np.full(values.shape, fallback, dtype=float)
    for regime, threshold in thresholds.items():
        limits[labels == regime] = threshold
    return np.asarray(values > limits, dtype=bool)


def quantile_regimes(values: FloatArray, *, n_regimes: int = 3) -> IntArray:
    """Derive coarse operating regimes as quantile bins of a load proxy.

    The prepared compressor tables carry no regime label, so the regime-conditioned
    baseline needs a declared derivation rather than an assumed column. Bin edges are
    fitted on training data by the caller and reused unchanged on the test split.
    """

    if n_regimes < 2:
        raise ValueError("n_regimes must be at least two")
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    edges = np.quantile(clean, np.linspace(0.0, 1.0, n_regimes + 1)[1:-1])
    return np.asarray(np.digitize(clean, edges), dtype=int)


def derived_run_length(exceedances: BoolArray, *, estimator: str) -> int:
    """Return a declustering run length implied by an estimated extremal index.

    A smaller extremal index means longer clusters, so the run length needed to merge a
    cluster scales like ``1 / theta``. This is what makes the Ferro-Segers and K-gaps
    event policies genuinely different from a fixed run length: the parameter is
    estimated from the training exceedance pattern instead of declared.
    """

    # Validated before the try block: an unrecognised estimator is a programming error
    # and must not be absorbed by the estimator-failure fallback below.
    if estimator not in {"ferro_segers", "k_gaps"}:
        raise ValueError(f"unknown run-length estimator: {estimator}")

    flags = np.asarray(exceedances, dtype=bool)
    indices = np.flatnonzero(flags).astype(np.int64)
    if indices.size < 2:
        return MIN_DERIVED_RUN_LENGTH
    try:
        if estimator == "ferro_segers":
            theta = intervals_extremal_index(indices)
        else:
            theta = k_gaps_extremal_index(indices, run_length=1, n_samples=int(len(flags)))
    except ValueError:
        return MIN_DERIVED_RUN_LENGTH
    if not np.isfinite(theta) or theta <= 0.0:
        return MIN_DERIVED_RUN_LENGTH
    return int(np.clip(round(1.0 / theta), MIN_DERIVED_RUN_LENGTH, MAX_DERIVED_RUN_LENGTH))


def declustered_flags(exceedances: BoolArray, *, run_length: int) -> BoolArray:
    """Collapse each extreme cluster to its first sample.

    Declustering changes the event conversion rather than the score: a cluster of
    exceedances becomes one alarm onset instead of many, which is precisely the
    duplicate-alarm reduction the declustering baselines exist to test.
    """

    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    flags = np.asarray(exceedances, dtype=bool)
    if not flags.any():
        return np.zeros_like(flags, dtype=bool)
    clusters = extract_clusters(flags, run_length=run_length)
    reduced = np.zeros_like(flags, dtype=bool)
    for cluster in clusters:
        reduced[cluster.interval.start] = True
    return reduced
