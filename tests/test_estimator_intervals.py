"""Tests for the distinct estimator dispatch and bootstrap interval machinery."""

from __future__ import annotations

import numpy as np
import pytest

from dyn_evt_pdm.evt.extremal_index import (
    ESTIMATOR_NAMES,
    IntervalEstimate,
    bootstrap_extremal_index_interval,
    circular_block_bootstrap,
    default_block_size,
    default_bootstrap_block_length,
    extremal_index_point_estimate,
    reciprocal_block_cluster_extremal_index,
)
from dyn_evt_pdm.simulation.study import (
    ESTIMATOR_COLUMN_STEMS,
    SimulationStudyConfig,
    run_simulation_study,
)


def _clustered_flags(n: int = 2000, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random(n) < 0.02
    return np.asarray(base | np.roll(base, 1), dtype=bool)


def test_every_declared_estimator_is_reachable_through_dispatch() -> None:
    flags = _clustered_flags()
    block_size = default_block_size(len(flags))
    for name in ESTIMATOR_NAMES:
        estimate = extremal_index_point_estimate(name, flags, run_length=5, block_size=block_size)
        assert 0.0 < estimate <= 1.0


def test_dispatch_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="unknown estimator"):
        extremal_index_point_estimate("nope", _clustered_flags(), run_length=5, block_size=45)


def test_estimators_are_pairwise_distinct_across_replicates() -> None:
    """Regression test: block and reciprocal-mean-cluster once duplicated runs exactly.

    ``block_theta`` was computed as ``n_clusters / n_exceedances``, which is the runs
    estimator by definition, and the reciprocal mean cluster size reduces to the same
    quantity when clusters come from runs declustering. Three of the six reported
    estimators were therefore the same number.
    """

    config = SimulationStudyConfig(
        systems=("iid_light_tail", "logistic_periodic_target"),
        sample_sizes=(2000,),
        threshold_quantiles=(0.98,),
        run_lengths=(5,),
        noise_scales=(0.0,),
        missing_rates=(0.0,),
        repetitions=8,
        seed=42,
        bootstrap_resamples=0,
        n_jobs=1,
    )
    frame = run_simulation_study(config)
    stems = [ESTIMATOR_COLUMN_STEMS[name] for name in ESTIMATOR_NAMES]

    duplicates: list[tuple[str, str]] = []
    for index, left in enumerate(stems):
        for right in stems[index + 1 :]:
            left_values = frame[f"{left}_theta"].to_numpy(dtype=float)
            right_values = frame[f"{right}_theta"].to_numpy(dtype=float)
            both = np.isfinite(left_values) & np.isfinite(right_values)
            if both.sum() and np.allclose(left_values[both], right_values[both], atol=1e-12):
                duplicates.append((left, right))
    assert duplicates == []


def test_reciprocal_block_cluster_differs_from_runs_ratio() -> None:
    flags = _clustered_flags()
    reciprocal = reciprocal_block_cluster_extremal_index(flags, block_size=45)
    runs_ratio = extremal_index_point_estimate("runs", flags, run_length=5, block_size=45)
    assert reciprocal != pytest.approx(runs_ratio, abs=1e-9)


def test_circular_block_bootstrap_preserves_length_and_wraps() -> None:
    flags = _clustered_flags(n=500)
    rng = np.random.default_rng(0)
    resampled = circular_block_bootstrap(flags, block_length=8, rng=rng)
    assert resampled.shape == flags.shape
    assert resampled.dtype == np.bool_


def test_circular_block_bootstrap_rejects_oversized_block() -> None:
    with pytest.raises(ValueError, match="at least as long"):
        circular_block_bootstrap(
            _clustered_flags(n=10), block_length=50, rng=np.random.default_rng(0)
        )


def test_block_bootstrap_retains_clustering_better_than_iid_resampling() -> None:
    """A block bootstrap must not destroy the dependence the estimator measures."""

    flags = _clustered_flags()
    rng = np.random.default_rng(3)
    block_theta = [
        extremal_index_point_estimate(
            "runs",
            circular_block_bootstrap(flags, block_length=13, rng=rng),
            run_length=5,
            block_size=45,
        )
        for _ in range(30)
    ]
    iid_theta = [
        extremal_index_point_estimate("runs", rng.permutation(flags), run_length=5, block_size=45)
        for _ in range(30)
    ]
    # Permuting individual samples removes clustering and pushes theta towards one.
    assert float(np.mean(block_theta)) < float(np.mean(iid_theta))


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    flags = _clustered_flags()
    block_size = default_block_size(len(flags))
    block_length = default_bootstrap_block_length(len(flags))
    point = extremal_index_point_estimate("runs", flags, run_length=5, block_size=block_size)
    interval = bootstrap_extremal_index_interval(
        "runs",
        flags,
        run_length=5,
        block_size=block_size,
        n_resamples=200,
        block_length=block_length,
        level=0.95,
        rng=np.random.default_rng(11),
    )
    assert interval.lower <= point <= interval.upper
    assert interval.width == pytest.approx(interval.upper - interval.lower)
    assert interval.n_valid > 0


def test_wider_level_produces_wider_interval() -> None:
    flags = _clustered_flags()
    kwargs = {
        "run_length": 5,
        "block_size": 45,
        "n_resamples": 300,
        "block_length": 13,
    }
    narrow = bootstrap_extremal_index_interval(
        "runs", flags, level=0.50, rng=np.random.default_rng(5), **kwargs
    )
    wide = bootstrap_extremal_index_interval(
        "runs", flags, level=0.99, rng=np.random.default_rng(5), **kwargs
    )
    assert wide.width > narrow.width


def test_interval_estimate_reports_non_estimable_interval() -> None:
    estimate = IntervalEstimate(
        lower=float("nan"), upper=float("nan"), n_resamples=10, n_valid=0, n_clipped=0
    )
    assert np.isnan(estimate.width)
    assert estimate.covers(0.5) is False


def test_bootstrap_rejects_degenerate_arguments() -> None:
    flags = _clustered_flags(n=200)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_extremal_index_interval(
            "runs",
            flags,
            run_length=5,
            block_size=14,
            n_resamples=1,
            block_length=6,
            level=0.95,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="level"):
        bootstrap_extremal_index_interval(
            "runs",
            flags,
            run_length=5,
            block_size=14,
            n_resamples=10,
            block_length=6,
            level=1.5,
            rng=np.random.default_rng(0),
        )


def test_no_declustering_reference_is_constant_one() -> None:
    flags = _clustered_flags()
    assert extremal_index_point_estimate(
        "no_declustering", flags, run_length=5, block_size=45
    ) == pytest.approx(1.0)


def test_study_records_interval_columns_and_failure_flags() -> None:
    config = SimulationStudyConfig(
        systems=("logistic_periodic_target",),
        sample_sizes=(2000,),
        threshold_quantiles=(0.98,),
        run_lengths=(5,),
        noise_scales=(0.0,),
        missing_rates=(0.0,),
        repetitions=6,
        seed=7,
        bootstrap_resamples=50,
        n_jobs=1,
    )
    frame = run_simulation_study(config)
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        for suffix in (
            "theta",
            "bias",
            "tuning_parameter",
            "lower",
            "upper",
            "interval_width",
            "covered",
            "bootstrap_valid",
            "bootstrap_clipped",
            "failed",
            "failure_reason",
        ):
            assert f"{stem}_{suffix}" in frame.columns
    assert (frame["bootstrap_resamples"] == 50).all()
    assert (frame["block_size"] > 0).all()


def test_coverage_denominator_counts_only_estimable_replicates() -> None:
    """Failed fits must not be silently counted as covered or uncovered."""

    config = SimulationStudyConfig(
        systems=("logistic_periodic_target",),
        sample_sizes=(2000,),
        threshold_quantiles=(0.98,),
        run_lengths=(5,),
        noise_scales=(0.0,),
        missing_rates=(0.0,),
        repetitions=6,
        seed=13,
        bootstrap_resamples=50,
        n_jobs=1,
    )
    frame = run_simulation_study(config)
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        estimable = np.isfinite(frame[f"{stem}_theta"].to_numpy(dtype=float))
        widths = frame[f"{stem}_interval_width"].to_numpy(dtype=float)
        # An interval may only exist where a point estimate exists.
        assert not np.any(np.isfinite(widths) & ~estimable)


def test_tuning_parameter_matches_estimator_family() -> None:
    config = SimulationStudyConfig(
        systems=("logistic_periodic_target",),
        sample_sizes=(2000,),
        threshold_quantiles=(0.98,),
        run_lengths=(7,),
        noise_scales=(0.0,),
        missing_rates=(0.0,),
        repetitions=2,
        seed=3,
        bootstrap_resamples=0,
        n_jobs=1,
    )
    frame = run_simulation_study(config)
    block_size = float(frame["block_size"].iloc[0])
    assert frame["runs_tuning_parameter"].iloc[0] == pytest.approx(7.0)
    assert frame["k_gaps_tuning_parameter"].iloc[0] == pytest.approx(7.0)
    assert frame["block_tuning_parameter"].iloc[0] == pytest.approx(block_size)
    assert frame["reciprocal_mean_cluster_tuning_parameter"].iloc[0] == pytest.approx(block_size)
    assert np.isnan(frame["no_declustering_tuning_parameter"].iloc[0])
