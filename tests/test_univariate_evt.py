import numpy as np
import pytest

from dyn_evt_pdm.evt.extremal_index import disjoint_blocks_extremal_index
from dyn_evt_pdm.evt.univariate import (
    bootstrap_runs_extremal_index,
    fit_univariate_evt,
    threshold_diagnostics,
    threshold_run_stability,
)


def test_fit_univariate_evt_returns_structured_tail_result() -> None:
    rng = np.random.default_rng(123)
    values = rng.pareto(2.0, size=2000) + 1.0

    result = fit_univariate_evt(
        values,
        threshold_quantile=0.95,
        run_length=1,
        min_exceedances=30,
        bootstrap_repetitions=20,
        random_seed=7,
    )

    assert result.n_samples == 2000
    assert result.n_exceedances == 100
    assert result.n_clusters > 0
    assert result.gpd is not None
    assert result.gpd.n_excesses == result.n_exceedances
    assert result.gpd.scale > 0.0
    assert result.extremal_index.runs is not None
    assert result.extremal_index.intervals is not None
    assert result.extremal_index.disjoint_blocks is not None
    assert result.extremal_index.k_gaps is not None
    assert {item.estimator_name for item in result.estimator_diagnostics} == {
        "runs",
        "ferro_segers_intervals",
        "disjoint_blocks",
        "k_gaps",
    }
    assert {item.status for item in result.estimator_diagnostics} == {"succeeded"}
    assert result.runs_theta_ci is not None
    assert 0.0 < result.runs_theta_ci.lower <= result.runs_theta_ci.upper <= 1.0
    assert result.to_dict()["n_exceedances"] == 100


def test_univariate_evt_warns_when_too_few_exceedances() -> None:
    values = np.arange(100, dtype=float)

    result = fit_univariate_evt(values, threshold_quantile=0.99, run_length=0, min_exceedances=10)

    assert result.n_exceedances == 1
    assert result.gpd is None
    assert result.warnings
    assert any(item.status == "failed" for item in result.estimator_diagnostics)


def test_threshold_diagnostics_and_stability_grid() -> None:
    values = np.linspace(0.0, 100.0, 500)
    diagnostics = threshold_diagnostics(values, diagnostic_quantiles=(0.8, 0.9))
    stability = threshold_run_stability(
        values,
        threshold_quantiles=(0.8, 0.9),
        run_lengths=(0, 2),
        min_exceedances=3,
    )

    assert [diagnostic.exceedance_count for diagnostic in diagnostics] == [100, 50]
    assert stability.shape[0] == 4
    assert {"runs_theta", "intervals_theta", "blocks_theta", "k_gaps_theta"}.issubset(
        stability.columns
    )


def test_extremal_index_estimators_are_bounded_and_affine_invariant() -> None:
    rng = np.random.default_rng(55)
    values = rng.normal(size=1000)
    shifted = 4.0 + 3.0 * values

    first = fit_univariate_evt(values, threshold_quantile=0.95, run_length=2)
    second = fit_univariate_evt(shifted, threshold_quantile=0.95, run_length=2)

    assert first.extremal_index.runs == pytest.approx(second.extremal_index.runs)
    assert first.extremal_index.intervals == pytest.approx(second.extremal_index.intervals)
    assert first.extremal_index.disjoint_blocks == pytest.approx(
        second.extremal_index.disjoint_blocks
    )
    assert first.extremal_index.k_gaps == pytest.approx(second.extremal_index.k_gaps)


def test_disjoint_blocks_and_bootstrap_validation() -> None:
    flags = np.array([True, False, False, True, False, True, False, False])

    estimate = disjoint_blocks_extremal_index(flags, block_size=2)
    interval = bootstrap_runs_extremal_index(
        flags,
        run_length=1,
        repetitions=10,
        random_seed=12,
        block_length=2,
    )

    assert 0.0 < estimate <= 1.0
    assert 0.0 < interval.lower <= interval.upper <= 1.0
    with pytest.raises(ValueError):
        disjoint_blocks_extremal_index(flags, block_size=0)
