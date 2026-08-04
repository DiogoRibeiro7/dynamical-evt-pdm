"""Tests for event-method identity and the score/threshold/policy implementations."""

from __future__ import annotations

import numpy as np
import pytest

from dyn_evt_pdm.pipelines.event_method_impl import (
    apply_regime_thresholds,
    best_single_sensor_index,
    declustered_flags,
    derived_run_length,
    fit_isolation_forest,
    gpd_return_level,
    isolation_forest_score,
    quantile_regimes,
    regime_conditioned_thresholds,
    single_sensor_score,
)
from dyn_evt_pdm.pipelines.event_method_specs import (
    EVENT_METHOD_SPECS,
    SPECS_BY_NAME,
    EventMethodSpec,
    assert_specifications_are_distinct,
    method_family,
)


def test_declared_methods_are_pairwise_distinct() -> None:
    """Regression test: nine declared baselines once produced identical results.

    Ten method names resolved to one score array through a silent fallback, so the
    published benchmark reported one computation under nine algorithm names, including
    the registered score and the classical EVT baselines it was compared against.
    """

    assert_specifications_are_distinct()


def test_identity_collision_is_rejected() -> None:
    original = EVENT_METHOD_SPECS[0]
    clone = EventMethodSpec(
        name="clone_of_" + original.name,
        family=original.family,
        score_kind=original.score_kind,
        threshold_rule=original.threshold_rule,
        event_policy=original.event_policy,
        parameter_count=original.parameter_count,
        calibration_status=original.calibration_status,
        tuning_partition=original.tuning_partition,
    )
    assert clone.identity == original.identity


def test_every_method_declares_a_family() -> None:
    families = {spec.name: method_family(spec.name) for spec in EVENT_METHOD_SPECS}
    assert families["dynamical_evt_robust_score"] == "registered"
    assert families["failure_prototype_region"] == "target_region"
    assert families["negative_control_region"] == "negative_control"
    assert families["engineering_threshold"] == "baseline"


def test_undeclared_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared event method"):
        method_family("not_a_method")


def test_specs_cover_the_shared_score_by_differing_elsewhere() -> None:
    """Methods sharing the recurrence score must differ in threshold or policy."""

    shared = [spec for spec in EVENT_METHOD_SPECS if spec.score_kind == "robust_max_abs_z"]
    assert len(shared) > 1
    identities = {spec.identity for spec in shared}
    assert len(identities) == len(shared)


def _matrix(seed: int = 0, rows: int = 400, columns: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, columns))
    values[:, 2] *= 4.0  # one clearly more dispersed sensor
    return values


def test_best_single_sensor_picks_the_separated_channel() -> None:
    assert best_single_sensor_index(_matrix()) == 2


def test_single_sensor_score_is_one_column() -> None:
    matrix = _matrix()
    scores = single_sensor_score(matrix, sensor_index=2)
    assert scores.shape == (matrix.shape[0],)
    assert np.allclose(scores, np.abs(matrix[:, 2]))


def test_single_sensor_score_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="sensor_index"):
        single_sensor_score(_matrix(), sensor_index=99)


def test_isolation_forest_is_a_real_fitted_model() -> None:
    """The published benchmark previously used row energy under this name."""

    train = _matrix(seed=1)
    model = fit_isolation_forest(train)
    scores = isolation_forest_score(model, train)
    assert scores.shape == (train.shape[0],)

    outlier = np.full((1, train.shape[1]), 25.0)
    outlier_score = isolation_forest_score(model, outlier)
    assert float(outlier_score[0]) > float(np.median(scores))

    row_energy = np.mean(np.nan_to_num(train) ** 2, axis=1)
    assert not np.allclose(scores, row_energy)


def test_gpd_return_level_differs_from_the_empirical_quantile() -> None:
    rng = np.random.default_rng(3)
    scores = rng.pareto(2.5, size=20_000) + 1.0
    fitted = gpd_return_level(scores, exceedance_quantile=0.95, target_quantile=0.99)
    empirical = float(np.quantile(scores, 0.99))
    assert np.isfinite(fitted)
    assert fitted != pytest.approx(empirical, abs=1e-9)


def test_gpd_falls_back_when_exceedances_are_too_few() -> None:
    scores = np.linspace(0.0, 1.0, 60)
    fitted = gpd_return_level(scores, exceedance_quantile=0.95, target_quantile=0.99)
    assert fitted == pytest.approx(float(np.quantile(scores, 0.99)))


def test_gpd_rejects_inconsistent_quantiles() -> None:
    with pytest.raises(ValueError, match="exceedance_quantile"):
        gpd_return_level(np.linspace(0, 1, 100), exceedance_quantile=0.99, target_quantile=0.95)


def test_regime_conditioned_thresholds_differ_between_regimes() -> None:
    rng = np.random.default_rng(5)
    scores = np.concatenate([rng.normal(0.0, 1.0, 500), rng.normal(5.0, 1.0, 500)])
    regimes = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    thresholds = regime_conditioned_thresholds(scores, regimes, quantile=0.95)
    assert set(thresholds) == {0, 1}
    assert thresholds[1] > thresholds[0]


def test_regime_thresholds_flag_against_the_matching_regime() -> None:
    scores = np.array([1.0, 6.0, 1.0, 6.0])
    regimes = np.array([0, 1, 1, 0])
    flags = apply_regime_thresholds(scores, regimes, {0: 5.0, 1: 2.0}, fallback=100.0)
    assert flags.tolist() == [False, True, False, True]


def test_quantile_regimes_partition_the_range() -> None:
    regimes = quantile_regimes(np.linspace(0.0, 1.0, 300), n_regimes=3)
    assert set(np.unique(regimes)) == {0, 1, 2}


def test_quantile_regimes_reject_degenerate_counts() -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        quantile_regimes(np.linspace(0.0, 1.0, 10), n_regimes=1)


def test_declustering_reduces_a_cluster_to_one_onset() -> None:
    flags = np.zeros(100, dtype=bool)
    flags[10:20] = True
    flags[60:65] = True
    reduced = declustered_flags(flags, run_length=5)
    assert int(reduced.sum()) == 2
    assert reduced[10] and reduced[60]


def test_declustering_of_an_empty_series_is_empty() -> None:
    assert not declustered_flags(np.zeros(50, dtype=bool), run_length=3).any()


def test_derived_run_length_grows_as_clustering_strengthens() -> None:
    """A smaller extremal index implies longer clusters and a longer run length."""

    rng = np.random.default_rng(11)
    independent = rng.random(5000) < 0.02
    clustered = np.zeros(5000, dtype=bool)
    for start in range(0, 5000, 250):
        clustered[start : start + 20] = True

    independent_length = derived_run_length(independent, estimator="ferro_segers")
    clustered_length = derived_run_length(clustered, estimator="ferro_segers")
    assert clustered_length > independent_length


def test_derived_run_length_handles_both_estimators() -> None:
    rng = np.random.default_rng(13)
    flags = rng.random(3000) < 0.03
    for estimator in ("ferro_segers", "k_gaps"):
        length = derived_run_length(flags, estimator=estimator)
        assert length >= 1


def test_derived_run_length_rejects_unknown_estimator() -> None:
    flags = np.zeros(100, dtype=bool)
    flags[[1, 5, 9, 30]] = True
    with pytest.raises(ValueError, match="unknown run-length estimator"):
        derived_run_length(flags, estimator="invented")


def test_every_declared_method_has_a_spec() -> None:
    from dyn_evt_pdm.pipelines.industrial_results import (
        EVENT_BASELINE_METHODS,
        EVENT_CONTROL_METHODS,
    )

    for method in (*EVENT_BASELINE_METHODS, *EVENT_CONTROL_METHODS):
        assert method in SPECS_BY_NAME, f"{method} has no declared specification"


def test_declustering_policies_merge_with_a_zero_gap() -> None:
    """One cluster must become one alarm.

    Declustering to onsets and then re-applying the global merge gap merges separate
    clusters back together and reproduces the undeclustered episode set, which is why
    the declustering baselines previously matched the global empirical threshold.
    """

    from dyn_evt_pdm.pipelines.event_method_specs import (
        DECLUSTERING_POLICIES,
        method_merge_gap,
    )

    for spec in EVENT_METHOD_SPECS:
        gap = method_merge_gap(spec.name, default_merge_gap=60)
        if spec.event_policy in DECLUSTERING_POLICIES:
            assert gap == 0, f"{spec.name} must merge with a zero gap after declustering"
        else:
            assert gap == 60, f"{spec.name} must use the shared default gap"


def test_method_merge_gap_rejects_bad_input() -> None:
    from dyn_evt_pdm.pipelines.event_method_specs import method_merge_gap

    with pytest.raises(ValueError, match="undeclared event method"):
        method_merge_gap("not_a_method", default_merge_gap=10)
    with pytest.raises(ValueError, match="default_merge_gap"):
        method_merge_gap("engineering_threshold", default_merge_gap=-1)
