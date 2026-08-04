"""Tests for the measures that separate a score failure from a threshold failure.

The reclassification of the MetroPT result rests on these two functions, so they are
tested directly against cases whose answer is known analytically rather than only
through the artifacts they are applied to.
"""

from __future__ import annotations

import numpy as np
import pytest

from dyn_evt_pdm.pipelines.industrial_results import (
    IndustrialResultsConfig,
    _chance_match_probability,
    _method_run_length,
    _warning_window_separation,
)


def _flags(total: int, start: int, end: int) -> np.ndarray:
    flags = np.zeros(total, dtype=bool)
    flags[start : end + 1] = True
    return flags


def test_separation_is_one_when_the_window_holds_the_highest_scores() -> None:
    total, horizon = 1_000, 100
    targets = _flags(total, 500, 600)
    scores = np.zeros(total, dtype=float)
    # Every sample in [400, 500] outranks every other sample.
    scores[400:501] = 10.0
    assert _warning_window_separation(scores, targets, horizon=horizon) == pytest.approx(1.0)


def test_separation_is_zero_when_the_window_holds_the_lowest_scores() -> None:
    total, horizon = 1_000, 100
    targets = _flags(total, 500, 600)
    scores = np.ones(total, dtype=float)
    scores[400:501] = 0.0
    assert _warning_window_separation(scores, targets, horizon=horizon) == pytest.approx(0.0)


def test_separation_is_one_half_for_a_constant_score() -> None:
    """A score carrying no information must land on chance, not on a boundary."""
    total, horizon = 1_000, 100
    targets = _flags(total, 500, 600)
    scores = np.full(total, 3.0, dtype=float)
    assert _warning_window_separation(scores, targets, horizon=horizon) == pytest.approx(0.5)


def test_separation_uses_the_horizon_before_onset_not_the_failure_interval() -> None:
    """Scoring the failure itself would reward reacting after the fact."""
    total, horizon = 1_000, 100
    targets = _flags(total, 500, 900)
    scores = np.zeros(total, dtype=float)
    # High scores only inside the failure interval, none before onset.
    scores[600:900] = 10.0
    separation = _warning_window_separation(scores, targets, horizon=horizon)
    assert separation < 0.5


def test_separation_is_not_computable_without_a_labelled_failure() -> None:
    scores = np.arange(100, dtype=float)
    assert np.isnan(_warning_window_separation(scores, np.zeros(100, dtype=bool), horizon=10))
    assert np.isnan(_warning_window_separation(np.array([]), np.array([], dtype=bool), horizon=10))


def test_separation_ignores_non_finite_scores() -> None:
    total, horizon = 200, 20
    targets = _flags(total, 100, 150)
    scores = np.zeros(total, dtype=float)
    scores[80:101] = 10.0
    scores[0:10] = np.nan
    assert _warning_window_separation(scores, targets, horizon=horizon) == pytest.approx(1.0)


def test_chance_match_probability_rises_with_the_alarm_rate() -> None:
    """A busier detector matches by coincidence more often."""
    targets = _flags(100_000, 50_000, 50_100)
    sparse = _chance_match_probability(
        alarm_count=1, target_flags=targets, horizon=3_600, tolerance_after=300
    )
    busy = _chance_match_probability(
        alarm_count=2_631, target_flags=targets, horizon=3_600, tolerance_after=300
    )
    # The busy value saturates to 1.0 in float64, so it is bounded from below only.
    assert 0.0 < sparse < busy
    assert sparse < 0.05
    assert busy > 0.99


def test_chance_match_probability_is_not_computable_without_alarms_or_failures() -> None:
    targets = _flags(1_000, 500, 600)
    assert np.isnan(
        _chance_match_probability(
            alarm_count=0, target_flags=targets, horizon=100, tolerance_after=10
        )
    )
    assert np.isnan(
        _chance_match_probability(
            alarm_count=5,
            target_flags=np.zeros(1_000, dtype=bool),
            horizon=100,
            tolerance_after=10,
        )
    )


def test_run_length_reflects_the_method_event_policy() -> None:
    """Estimator-driven policies derive the run length rather than declaring it."""
    config = IndustrialResultsConfig()
    exceedances = np.zeros(10_000, dtype=bool)
    exceedances[::50] = True

    assert _method_run_length("fixed_run_declustering", exceedances, config) == config.merge_gap
    # A merge-gap policy declusters nothing, so it reports no run length.
    assert _method_run_length("global_empirical_threshold", exceedances, config) == 0
    for method in ("ferro_segers_event_policy", "k_gaps_event_policy"):
        assert _method_run_length(method, exceedances, config) >= 1
    assert _method_run_length("dynamical_evt_robust_score", exceedances, config) >= 1
