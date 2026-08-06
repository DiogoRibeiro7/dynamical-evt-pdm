"""Tests for the circular-shift chance-match null.

The null asks how often a detector still matches the failure once its alignment to that
failure is randomised, while everything else about it is preserved. It replaced a
homogeneous Poisson calculation, which assumed independent arrivals at a constant rate
and so contradicted the clustering this paper is about.

The computation is exact rather than sampled, so these cases can be checked against
answers worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from dyn_evt_pdm.pipelines.industrial_results import _chance_match_probability


def _targets(total: int, start: int, end: int) -> np.ndarray:
    flags = np.zeros(total, dtype=bool)
    flags[start : end + 1] = True
    return flags


def _alarms(total: int, starts: tuple[int, ...], length: int = 1) -> np.ndarray:
    flags = np.zeros(total, dtype=bool)
    for start in starts:
        flags[start : start + length] = True
    return flags


def test_single_instant_alarm_hits_a_window_of_known_width() -> None:
    """One zero-length alarm and one window: the answer is the window's share of the series."""
    total, horizon, tolerance = 1_000, 99, 0
    probability = _chance_match_probability(
        alarm_flags=_alarms(total, (10,)),
        target_flags=_targets(total, 500, 520),
        horizon=horizon,
        tolerance_after=tolerance,
        merge_gap=0,
    )
    # Window spans [401, 500]: 100 of the 1000 shifts place the alarm inside it.
    assert probability == pytest.approx(100 / total)


def test_probability_grows_with_the_number_of_episodes() -> None:
    """A busier detector is more likely to hit the window by alignment alone."""
    total, horizon = 10_000, 99
    values = [
        _chance_match_probability(
            alarm_flags=_alarms(total, tuple(range(0, total, total // count))),
            target_flags=_targets(total, 5_000, 5_100),
            horizon=horizon,
            tolerance_after=0,
            merge_gap=0,
        )
        for count in (1, 10, 50)
    ]
    assert values[0] < values[1] < values[2]
    assert all(0.0 < value <= 1.0 for value in values)


def test_saturates_when_the_detector_is_always_on() -> None:
    """An alarm covering the series matches under every possible shift."""
    total = 500
    probability = _chance_match_probability(
        alarm_flags=np.ones(total, dtype=bool),
        target_flags=_targets(total, 200, 210),
        horizon=10,
        tolerance_after=0,
        merge_gap=0,
    )
    assert probability == 1.0


def test_overlapping_hitting_intervals_are_not_double_counted() -> None:
    """Two adjacent alarms hit overlapping shift sets; the union must not exceed one."""
    total, horizon = 1_000, 199
    probability = _chance_match_probability(
        alarm_flags=_alarms(total, (10, 12)),
        target_flags=_targets(total, 500, 520),
        horizon=horizon,
        tolerance_after=0,
        merge_gap=0,
    )
    single = _chance_match_probability(
        alarm_flags=_alarms(total, (10,)),
        target_flags=_targets(total, 500, 520),
        horizon=horizon,
        tolerance_after=0,
        merge_gap=0,
    )
    # The second alarm adds only the two shifts its offset contributes beyond the first.
    assert single < probability <= single + 2 / total


def test_is_not_computable_without_alarms_or_failures() -> None:
    total = 1_000
    assert np.isnan(
        _chance_match_probability(
            alarm_flags=np.zeros(total, dtype=bool),
            target_flags=_targets(total, 500, 520),
            horizon=10,
            tolerance_after=0,
            merge_gap=0,
        )
    )
    assert np.isnan(
        _chance_match_probability(
            alarm_flags=_alarms(total, (10,)),
            target_flags=np.zeros(total, dtype=bool),
            horizon=10,
            tolerance_after=0,
            merge_gap=0,
        )
    )


def test_clustered_and_spread_alarms_differ_at_equal_count() -> None:
    """The point of the change: the null responds to structure, not just to the rate.

    A Poisson null depends only on the episode count, so these two detectors would
    receive the same value despite occupying the series completely differently.
    """
    total, horizon = 10_000, 199
    clustered = _chance_match_probability(
        alarm_flags=_alarms(total, tuple(range(100, 140, 4))),
        target_flags=_targets(total, 5_000, 5_100),
        horizon=horizon,
        tolerance_after=0,
        merge_gap=0,
    )
    spread = _chance_match_probability(
        alarm_flags=_alarms(total, tuple(range(0, total, total // 10))),
        target_flags=_targets(total, 5_000, 5_100),
        horizon=horizon,
        tolerance_after=0,
        merge_gap=0,
    )
    assert clustered < spread
