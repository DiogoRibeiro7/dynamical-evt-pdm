import numpy as np
import pytest

from dyn_evt_pdm.evaluation.calibration import brier_score, reliability_bins
from dyn_evt_pdm.evt.hitting_times import (
    empirical_hit_probability,
    first_hitting_time,
    return_times,
)


def test_hitting_time_utilities() -> None:
    flags = np.array([False, False, True, False, True, False])
    assert first_hitting_time(flags, start=0) == 2
    assert first_hitting_time(flags, start=3) == 1
    assert first_hitting_time(flags, start=5) is None
    assert return_times(flags).tolist() == [2]
    assert empirical_hit_probability(flags, horizon=2) == pytest.approx(1.0)


def test_hitting_time_validation() -> None:
    flags = np.array([False, True])
    with pytest.raises(ValueError):
        first_hitting_time(flags, start=2)
    with pytest.raises(ValueError):
        empirical_hit_probability(flags, horizon=3)


def test_calibration_metrics() -> None:
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0, 0, 1, 1])
    assert brier_score(probabilities, outcomes) == pytest.approx(0.025)
    bins = reliability_bins(probabilities, outcomes, n_bins=2)
    assert len(bins) == 2
    assert bins[0].event_rate == 0.0
    assert bins[1].event_rate == 1.0
    with pytest.raises(ValueError):
        brier_score(np.array([1.2]), np.array([1]))
