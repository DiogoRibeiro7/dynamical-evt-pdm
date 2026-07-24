import numpy as np
import pytest

from dyn_evt_pdm.evt.extremal_index import (
    intervals_extremal_index,
    mean_cluster_size_from_extremal_index,
    runs_extremal_index,
)


def test_runs_extremal_index_matches_cluster_ratio() -> None:
    flags = np.array([True, True, False, False, True, False, False, False, True])
    assert runs_extremal_index(flags, run_length=1) == pytest.approx(3.0 / 4.0)


def test_intervals_estimator_is_bounded() -> None:
    estimate = intervals_extremal_index(np.array([1, 2, 10, 11, 30]))
    assert 0.0 < estimate <= 1.0


def test_mean_cluster_size() -> None:
    assert mean_cluster_size_from_extremal_index(0.25) == pytest.approx(4.0)
