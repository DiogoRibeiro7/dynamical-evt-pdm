import numpy as np

from dyn_evt_pdm.evt.multivariate import LaggedExtremePattern, detect_lagged_pattern


def test_detect_lagged_pattern() -> None:
    matrix = np.zeros((8, 2), dtype=bool)
    matrix[2, 0] = True
    matrix[4, 1] = True
    pattern = LaggedExtremePattern(components=(0, 1), lags=(0, 2))

    detected = detect_lagged_pattern(matrix, pattern)

    assert detected[2]
    assert int(detected.sum()) == 1
