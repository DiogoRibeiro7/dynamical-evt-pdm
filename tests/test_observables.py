import numpy as np

from dyn_evt_pdm.evt.observables import negative_log_distance


def test_negative_log_distance_increases_near_reference() -> None:
    states = np.array([[0.0, 0.0], [0.9, 0.9], [0.99, 0.99]])
    references = np.array([[1.0, 1.0]])

    observable = negative_log_distance(states, references)

    assert observable[2] > observable[1] > observable[0]
