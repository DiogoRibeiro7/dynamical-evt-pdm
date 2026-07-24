import numpy as np

from dyn_evt_pdm.evt.clusters import extract_clusters


def test_extract_clusters_respects_run_length() -> None:
    flags = np.array([False, True, True, False, False, True, False, False, False, True])

    clusters = extract_clusters(flags, run_length=2)

    assert len(clusters) == 2
    assert clusters[0].exceedance_indices == (1, 2, 5)
    assert clusters[1].exceedance_indices == (9,)


def test_extract_clusters_empty() -> None:
    assert extract_clusters(np.zeros(5, dtype=bool), run_length=1) == []
