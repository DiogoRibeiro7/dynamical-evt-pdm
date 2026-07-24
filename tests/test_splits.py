from dyn_evt_pdm.data.splits import ordered_split_indices


def test_ordered_split_indices_are_disjoint() -> None:
    split = ordered_split_indices(100)
    assert split.train[-1] < split.validation[0]
    assert split.validation[-1] < split.test[0]
    assert len(split.train) + len(split.validation) + len(split.test) == 100
