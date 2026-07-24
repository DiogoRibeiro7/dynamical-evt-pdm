from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine


def test_simulation_is_reproducible() -> None:
    config = CyclicSimulationConfig(n_steps=1000, seed=7)
    first = simulate_cyclic_machine(config)
    second = simulate_cyclic_machine(config)
    assert first.equals(second)
    assert first["is_fault"].any()
    assert set(first["regime"].unique()) == {"off", "loaded", "recovery"}
    assert first["split"].tolist().count("train") == 600
    assert set(first["split"].unique()) == {"train", "validation", "test"}
