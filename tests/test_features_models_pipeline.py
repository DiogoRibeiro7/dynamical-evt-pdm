import numpy as np
import pandas as pd
import pytest

from dyn_evt_pdm.features.regimes import CompressorRegimeRules, infer_compressor_regime
from dyn_evt_pdm.features.state_space import StateSpaceTransformer
from dyn_evt_pdm.models.baselines import IsolationForestBaseline
from dyn_evt_pdm.pipelines.analyse import analyse_series
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine
from dyn_evt_pdm.simulation.maps import logistic_map


def test_regime_inference() -> None:
    current = pd.Series([0.0, 2.0, 2.0])
    pressure = pd.Series([1.0, 1.1, 2.0])
    rules = CompressorRegimeRules(current_on_threshold=1.0, pressure_recovery_derivative=0.5)
    assert infer_compressor_regime(current, pressure, rules).tolist() == ["off", "loaded", "recovery"]
    with pytest.raises(ValueError):
        infer_compressor_regime(current.iloc[:2], pressure, rules)


def test_state_space_transformer() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    transformer = StateSpaceTransformer(["x", "y"])
    with pytest.raises(RuntimeError):
        transformer.transform(frame)
    transformed = transformer.fit(frame).transform(frame)
    assert transformed.shape == (3, 2)
    with pytest.raises(ValueError):
        StateSpaceTransformer([])


def test_isolation_forest_baseline() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(100, 2))
    model = IsolationForestBaseline(contamination=0.05, random_state=4)
    with pytest.raises(RuntimeError):
        model.score(train)
    scores = model.fit(train).score(train[:5])
    assert scores.shape == (5,)
    with pytest.raises(ValueError):
        IsolationForestBaseline(contamination=0.0)


def test_analysis_pipeline() -> None:
    frame = simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=3))
    summary = analyse_series(
        frame,
        value_column="observable",
        regime_column="regime",
        quantile=0.98,
        run_length=5,
    )
    assert summary["n_samples"] == 1000
    assert summary["n_exceedances"] > 0
    assert summary["n_clusters"] > 0
    with pytest.raises(ValueError):
        analyse_series(frame, value_column="missing", regime_column="regime", quantile=0.98, run_length=5)


def test_logistic_map_is_bounded_and_validated() -> None:
    values = logistic_map(100, x0=0.2, burn_in=10)
    assert np.all((values >= 0.0) & (values <= 1.0))
    with pytest.raises(ValueError):
        logistic_map(0)
