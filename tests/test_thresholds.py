import pandas as pd

from dyn_evt_pdm.evt.thresholds import apply_regime_thresholds, fit_regime_thresholds


def test_regime_thresholds_use_different_scales() -> None:
    values = pd.Series(list(range(100)) + list(range(1000, 1100)))
    regimes = pd.Series(["a"] * 100 + ["b"] * 100, dtype="string")

    fitted = fit_regime_thresholds(values, regimes, quantile=0.9, min_regime_samples=20)
    flags = apply_regime_thresholds(values, regimes, fitted)

    assert fitted.values["b"] > fitted.values["a"]
    assert 15 <= int(flags.sum()) <= 25
