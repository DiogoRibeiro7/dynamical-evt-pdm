import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.features.regimes import (
    ChangePointRegimeConfig,
    HiddenStateRegimeModel,
    infer_changepoint_regime,
    pool_tiny_regimes,
    regime_diagnostics,
    regime_report,
    regime_threshold_comparison,
)


def test_hidden_state_regime_model_uses_neutral_labels_and_fallback_pooling() -> None:
    features = pd.DataFrame(
        {
            "pressure": np.r_[np.zeros(30), np.ones(30) * 10.0, [100.0]],
            "current": np.r_[np.zeros(30), np.ones(30) * 5.0, [100.0]],
        }
    )

    model = HiddenStateRegimeModel(n_states=3, random_state=4, min_regime_fraction=0.05).fit(
        features
    )
    labels = model.predict(features)

    assert set(labels.dropna().unique()).issubset({"state_0", "state_1", "state_2", "unknown"})
    assert "unknown" in set(labels)
    assert not any("failure" in label for label in labels.astype(str))


def test_changepoint_regime_is_causal_and_detects_large_shift() -> None:
    frame = pd.DataFrame({"x": [0.0] * 20 + [10.0] + [0.0] * 5})

    labels = infer_changepoint_regime(
        frame,
        ChangePointRegimeConfig(value_column="x", window=5, z_threshold=2.0),
    )

    assert labels.iloc[:5].eq("stable").all()
    assert labels.iloc[20] == "high_shift"


def test_regime_diagnostics_pooling_and_threshold_comparison() -> None:
    regimes = pd.Series(["a"] * 10 + ["b"] * 2 + ["c"])
    values = pd.Series(list(range(len(regimes))), dtype=float)

    pooled = pool_tiny_regimes(regimes, min_regime_fraction=0.2)
    diagnostics = regime_diagnostics(pooled, min_regime_fraction=0.2)
    comparison = regime_threshold_comparison(values, pooled, quantile=0.8, min_regime_samples=1)
    report = regime_report(
        values, regimes, quantile=0.8, min_regime_fraction=0.2, min_regime_samples=1
    )

    assert "pooled" in set(pooled)
    assert diagnostics.occupancy["pooled"] == 3
    assert diagnostics.transition_matrix["a"]["pooled"] == 1
    assert comparison["global"] > comparison["regime_specific"]["a"]
    assert "mixture_gaps" in comparison
    assert "diagnostics" in report
    assert "thresholds" in report


def test_analyse_regimes_cli_rules_and_hidden_state(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "observable": np.linspace(0.0, 1.0, 80),
            "current": np.r_[np.zeros(20), np.ones(60) * 2.0],
            "pressure": np.r_[np.zeros(40), np.linspace(0.0, 4.0, 40)],
        }
    )
    input_path = tmp_path / "frame.csv"
    frame.to_csv(input_path, index=False)

    rules_output = tmp_path / "rules.json"
    result = CliRunner().invoke(
        app,
        [
            "analyse-regimes",
            "--input",
            str(input_path),
            "--value-column",
            "observable",
            "--output",
            str(rules_output),
            "--method",
            "rules",
        ],
    )
    assert result.exit_code == 0, result.output
    rules_payload = json.loads(rules_output.read_text())
    assert "off" in rules_payload["diagnostics"]["occupancy"]

    hidden_output = tmp_path / "hidden.json"
    result = CliRunner().invoke(
        app,
        [
            "analyse-regimes",
            "--input",
            str(input_path),
            "--value-column",
            "observable",
            "--output",
            str(hidden_output),
            "--method",
            "hidden-state",
            "--feature-columns",
            "current,pressure",
            "--n-states",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    hidden_payload = json.loads(hidden_output.read_text())
    assert hidden_payload["diagnostics"]["occupancy"]
