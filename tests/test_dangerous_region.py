import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.evt.dangerous_region import (
    StateProvenance,
    dangerous_region_from_constraints,
    dangerous_region_from_density_level,
    dangerous_region_from_training_failures,
    estimate_horizon_risk,
    horizon_entry_outcomes,
    nearest_medoid_references,
    score_dangerous_region,
)
from dyn_evt_pdm.evt.observables import distance_observable_report, nearest_reference_distance
from dyn_evt_pdm.features.state_space import (
    StateSpaceConfig,
    StateSpaceTransformer,
    add_causal_delay_embeddings,
)


def test_state_space_handles_categorical_controls_and_causal_lags() -> None:
    frame = pd.DataFrame(
        {
            "pressure": [1.0, 2.0, 3.0],
            "current": [0.5, 0.7, 0.9],
            "mode": ["off", "loaded", "off"],
        }
    )
    transformer = StateSpaceTransformer(
        config=StateSpaceConfig(
            continuous_columns=("pressure", "current"),
            categorical_columns=("mode",),
            delay_lags=(1,),
        )
    ).fit(frame.iloc[:2])

    states = transformer.transform(frame)

    assert states.shape[0] == 3
    assert states.shape[1] == 8
    assert np.isnan(states[0, 4:]).all()
    assert np.isfinite(states[1:, :]).all()
    with pytest.raises(ValueError):
        add_causal_delay_embeddings(np.array([1.0]), (1,))


def test_dangerous_region_rejects_held_out_failure_provenance() -> None:
    states = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    provenance = (
        StateProvenance(index=0, source="fixture", failure_id="f_train", split="train"),
        StateProvenance(index=1, source="fixture", failure_id="f_test", split="test"),
        StateProvenance(index=2, source="fixture", failure_id=None, split="train"),
    )

    region = dangerous_region_from_training_failures(
        states,
        provenance,
        allowed_failure_ids={"f_train"},
    )

    assert region.references.shape == (1, 2)
    assert region.provenance[0].failure_id == "f_train"
    with pytest.raises(ValueError):
        dangerous_region_from_training_failures(states, provenance, allowed_failure_ids={"f_test"})


def test_region_definitions_scoring_and_horizon_risk() -> None:
    states = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [1.1, 1.1]])
    provenance = tuple(StateProvenance(index=i, source="fixture", split="train") for i in range(4))
    constraint_region = dangerous_region_from_constraints(
        states,
        np.array([False, True, False, True]),
        provenance,
    )
    density_region = dangerous_region_from_density_level(states, quantile=0.75)

    scores = score_dangerous_region(states, constraint_region)
    risk = estimate_horizon_risk(
        scores["distance_to_dangerous_region"].to_numpy(),
        hit_radius=0.15,
        horizons=(1, 2),
    )
    report = distance_observable_report(states, constraint_region.references, chunk_size=2)

    assert density_region.references.shape[1] == 2
    assert scores["exact_reference_duplicate"].sum() == 2
    assert risk.probabilities[1] >= 0.0
    assert risk.brier_scores[2] >= 0.0
    assert report["exact_duplicate_count"] == 2
    assert nearest_reference_distance(states, constraint_region.references, chunk_size=2).shape == (
        4,
    )
    assert horizon_entry_outcomes(np.array([False, True, False]), horizon=1).tolist() == [
        True,
        False,
        False,
    ]


def test_nearest_medoid_references_is_deterministic() -> None:
    states = np.array([[0.0], [1.0], [10.0], [11.0]])

    first = nearest_medoid_references(states, n_medoids=2)
    second = nearest_medoid_references(states, n_medoids=2)

    assert np.array_equal(first, second)
    assert first.shape == (2, 1)


def test_analyse_dangerous_region_cli(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 1.1],
            "y": [0.0, 1.0, np.nan, 1.1],
            "target": [False, True, False, False],
            "split": ["train", "train", "test", "test"],
            "failure_id": [None, "f1", "f2", "f2"],
        }
    )
    input_path = tmp_path / "states.csv"
    output = tmp_path / "scores.parquet"
    report = tmp_path / "report.json"
    frame.to_csv(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "analyse-dangerous-region",
            "--input",
            str(input_path),
            "--state-columns",
            "x,y",
            "--target-column",
            "target",
            "--split-column",
            "split",
            "--failure-id-column",
            "failure_id",
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--horizons",
            "1,2",
        ],
    )

    assert result.exit_code == 0, result.output
    scores = pd.read_parquet(output)
    payload = json.loads(report.read_text())
    assert scores.shape[0] == len(frame)
    assert payload["n_references"] == 1
    assert payload["imputed_state_values"] == 1
    assert payload["metadata"]["allowed_failure_ids"] == ["f1"]
