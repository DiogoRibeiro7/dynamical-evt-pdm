import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.models import baseline_runner
from dyn_evt_pdm.models.baseline_runner import (
    BaselineRunConfig,
    baseline_metadata_to_frame,
    run_baseline_experiment,
)
from dyn_evt_pdm.simulation.cyclic import CyclicSimulationConfig, simulate_cyclic_machine


def test_baseline_runner_outputs_standardized_prediction_table() -> None:
    frame = simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=21))
    frame["asset_id"] = "compressor_001"
    frame["exclude"] = False
    frame.loc[frame.index[-10:], "exclude"] = True
    config = BaselineRunConfig(
        feature_columns=("pressure", "current", "temperature"),
        dataset_id="synthetic_cyclic",
        entity_id_column="asset_id",
        timestamp_column="time",
        regime_column="regime",
        target_column="is_fault",
        exclusion_column="exclude",
        window_size=2,
        threshold_quantiles=(0.9, 0.95),
        isolation_contaminations=(0.03,),
        isolation_n_estimators=20,
        changepoint_windows=(8,),
        autoencoder_components=(1, 2),
        fixed_run_lengths=(0, 4),
        horizon=20,
        random_state=21,
    )

    result = run_baseline_experiment(frame, config)

    required = {
        "timestamp",
        "observation_index",
        "dataset_id",
        "entity_id",
        "partition",
        "model",
        "model_id",
        "raw_score",
        "score",
        "transformed_risk_score",
        "threshold",
        "alarm_flag",
        "alarm_episode_id",
        "episode_id",
        "regime",
        "prediction_horizon",
        "horizon_risk",
        "training_config_hash",
        "calibration_config_hash",
        "provenance",
        "runtime_seconds",
        "peak_memory_bytes",
        "parameter_count",
    }
    assert required.issubset(result.predictions.columns)
    assert set(result.predictions["model"]) == {
        "engineering_threshold",
        "global_empirical_threshold",
        "pot_gpd",
        "fixed_run_declustering",
        "spot",
        "isolation_forest",
        "robust_changepoint",
        "autoencoder_reconstruction",
        "conformal_anomaly_score",
    }
    assert len(result.predictions) == len(frame) * 9
    assert set(result.predictions["dataset_id"]) == {"synthetic_cyclic"}
    assert set(result.predictions["entity_id"]) == {"compressor_001"}
    assert set(result.predictions["partition"]) == {"train", "validation", "test"}
    assert result.predictions["prediction_horizon"].eq(20).all()
    assert result.predictions["model_id"].str.contains(":").all()
    excluded = result.predictions[result.predictions["timestamp"].isin(frame["time"].tail(10))]
    assert not excluded["alarm_flag"].any()
    assert result.predictions["horizon_risk"].between(0.0, 1.0).all()

    metadata = baseline_metadata_to_frame(result.metadata)
    assert metadata["runtime_seconds"].ge(0.0).all()
    assert metadata["parameter_count"].gt(0).all()
    assert metadata["model_id"].str.contains(":").all()
    assert metadata["model_card"].map(lambda value: isinstance(value, dict)).all()
    assert metadata["training_config_hash"].str.len().eq(16).all()
    assert metadata["calibration_config_hash"].str.len().eq(16).all()


def test_run_baselines_cli_writes_predictions_and_metadata(tmp_path: Path) -> None:
    frame = simulate_cyclic_machine(CyclicSimulationConfig(n_steps=1000, seed=9))
    frame["asset_id"] = "compressor_cli"
    input_path = tmp_path / "simulation.csv"
    output_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    frame.to_csv(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "run-baselines",
            "--input",
            str(input_path),
            "--feature-columns",
            "pressure,current,temperature",
            "--dataset-id",
            "synthetic_cli",
            "--entity-id-column",
            "asset_id",
            "--timestamp-column",
            "time",
            "--regime-column",
            "regime",
            "--split-column",
            "split",
            "--target-column",
            "is_fault",
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
            "--window-size",
            "2",
            "--horizon",
            "20",
            "--isolation-estimators",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    predictions = pd.read_parquet(output_path)
    assert {
        "dataset_id",
        "entity_id",
        "model",
        "model_id",
        "raw_score",
        "score",
        "threshold",
        "alarm_flag",
        "alarm_episode_id",
        "episode_id",
    }.issubset(predictions.columns)
    assert set(predictions["dataset_id"]) == {"synthetic_cli"}
    assert set(predictions["entity_id"]) == {"compressor_cli"}
    assert metadata_path.exists()
    metadata = pd.DataFrame(json.loads(metadata_path.read_text()))
    assert metadata["model_card"].map(lambda value: isinstance(value, dict)).all()


def test_horizon_risk_matches_tail_mean_definition() -> None:
    scores = np.array([0.1, 0.5, 0.9, 0.4, 0.8], dtype=np.float64)
    target = np.array([False, False, True, False, False])
    validation_mask = np.array([False, True, True, True, False])
    bundle = baseline_runner._FeatureBundle(
        matrix=np.zeros((5, 1), dtype=np.float64),
        signal=scores,
        names=("x",),
        train_mask=np.array([True, False, False, False, False]),
        validation_mask=validation_mask,
        test_mask=np.array([False, False, False, False, True]),
        allowed_mask=np.ones(5, dtype=bool),
        target=target,
    )
    config = BaselineRunConfig(feature_columns=("x",), horizon=1)

    risks = baseline_runner._score_to_horizon_risk(scores, bundle, config)
    validation_scores = scores[validation_mask]
    validation_outcomes = baseline_runner._future_positive(target, horizon=1)[validation_mask]
    expected = np.array(
        [
            validation_outcomes[validation_scores >= score].mean()
            if np.any(validation_scores >= score)
            else validation_outcomes.mean()
            for score in scores
        ]
    )

    np.testing.assert_allclose(risks, expected)
