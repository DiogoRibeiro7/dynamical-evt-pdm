from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from dyn_evt_pdm.cli import app
from dyn_evt_pdm.simulation.plots import plot_theta_bias_by_quantile
from dyn_evt_pdm.simulation.study import (
    SimulationStudyConfig,
    run_simulation_study,
    simulation_decision_records,
    simulation_experiment_id,
    write_simulation_study,
)
from dyn_evt_pdm.simulation.systems import (
    simulate_iid_bounded_tail,
    simulate_iid_light_tail,
    simulate_iid_pareto,
    simulate_lagged_multivariate_extremes,
    simulate_logistic_nonperiodic_observable,
    simulate_logistic_target_observable,
    simulate_regime_mixture_series,
)


def test_simulation_systems_are_generator_driven_and_structured() -> None:
    rng = np.random.default_rng(123)
    light = simulate_iid_light_tail(200, rng=rng)
    bounded = simulate_iid_bounded_tail(200, rng=rng)
    iid = simulate_iid_pareto(200, rng=rng)
    nonperiodic = simulate_logistic_nonperiodic_observable(200, rng=rng)
    logistic = simulate_logistic_target_observable(200, rng=rng)
    mixture = simulate_regime_mixture_series(200, rng=rng)
    lagged = simulate_lagged_multivariate_extremes(200, rng=rng)

    assert light.true_theta == 1.0
    assert bounded.true_theta == 1.0
    assert iid.true_theta == 1.0
    assert nonperiodic.true_theta == 1.0
    assert logistic.true_theta == 0.5
    assert mixture.true_theta is None
    assert iid.frame["observable"].notna().all()
    assert {"low_scale", "high_scale"}.issuperset(set(mixture.frame["regime"]))
    assert lagged.frame["pattern_start"].any()
    assert lagged.values.shape == (200,)


def test_simulation_study_is_reproducible_and_tidy() -> None:
    config = SimulationStudyConfig(
        systems=("iid_pareto", "logistic_periodic_target"),
        sample_sizes=(500,),
        threshold_quantiles=(0.95,),
        run_lengths=(0, 2),
        noise_scales=(0.0,),
        repetitions=2,
        seed=99,
    )

    first = run_simulation_study(config)
    second = run_simulation_study(config)

    pd.testing.assert_frame_equal(first, second)
    assert simulation_experiment_id(config) == first["experiment_id"].iloc[0]
    assert set(first["system"]) == {"iid_pareto", "logistic_periodic_target"}
    assert set(first["run_length"]) == {0, 2}
    assert first["runs_theta"].between(0.0, 1.0).all()
    assert first["k_gaps_theta"].between(0.0, 1.0).all()
    assert first["failed"].eq(False).all()
    decisions = simulation_decision_records(first)
    assert {decision.estimator for decision in decisions} == {
        "runs",
        "ferro_segers_intervals",
        "k_gaps",
    }
    assert {decision.hypothesis_id for decision in decisions} == {"SIM-EI-RECOVERY"}


def test_write_simulation_study_and_plot(tmp_path: Path) -> None:
    config = SimulationStudyConfig(
        systems=("iid_pareto",),
        sample_sizes=(500,),
        threshold_quantiles=(0.9, 0.95),
        run_lengths=(0,),
        noise_scales=(0.0,),
        repetitions=2,
        seed=12,
    )
    output = tmp_path / "study.parquet"
    result = write_simulation_study(config, output)

    assert output.exists()
    assert output.with_suffix(".parquet.manifest.json").exists()
    assert output.with_suffix(".parquet.decisions.json").exists()
    assert pd.read_parquet(output).shape == result.shape

    figure = tmp_path / "theta_bias.png"
    plot_theta_bias_by_quantile(result, figure)
    assert figure.exists()
    assert figure.stat().st_size > 0


def test_simulation_study_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
simulation:
  n_steps: 2000
  noise_scale: 0.05
  seed: 7
experiment:
  thresholds: [0.95]
  run_lengths: [0, 2]
  repetitions: 2
""",
        encoding="utf-8",
    )
    output = tmp_path / "smoke.parquet"
    result = CliRunner().invoke(
        app,
        [
            "run-simulation-study",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    frame = pd.read_parquet(output)
    assert len(frame) == 16
    assert "k_gaps_theta" in frame
    assert {
        "iid_light_tail",
        "iid_pareto",
        "logistic_periodic_target",
        "cyclic_degradation",
    } == set(frame["system"])
    assert output.with_suffix(".parquet.decisions.json").exists()
