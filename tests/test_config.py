from pathlib import Path

import pytest

from dyn_evt_pdm.config import ExperimentConfig, SplitConfig, load_experiment_config


def test_load_experiment_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "random_seed: 7\nevt:\n  quantile: 0.99\nsplit:\n  train: 0.5\n  validation: 0.25\n  test: 0.25\n"
    )
    config = load_experiment_config(path)
    assert isinstance(config, ExperimentConfig)
    assert config.random_seed == 7
    assert config.evt.quantile == 0.99


def test_invalid_split_total() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train=0.5, validation=0.3, test=0.3)


def test_load_config_rejects_missing_and_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "missing.yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(TypeError):
        load_experiment_config(path)
