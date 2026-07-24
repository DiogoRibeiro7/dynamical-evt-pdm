"""Validated configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class EVTConfig(BaseModel):
    """Configuration for thresholding and extremal-index estimation."""

    quantile: float = Field(default=0.98, gt=0.5, lt=1.0)
    run_length: int = Field(default=10, ge=1)
    min_exceedances: int = Field(default=30, ge=3)
    estimator: str = Field(default="intervals", pattern="^(intervals|runs)$")


class SplitConfig(BaseModel):
    """Ordered temporal split proportions."""

    train: float = Field(default=0.60, gt=0.0, lt=1.0)
    validation: float = Field(default=0.20, gt=0.0, lt=1.0)
    test: float = Field(default=0.20, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> SplitConfig:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split proportions must sum to 1")
        return self


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    random_seed: int = 42
    evt: EVTConfig = Field(default_factory=EVTConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""

    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    return ExperimentConfig.model_validate(raw)
