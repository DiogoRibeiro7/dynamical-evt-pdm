"""Tests for theoretical and numerical reference extremal indices."""

from __future__ import annotations

import numpy as np
import pytest

from dyn_evt_pdm.simulation.reference_theta import (
    MIN_REFERENCE_BLOCK_SIZE,
    NOT_ESTABLISHED,
    NUMERICAL,
    THEORETICAL,
    ReferenceTheta,
    numerical_reference_theta,
    reference_block_size,
    theoretical_reference_theta,
)
from dyn_evt_pdm.simulation.study import _simulate_system, resolve_reference_theta


def _long_series(system: str, *, noise_scale: float, n_steps: int = 200_000) -> np.ndarray:
    simulated = _simulate_system(
        system,
        n_steps=n_steps,
        rng=np.random.default_rng(20260728),
        noise_scale=noise_scale,
        missing_rate=0.0,
    )
    return np.asarray(simulated.values, dtype=np.float64)


def test_theoretical_reference_covers_independent_systems() -> None:
    for system in ("iid_light_tail", "iid_bounded_tail", "iid_pareto"):
        assert theoretical_reference_theta(
            system, noise_scale=0.0, missing_rate=0.0
        ) == pytest.approx(1.0)


def test_theoretical_reference_declines_when_noise_removes_the_closed_form() -> None:
    assert theoretical_reference_theta(
        "logistic_periodic_target", noise_scale=0.0, missing_rate=0.0
    ) == pytest.approx(0.5)
    assert (
        theoretical_reference_theta("logistic_periodic_target", noise_scale=0.08, missing_rate=0.0)
        is None
    )


def test_reference_block_size_scales_inversely_with_exceedance_rate() -> None:
    """A fixed block saturates the blocks estimator at high exceedance rates."""

    assert reference_block_size(0.02) == 50
    assert reference_block_size(0.01) == 100
    assert reference_block_size(0.5) == MIN_REFERENCE_BLOCK_SIZE


def test_reference_block_size_rejects_degenerate_rates() -> None:
    for rate in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="exceedance_rate"):
            reference_block_size(rate)


def test_numerical_reference_recovers_the_known_periodic_value() -> None:
    """Calibration check: the construction must reproduce a value theory already knows."""

    values = _long_series("logistic_periodic_target", noise_scale=0.0)
    reference = numerical_reference_theta(
        values,
        system="logistic_periodic_target",
        noise_scale=0.0,
        missing_rate=0.0,
        threshold_quantile=0.98,
        seed=1,
    )
    assert reference.kind == NUMERICAL
    assert reference.value == pytest.approx(0.5, abs=0.1)


def test_numerical_reference_recovers_independence() -> None:
    values = _long_series("iid_light_tail", noise_scale=0.0)
    reference = numerical_reference_theta(
        values,
        system="iid_light_tail",
        noise_scale=0.0,
        missing_rate=0.0,
        threshold_quantile=0.98,
        seed=1,
    )
    assert reference.kind == NUMERICAL
    assert reference.value == pytest.approx(1.0, abs=0.1)


def test_numerical_reference_declines_on_short_series() -> None:
    reference = numerical_reference_theta(
        np.random.default_rng(0).normal(size=500),
        system="iid_light_tail",
        noise_scale=0.0,
        missing_rate=0.0,
        threshold_quantile=0.98,
        seed=1,
    )
    assert reference.kind == NOT_ESTABLISHED
    assert reference.value is None
    assert "insufficient" in reference.reason


def test_numerical_reference_declines_when_long_run_estimates_disagree() -> None:
    values = _long_series("logistic_periodic_target", noise_scale=0.0)
    reference = numerical_reference_theta(
        values,
        system="logistic_periodic_target",
        noise_scale=0.0,
        missing_rate=0.0,
        threshold_quantile=0.98,
        seed=1,
        tolerance=1e-6,
    )
    assert reference.kind == NOT_ESTABLISHED
    assert reference.value is None
    assert "disagree" in reference.reason


def test_reference_theta_rejects_inconsistent_kind_and_value() -> None:
    with pytest.raises(ValueError, match="must not carry a value"):
        ReferenceTheta(
            system="s",
            noise_scale=0.0,
            missing_rate=0.0,
            threshold_quantile=0.98,
            kind=NOT_ESTABLISHED,
            value=0.5,
            runs_reference=None,
            blocks_reference=None,
            disagreement=None,
            n_steps=0,
            seed=0,
            reason="",
        )
    with pytest.raises(ValueError, match="must carry a value"):
        ReferenceTheta(
            system="s",
            noise_scale=0.0,
            missing_rate=0.0,
            threshold_quantile=0.98,
            kind=NUMERICAL,
            value=None,
            runs_reference=None,
            blocks_reference=None,
            disagreement=None,
            n_steps=0,
            seed=0,
            reason="",
        )
    with pytest.raises(ValueError, match="unknown reference kind"):
        ReferenceTheta(
            system="s",
            noise_scale=0.0,
            missing_rate=0.0,
            threshold_quantile=0.98,
            kind="guessed",
            value=0.5,
            runs_reference=None,
            blocks_reference=None,
            disagreement=None,
            n_steps=0,
            seed=0,
            reason="",
        )


def test_resolve_prefers_theory_and_never_relabels_provenance() -> None:
    theoretical = resolve_reference_theta(
        "iid_light_tail",
        noise_scale=0.0,
        missing_rate=0.0,
        threshold_quantile=0.98,
        seed=5,
    )
    assert theoretical.kind == THEORETICAL
    assert theoretical.n_steps == 0  # no simulation was needed

    numerical = resolve_reference_theta(
        "smoothing",
        noise_scale=0.08,
        missing_rate=0.05,
        threshold_quantile=0.98,
        seed=5,
        n_steps=200_000,
    )
    assert numerical.kind in {NUMERICAL, NOT_ESTABLISHED}
    assert numerical.kind != THEORETICAL
