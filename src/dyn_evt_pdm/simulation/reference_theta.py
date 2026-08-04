"""Theoretical and numerical reference extremal indices for the simulation grid.

A finite-sample estimator study needs a reference value to measure bias, RMSE and
coverage against. Closed-form values exist only for the independent systems and for
the noiseless logistic map at its periodic target. Every other configuration in the
grid -- including every noisy dynamical system -- previously carried no reference at
all, which silently made bias, RMSE and coverage undefined for those rows.

This module supplies numerical references by long-run simulation. The construction is
deliberately conservative: a numerical reference is only accepted when two structurally
different long-run estimators agree. A reference obtained from a single declustering
rule is not independent of the estimators under test, so accepting it uncritically
would flatter whichever estimator family shares that rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.evt.extremal_index import (
    disjoint_blocks_extremal_index,
    runs_extremal_index,
)
from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold
from dyn_evt_pdm.types import FloatArray

#: Reference kinds, ordered from strongest to weakest evidence.
THEORETICAL = "theoretical"
NUMERICAL = "numerical"
NOT_ESTABLISHED = "not_established"

#: Series length used for long-run numerical reference estimation.
REFERENCE_N_STEPS = 400_000

#: Run length used by the long-run runs-declustering reference.
#:
#: Runs declustering is only reference-grade at a short run length. The estimate decays
#: monotonically as the run length grows because separate clusters get chained
#: together, so a long run length reports strong clustering for any process, including
#: an independent one. Calibration against the two systems with closed-form values
#: confirms the short setting: at ``K = 1`` the long-run estimate is 0.495 for the
#: noiseless logistic map at its periodic target (theory 0.5) and 0.961 for the
#: independent light-tailed process (theory 1.0), while ``K = 50`` reports 0.303 and
#: 0.358 for the same two systems.
REFERENCE_RUN_LENGTH = 1

#: Minimum block size for the long-run disjoint-blocks reference.
MIN_REFERENCE_BLOCK_SIZE = 10


def reference_block_size(exceedance_rate: float) -> int:
    """Return a block size holding roughly one expected exceedance per block.

    The disjoint-blocks estimator reads the extremal index out of the probability that
    a block contains no exceedance. If ``block_size * exceedance_rate`` is large then
    almost every block is occupied, that probability collapses towards zero, and the
    estimate saturates at one regardless of the true clustering. Scaling the block with
    the exceedance rate keeps the statistic in its informative range.
    """

    if not 0.0 < exceedance_rate < 1.0:
        raise ValueError("exceedance_rate must lie in (0, 1)")
    return max(MIN_REFERENCE_BLOCK_SIZE, int(round(1.0 / exceedance_rate)))


#: Maximum absolute disagreement between the two long-run references.
REFERENCE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class ReferenceTheta:
    """Reference extremal index for one simulation configuration."""

    system: str
    noise_scale: float
    missing_rate: float
    threshold_quantile: float
    kind: str
    value: float | None
    runs_reference: float | None
    blocks_reference: float | None
    disagreement: float | None
    n_steps: int
    seed: int
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in {THEORETICAL, NUMERICAL, NOT_ESTABLISHED}:
            raise ValueError(f"unknown reference kind: {self.kind}")
        if self.kind == NOT_ESTABLISHED and self.value is not None:
            raise ValueError("an unestablished reference must not carry a value")
        if self.kind != NOT_ESTABLISHED and self.value is None:
            raise ValueError("an established reference must carry a value")


def theoretical_reference_theta(
    system: str, *, noise_scale: float, missing_rate: float
) -> float | None:
    """Return the closed-form extremal index when one is available.

    Independent systems have ``theta = 1`` by definition, and missingness applied
    independently of the process does not induce clustering. The noiseless logistic
    map at a periodic target has ``theta = 1 - 1 / |Lambda|`` which equals ``0.5`` for
    the configuration used by the study. Perturbing the dynamics with noise removes
    the closed form, so this function returns ``None`` rather than guessing.
    """

    independent = {"iid_light_tail", "iid_bounded_tail", "iid_pareto"}
    if system in independent:
        return 1.0
    if system in {"missing_at_random", "burst_missingness"}:
        # These wrap an independent base process with a missingness mechanism.
        return 1.0
    if system == "logistic_periodic_target" and noise_scale == 0.0:
        return 0.5
    if system == "logistic_nonperiodic_target" and noise_scale == 0.0:
        return 1.0
    return None


def _long_run_references(
    values: FloatArray, *, threshold_quantile: float
) -> tuple[float | None, float | None, str]:
    """Return runs-based and blocks-based long-run estimates for one realisation."""

    finite = values[np.isfinite(values)]
    if finite.size < 10_000:
        return None, None, "insufficient finite samples for a long-run reference"
    try:
        threshold = fit_quantile_threshold(finite, quantile=threshold_quantile)
    except ValueError as exc:
        return None, None, f"threshold fit failed: {exc}"

    exceedances = np.isfinite(values) & (values > threshold)
    n_exceedances = int(np.count_nonzero(exceedances))
    if n_exceedances < 100:
        return None, None, "fewer than 100 exceedances in the long-run realisation"

    exceedance_rate = n_exceedances / float(len(exceedances))
    block_size = reference_block_size(exceedance_rate)

    runs_value: float | None
    blocks_value: float | None
    try:
        runs_value = runs_extremal_index(exceedances, run_length=REFERENCE_RUN_LENGTH)
    except ValueError:
        runs_value = None
    try:
        blocks_value = disjoint_blocks_extremal_index(exceedances, block_size=block_size)
    except ValueError:
        blocks_value = None
    return runs_value, blocks_value, ""


def numerical_reference_theta(
    values: FloatArray,
    *,
    system: str,
    noise_scale: float,
    missing_rate: float,
    threshold_quantile: float,
    seed: int,
    tolerance: float = REFERENCE_TOLERANCE,
) -> ReferenceTheta:
    """Derive a numerical reference from one long realisation, or decline to.

    The two long-run estimates use different declustering geometry. Requiring them to
    agree guards against adopting a reference that merely restates one estimator's own
    convention. When they disagree the configuration is reported as one where the
    extremal index could not be pinned down, which is itself a reportable result.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    runs_value, blocks_value, reason = _long_run_references(
        values, threshold_quantile=threshold_quantile
    )

    def _build(
        *,
        kind: str,
        value: float | None,
        disagreement: float | None,
        reason: str,
    ) -> ReferenceTheta:
        return ReferenceTheta(
            system=system,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
            threshold_quantile=threshold_quantile,
            kind=kind,
            value=value,
            runs_reference=runs_value,
            blocks_reference=blocks_value,
            disagreement=disagreement,
            n_steps=int(len(values)),
            seed=seed,
            reason=reason,
        )

    if runs_value is None or blocks_value is None:
        return _build(
            kind=NOT_ESTABLISHED,
            value=None,
            disagreement=None,
            reason=reason or "at least one long-run reference estimator failed",
        )

    disagreement = abs(runs_value - blocks_value)
    if disagreement > tolerance:
        return _build(
            kind=NOT_ESTABLISHED,
            value=None,
            disagreement=disagreement,
            reason=(
                f"long-run references disagree by {disagreement:.3f} "
                f"(runs {runs_value:.3f} vs blocks {blocks_value:.3f}), "
                f"above the {tolerance:.3f} tolerance"
            ),
        )
    return _build(
        kind=NUMERICAL,
        value=float(0.5 * (runs_value + blocks_value)),
        disagreement=disagreement,
        reason="mean of two agreeing long-run references",
    )
