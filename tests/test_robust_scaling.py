"""Guards on the robust scaling denominator.

A near-constant channel divided by its own interquartile range produces z-scores in the
thousands and takes over a max-abs aggregation. MetroPT's ``pressure_tp2`` does exactly
this: its central half spans 0.004 against a 1st-to-99th-percentile range near 10, and
the resulting score threshold was 2,555 while the same score's median was 1.5. The
warning-window separation on that dataset moved from 0.45 to 0.78 once the denominator
was floored, which is the difference between "this score carries no ordering information"
and "the scaling discarded the ordering information it had".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dyn_evt_pdm.pipelines.industrial_results import _MIN_RELATIVE_IQR, _robust_center_scale

FEATURES = ("well_behaved", "near_constant")


def _frame(seed: int = 0, n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    well_behaved = rng.normal(10.0, 2.0, size=n)
    # Idle at a constant value with rare large excursions: the shape that breaks an
    # IQR denominator, since the central half is quantisation noise.
    near_constant = np.full(n, 0.5)
    near_constant += rng.normal(0.0, 0.001, size=n)
    # Excursions at half a percent: rarer than the 1st-to-99th-percentile range can
    # see, which is why the floor is measured on a wider range.
    near_constant[rng.choice(n, size=n // 200, replace=False)] += 8.0
    return pd.DataFrame({"well_behaved": well_behaved, "near_constant": near_constant})


def test_well_behaved_feature_keeps_its_interquartile_scale() -> None:
    """The floor must not disturb features whose IQR is already meaningful."""
    frame = _frame()
    _center, scale = _robust_center_scale(frame, FEATURES)
    values = frame["well_behaved"]
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    assert float(scale["well_behaved"]) == iqr


def test_near_constant_feature_is_floored() -> None:
    frame = _frame()
    _center, scale = _robust_center_scale(frame, FEATURES)
    values = frame["near_constant"]
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    spread = float(values.quantile(0.999) - values.quantile(0.001))
    assert iqr < _MIN_RELATIVE_IQR * spread, "fixture is not degenerate enough to test"
    assert float(scale["near_constant"]) > iqr
    assert float(scale["near_constant"]) == _MIN_RELATIVE_IQR * spread


def test_floor_bounds_the_score_a_degenerate_feature_can_reach() -> None:
    """Without the floor this channel alone reaches z-scores in the thousands."""
    frame = _frame()
    values = frame["near_constant"]
    centre = float(values.median())
    raw_iqr = float(values.quantile(0.75) - values.quantile(0.25))
    unfloored = float((values - centre).abs().max() / raw_iqr)

    _center, scale = _robust_center_scale(frame, FEATURES)
    floored = float((values - centre).abs().max() / float(scale["near_constant"]))

    assert unfloored > 1_000
    assert floored < 200


def test_zero_spread_feature_still_falls_back_to_unit_scale() -> None:
    """A constant column has no scale at all; it must not divide by zero."""
    frame = pd.DataFrame({"well_behaved": [1.0, 2.0, 3.0, 4.0], "near_constant": [7.0] * 4})
    _center, scale = _robust_center_scale(frame, FEATURES)
    assert float(scale["near_constant"]) == 1.0
    assert np.isfinite(scale.to_numpy(dtype=float)).all()
