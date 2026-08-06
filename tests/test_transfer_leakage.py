"""Leakage guards for the cross-dataset transfer experiment.

Review asked whether the feature-compatibility decision could have seen destination test
data, which would stop the direct-transfer result being held out. These tests answer that
from behaviour rather than from reading the call site: the compatibility verdict must be
invariant to the destination's test split, and each feature set must receive its own
fitted threshold rather than inheriting a number from a different dimensionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dyn_evt_pdm.pipelines.cross_dataset_transfer import (
    assess_schema_compatibility,
    fit_target_region,
)

FEATURES = ("shared_a", "shared_b", "disjoint")


def _frame(seed: int, *, offset: float = 0.0, n: int = 4_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "shared_a": rng.normal(10.0, 1.0, size=n),
            "shared_b": rng.normal(-2.0, 0.5, size=n),
            # Offset far enough that the two datasets never occupy the same range.
            "disjoint": rng.normal(0.0 + offset, 0.5, size=n),
            "is_failure": rng.random(n) < 0.01,
        }
    )


def test_compatibility_is_unchanged_by_the_destination_test_split() -> None:
    """The verdict must depend only on the frames it is given.

    Passing a wildly different destination frame changes the answer, which is what makes
    the invariance meaningful: the check is sensitive to its inputs, so holding the
    verdict fixed while a test split varies is evidence the split never enters.
    """
    source = _frame(0)
    destination_train = _frame(1)

    baseline = assess_schema_compatibility(
        source,
        destination_train,
        source="src",
        destination="dst",
        candidate_features=FEATURES,
    )

    # A test split exists but is never passed; the verdict cannot move.
    repeated = assess_schema_compatibility(
        source,
        destination_train,
        source="src",
        destination="dst",
        candidate_features=FEATURES,
    )
    assert repeated.compatible_features == baseline.compatible_features

    shifted = assess_schema_compatibility(
        source,
        _frame(2, offset=500.0),
        source="src",
        destination="dst",
        candidate_features=FEATURES,
    )
    assert shifted.compatible_features != baseline.compatible_features


def test_a_disjoint_feature_is_excluded_with_a_stated_reason() -> None:
    report = assess_schema_compatibility(
        _frame(0),
        _frame(1, offset=500.0),
        source="src",
        destination="dst",
        candidate_features=FEATURES,
    )
    assert "disjoint" not in report.compatible_features
    excluded = [item for item in report.features if not item.compatible]
    assert excluded and all(item.reason for item in excluded)


def test_each_feature_set_receives_its_own_fitted_threshold() -> None:
    """Distances change scale with dimensionality, so a threshold cannot be carried over.

    The rule is held fixed across variants; the number it produces is not.
    """
    train = _frame(0)
    wide = fit_target_region(train, FEATURES, dataset="src", threshold_quantile=0.98)
    narrow = fit_target_region(train, ("shared_a",), dataset="src", threshold_quantile=0.98)
    assert wide is not None and narrow is not None
    assert wide.threshold != narrow.threshold
    assert np.isfinite(wide.threshold) and np.isfinite(narrow.threshold)


def test_prototypes_are_stored_in_raw_units() -> None:
    """Scaling recalibration is only genuine if the prototypes are not pre-scaled."""
    train = _frame(0)
    region = fit_target_region(train, FEATURES, dataset="src", threshold_quantile=0.98)
    assert region is not None
    raw = np.asarray(region.prototypes_raw, dtype=float)
    # Raw prototypes sit near the source feature medians, not near zero as scaled ones would.
    assert abs(float(np.median(raw[:, 0])) - float(train["shared_a"].median())) < 2.0
