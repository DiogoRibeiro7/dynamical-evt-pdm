"""Declared specifications for every event-level baseline method.

A method is defined by three independent choices: the score it computes, the rule that
turns that score into a threshold, and the policy that turns threshold exceedances into
alarm episodes. Two methods that share all three are the same method under two names.

This registry exists because they previously did. Ten declared method names resolved to
one score array through a silent fallback, so nine of the nineteen published baselines
were bit-identical -- including the registered recurrence score and the classical
POT/GPD, Ferro-Segers and K-gaps entries it was compared against. Making the three axes
explicit means a collision is visible in the specification rather than buried in a
result table, and :func:`assert_specifications_are_distinct` refuses to let one recur.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How the per-sample score is computed.
ScoreKind = str

#: How the alarm threshold is chosen from training scores.
ThresholdRule = str

#: How threshold exceedances become alarm episodes.
EventPolicy = str


@dataclass(frozen=True, slots=True)
class EventMethodSpec:
    """The declared identity of one event-level method."""

    name: str
    family: str
    score_kind: ScoreKind
    threshold_rule: ThresholdRule
    event_policy: EventPolicy
    parameter_count: int
    calibration_status: str
    tuning_partition: str
    notes: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the triple that makes this method distinct from another."""

        return (self.score_kind, self.threshold_rule, self.event_policy)


#: Every declared event-level method, with its three defining axes.
EVENT_METHOD_SPECS: tuple[EventMethodSpec, ...] = (
    EventMethodSpec(
        name="engineering_threshold",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="fixed_physical_3z",
        event_policy="merge_gap",
        parameter_count=1,
        calibration_status="uncalibrated",
        tuning_partition="none (fixed a priori)",
        notes="Mandatory simple baseline; the threshold is declared, not fitted.",
    ),
    EventMethodSpec(
        name="best_individual_sensor_threshold",
        family="baseline",
        score_kind="best_single_sensor_z",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=2,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Selects the single sensor with the highest training separation.",
    ),
    EventMethodSpec(
        name="global_empirical_threshold",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=1,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="regime_conditioned_empirical_threshold",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="regime_conditioned_quantile",
        event_policy="merge_gap",
        parameter_count=3,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="One threshold per operating regime rather than one global threshold.",
    ),
    EventMethodSpec(
        name="classical_pot_gpd",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="gpd_return_level",
        event_policy="merge_gap",
        parameter_count=3,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Generalised Pareto fit to training exceedances; threshold is a return level.",
    ),
    EventMethodSpec(
        name="fixed_run_declustering",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="train_quantile",
        event_policy="fixed_run_declustering",
        parameter_count=2,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Same score and threshold as the global rule; differs in event conversion.",
    ),
    EventMethodSpec(
        name="ferro_segers_event_policy",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="train_quantile",
        event_policy="ferro_segers_run_length",
        parameter_count=2,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Run length derived from the Ferro-Segers extremal index on training data.",
    ),
    EventMethodSpec(
        name="k_gaps_event_policy",
        family="baseline",
        score_kind="robust_max_abs_z",
        threshold_rule="train_quantile",
        event_policy="k_gaps_run_length",
        parameter_count=2,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Run length derived from the K-gaps extremal index on training data.",
    ),
    EventMethodSpec(
        name="spot",
        family="baseline",
        score_kind="spot_streaming_excess",
        threshold_rule="spot_zero",
        event_policy="merge_gap",
        parameter_count=3,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="isolation_forest",
        family="baseline",
        score_kind="isolation_forest",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=100,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="Fitted sklearn IsolationForest on training rows only.",
    ),
    EventMethodSpec(
        name="robust_online_changepoint",
        family="baseline",
        score_kind="absolute_first_difference",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=1,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="linear_autoencoder",
        family="baseline",
        score_kind="linear_projection_residual",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=8,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="compact_nonlinear_autoencoder",
        family="baseline",
        score_kind="nonlinear_projection_residual",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=8,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="conformal_anomaly",
        family="baseline",
        score_kind="conformal_tail_probability",
        threshold_rule="conformal_level",
        event_policy="merge_gap",
        parameter_count=1,
        calibration_status="conformal (marginal, split)",
        tuning_partition="train (calibration split)",
    ),
    EventMethodSpec(
        name="empirical_horizon_risk",
        family="baseline",
        score_kind="empirical_horizon_risk",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=2,
        calibration_status="empirical frequency",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="dynamical_evt_robust_score",
        family="registered",
        score_kind="robust_max_abs_z",
        threshold_rule="train_quantile",
        event_policy="extremal_index_declustering",
        parameter_count=3,
        calibration_status="uncalibrated",
        tuning_partition="train",
        notes="The registered method: recurrence score with extremal-index declustering.",
    ),
    EventMethodSpec(
        name="failure_prototype_region",
        family="target_region",
        score_kind="negative_prototype_distance",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=4,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="rare_state_region",
        family="target_region",
        score_kind="negative_rare_state_distance",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=4,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
    EventMethodSpec(
        name="negative_control_region",
        family="negative_control",
        score_kind="negative_control_distance",
        threshold_rule="train_quantile",
        event_policy="merge_gap",
        parameter_count=4,
        calibration_status="uncalibrated",
        tuning_partition="train",
    ),
)

#: Specification lookup by method name.
SPECS_BY_NAME: dict[str, EventMethodSpec] = {spec.name: spec for spec in EVENT_METHOD_SPECS}


def assert_specifications_are_distinct() -> None:
    """Raise if two declared methods share score, threshold rule and event policy.

    Reporting two names for one computation as two baselines overstates the breadth of
    the comparison, so a collision is a hard error rather than a warning.
    """

    seen: dict[tuple[str, str, str], str] = {}
    collisions: list[str] = []
    for spec in EVENT_METHOD_SPECS:
        previous = seen.get(spec.identity)
        if previous is not None:
            collisions.append(f"{previous} and {spec.name} share identity {spec.identity}")
        else:
            seen[spec.identity] = spec.name
    if collisions:
        raise ValueError("event method specifications collide: " + "; ".join(collisions))


def method_family(method: str) -> str:
    """Return the declared family for one method name."""

    spec = SPECS_BY_NAME.get(method)
    if spec is None:
        raise ValueError(f"undeclared event method: {method}")
    return spec.family


#: Event policies that reduce each extreme cluster to a single alarm onset.
DECLUSTERING_POLICIES: frozenset[str] = frozenset(
    {
        "fixed_run_declustering",
        "ferro_segers_run_length",
        "k_gaps_run_length",
        "extremal_index_declustering",
    }
)


def method_merge_gap(method: str, *, default_merge_gap: int) -> int:
    """Return the alarm merge gap belonging to one method's event policy.

    Alarm merging is part of the detector, not part of the evaluation protocol, so it
    varies by method while matching, warning horizon, tolerance and operating time stay
    shared. The audit reports the variation rather than forbidding it.

    A declustering policy has already reduced each extreme cluster to one onset, so it
    merges with a gap of zero: one cluster becomes one alarm. Re-applying the global gap
    on top would merge separate clusters back together and reproduce the undeclustered
    episode set, which is exactly why the declustering baselines previously returned
    results identical to the global empirical threshold.
    """

    if default_merge_gap < 0:
        raise ValueError("default_merge_gap must be non-negative")
    spec = SPECS_BY_NAME.get(method)
    if spec is None:
        raise ValueError(f"undeclared event method: {method}")
    if spec.event_policy in DECLUSTERING_POLICIES:
        return 0
    return default_merge_gap


#: What each score is a measurement of. Review asked for this because a method's name and
#: its family do not say whether it measures proximity to a recurrent state, the size of
#: an anomaly, or a probability in a tail, and those are not interchangeable claims.
SCORE_SEMANTICS: dict[ScoreKind, str] = {
    "robust_max_abs_z": "anomaly magnitude",
    "best_single_sensor_z": "anomaly magnitude",
    "isolation_forest": "anomaly magnitude",
    "absolute_first_difference": "change magnitude",
    "linear_projection_residual": "reconstruction error",
    "nonlinear_projection_residual": "reconstruction error",
    "spot_streaming_excess": "tail probability",
    "conformal_tail_probability": "tail probability",
    "empirical_horizon_risk": "estimated event probability",
    "negative_prototype_distance": "recurrence proximity",
    "negative_rare_state_distance": "recurrence proximity",
    "negative_control_distance": "recurrence proximity",
}

#: Threshold rules that are derived from an extreme-value model.
_EVT_THRESHOLD_RULES = frozenset({"gpd_return_level", "spot_zero"})


def score_semantics(method: str) -> str:
    """Return what this method's score measures."""
    spec = SPECS_BY_NAME.get(method)
    if spec is None:
        raise ValueError(f"undeclared event method: {method}")
    semantics = SCORE_SEMANTICS.get(spec.score_kind)
    if semantics is None:
        raise ValueError(f"score kind {spec.score_kind!r} has no declared semantics")
    return semantics


def evt_role(method: str) -> str:
    """Return where extreme-value theory enters this method, if anywhere.

    The distinction review asked for is whether an extreme-value quantity changes the
    alarms a method raises or is only reported alongside them. It changes the alarms in
    two ways here: a tail model can set the threshold, or an estimated extremal index can
    set the declustering run length. Everywhere else the extremal-index estimates in this
    paper are diagnostics computed on the side, and the alarms would be identical without
    them.
    """
    spec = SPECS_BY_NAME.get(method)
    if spec is None:
        raise ValueError(f"undeclared event method: {method}")
    sets_threshold = spec.threshold_rule in _EVT_THRESHOLD_RULES
    sets_declustering = spec.event_policy in {
        "ferro_segers_run_length",
        "k_gaps_run_length",
        "extremal_index_declustering",
    }
    if sets_threshold and sets_declustering:
        return "sets threshold and declustering"
    if sets_threshold:
        return "sets threshold"
    if sets_declustering:
        return "sets declustering"
    return "diagnostic only"


def assert_semantics_are_declared() -> None:
    """Every declared method must have score semantics and an EVT role."""
    missing = [spec.name for spec in EVENT_METHOD_SPECS if spec.score_kind not in SCORE_SEMANTICS]
    if missing:
        raise ValueError(f"methods without declared score semantics: {sorted(missing)}")
