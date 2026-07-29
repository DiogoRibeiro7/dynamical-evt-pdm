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
