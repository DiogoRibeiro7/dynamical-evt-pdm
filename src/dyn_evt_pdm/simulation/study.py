"""Deterministic finite-sample simulation study runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dyn_evt_pdm.evt.clusters import extract_clusters
from dyn_evt_pdm.evt.extremal_index import (
    BLOCK_SIZE_ESTIMATORS,
    ESTIMATOR_NAMES,
    RUN_LENGTH_ESTIMATORS,
    IntervalEstimate,
    bootstrap_extremal_index_interval,
    default_block_size,
    default_bootstrap_block_length,
    extremal_index_point_estimate,
)
from dyn_evt_pdm.evt.hitting_times import empirical_hit_probability
from dyn_evt_pdm.evt.thresholds import fit_quantile_threshold
from dyn_evt_pdm.simulation.reference_theta import (
    REFERENCE_N_STEPS,
    THEORETICAL,
    ReferenceTheta,
    numerical_reference_theta,
    theoretical_reference_theta,
)
from dyn_evt_pdm.simulation.systems import (
    SimulatedSeries,
    simulate_cyclic_degradation_series,
    simulate_iid_bounded_tail,
    simulate_iid_light_tail,
    simulate_iid_pareto,
    simulate_lagged_multivariate_extremes,
    simulate_logistic_nonperiodic_observable,
    simulate_logistic_target_observable,
    simulate_regime_mixture_series,
)
from dyn_evt_pdm.types import BoolArray

#: Stable artifact column stem for each estimator name.
ESTIMATOR_COLUMN_STEMS: dict[str, str] = {
    "runs": "runs",
    "ferro_segers_intervals": "intervals",
    "k_gaps": "k_gaps",
    "reciprocal_mean_cluster": "reciprocal_mean_cluster",
    "block": "block",
    "no_declustering": "no_declustering",
}


@dataclass(frozen=True, slots=True)
class SimulationStudyConfig:
    """Grid configuration for a reproducible simulation study."""

    systems: tuple[str, ...] = (
        "iid_light_tail",
        "iid_bounded_tail",
        "iid_pareto",
        "logistic_nonperiodic_target",
        "logistic_periodic_target",
        "regime_mixture",
        "cyclic_degradation",
        "lagged_multivariate",
    )
    sample_sizes: tuple[int, ...] = (1_000, 5_000)
    threshold_quantiles: tuple[float, ...] = (0.95, 0.98)
    run_lengths: tuple[int, ...] = (0, 5, 10)
    noise_scales: tuple[float, ...] = (0.0, 0.08)
    missing_rates: tuple[float, ...] = (0.0,)
    repetitions: int = 10
    seed: int = 42
    hit_horizon: int = 20
    n_jobs: int = 1
    bootstrap_resamples: int = 0
    bootstrap_level: float = 0.95

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.hit_horizon < 1:
            raise ValueError("hit_horizon must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples must be non-negative")
        if self.bootstrap_resamples == 1:
            raise ValueError("bootstrap_resamples must be zero or at least two")
        if not 0.0 < self.bootstrap_level < 1.0:
            raise ValueError("bootstrap_level must lie in (0, 1)")
        for sample_size in self.sample_sizes:
            if sample_size < 100:
                raise ValueError("sample sizes must be at least 100")
        for quantile in self.threshold_quantiles:
            if not 0.0 < quantile < 1.0:
                raise ValueError("threshold quantiles must lie in (0, 1)")
        for run_length in self.run_lengths:
            if run_length < 0:
                raise ValueError("run lengths must be non-negative")
        for rate in self.missing_rates:
            if not 0.0 <= rate < 1.0:
                raise ValueError("missing rates must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class SimulationDecision:
    """Machine-readable hypothesis decision for one simulation cell."""

    decision_id: str
    hypothesis_id: str
    system: str
    estimator: str
    n_steps: int
    threshold_quantile: float
    run_length: int
    noise_scale: float
    missing_rate: float
    repetitions: int
    true_theta: float | None
    mean_estimate: float | None
    bias: float | None
    rmse: float | None
    monte_carlo_standard_error: float | None
    failure_rate: float
    status: str
    reason: str


def simulation_experiment_id(config: SimulationStudyConfig) -> str:
    """Return a stable ID derived from normalized configuration content."""

    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_simulation_study(config: SimulationStudyConfig) -> pd.DataFrame:
    """Evaluate extremal-index estimators across the configured simulation grid."""

    tasks = list(
        product(
            config.systems,
            config.sample_sizes,
            config.threshold_quantiles,
            config.run_lengths,
            config.noise_scales,
            config.missing_rates,
            range(config.repetitions),
        )
    )
    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = [int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(len(tasks))]
    experiment_id = simulation_experiment_id(config)

    records = Parallel(n_jobs=config.n_jobs)(
        delayed(_evaluate_task)(task, child_seed=seed, experiment_id=experiment_id, config=config)
        for task, seed in zip(tasks, child_seeds, strict=True)
    )
    frame = pd.DataFrame.from_records(records)
    return attach_reference_theta(frame, config=config)


def attach_reference_theta(
    frame: pd.DataFrame,
    *,
    config: SimulationStudyConfig,
    reference_n_steps: int = REFERENCE_N_STEPS,
) -> pd.DataFrame:
    """Attach reference extremal indices and recompute bias and coverage against them.

    ``true_theta`` carries only the closed-form value emitted by the simulator, which is
    absent for every noisy configuration. Bias and coverage are therefore recomputed
    here against the resolved reference so that numerically referenced systems
    contribute to the study instead of silently dropping out.
    """

    if frame.empty:
        return frame

    keys_per_row: list[tuple[str, float, float, float]] = list(
        zip(
            [str(value) for value in frame["system"].tolist()],
            [float(value) for value in frame["noise_scale"].tolist()],
            [float(value) for value in frame["missing_rate"].tolist()],
            [float(value) for value in frame["threshold_quantile"].tolist()],
            strict=True,
        )
    )
    resolved: dict[tuple[str, float, float, float], ReferenceTheta] = {}
    for key in dict.fromkeys(keys_per_row):
        system, noise_scale, missing_rate, threshold_quantile = key
        resolved[key] = resolve_reference_theta(
            system,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
            threshold_quantile=threshold_quantile,
            seed=config.seed,
            n_steps=reference_n_steps,
        )
    references = [resolved[key] for key in keys_per_row]

    updated = frame.copy()
    updated["reference_theta"] = [
        reference.value if reference.value is not None else np.nan for reference in references
    ]
    updated["reference_kind"] = [reference.kind for reference in references]
    updated["reference_runs"] = [
        reference.runs_reference if reference.runs_reference is not None else np.nan
        for reference in references
    ]
    updated["reference_blocks"] = [
        reference.blocks_reference if reference.blocks_reference is not None else np.nan
        for reference in references
    ]
    updated["reference_disagreement"] = [
        reference.disagreement if reference.disagreement is not None else np.nan
        for reference in references
    ]
    updated["reference_reason"] = [reference.reason for reference in references]

    reference_values = updated["reference_theta"].to_numpy(dtype=np.float64)
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        estimates = updated[f"{stem}_theta"].to_numpy(dtype=np.float64)
        updated[f"{stem}_bias"] = estimates - reference_values
        lower = updated[f"{stem}_lower"].to_numpy(dtype=np.float64)
        upper = updated[f"{stem}_upper"].to_numpy(dtype=np.float64)
        covered = (
            np.isfinite(lower)
            & np.isfinite(upper)
            & np.isfinite(reference_values)
            & (lower <= reference_values)
            & (reference_values <= upper)
        )
        estimable = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(reference_values)
        updated[f"{stem}_covered"] = pd.array(
            [
                bool(value) if known else None
                for value, known in zip(covered, estimable, strict=True)
            ],
            dtype="boolean",
        )
    return updated


def write_simulation_study(
    config: SimulationStudyConfig,
    output_path: Path,
) -> pd.DataFrame:
    """Run a study and persist tidy Parquet results."""

    result = run_simulation_study(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    decisions = simulation_decision_records(result)
    decision_path = output_path.with_suffix(output_path.suffix + ".decisions.json")
    decision_path.write_text(
        json.dumps([asdict(decision) for decision in decisions], indent=2),
        encoding="utf-8",
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "experiment_id": simulation_experiment_id(config),
        "config": asdict(config),
        "rows": len(result),
        "output_path": str(output_path),
        "decision_path": str(decision_path),
        "decision_count": len(decisions),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def simulation_decision_records(
    results: pd.DataFrame,
    *,
    rmse_tolerance: float = 0.25,
    max_failure_rate: float = 0.10,
) -> tuple[SimulationDecision, ...]:
    """Aggregate replicate rows into predeclared simulation hypothesis decisions."""

    required = {
        "experiment_id",
        "system",
        "n_steps",
        "threshold_quantile",
        "run_length",
        "noise_scale",
        "missing_rate",
        "true_theta",
        "failed",
    }
    estimator_columns = {
        "runs": "runs_theta",
        "ferro_segers_intervals": "intervals_theta",
        "k_gaps": "k_gaps_theta",
        "reciprocal_mean_cluster": "reciprocal_mean_cluster_theta",
        "block": "block_theta",
        "no_declustering": "no_declustering_theta",
    }
    missing = sorted(required.union(estimator_columns.values()).difference(results.columns))
    if missing:
        raise ValueError(f"missing simulation result columns: {missing}")
    if rmse_tolerance <= 0.0:
        raise ValueError("rmse_tolerance must be positive")
    if not 0.0 <= max_failure_rate <= 1.0:
        raise ValueError("max_failure_rate must lie in [0, 1]")

    group_columns = [
        "experiment_id",
        "system",
        "n_steps",
        "threshold_quantile",
        "run_length",
        "noise_scale",
        "missing_rate",
    ]
    decisions: list[SimulationDecision] = []
    for keys, group in results.groupby(group_columns, dropna=False, sort=True):
        key_values = dict(zip(group_columns, keys, strict=True))
        true_values = group["true_theta"].to_numpy(dtype=np.float64)
        finite_truth = true_values[np.isfinite(true_values)]
        true_theta = float(finite_truth[0]) if len(finite_truth) else None
        for estimator, column in estimator_columns.items():
            decision = _simulation_estimator_decision(
                group,
                key_values=key_values,
                estimator=estimator,
                column=column,
                true_theta=true_theta,
                rmse_tolerance=rmse_tolerance,
                max_failure_rate=max_failure_rate,
            )
            decisions.append(decision)
    return tuple(decisions)


def simulation_study_config_from_mapping(
    raw: dict[str, Any],
    *,
    smoke: bool = False,
    n_jobs: int | None = None,
) -> SimulationStudyConfig:
    """Build a study config from project YAML dictionaries."""

    simulation = raw.get("simulation", {})
    experiment = raw.get("experiment", {})
    sample_sizes = tuple(int(value) for value in experiment.get("sample_sizes", []))
    if not sample_sizes:
        sample_sizes = (int(simulation.get("n_steps", 20_000)),)
    threshold_quantiles = tuple(
        float(value) for value in experiment.get("thresholds", (0.95, 0.98))
    )
    run_lengths = tuple(int(value) for value in experiment.get("run_lengths", (0, 5, 10)))
    repetitions = int(experiment.get("repetitions", 10))
    systems = tuple(str(value) for value in experiment.get("systems", ()))
    noise_scales = tuple(float(value) for value in experiment.get("noise_scales", ()))
    if not noise_scales:
        noise_scales = (0.0, float(simulation.get("noise_scale", 0.08)))
    missing_rates = tuple(float(value) for value in experiment.get("missing_rates", (0.0,)))
    bootstrap_resamples = int(experiment.get("bootstrap_resamples", 0))
    bootstrap_level = float(experiment.get("bootstrap_level", 0.95))
    config = SimulationStudyConfig(
        systems=systems or SimulationStudyConfig().systems,
        sample_sizes=sample_sizes,
        threshold_quantiles=threshold_quantiles,
        run_lengths=run_lengths,
        noise_scales=noise_scales,
        missing_rates=missing_rates,
        repetitions=repetitions,
        seed=int(simulation.get("seed", 42)),
        n_jobs=1 if n_jobs is None else n_jobs,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_level=bootstrap_level,
    )
    if not smoke:
        return config
    return SimulationStudyConfig(
        systems=("iid_light_tail", "iid_pareto", "logistic_periodic_target", "cyclic_degradation"),
        sample_sizes=(1_000,),
        threshold_quantiles=threshold_quantiles[:1],
        run_lengths=run_lengths[:2],
        noise_scales=(0.0,),
        missing_rates=(0.0,),
        repetitions=min(repetitions, 3),
        seed=config.seed,
        hit_horizon=config.hit_horizon,
        n_jobs=config.n_jobs,
        bootstrap_resamples=min(bootstrap_resamples, 20),
        bootstrap_level=bootstrap_level,
    )


def _evaluate_task(
    task: tuple[str, int, float, int, float, float, int],
    *,
    child_seed: int,
    experiment_id: str,
    config: SimulationStudyConfig,
) -> dict[str, object]:
    system, n_steps, quantile, run_length, noise_scale, missing_rate, repetition = task
    rng = np.random.default_rng(child_seed)
    simulated = _simulate_system(
        system,
        n_steps=n_steps,
        rng=rng,
        noise_scale=noise_scale,
        missing_rate=missing_rate,
    )
    values = np.asarray(simulated.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    failed = False
    failure_reason = ""
    threshold = np.nan
    n_exceedances = 0
    n_clusters = 0
    mean_cluster_size = np.nan
    hit_probability = np.nan
    exceedances: BoolArray | None = None
    block_size = default_block_size(n_steps)
    bootstrap_block_length = default_bootstrap_block_length(n_steps)

    try:
        threshold = fit_quantile_threshold(finite, quantile=quantile)
        exceedances = np.isfinite(values) & (values > threshold)
        exceedance_indices = np.flatnonzero(exceedances).astype(np.int64)
        clusters = extract_clusters(exceedances, run_length=run_length)
        n_exceedances = int(len(exceedance_indices))
        n_clusters = int(len(clusters))
        if n_clusters:
            mean_cluster_size = float(np.mean([cluster.size for cluster in clusters]))
        if len(values) > config.hit_horizon:
            hit_probability = empirical_hit_probability(exceedances, horizon=config.hit_horizon)
    except ValueError as exc:
        failed = True
        failure_reason = str(exc)

    true_theta = simulated.true_theta
    estimates: dict[str, float] = dict.fromkeys(ESTIMATOR_NAMES, np.nan)
    intervals: dict[str, IntervalEstimate | None] = dict.fromkeys(ESTIMATOR_NAMES)
    estimator_failures: dict[str, str] = {}
    if exceedances is not None:
        bootstrap_rng = np.random.default_rng(child_seed ^ 0x9E3779B9)
        for name in ESTIMATOR_NAMES:
            try:
                estimates[name] = extremal_index_point_estimate(
                    name, exceedances, run_length=run_length, block_size=block_size
                )
            except ValueError as exc:
                estimator_failures[name] = str(exc)
                continue
            if config.bootstrap_resamples >= 2:
                try:
                    intervals[name] = bootstrap_extremal_index_interval(
                        name,
                        exceedances,
                        run_length=run_length,
                        block_size=block_size,
                        n_resamples=config.bootstrap_resamples,
                        block_length=bootstrap_block_length,
                        level=config.bootstrap_level,
                        rng=bootstrap_rng,
                    )
                except ValueError as exc:
                    estimator_failures.setdefault(name, str(exc))

    estimator_columns: dict[str, object] = {}
    for name in ESTIMATOR_NAMES:
        stem = ESTIMATOR_COLUMN_STEMS[name]
        estimate = estimates[name]
        interval = intervals[name]
        estimator_columns[f"{stem}_theta"] = estimate
        estimator_columns[f"{stem}_bias"] = (
            estimate - true_theta if true_theta is not None else np.nan
        )
        estimator_columns[f"{stem}_tuning_parameter"] = (
            float(run_length)
            if name in RUN_LENGTH_ESTIMATORS
            else float(block_size)
            if name in BLOCK_SIZE_ESTIMATORS
            else np.nan
        )
        estimator_columns[f"{stem}_lower"] = interval.lower if interval else np.nan
        estimator_columns[f"{stem}_upper"] = interval.upper if interval else np.nan
        estimator_columns[f"{stem}_interval_width"] = interval.width if interval else np.nan
        estimator_columns[f"{stem}_covered"] = (
            bool(interval.covers(true_theta))
            if interval is not None and true_theta is not None
            else None
        )
        estimator_columns[f"{stem}_bootstrap_valid"] = interval.n_valid if interval else 0
        estimator_columns[f"{stem}_bootstrap_clipped"] = interval.n_clipped if interval else 0
        estimator_columns[f"{stem}_failed"] = name in estimator_failures
        estimator_columns[f"{stem}_failure_reason"] = estimator_failures.get(name, "")

    return {
        "experiment_id": experiment_id,
        "system": simulated.system,
        "repetition": repetition,
        "seed": child_seed,
        "n_steps": n_steps,
        "threshold_quantile": quantile,
        "threshold": threshold,
        "run_length": run_length,
        "noise_scale": noise_scale,
        "missing_rate": missing_rate,
        "n_finite": int(len(finite)),
        "n_missing": int(len(values) - len(finite)),
        "n_exceedances": n_exceedances,
        "n_clusters": n_clusters,
        "block_size": block_size,
        "bootstrap_block_length": bootstrap_block_length,
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_level": config.bootstrap_level,
        "true_theta": true_theta if true_theta is not None else np.nan,
        **estimator_columns,
        "mean_cluster_size": mean_cluster_size,
        "hit_probability": hit_probability,
        "failed": failed,
        "failure_reason": failure_reason,
    }


def _simulation_estimator_decision(
    group: pd.DataFrame,
    *,
    key_values: dict[str, object],
    estimator: str,
    column: str,
    true_theta: float | None,
    rmse_tolerance: float,
    max_failure_rate: float,
) -> SimulationDecision:
    estimates = group[column].to_numpy(dtype=np.float64)
    valid = np.isfinite(estimates)
    failure_flags = group["failed"].astype(bool).to_numpy() | ~valid
    failure_rate = float(np.mean(failure_flags))
    repetition_count = int(len(group))
    mean_estimate = float(np.mean(estimates[valid])) if np.any(valid) else None
    if true_theta is None:
        return _decision_from_values(
            key_values=key_values,
            estimator=estimator,
            repetitions=repetition_count,
            true_theta=None,
            mean_estimate=mean_estimate,
            bias=None,
            rmse=None,
            monte_carlo_standard_error=None,
            failure_rate=failure_rate,
            status="inconclusive",
            reason="simulation cell has no declared theoretical or numerical theta target",
        )
    if not np.any(valid):
        return _decision_from_values(
            key_values=key_values,
            estimator=estimator,
            repetitions=repetition_count,
            true_theta=true_theta,
            mean_estimate=None,
            bias=None,
            rmse=None,
            monte_carlo_standard_error=None,
            failure_rate=failure_rate,
            status="inconclusive",
            reason="all estimator replicates failed or returned non-finite values",
        )
    errors = estimates[valid] - true_theta
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mcse = float(np.std(errors, ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else 0.0
    status = (
        "supported"
        if rmse <= rmse_tolerance and failure_rate <= max_failure_rate
        else "not_supported"
    )
    reason = (
        f"rmse={rmse:.4g}, failure_rate={failure_rate:.4g}, "
        f"thresholds=({rmse_tolerance:.4g}, {max_failure_rate:.4g})"
    )
    return _decision_from_values(
        key_values=key_values,
        estimator=estimator,
        repetitions=repetition_count,
        true_theta=true_theta,
        mean_estimate=mean_estimate,
        bias=bias,
        rmse=rmse,
        monte_carlo_standard_error=mcse,
        failure_rate=failure_rate,
        status=status,
        reason=reason,
    )


def _decision_from_values(
    *,
    key_values: dict[str, object],
    estimator: str,
    repetitions: int,
    true_theta: float | None,
    mean_estimate: float | None,
    bias: float | None,
    rmse: float | None,
    monte_carlo_standard_error: float | None,
    failure_rate: float,
    status: str,
    reason: str,
) -> SimulationDecision:
    payload = json.dumps(
        {
            **key_values,
            "estimator": estimator,
        },
        sort_keys=True,
        default=str,
    )
    return SimulationDecision(
        decision_id=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        hypothesis_id="SIM-EI-RECOVERY",
        system=str(key_values["system"]),
        estimator=estimator,
        n_steps=_object_to_int(key_values["n_steps"]),
        threshold_quantile=_object_to_float(key_values["threshold_quantile"]),
        run_length=_object_to_int(key_values["run_length"]),
        noise_scale=_object_to_float(key_values["noise_scale"]),
        missing_rate=_object_to_float(key_values["missing_rate"]),
        repetitions=repetitions,
        true_theta=true_theta,
        mean_estimate=mean_estimate,
        bias=bias,
        rmse=rmse,
        monte_carlo_standard_error=monte_carlo_standard_error,
        failure_rate=failure_rate,
        status=status,
        reason=reason,
    )


def _object_to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def _object_to_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


def resolve_reference_theta(
    system: str,
    *,
    noise_scale: float,
    missing_rate: float,
    threshold_quantile: float,
    seed: int,
    n_steps: int = REFERENCE_N_STEPS,
) -> ReferenceTheta:
    """Return the theoretical reference when one exists, otherwise a numerical one.

    Theoretical values are never overwritten by simulation, and numerical values are
    never relabelled as theoretical, so the two provenances stay separable downstream.
    """

    theoretical = theoretical_reference_theta(
        system, noise_scale=noise_scale, missing_rate=missing_rate
    )
    if theoretical is not None:
        return ReferenceTheta(
            system=system,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
            threshold_quantile=threshold_quantile,
            kind=THEORETICAL,
            value=theoretical,
            runs_reference=None,
            blocks_reference=None,
            disagreement=None,
            n_steps=0,
            seed=seed,
            reason="closed-form extremal index for this system",
        )

    rng = np.random.default_rng(seed)
    simulated = _simulate_system(
        system,
        n_steps=n_steps,
        rng=rng,
        noise_scale=noise_scale,
        missing_rate=missing_rate,
    )
    return numerical_reference_theta(
        np.asarray(simulated.values, dtype=np.float64),
        system=system,
        noise_scale=noise_scale,
        missing_rate=missing_rate,
        threshold_quantile=threshold_quantile,
        seed=seed,
    )


def _simulate_system(
    system: str,
    *,
    n_steps: int,
    rng: np.random.Generator,
    noise_scale: float,
    missing_rate: float,
) -> SimulatedSeries:
    if system == "iid_pareto":
        return simulate_iid_pareto(n_steps, rng=rng, missing_rate=missing_rate)
    if system == "iid_light_tail":
        return simulate_iid_light_tail(n_steps, rng=rng, missing_rate=missing_rate)
    if system == "iid_bounded_tail":
        return simulate_iid_bounded_tail(n_steps, rng=rng, missing_rate=missing_rate)
    if system == "logistic_nonperiodic_target":
        return simulate_logistic_nonperiodic_observable(
            n_steps,
            rng=rng,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
        )
    if system == "logistic_periodic_target":
        return simulate_logistic_target_observable(
            n_steps,
            rng=rng,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
        )
    if system == "regime_mixture":
        return simulate_regime_mixture_series(n_steps, rng=rng, missing_rate=missing_rate)
    if system == "cyclic_degradation":
        return simulate_cyclic_degradation_series(
            n_steps,
            rng=rng,
            noise_scale=max(noise_scale, 0.001),
            missing_rate=missing_rate,
        )
    if system == "lagged_multivariate":
        return simulate_lagged_multivariate_extremes(n_steps, rng=rng, missing_rate=missing_rate)
    if system == "noisy_periodic_dynamics":
        return _renamed_series(
            simulate_logistic_target_observable(
                n_steps,
                rng=rng,
                noise_scale=max(noise_scale, 0.12),
                missing_rate=missing_rate,
            ),
            system,
        )
    if system == "persistent_shift":
        return _persistent_shift_series(
            n_steps,
            rng=rng,
            noise_scale=noise_scale,
            missing_rate=missing_rate,
        )
    if system == "smoothing":
        return _smoothed_series(
            simulate_iid_pareto(n_steps, rng=rng, missing_rate=missing_rate),
            window=5,
            system=system,
        )
    if system == "downsampling":
        return _downsampled_series(
            simulate_iid_pareto(max(n_steps * 2, n_steps + 1), rng=rng, missing_rate=missing_rate),
            n_steps=n_steps,
            system=system,
        )
    if system == "missing_at_random":
        return _renamed_series(
            simulate_iid_pareto(n_steps, rng=rng, missing_rate=max(missing_rate, 0.10)),
            system,
        )
    if system == "burst_missingness":
        return _burst_missingness_series(
            simulate_iid_pareto(n_steps, rng=rng, missing_rate=missing_rate),
            rng=rng,
            system=system,
        )
    raise ValueError(f"unknown simulation system: {system}")


def _renamed_series(series: SimulatedSeries, system: str) -> SimulatedSeries:
    return SimulatedSeries(
        system=system,
        values=series.values,
        true_theta=series.true_theta,
        frame=series.frame.assign(system=system),
    )


def _smoothed_series(series: SimulatedSeries, *, window: int, system: str) -> SimulatedSeries:
    values = pd.Series(series.values).rolling(window=window, min_periods=1).mean().to_numpy()
    return SimulatedSeries(
        system=system,
        values=values.astype(np.float64),
        true_theta=None,
        frame=pd.DataFrame({"time": np.arange(len(values), dtype=np.int64), "observable": values}),
    )


def _downsampled_series(series: SimulatedSeries, *, n_steps: int, system: str) -> SimulatedSeries:
    values = np.asarray(series.values[::2][:n_steps], dtype=np.float64)
    if len(values) < n_steps:
        values = np.pad(values, (0, n_steps - len(values)), mode="edge")
    return SimulatedSeries(
        system=system,
        values=values,
        true_theta=None,
        frame=pd.DataFrame({"time": np.arange(len(values), dtype=np.int64), "observable": values}),
    )


def _burst_missingness_series(
    series: SimulatedSeries,
    *,
    rng: np.random.Generator,
    system: str,
) -> SimulatedSeries:
    values = np.asarray(series.values, dtype=np.float64).copy()
    burst_count = max(1, len(values) // 500)
    burst_width = max(2, len(values) // 200)
    starts = rng.integers(0, max(1, len(values) - burst_width), size=burst_count)
    for start in starts:
        values[int(start) : int(start) + burst_width] = np.nan
    return SimulatedSeries(
        system=system,
        values=values,
        true_theta=series.true_theta,
        frame=pd.DataFrame({"time": np.arange(len(values), dtype=np.int64), "observable": values}),
    )


def _persistent_shift_series(
    n_steps: int,
    *,
    rng: np.random.Generator,
    noise_scale: float,
    missing_rate: float,
) -> SimulatedSeries:
    values = rng.normal(0.0, 1.0, size=n_steps).astype(np.float64)
    shift_starts = np.arange(max(10, n_steps // 8), n_steps, max(10, n_steps // 4))
    shift_width = max(5, n_steps // 20)
    for start in shift_starts:
        values[int(start) : min(n_steps, int(start) + shift_width)] += 4.0
    if noise_scale:
        values += rng.normal(0.0, noise_scale, size=n_steps)
    if missing_rate:
        missing = rng.random(n_steps) < missing_rate
        values[missing] = np.nan
    return SimulatedSeries(
        system="persistent_shift",
        values=values,
        true_theta=None,
        frame=pd.DataFrame({"time": np.arange(n_steps, dtype=np.int64), "observable": values}),
    )
