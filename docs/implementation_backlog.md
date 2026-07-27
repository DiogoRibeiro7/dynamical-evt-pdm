# Research Extension Backlog

This document records extension criteria for broader claims than the current submission-ready package makes. Many foundations listed here are already implemented in the executable core; the authoritative current status is [`IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md). Items below should be read as future broadening work, larger validation surfaces, or stricter acceptance criteria for stronger methodological or deployment claims.

## Data Ingestion And Contract Extensions

MetroPT and MetroPT2:

- Read CSV data in chunks and convert it to partitioned Parquet.
- Detect source timestamps and document the explicit mapping.
- Preserve original columns and attach schema manifests with inferred dtype, missing rate, range and unique count.
- Verify monotonicity, duplicates, sampling gaps and file checksums.
- Create canonical sensor aliases without discarding source names.
- Maintain source-verified failure labels in machine-readable YAML.
- Generate exclusion masks for maintenance, corrupt sequences and configurable warm-up periods.

SCANIA Component X:

- Join operational, repair, time-to-event and specification data by `vehicle_id` without duplicating readouts.
- Sort by `vehicle_id,time_step`.
- Detect counter resets and irregular intervals.
- Convert cumulative counters to increments under an explicit reset policy.
- Preserve histogram groups as structured feature families.
- Create vehicle-level train, validation and test splits with no entity leakage.

Extension deliverables:

- Typed adapter classes.
- Dataset manifests and checksum validation.
- CLI commands `prepare-metropt`, `prepare-metropt2` and `prepare-scania`.
- Small synthetic fixtures mirroring each schema.
- Tests for duplicate timestamps, missing columns, counter resets and split leakage.
- Updated data dictionary and acquisition documentation.

Extension acceptance criteria:

- No real-data path is hard-coded.
- Processing 10 million rows is chunked and bounded in memory.
- Transformations are auditable from generated manifests.

## Simulation Study Extensions

Systems:

- IID Pareto or GPD sequence with `theta=1`.
- Logistic map at chaotic parameter values.
- Target points that are non-periodic and periodic where theoretical behaviour is known.
- Additive observational and dynamical noise.
- Cyclic industrial surrogate with operating regimes.
- Persistent degradation episode with a known start, end and severity trajectory.
- Multivariate lagged extreme sequence with configurable component order and lags.

Experimental grid:

- Sample size.
- Threshold quantile.
- Run length.
- Noise intensity.
- Missingness pattern.
- Regime imbalance.
- Fault duration and recurrence.
- Target-region radius.

Estimator outputs:

- Bias.
- RMSE.
- Empirical variance.
- Confidence-interval coverage.
- Failure rate.
- Cluster-size error.
- Hitting-probability calibration.

Engineering requirements for larger studies:

- Use `numpy.random.Generator` only.
- Derive deterministic experiment IDs from normalized configs.
- Run parallel repetitions with stable seed spawning.
- Save tidy Parquet results.
- Provide plotting functions outside notebooks.
- Keep CI smoke mode under 30 seconds.

Extension acceptance criteria:

- Every simulation figure can be recreated by one CLI command.
- Tests verify known trivial cases and seeded reproducibility.

## Univariate EVT And Extremal Index

Method extensions:

- Empirical and diagnostic threshold selection.
- Generalized Pareto fitting to excesses.
- Cluster maxima or peak extraction.
- Intervals estimator.
- Runs estimator.
- At least one additional established extremal-index estimator after verifying its primary source.
- Block or stationary bootstrap confidence intervals.
- Return-time and inter-exceedance diagnostics.
- Threshold and run-length stability analysis.

Guardrails:

- Validate finite values and minimum exceedance counts.
- Return structured result objects containing estimates, uncertainty, diagnostics and warnings.
- Avoid presenting estimates when regularity or sample-size diagnostics fail.
- Distinguish exceedance count, cluster size and cluster duration.
- Fit every threshold and tail model on training data only.

Tests:

- Use analytically simple IID and clustered synthetic processes.
- Add property tests for parameter bounds and invariance under strictly increasing affine transformations where applicable.

## Operating Regimes

Compare or expand three regime definitions:

- Auditable engineering rules using control signals and derivatives.
- Unsupervised hidden-state model fit only on training data.
- Change-point-derived piecewise stationary regimes.

Requirements:

- Regimes are causal at scoring time.
- No state is named using test failure labels.
- Expose regime occupancy, transition matrix and duration distributions.
- Detect tiny or unstable regimes and pool them using a documented rule.
- Compare global, regime-specific and hierarchical shrinkage thresholds.
- Quantify how much global thresholds mix incompatible distributions.
- Provide regime-definition ablations in every real-data result.

Extension acceptance criteria:

- The same EVT interface works with all regime methods.
- Unknown regimes at inference time use an explicit fallback with a logged warning.

## Dangerous Region And Hitting-Time Model

State construction:

- Robust scale continuous sensors on training data.
- Retain categorical and control-state information through suitable encoding.
- Optionally add causal delay embeddings.
- Prevent redundant dimensions from dominating Euclidean distance.

Dangerous-region definitions:

- Engineering constraint set.
- Pre-failure state prototypes derived from training failures only.
- Robust density level set.
- Medoid or nearest-neighbour target set.

Metric requirements:

- Support Euclidean, Mahalanobis and learned metric ablations.
- Fit learned metrics inside nested validation.

Observable:

- Implement `-log(max(distance, epsilon))` with chunked nearest-neighbour search.
- Track whether a high observable results from an exact duplicate, small metric scale or genuine proximity.

Hitting risk:

- Estimate probability of entry within horizons such as 15 min, 30 min, 1 h and 2 h.
- Add reliability diagrams, Brier score, calibration slope and bootstrap uncertainty.

Extension acceptance criteria:

- No target state from a held-out failure may enter the training dangerous region.
- Tests enforce provenance metadata for dangerous-region construction.

## Multivariate And Lagged Extremes

Functionality extensions:

- Component-specific regime-conditioned thresholds.
- Simultaneous exceedance vectors.
- Lagged component sequences.
- Empirical stable tail dependence diagnostics where justified.
- Multivariate extremal-index surface or a clearly labelled empirical analogue.
- Cluster construction for overlapping patterns.
- Permutation and time-shift null models that preserve univariate marginal clustering.
- Lag selection inside nested validation.

Main hypothesis:

- A lagged sequence across current, temperature and pressure-recovery extremes can provide earlier warning than simultaneous thresholds.

Extension acceptance criteria:

- Report uncertainty and multiple-comparison controls over tested lags.
- A pattern discovered on a held-out failure is exploratory and is not evaluated on the same event.

## Baselines

Baseline extensions:

- Simple engineering thresholds.
- Global empirical threshold.
- Classical POT/GPD.
- Fixed-run declustering.
- SPOT or DSPOT from a verified primary implementation or paper.
- Isolation Forest.
- Robust change-point detection.
- Autoencoder reconstruction score.
- Conformal anomaly score or horizon-risk calibration baseline.

Fairness rules:

- Select all hyperparameters on validation only.
- Use the same causal feature window.
- Use the same exclusion masks.
- Use the same event matching tolerance.
- Report pointwise score only as secondary.
- Include runtime, peak memory and parameter count.
- Keep strong simple baselines even when they outperform the proposed method.

Extension acceptance criteria:

- One experiment runner produces a standardized prediction table for every model with timestamp, score, threshold, alarm flag, episode ID, regime, horizon risk and provenance.

## Event-Level Evaluation

Definitions:

- Alarm event: configurable merge gap over pointwise flags.
- Failure event: documented maintenance interval.
- Early-warning window: interval ending at failure onset.
- Duplicate alarm: more than one alarm episode assigned to one failure.
- False alarm: alarm with no assignment under the selected tolerance.

Matching:

- Implement greedy and optimal bipartite matching.
- Make tolerance and early-warning policy explicit.
- Ensure one prediction cannot satisfy multiple failures unless the protocol explicitly allows it.

Metrics:

- Event recall and precision.
- F1.
- False alarm events per operating day.
- Duplicate alarms per failure.
- Earliest and median warning lead time.
- Time under warning.
- Normalized time-to-detection.
- Horizon-risk Brier score and calibration.
- Event utility under configurable costs.

Uncertainty:

- Bootstrap complete failures, days or vehicles.
- Do not bootstrap individual seconds as independent observations.

Extension acceptance criteria:

- Tests cover touching intervals, nested alarms, no alarms, no failures, alarms before the early-warning window and multiple alarms around one failure.

## Experiments, Figures And Paper Assets

Extension matrix:

- Simulation recovery.
- MetroPT leave-one-failure-out case studies.
- MetroPT2 temporal replication.
- SCANIA vehicle-level external validation.
- Threshold and run-length sensitivity.
- Regime ablation.
- Dangerous-region ablation.
- Noise and missingness stress tests.
- Multivariate lag ablation.
- Computational benchmark.

Extension outputs:

- Threshold stability.
- Extremal index by regime and period.
- Cluster-size distributions.
- Event timelines for every failure.
- Lead-time versus false-alarm frontier.
- Reliability diagrams.
- Simulation bias/RMSE.
- Ablation tables.
- Dataset and split summary.

Reproducibility:

- Each output embeds or accompanies experiment ID, commit hash and configuration hash.
- Figures are generated by library code, not manually edited notebook cells.

Extension acceptance criteria:

- A single `make paper-assets` command rebuilds every non-text artifact used in the manuscript from cached processed data.

## Scientific Review Checklist For Broader Claims

Before broadening the manuscript beyond the current negative-result scope, check:

- Whether the method is genuinely dynamical EVT or conventional declustering with new terminology.
- Whether the extremal index is estimated reliably at the available sample sizes and thresholds.
- Whether regimes are selected in a way that mechanically creates the reported differences.
- Whether dangerous-region construction leaks test-failure information.
- Whether independent sample sizes are overstated by counting seconds instead of failures or vehicles.
- Whether simple rules and strong anomaly baselines are treated fairly.
- Whether the event matching protocol favours the proposed method.
- Whether threshold, run length, target radius and lag choices are stable.
- Whether multivariate claims are supported by theory or clearly labelled empirical.
- Whether SCANIA validates the same mechanism, given its irregular entity-level structure.
- Whether uncertainty intervals are meaningful under only a handful of failures.
- Whether preprocessing, sensor smoothing or missingness creates apparent clustering.

Review outputs:

- Fatal flaws.
- Major revisions.
- Minor revisions.
- Further extensions.
- Claims kept restricted.
- Evidence that would broaden claims.
- Final recommendation with confidence.
