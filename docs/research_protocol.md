# Research protocol

## 1. Target contribution

The paper introduces a **regime-conditioned dynamical EVT monitoring framework** that converts pointwise extremes into physically interpretable episodes and estimates recurrence to dangerous operating regions.

The method is considered successful only when it demonstrates all of the following:

- valid or diagnostically explainable extremal-index behaviour in controlled simulations;
- fewer duplicate alarm episodes than pointwise and fixed-threshold baselines;
- comparable or better event recall at an operationally useful warning horizon;
- stable conclusions over threshold, run-length and regime specifications;
- credible transfer from a single APU to a fleet-level dataset.

## 2. Estimands

### Univariate recurrence

For scalar observable `X_t`, estimate:

- high threshold `u_r` within regime `r`;
- exceedance rate;
- extremal index `theta_r`;
- empirical cluster-size distribution;
- return-time distribution;
- generalized Pareto tail parameters after declustering.

### Dangerous-region recurrence

Construct a state vector `Z_t`. Define a dangerous region `A` from training-available pre-failure states or engineering constraints. Use:

```text
X_t = -log(d(Z_t, A))
```

Estimate the probability of entering `A` within horizon `h` and evaluate calibration.

### Multivariate recurrence

Evaluate simultaneous and lagged component exceedances. Candidate sequences include:

1. current extreme;
2. temperature extreme after lag `l1`;
3. pressure-recovery extreme after lag `l2`.

All lag patterns must be specified before test evaluation or selected inside nested validation.

## 3. Simulation ladder

1. IID heavy-tail reference with theoretical `theta = 1`.
2. Logistic map with recurrence near non-periodic and periodic targets.
3. Additive-noise sensitivity study.
4. Cyclic industrial surrogate with regime changes.
5. Degradation surrogate where fault persistence changes cluster structure.
6. Multivariate lagged-fault surrogate.

For each level, vary sample size, threshold, noise and missingness. Report bias, variance, coverage where confidence intervals are available, and estimator failure rate.

## 4. Real-data studies

### MetroPT

Primary operational case study. Three reported failures limit inferential power. Use leave-one-failure-out analysis where feasible and report each failure separately.

### MetroPT2

Temporal replication on two reported failures. Do not merge failures into pointwise rows and claim a large sample size.

### SCANIA Component X

External fleet-level validation. Unit of independence is the vehicle, not a readout row. Evaluate entity-level forward risk and repair outcomes.

## 5. Leakage controls

- No random row splitting.
- No dangerous-region prototypes built using test failures.
- No threshold selection using test false alarms.
- No regime model trained with future failure status unless explicitly treated as supervised and nested.
- No interpolation across maintenance boundaries.
- No retrospective feature windows extending beyond the prediction timestamp.

## 6. Primary comparison table

| Family | Required methods |
|---|---|
| Engineering | published or reconstructed pressure/current rules |
| Classical EVT | global POT, GPD, fixed-run declustering |
| Sequential EVT | SPOT/DSPOT-style threshold adaptation |
| Dynamical EVT | intervals EI, runs EI, regime-conditioned EI, dangerous-region hitting |
| ML anomaly | Isolation Forest, change point, autoencoder |
| Calibration | conformal or empirical horizon-risk baseline |

## 7. Primary metrics

- failure episodes detected;
- false alarm episodes per operating day;
- duplicate alarms per failure episode;
- earliest warning lead time;
- time under warning;
- event precision, recall and F1;
- Brier score for failure within horizon;
- calibration slope/intercept or reliability curves;
- computational latency and memory.

Pointwise metrics may be reported only as secondary diagnostics.

## 8. Statistical reporting

- Bootstrap blocks or entities, not individual rows.
- Report uncertainty by failure and by entity.
- Treat threshold and run length as sensitivity dimensions.
- Correct for repeated model-selection comparisons or state that results are exploratory.
- Publish all negative results, estimator failures and unstable regions.
