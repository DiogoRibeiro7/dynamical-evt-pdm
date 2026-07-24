# Paper outline

## Working title

**From Extreme Samples to Failure Episodes: Regime-Conditioned Dynamical Extreme Value Theory for Predictive Maintenance**

## Abstract logic

1. Industrial telemetry is cyclic and produces dependent alarms.
2. Pointwise methods over-count physical events and often ignore recurrence structure.
3. Introduce a regime-conditioned, dynamical-EVT-inspired finite-sample workflow with extremal-index-aware clustering and training-defined dangerous-region hitting probabilities.
4. Validate estimator behaviour in deterministic, noisy and degradation simulations.
5. Evaluate on MetroPT/MetroPT2 and transfer to SCANIA Component X.
6. Report event-level alarm burden, warning lead time and calibration.

## 1. Introduction

- Operational problem: alarms correspond poorly to physical episodes.
- Statistical problem: extremes are dependent because machinery has cycles and control loops.
- Gap: predictive-maintenance EVT often stops at POT or adaptive thresholds.
- Contribution list with no inflated claims.

## 2. Related work

- predictive maintenance and anomaly detection;
- classical univariate and multivariate EVT;
- dynamical EVT, hitting times and periodicity;
- extremal index and declustering;
- event-level time-series evaluation.

## 3. Method

### 3.1 State and regime representation
### 3.2 Dangerous-region observable
### 3.3 Regime-conditioned thresholds
### 3.4 Extremal-index estimation
### 3.5 Episode construction
### 3.6 Lagged multivariate extremes
### 3.7 Horizon-risk calibration

## 4. Simulation study

- systems and theoretical expectations;
- finite-sample bias and estimator stability;
- noise, regime mixture and missingness;
- recovery of episode properties.

## 5. Industrial datasets

- MetroPT and MetroPT2;
- SCANIA Component X;
- failure labels, splitting and preprocessing;
- limitations and independence units.

## 6. Experiments

- baseline definitions;
- nested temporal validation;
- primary event metrics;
- computational setup.

## 7. Results

- simulation tables first;
- per-failure case studies;
- aggregate event metrics with uncertainty;
- calibration;
- threshold/run-length stability;
- external fleet validation.

## 8. Discussion

- when the extremal index is mechanically informative;
- when it is only an empirical dependence diagnostic;
- operational trade-offs;
- dataset scarcity and label uncertainty;
- future theoretical work.

## 9. Conclusion

State the narrow supported contribution: recurrence diagnostics can provide a useful episode representation and risk diagnostic for cyclic industrial telemetry under the studied conditions.
