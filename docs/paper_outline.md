# Paper outline

## Working title

**When Clustered Extremes Do Not Become Early Warnings: A Negative-Result Dynamical-EVT Study for Public Industrial Monitoring Data**

## Abstract logic

1. Industrial telemetry can produce clustered extreme scores.
2. Maintenance decisions happen at alarm-episode, failure, vehicle or repair-action level.
3. Evaluate a regime-conditioned, dynamical-EVT-inspired workflow with training-only state construction, threshold sensitivity and event conversion.
4. Use bounded finite-sample simulations and five public industrial datasets.
5. Report negative and weak event-level outcomes instead of converting row counts into broad performance claims.
6. State the conditions required before recurrence diagnostics can support operational early warning.

## 1. Introduction

- Operational problem: pointwise extreme scores can become unusable alarm streams.
- Statistical problem: clustered extremes are dependence diagnostics, not maintenance decisions.
- Gap: industrial studies often lack traceable episode conversion, independent-unit accounting and negative-result preservation.
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

- MetroPT and MetroPT2 as compressor event case studies;
- SCANIA Component X as a vehicle-level repair-risk stress test;
- Hydraulic Systems and SECOM as supplementary condition-monitoring demonstrations;
- failure labels, splitting, preprocessing, limitations and independence units.

## 6. Experiments

- baseline definitions;
- nested temporal validation;
- primary event metrics;
- computational setup.

## 7. Results

- bounded simulation diagnostics;
- real compressor event baseline comparison;
- target-region variants and negative controls;
- pointwise-to-event metric collapse;
- score stratification rather than probability calibration;
- condition-monitoring and vehicle-level stress-test outcomes.

## 8. Discussion

- when the extremal index is mechanically informative;
- when it is only an empirical dependence diagnostic;
- operational trade-offs;
- dataset scarcity and label uncertainty;
- future theoretical work.

## 9. Conclusion

State the narrow supported contribution: recurrence diagnostics can expose dependence and alarm-conversion failure modes, but the evaluated public datasets do not establish reliable operational early warning.
