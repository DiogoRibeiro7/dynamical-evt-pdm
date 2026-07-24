# Adversarial scientific review

This document is written as a hostile but technically fair pre-submission review. It assumes the current repository is the evidence package and asks what would make a reviewer reject the paper.

## Fatal flaws

1. The paper must not claim that the industrial results prove a genuinely new dynamical-EVT theorem. The implemented industrial procedure is a finite-data monitoring workflow using regime thresholds, declustering, hitting diagnostics and empirical lag patterns. Without a theorem linking the specific fitted observables, regimes and target sets to dynamical-EVT limits, the contribution is methodological and empirical.
2. MetroPT and MetroPT2 have too few independent failures for broad predictive-maintenance superiority claims. Counting 1 Hz rows as independent evidence would be a fatal statistical error.
3. SCANIA Component X cannot be treated as a direct mechanistic replication of MetroPT unless the paper proves that the readout cadence, repair labels, component semantics and entity-level sampling support the same recurrence mechanism. At present it is external operational validation, not proof of the same dynamical process.
4. Dangerous-region results are rejectable if any target prototype, distance transform, threshold, radius or lag is selected using the held-out failure later used for evaluation. The current code has provenance checks for one construction path, but the paper must show the same audit for every experiment.
5. If regimes are selected because they maximize extremal-index separation, the regime result is circular. Regime definitions must be causal and selected inside validation, or reported as exploratory.

## Major revisions

1. Reframe the central claim as: regime-conditioned recurrence and event-level evaluation can expose useful finite-sample dependence diagnostics in cyclic telemetry. Avoid wording that says the method has proved causal degradation detection.
2. Report every real-data result by independent unit: failure for MetroPT/MetroPT2 and vehicle for SCANIA. Include aggregate row counts only as computational scale.
3. Add a table showing, for every experiment, which split selected thresholds, run lengths, regimes, lag sets, dangerous-region radius and baseline hyperparameters.
4. Include negative controls: shuffled regimes, time-shifted alarms, shifted failure labels and missingness/noise perturbations. These are needed to test whether preprocessing or cyclic smoothness creates apparent clustering.
5. Add sensitivity surfaces for threshold, run length, dangerous-region radius and lag. A single best setting is not enough.
6. Distinguish classical EVT, dynamical-EVT motivation and empirical operational diagnostics in the methods section. Multivariate lag results must remain finite-sample empirical tests unless stronger theory is supplied.
7. Show event-matching tolerance ablations. A reviewer will suspect the protocol was tuned to favor the proposed method unless false alarms, duplicate alarms and lead times are stable across tolerances.
8. Report baseline failures and strong baseline wins. Do not suppress simple rules if they outperform the proposed method.

## Minor revisions

1. Replace any phrase implying "the system estimates whether an episode is beginning" with "the system scores evidence consistent with episode onset" unless prospective validation is shown.
2. Use "empirical extremal-index analogue" consistently for multivariate lag surfaces.
3. In all plots, annotate the independent unit and split used for model selection.
4. State that SPOT-style implementation follows the published algorithmic idea but is not a claim of bitwise equivalence with a reference package.
5. Explain how maintenance intervals are constructed and whether labels mark onset, repair completion or an interval of uncertainty.
6. In SCANIA, separate vehicle-level forecasting from readout-level anomaly scoring.

## Missing experiments

1. Leave-one-failure-out MetroPT case studies with all hyperparameters selected without the held-out failure.
2. MetroPT2 temporal replication with the same pre-registered settings used on MetroPT where feasible.
3. SCANIA vehicle-level validation with no vehicle leakage and uncertainty over vehicles, not rows.
4. Noise and missingness stress tests that preserve temporal order and maintenance boundaries.
5. Threshold and run-length sensitivity for univariate EVT, dangerous-region observables and fixed-run baselines.
6. Regime ablations: global threshold, rule regimes, hidden states, changepoints and shuffled regimes.
7. Dangerous-region ablations: engineering constraints, pre-failure prototypes, density level set, Mahalanobis transform and radius grid.
8. Multivariate lag ablation with nested lag selection and multiple-comparison correction.
9. Computational benchmark on the same hardware for all baselines and proposed variants.
10. Label-shift/null experiments: failure labels shifted outside plausible warning windows and component series time-shifted relative to failures.

## Claims that must be weakened

1. "Regime conditioning improves calibration" must become "regime conditioning improved or changed calibration under the reported splits" unless replicated across datasets and uncertainty intervals exclude no improvement.
2. "Lagged multivariate extreme patterns warn earlier" must become "selected lagged patterns are evaluated as empirical early-warning candidates."
3. "Dangerous region" must be described as a training-defined target set or engineering constraint, not an objectively known hazardous state unless domain evidence is provided.
4. "Dynamical EVT monitoring framework" is acceptable as motivation, but industrial claims should say "dynamical-EVT-inspired finite-sample diagnostics."
5. "External validation on SCANIA" must be narrowed to "vehicle-level operational validation on a structurally different dataset."
6. "Independent observations" must refer to failures, days, temporal blocks or vehicles, never raw seconds.

## Decision-changing tests or analyses

1. A leave-one-failure-out table where the proposed method improves duplicate-alarm burden or calibration without materially reducing recall for each MetroPT failure, with all choices made before the held-out failure.
2. A threshold/run-length/radius/lag stability grid showing the main conclusion survives a broad neighborhood of settings.
3. A regime-null experiment where shuffled or future-informed regimes do not explain the reported extremal-index separation.
4. A dangerous-region provenance audit listing every prototype source and proving none come from evaluated failures.
5. A vehicle-level SCANIA result with bootstrap intervals over vehicles and a simple rule baseline included.
6. A matching-protocol ablation showing conclusions are not reversed by reasonable early-warning windows and tolerances.
7. A missingness and smoothing stress test showing clustering is not created by preprocessing artifacts.
8. A baseline table where engineering rules, POT/GPD, SPOT-style thresholding, Isolation Forest, change point, autoencoder and conformal methods use the same feature windows and validation splits.

## Final recommendation

Recommendation: major revision before submission.

Confidence: high for the statistical concerns and medium for the implementation concerns. The repository now contains many of the required safeguards, but the paper should not be submitted until the real-data experiments, uncertainty units and claim language are aligned with the limitations above.
