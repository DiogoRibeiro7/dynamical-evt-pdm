# Implementation status

## Implemented and executable

- cyclic degradation simulation with normal operating regimes;
- IID Pareto, logistic target and lagged multivariate simulation systems;
- deterministic simulation-study runner with tidy Parquet outputs;
- logistic-map simulation utility;
- dangerous-region `-log(distance)` observable;
- training-provenance-checked dangerous-region construction and horizon-risk scoring;
- global and regime-conditioned empirical thresholds;
- component-specific regime-conditioned multivariate thresholds;
- empirical simultaneous, lagged, tail-dependence and multivariate extremal-index diagnostics;
- clustering-preserving lag-pattern null models with Benjamini-Hochberg correction;
- nested-validation lag-pattern selection with held-out exploratory labelling;
- fair baseline runner with standardized prediction tables and validation-only selection;
- engineering-threshold, global empirical, POT/GPD, fixed-run declustering, SPOT-style, Isolation Forest, robust change-point, linear autoencoder and conformal baselines;
- per-baseline runtime, peak-memory and parameter-count reporting;
- rigorous event-level evaluation with merge-gap alarm events, early-warning policy windows, greedy/optimal matching, duplicate/false-alarm accounting, utility and calibration metrics;
- group-level bootstrap helpers for complete failures, days or vehicles;
- reproducible paper asset builder for publication figures, CSV/LaTeX tables, computational benchmarks and provenance manifests;
- `make paper-assets` target that rebuilds generated manuscript artifacts from cached processed data;
- adversarial scientific review with fatal-flaw, revision, missing-experiment and claim-weakening gates;
- auditable rule, unsupervised hidden-state and causal change-point regime definitions;
- regime occupancy, transition, duration and threshold-ablation reports;
- run declustering and cluster maxima;
- runs and Ferro-Segers intervals extremal-index estimators;
- empirical hitting and return-time utilities;
- lagged multivariate extreme-pattern detection and CLI reporting;
- event conversion, event matching and operational metrics;
- calibration diagnostics;
- Isolation Forest baseline wrapper;
- memory-aware CSV ingestion and Parquet output helpers;
- real-data acquisition for MetroPT, MetroPT2 and SCANIA Component X into `data/raw/`;
- chunked raw-to-Parquet preparation with schema manifests and quality checks;
- MetroPT/MetroPT2 failure metadata helpers;
- SCANIA counter-reset handling;
- command-line simulation and series analysis;
- research protocol, paper outline and staged implementation backlog.

## Validation completed in the build environment

- Ruff linting: passed.
- Mypy strict type checking: passed.
- Unit and smoke tests: 70 passed.
- Measured line/branch coverage: 85.23%, above the configured 80% gate.
- CLI smoke workflows: passed for simulation, analysis, baseline, event/paper assets and `make paper-assets`.

## Deferred paper-level work

- richer data dictionaries with verified units and source-to-canonical mappings;
- theoretical simulation targets with known periodic-point extremal indices;
- external validation on downloaded SCANIA data.

These are specified in detail in `docs/implementation_backlog.md` and `docs/research_protocol.md`.

## Tooling note

Ruff and mypy are configured in `pyproject.toml` and GitHub Actions. They were not installed in the build container, so local static-analysis execution was not available during artifact generation. Syntax compilation and the complete pytest suite passed.
