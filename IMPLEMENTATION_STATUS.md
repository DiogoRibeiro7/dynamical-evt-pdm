# Implementation status

## Implemented and executable

- cyclic degradation simulation with normal operating regimes;
- IID Pareto, logistic target and lagged multivariate simulation systems;
- deterministic simulation-study runner with tidy Parquet outputs;
- logistic-map simulation utility;
- dangerous-region `-log(distance)` observable;
- training-provenance-checked dangerous-region construction and horizon-risk scoring;
- global and regime-conditioned empirical thresholds;
- auditable rule, unsupervised hidden-state and causal change-point regime definitions;
- regime occupancy, transition, duration and threshold-ablation reports;
- run declustering and cluster maxima;
- runs and Ferro-Segers intervals extremal-index estimators;
- empirical hitting and return-time utilities;
- lagged multivariate extreme-pattern detection;
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

- Python compilation: passed.
- Unit and smoke tests: 29 passed.
- Measured line/branch coverage: 83.19%, above the configured 80% gate.
- CLI smoke workflow: passed for simulation and analysis.

## Deferred paper-level work

- richer data dictionaries with verified units and source-to-canonical mappings;
- theoretical simulation targets with known periodic-point extremal indices;
- SPOT/DSPOT, autoencoder and conformal baselines;
- optimal event matching and early-warning-window utilities;
- nested temporal model selection;
- paper figures, tables and manuscript assets;
- external validation on downloaded SCANIA data.

These are specified in detail in `docs/implementation_backlog.md` and `docs/research_protocol.md`.

## Tooling note

Ruff and mypy are configured in `pyproject.toml` and GitHub Actions. They were not installed in the build container, so local static-analysis execution was not available during artifact generation. Syntax compilation and the complete pytest suite passed.
