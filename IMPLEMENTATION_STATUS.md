# Implementation Status

## Implemented And Executable

- deterministic cyclic degradation simulation and finite-sample simulation studies;
- IID, logistic-map and lagged multivariate simulation systems;
- univariate EVT diagnostics, run declustering, cluster maxima and extremal-index estimators;
- empirical hitting, return-time and dangerous-region recurrence utilities;
- training-provenance-checked dangerous-region construction and horizon-risk scoring;
- rule-based, hidden-state and changepoint operating-regime definitions;
- global and regime-conditioned thresholds with threshold/run-length sensitivity assets;
- lagged multivariate extreme-pattern diagnostics with clustering-preserving nulls;
- fair baseline runner covering engineering thresholds, empirical thresholds, POT/GPD, fixed-run declustering, SPOT-style thresholding, Isolation Forest, robust change point, linear autoencoder and conformal anomaly detection;
- rigorous event-level evaluation with alarm merge gaps, early-warning windows, greedy/optimal matching, duplicate alarms, false alarms, lead time, utility and calibration diagnostics;
- real-data acquisition for MetroPT, MetroPT2 and SCANIA Component X into ignored `data/raw/`;
- chunked raw-to-Parquet preparation with schema manifests, quality checks and checksum validation;
- real-data matrix verification, evidence-scope summaries and conservative industrial diagnostic result artifacts;
- paper asset builder for figures, CSV tables, LaTeX tables, claim ledger, provenance, benchmarks and result macros;
- manuscript, supplement and submission-package assembly without copying raw third-party datasets;
- generated adversarial review, revision matrix, submission statements and final readiness decision.

## Primary Commands

```bash
make fetch-data
make prepare-data
make real-data-matrix
make paper-assets
make submission-package
```

For a CI-sized synthetic workflow:

```bash
make end-to-end-smoke
```

For only the industrial result artifacts after data preparation:

```bash
make industrial-results
```

## Current Validation

- Poetry strict metadata check: passed.
- Package wheel and source distribution build: passed.
- Ruff formatting: passed.
- Ruff linting: passed.
- Mypy strict type checking: passed.
- Pytest with configured coverage gate: 93 passed.
- Measured coverage: 82.87%, above the configured 80% gate.
- `make real-data-matrix`: succeeded locally with prepared MetroPT, MetroPT2 and SCANIA Component X data.
- `make submission-package`: succeeded locally with decision `submission ready` and zero blockers.

## Scientific Boundaries

- The package supports a reproducible diagnostic workflow for the current datasets and workflows; broader industrial claims should stay tied to the generated evidence-scope and result artifacts.
- MetroPT and MetroPT2 contain few independent failure episodes; raw timestamp rows are not treated as independent evidence.
- SCANIA Component X is evaluated as a vehicle-level repair-risk resource, not as a direct compressor-event replication.
- Calibration and early-warning artifacts are diagnostics unless prospective calibrated probability evidence is added.
- Weak, failed or non-estimable real-data results are preserved as part of the evidence package.
