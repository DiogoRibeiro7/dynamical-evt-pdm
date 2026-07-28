# Dynamical EVT for Predictive Maintenance

Research repository for **regime-conditioned, cluster-aware predictive maintenance using dynamical extreme value theory (EVT)**.

The project translates ideas from dynamical EVT—extremal indices, hitting and return times, recurrence near periodic states, and multivariate lagged extremes—into an event-level monitoring framework for cyclic industrial systems.

## Research question

> Does dynamical recurrence information distinguish ordinary cyclic extremes from persistent pre-failure episodes, while reducing duplicate alarms and preserving useful warning lead time?

## Core hypotheses

1. **Fault-related extremes cluster more strongly** than normal operating extremes, so their extremal index is lower.
2. **Dynamical declustering reduces duplicate alarms** without materially reducing failure-episode recall.
3. **Lagged multivariate extreme patterns warn earlier** than simultaneous thresholds and univariate EVT.
4. **Regime conditioning improves calibration** by avoiding thresholds that mix incompatible operating states.

## Scope

The repository contains:

- deterministic and noisy simulation systems with known recurrence structure;
- univariate peaks-over-threshold and extremal-index estimators;
- regime-conditioned thresholds and cluster extraction;
- hitting-time and dangerous-region observables;
- lagged multivariate extreme signatures;
- event-level predictive-maintenance metrics;
- adapters for MetroPT, MetroPT2, SCANIA Component X, hydraulic-system, and SECOM datasets;
- real-data acquisition commands for all five public datasets;
- real-data verification, evidence-scope and industrial diagnostic result artifacts;
- classical anomaly-detection baselines;
- reproducible experiment configurations;
- a paper protocol, claim ledger, and submission package builder.

## Why this is not a standard anomaly-detection project

Pointwise anomaly scores can produce hundreds of alarms during one physical event. Here, the primary unit of evaluation is the **extreme episode**. The system estimates whether an episode is beginning, how severe and persistent it is, and how soon the trajectory may enter a dangerous region.

## Repository layout

```text
configs/                 Reproducible experiment configurations
data/                    Local raw/interim/processed data; never committed
notebooks/               Research walkthroughs
src/dyn_evt_pdm/         Typed Python package
tests/                   Unit tests for statistical and event logic
docs/                    Research protocol, paper outline, claims and data notes
```

## Installation

Python 3.11 or 3.12 and Poetry 2.2 or newer are recommended.

```bash
poetry install --with dev
poetry run pre-commit install
```

## Real Data Acquisition

Fetch the public raw datasets into the ignored `data/raw/` tree:

```bash
poetry run dyn-evt fetch-data --dataset all
```

You can fetch one dataset at a time with `--dataset metropt`, `--dataset metropt2`, `--dataset scania_component_x`, `--dataset hydraulic_systems`, or `--dataset secom`. Each run writes a local `manifest.json` with source URLs, byte counts, SHA-256 hashes, and any published checksum.

Prepare the raw files into processed Parquet parts:

```bash
poetry run dyn-evt prepare-metropt
poetry run dyn-evt prepare-metropt2
poetry run dyn-evt prepare-scania
poetry run dyn-evt prepare-hydraulic-systems
poetry run dyn-evt prepare-secom
```

Verify the prepared datasets and generate conservative industrial diagnostic results:

```bash
make real-data-matrix
```

For only the industrial result tables, after preparation:

```bash
make industrial-results
```

These commands write ignored artifacts under `artifacts/real_data_matrix/`, including dataset verification, an experiment manifest, evidence-scope summaries, and dataset-specific diagnostic result summaries.

## Smoke Experiment

Run the complete synthetic smoke workflow:

```bash
make end-to-end-smoke
```

Generate a cyclic system with an injected degradation episode:

```bash
poetry run dyn-evt simulate \
  --output data/processed/synthetic_cyclic.csv \
  --n-steps 20000 \
  --seed 42
```

Estimate regime-conditioned thresholds, clusters, and extremal indices:

```bash
poetry run dyn-evt analyse-series \
  --input data/processed/synthetic_cyclic.csv \
  --value-column observable \
  --regime-column regime \
  --quantile 0.98 \
  --run-length 10 \
  --output artifacts/synthetic_summary.json
```

Fit structured univariate EVT diagnostics for a scalar column:

```bash
poetry run dyn-evt analyse-univariate \
  --input data/processed/synthetic_cyclic.csv \
  --value-column observable \
  --quantile 0.98 \
  --run-length 10 \
  --output artifacts/synthetic_univariate_evt.json
```

Compare operating-regime definitions and thresholds:

```bash
poetry run dyn-evt analyse-regimes \
  --input data/processed/synthetic_cyclic.csv \
  --value-column observable \
  --method rules \
  --current-column current \
  --pressure-column pressure \
  --output artifacts/synthetic_regime_report.json
```

Score recurrence to a training-defined dangerous region:

```bash
poetry run dyn-evt analyse-dangerous-region \
  --input data/processed/synthetic_cyclic.csv \
  --state-columns pressure,current,temperature \
  --target-column is_fault \
  --output artifacts/synthetic_dangerous_region.parquet \
  --report-output artifacts/synthetic_dangerous_region.json
```

Analyse simultaneous and lagged multivariate extreme patterns with train-only thresholds:

```bash
poetry run dyn-evt analyse-multivariate-extremes \
  --input data/processed/synthetic_cyclic.csv \
  --component-columns current,temperature,pressure \
  --regime-column regime \
  --split-column split \
  --max-lag 10 \
  --output artifacts/synthetic_multivariate_evt.json
```

Run the fair baseline suite with one shared causal feature window and split-aware selection:

```bash
poetry run dyn-evt run-baselines \
  --input data/processed/synthetic_cyclic.csv \
  --feature-columns pressure,current,temperature \
  --timestamp-column time \
  --regime-column regime \
  --split-column split \
  --target-column is_fault \
  --window-size 2 \
  --output artifacts/synthetic_baseline_predictions.parquet \
  --metadata-output artifacts/synthetic_baseline_metadata.json
```

Run tests:

```bash
poetry run pytest
```

Run the same local quality gate used by CI:

```bash
make check
```

Run the finite-sample simulation-study smoke grid:

```bash
poetry run dyn-evt run-simulation-study \
  --smoke \
  --output artifacts/simulation_study_smoke.parquet
```

Rebuild manuscript figures, CSV tables, LaTeX tables and provenance manifests:

```bash
make paper-assets
```

The generated assets are written under `reports/paper/` and each run records the experiment ID, Git commit hash and configuration hash in `asset_manifest.json`.

Build and audit the complete manuscript package:

```bash
make submission-package
```

This compiles the manuscript and supplement, verifies generated assets and claim references, and assembles `reports/submission/` without redistributing raw third-party datasets.

## Data

Large datasets are not committed. Acquisition notes and expected schemas are in [`data/README.md`](data/README.md) and dataset configurations are under [`configs/data`](configs/data).

- **MetroPT**: 1 Hz APU telemetry, roughly 10.98 million observations and three reported catastrophic failures.
- **MetroPT2**: 1 Hz APU telemetry, roughly 7.12 million observations and two reported failures.
- **SCANIA Component X**: fleet-level operational readouts, repair information and vehicle specifications.
- **Hydraulic Systems**: UCI hydraulic test-rig cycles with multi-rate sensor traces and component-condition labels.
- **SECOM**: UCI semiconductor process measurements with pass/fail yield labels.

## Scientific guardrails

- MetroPT has too few independent failures to support broad superiority claims by itself.
- Thresholds and model choices are fit on training periods only.
- Validation is temporal and event-level; random row splitting is prohibited.
- Simulations establish estimator behaviour under known ground truth.
- Real datasets broaden the evaluated estimands while keeping generalization claims evidence-bound.
- Simple engineering thresholds remain mandatory baselines.

See [`docs/statistical_claims.md`](docs/statistical_claims.md) before interpreting results.
The hypothesis-to-artifact implementation map is in [`docs/implementation_plan.md`](docs/implementation_plan.md).
The adversarial readiness review is in [`docs/adversarial_review.md`](docs/adversarial_review.md).

## Primary references

- Freitas, Freitas & Todd, *Hitting Time Statistics and Extreme Value Theory*.
- Freitas, Freitas & Todd, *Extremal Index, Hitting Time Statistics and Periodicity*.
- Faranda et al., *Extreme Value Statistics for Dynamical Systems with Noise*.
- Aimino, Freitas, Freitas & Todd, *Multivariate Extreme Values for Dynamical Systems*.
- Veloso et al., *The MetroPT Dataset for Predictive Maintenance*.
- Kharazian, Lindgren & Andersson Reyna, *SCANIA Component X Dataset*.
- Helwig, Pignanelli & Schutze, *Condition Monitoring of Hydraulic Systems*.
- McCann & Johnston, *SECOM*.

Full links and the role of each source are documented in [`docs/literature_map.md`](docs/literature_map.md).

## Status

The repository is a **complete research package that is not yet submission ready**. Every local scientific and quality gate passes once the public raw datasets have been fetched and prepared, but the generated readiness decision in `reports/submission/final_decision.md` remains `not submission ready` while the external DOI-backed software archive is missing. The release and archive runbook is in [`docs/release_archive.md`](docs/release_archive.md). Raw third-party datasets and generated artifacts remain ignored and are rebuilt from the documented commands.
