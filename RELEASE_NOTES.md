# Release Notes

## 0.2.2 - 2026-07-28

This release candidate corrects a stale manuscript build. The `paper/main.pdf` shipped in `0.2.0` and `0.2.1` was compiled before the final regeneration of the `reports/paper` assets, so its matched-negative-control table reported superseded values. Most consequentially, the MetroPT2 prototype-permutation control was typeset as `lower alarm burden than controls` when the generating artifacts report `not lower burden`.

- Rebuilt `paper/main.pdf` and `paper/supplement/supplement.pdf` from regenerated assets; both are byte-reproducible across clean rebuilds on the same toolchain.
- Corrected the MetroPT2 matched-negative-control rows: prototype permutation is `not lower burden` (control median 8.01, 95% interval 7.53-8.44, percentile 1), random occupancy control median is 8.01 rather than 0.637, and the rare-region, label-permutation and time-shifted intervals are restated.
- Abstract, Results narrative, and all other claims are unaffected; the reported headline figures already matched the generating artifacts.
- `paper/supplement/supplement.pdf` is now tracked and attached to the release alongside the manuscript.
- Documentation status text, coverage figure, and release runbook preconditions corrected.

No external software archive DOI is recorded for this release candidate. Do not cite a DOI until a Zenodo or equivalent archive record exists.

## 0.2.1 - 2026-07-28

This release candidate is DOI-archive ready: it adds `.zenodo.json` to the tagged tree and keeps the Prompt 61-70 scientific artifacts from `0.2.0`.

No external software archive DOI is recorded for this release candidate. Do not cite a DOI until a Zenodo or equivalent archive record exists.

## 0.2.0 - 2026-07-28

This release candidate contains the Prompt 61-70 research package.

- High-replication Monte Carlo artifact with 7,000 replicate rows: 500 repetitions across fourteen process families and six estimator diagnostics.
- Complete held-out event-level baseline matrix for MetroPT and MetroPT2.
- Target-region transferability, score-threshold-alarm decomposition, and local/global timeline reconciliation artifacts.
- Matched negative-control experiments with 6,000 draw-level rows: 500 draws for each dataset and control family.
- Rewritten Results, Discussion, Conclusion, QA, and response-to-review materials around the negative result and MetroPT/MetroPT2 target-region divergence.
- Submission package now reports only the external DOI-backed archive as unresolved.

No external software archive DOI is recorded for this release candidate. Do not cite a DOI until a Zenodo or equivalent archive record exists.

## 0.1.0 - 2026-07-26

This repository contains the reproducible research package for the current dynamical-EVT predictive-maintenance study.

- Real-data acquisition, preparation, and evidence-scope records for MetroPT, MetroPT2, SCANIA Component X, UCI Hydraulic Systems, and UCI SECOM.
- Event-level alarm evaluation for the compressor datasets and separate non-event diagnostics for cycle, vehicle, and wafer targets.
- Generated manuscript assets, provenance manifests, paper checks, and submission-package tooling.
- Baseline-runner smoke coverage for the shared causal prediction contract.

No external software archive DOI is recorded for this release. Cite the repository URL unless a DOI-backed archive is created separately.
