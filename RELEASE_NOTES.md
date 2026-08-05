# Release Notes

## 0.3.2 - 2026-08-05

Manuscript presentation. No code or results change.

- The abstract is condensed from 604 words across four paragraphs to 270 words in one,
  and now ends on page one. All eight elements required by the revision contract are
  retained, including the 0.45 warning-window separation and the 0.99 chance-match
  probability, which carry the paper's central finding. What was cut is detail the
  results section already reports at length.
- The manuscript and the software citation name this release as the state the reported
  numbers correspond to.

The `v0.3.2` snapshot is archived on Zenodo. Version DOI
[10.5281/zenodo.21803769](https://doi.org/10.5281/zenodo.21803769) names this snapshot
and is what the manuscript cites; concept DOI
[10.5281/zenodo.21803768](https://doi.org/10.5281/zenodo.21803768) names the software
across all versions and resolves to the latest. A deposit cannot contain its own
identifier, so the manuscript inside the archived `v0.3.2` tarball predates the DOI
minted for it; the DOI text is on `main` from this point forward.

## 0.3.1 - 2026-08-04

Metadata and availability release. No scientific results change.

- The repository is public. The Code and Data Availability section previously described
  it as private and stated that a reader could not retrieve the code; that is now
  resolved, and the section states that the results correspond to the tagged release.
- Author identity recorded across the citation metadata and the manuscript title page:
  ORCID 0009-0001-2022-7072, dfr@esmad.ipp.pt, ESMAD - School of Media Arts and Design,
  Polytechnic of Porto.
- The archival DOI remains unminted and is the one outstanding item. Now that the
  repository is public both the automated Zenodo integration and a direct upload of a
  release archive are available; either mints a DOI for the deposited snapshot. This is
  an account action outside the build, so the paper still does not claim a persistent
  software identifier.

## 0.3.0 - 2026-08-03

Fifth-review revision. This release changes a headline scientific conclusion, so it is a
minor version rather than a patch.

- **The MetroPT detection is reclassified as a score-layer failure.** A new
  warning-window rank separation measures how well each score ranks the hour before
  onset against the rest of the test split. The registered recurrence score separates at
  0.46 on MetroPT, below the chance value of 0.50, while raising 2,631 alarm episodes; a
  detector alarming that often matches the failure with probability 0.99 carrying no
  signal. Its recall of 1.0 there is not evidence of early warning. Fourteen of the
  nineteen MetroPT methods are in the same position. MetroPT2 separates at 0.86 and its
  methods do fail on alarm conversion, as previously reported.
- The rank separation uses midranks. An earlier form of this measure broke ties by
  position, which biases the statistic when a score is constant over long stretches, as
  industrial telemetry is during idle operation; a constant score returned something
  other than the chance value of 0.50 purely because the warning window sits at one end
  of the series.
- Root-cause analysis is now per dataset-method, with evidence, a competing explanation
  and a confidence level on every row, over an eight-layer taxonomy used consistently by
  the decomposition and root-cause tables.
- The failure timeline is reported at two scales. The local window around the MetroPT
  failure holds 23 alarm episodes; the full test split holds 2,631. Reconciliation tests
  tie the figure to the event tables.
- Cross-dataset target-region transfer replaces the previous cross-dataset comparison,
  under four protocols in both directions, with schema compatibility checked first.
- Matched controls are conditioned on detection, with a predeclared joint utility.
- The decomposition separates detector evidence from ground truth and carries raw
  exceedances, which had been reported as post-declustering onsets for four methods.
- **Fixed: the supplementary 38-row benchmark had been silently dropping rows from the
  published PDF.** A table float that overruns its page does not error; LaTeX warns and
  lets the surplus fall off the bottom. The writer now selects `longtable` by row count,
  and a test fails the build on any float overflow.
- Main text focused to five tables; non-event stress tests moved to the supplement.
- Twelve bibliography entries added; Related Work reorganised; false-alarm rates compared
  against alarm-management standards.
- Supplementary table numbers now resolve through `xr` rather than being hardcoded.

The repository is private and no archival DOI has been minted, so there is currently no
public code-availability route: the address cited in the manuscript does not resolve for
a reader. Making the repository public and depositing a release snapshot are two separate
steps, and neither is performed by the build. The automated GitHub-Zenodo integration is
unavailable while the repository is private, but a DOI does not require publishing the
development history; uploading a release archive directly to an archive service mints one
for that snapshot. The manuscript states both gaps rather than claiming public code or a
persistent identifier.

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
