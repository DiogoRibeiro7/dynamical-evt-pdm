# Revision Identity Decision Memo

## Decision

Primary identity: negative-result paper.

The current evidence does not support a methodological-performance paper. The simulation artifact has been broadened enough for bounded finite-sample validation, and the real-data baseline and variant comparisons are now generated artifacts, but the event-level evidence still has too few independent compressor failures and the strongest empirical findings remain weak or null. A software/reproducibility identity is also possible, but the revised manuscript should not make procedural auditability the central scientific result. The defensible paper identity is therefore:

> Under what conditions do dynamical-EVT-inspired clustering diagnostics fail to translate extreme recurrence into useful public-data industrial early warning?

## Decision Table

| Candidate identity | Minimum evidence required | Evidence currently available | Missing evidence | Feasibility | Overclaiming risk | Likely contribution | Venue class | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methodological paper | Full estimator validation, fair baselines, uncertainty, and positive empirical value | Broad finite-sample diagnostics, sensitivity assets, fair baseline artifacts, weak compressor event results | Prospective or larger independent-failure evidence with useful alarm burden | Medium to high effort | High | New EVT method only if future experiments succeed | Methods/statistical ML | Reject for current revision |
| Negative-result paper | Honest weak/null results, failure-mode analysis, valid independent units, claim restrictions | Zero-recall threshold grid, low-precision compressor event detection, F1=0 condition diagnostics, heterogeneous estimands, target-region variants and negative controls | Additional independent failures would broaden uncertainty estimates | Feasible | Low if claims stay narrow | Boundary conditions for using dynamical EVT in public industrial data | Applied ML, reliability, reproducibility, negative results | Select |
| Software/reproducibility paper | Architecture/API comparison, auditability, package design evidence | Strong build checks, manifests, data acquisition, claim ledger | Package comparison and user-facing API study | Feasible | Medium if presented as science | Reproducible evaluation package | Software/research tools | Secondary |

## Revised Research Questions

1. Does regime conditioning stabilise extremal-index diagnostics relative to global thresholding?
2. Does declustering reduce duplicate alarm episodes without destroying event recall or warning lead time?
3. Does the registered robust extreme-score diagnostic add operational value after thresholding, declustering, and episode conversion?
4. Which mechanisms explain failure: target-region misspecification, regime mixture, insufficient independent failures, non-extreme fault signatures, label ambiguity, calibration shift, or alarm conversion?

## Claims Removed From Main Narrative

- Dataset-count breadth as evidence of generality.
- Repository execution as scientific success.
- Claim-ledger counts or hashes in the abstract.
- Probability-calibration language for ranked anomaly scores.
- Useful lead-time frontier language when recall is zero.

## Dataset Roles

| Dataset | Role |
| --- | --- |
| MetroPT | Primary event-level case study |
| MetroPT2 | Primary event-level replication case study |
| SCANIA Component X | External vehicle-level stress test |
| UCI Hydraulic Systems | Supplementary condition-monitoring demonstration |
| UCI SECOM | Supplementary condition-monitoring demonstration |
