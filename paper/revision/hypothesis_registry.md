# Versioned Hypothesis Registry

Registry version: `2026-07-26-negative-result-revision`

| ID | Statement | Estimand | Unit | Dataset role | Comparator | Direction | Practical threshold | Uncertainty | Decision rule | Status | Section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | Regime-conditioned thresholds reduce extremal-index sensitivity across threshold quantiles compared with global thresholding. | Variation of extremal-index estimates over threshold/run-length grid | Operating interval or simulation replicate | Simulation and compressor telemetry | Global thresholding | Lower sensitivity | At least 10% lower grid dispersion | Replicate or interval bootstrap where available | Supported only if dispersion reduction exceeds threshold without higher failure rate | Exploratory | Results |
| H2 | Declustering reduces duplicate alarm episodes without reducing event recall by more than 5 percentage points. | Duplicate alarms per failure and event recall | Failure episode | MetroPT, MetroPT2 | No declustering or fixed-run declustering | Fewer duplicates with preserved recall | Duplicate alarms down at least 25%; recall drop no more than 5 pp | Failure-level exact limitation because independent events are sparse | Supported only if both criteria hold | Exploratory | Results |
| H3 | The dangerous-region observable improves warning utility relative to marginal sensor thresholds. | Event utility, false alarms per operating day, warning lead time | Failure episode or operating day | MetroPT, MetroPT2 | Best marginal-sensor threshold | Higher utility at fixed burden | Positive utility difference with nonzero recall | Failure-level exact limitation | Supported only if recall is nonzero and false-alarm burden is acceptable | Exploratory | Results |
| H4 | Public industrial datasets with few independent failures do not support broad predictive superiority claims. | Presence of sufficient independent units and coherent estimands | Failure, vehicle, cycle, or wafer | All registered roles | Required evidence standard | Boundary condition | Independent units and estimands must be adequate for the claim | Dataset-role audit | Supported when evidence is insufficient or estimands differ | Confirmatory restriction | Discussion |

## Score Semantics

Current model outputs are anomaly or recurrence scores. They are not treated as calibrated probabilities unless a separate calibration partition contains enough positive independent units and the score-to-probability map is evaluated on a held-out test partition.
