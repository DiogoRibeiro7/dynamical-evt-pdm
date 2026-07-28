# Page-by-page visual and scientific QA

Decision: not submission ready.

The previously generated PDF remains available, and the main manuscript sources are internally consistent under the stricter Prompt 61-70 contract. The local scientific artifacts are now generated, but the package is not submission ready because a DOI-backed external software archive has not been created. A fresh LaTeX rebuild was attempted in this environment and blocked by MiKTeX setup, so fresh PDF visual readiness is not claimed here.

## Main Manuscript

| Page | Content inspected | Finding | Action |
|---:|---|---|---|
| 1 | Title, abstract, introduction start | Negative-result identity and numerical abstract present; margins OK | No action |
| 2 | Introduction and related work | No unsupported broad claim or float intrusion | No action |
| 3 | Method | Causal score target and independent-unit language present | No action |
| 4 | Leakage audit and simulation start | Leakage table fit in the previously generated PDF; simulation scope now described as high-replication bounded validation | Updated simulation scope |
| 5 | Dataset role table | Dataset roles readable; timestamps not treated as independent units | No action |
| 6 | Protocol and simulation results | Compact stability table fits; claims remain bounded | No action |
| 7 | Broad simulation figure and failed-detection text | Flat failed-detection figure remains absent from main text | Full threshold grid stays in supplement |
| 8 | Event and non-event result tables | Event rows and non-event rows are separated and readable | No action |
| 9 | Root-cause table and real comparison setup | Evidence status and competing explanations visible | No action |
| 10 | Real event-level comparison table | Compact method comparison uses the same held-out alarm policy | Added real comparison artifact |
| 11 | Transferability and matched-control tables | Target-region divergence and controls are now central scientific evidence | Added transfer/control artifacts |
| 12 | Score-threshold-alarm decomposition | Incompatible precision/FA-day plot has been replaced by count flow | Rebuilt figure from decomposition artifact |
| 13 | Real timeline figure | Real MetroPT held-out timeline is visible in the previous PDF and traceable in source artifacts | Replaced representative timeline |
| 14 | Discussion, conclusion, and availability | Claims remain tied to current evidence; repository cited by URL with no invented DOI | No action |
| 15 | References | Bibliography continuous in the previous PDF; fresh rebuild blocked by MiKTeX setup | External TeX setup required |

## Automated Checks

- Clean LaTeX rebuild: blocked by local MiKTeX setup before compilation.
- Citation and cross-reference audit: passed through `dyn-evt check-paper`.
- Float-after-references audit: passed through PDF-text check.
- Duplicate-label audit: passed through manuscript checker.
- Placeholder and unsupported-claim audit: passed.
- Numerical-provenance audit: generated assets verified through `verify-paper-assets`.
- Submission package decision under the Prompt 61-70 contract: `not submission ready`, with only the DOI-backed external archive blocker unresolved.
