# Page-by-page visual and scientific QA

Decision: submission ready.

The current PDF compiles cleanly, the main manuscript is internally consistent, and the previously listed study-base blockers now have generated artifacts. The software DOI remains venue-dependent: the repository contains citation metadata and a repository citation, but an external archive DOI must be created through the archive provider if a target venue requires it.

## Main Manuscript

| Page | Content inspected | Finding | Action |
|---:|---|---|---|
| 1 | Title, abstract, introduction start | Negative-result identity and numerical abstract present; margins OK | No action |
| 2 | Introduction and related work | No unsupported broad claim or float intrusion | No action |
| 3 | Method | Causal score target and independent-unit language present | No action |
| 4 | Leakage audit and simulation start | Leakage table fits page; simulation scope described as bounded validation | Updated simulation scope |
| 5 | Dataset role table | Dataset roles readable; timestamps not treated as independent units | No action |
| 6 | Protocol and simulation results | Compact stability table fits; claims remain bounded | No action |
| 7 | Broad simulation figure and failed-detection text | Flat failed-detection figure remains absent from main text | Full threshold grid stays in supplement |
| 8 | Event and non-event result tables | Event rows and non-event rows are separated and readable | No action |
| 9 | Root-cause table and real comparison setup | Evidence status and competing explanations visible | No action |
| 10 | Real event-level comparison table | Baselines and variants use the same held-out alarm policy | Added real comparison artifact |
| 11 | Variant and negative-control table | Target-region variants and negative controls are traceable | Added variant artifact |
| 12 | Score-stratification figures and metric-provenance table | Figures use score/rank semantics; metric provenance is real event-level | Removed synthetic smoke comparison from main text |
| 13 | Metric-collapse figure and method-scope table | Real event precision and false-alarm burden are visible; broader requested analyses show completed artifact status | Updated metric-collapse framing |
| 14 | Real timeline figure | Real MetroPT held-out timeline is visible and traceable | Replaced representative timeline |
| 15 | Discussion, conclusion, and availability | Claims remain tied to current evidence; repository cited by URL with no invented DOI | No action |
| 16 | References | Bibliography continuous; no floats after References | No action |

## Automated Checks

- Clean LaTeX rebuild: passed for main and supplement.
- Citation and cross-reference audit: passed through `dyn-evt check-paper`.
- Float-after-references audit: passed through PDF-text check.
- Duplicate-label audit: passed through manuscript checker.
- Placeholder and unsupported-claim audit: passed.
- Numerical-provenance audit: generated assets verified through `verify-paper-assets`.
- Submission package decision: `submission ready`, with zero unresolved blockers.
