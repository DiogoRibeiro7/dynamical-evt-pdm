# Page-by-page visual and scientific QA

Decision: not submission ready.

The current PDF compiles cleanly and the main manuscript is internally consistent, but the broader review-requested study base is incomplete. The remaining blockers are full Monte Carlo validation, real event-level baseline comparison, real target-region variant/negative-control matrices, a real failure-specific timeline, and an external DOI-backed software archive.

## Main Manuscript

| Page | Content inspected | Finding | Action |
|---:|---|---|---|
| 1 | Title, abstract, introduction start | Negative-result identity and numerical abstract present; margins OK | No action |
| 2 | Introduction and related work | No unsupported broad claim or float intrusion | No action |
| 3 | Method | Causal score target and independent-unit language present | Updated stale table reference |
| 4 | Leakage audit and simulation start | Leakage table fits page; simulation claim remains limited | No action |
| 5 | Dataset role table | Dataset roles readable; timestamps not treated as independent units | No action |
| 6 | Protocol and simulation results | Compact stability table fits; claims narrowed | No action |
| 7 | Limited simulation figure and failed-detection text | Flat failed-detection figure removed from main text | Moved full threshold grid to supplement |
| 8 | Event and non-event result tables | Event rows and non-event rows are separated and readable | Shortened generated headers |
| 9 | Root-cause table and calibration text | Evidence status and competing explanations visible | Added root-cause uncertainty columns |
| 10 | Score-stratification figures and metric-provenance text | Figures use score/rank semantics; no calibration diagonal | No action |
| 11 | Metric provenance and method-scope tables | Pointwise smoke metrics are separated from event metrics; blockers visible | Shortened provenance table |
| 12 | Metric-collapse figure and timeline traceability | Representative timeline status visible; not used as empirical evidence | Shortened timeline audit table |
| 13 | Discussion and conclusion | Claims remain tied to current evidence | No action |
| 14 | Availability and references start | Repository cited by URL; no invented DOI | No action |
| 15 | References continuation | Bibliography continuous; no floats after References | No action |

## Automated Checks

- Clean LaTeX rebuild: passed for main and supplement.
- Citation and cross-reference audit: passed through `dyn-evt check-paper`.
- Float-after-references audit: passed through PDF-text check.
- Duplicate-label audit: passed through manuscript checker.
- Placeholder and unsupported-claim audit: passed.
- Numerical-provenance audit: generated assets verified through `verify-paper-assets`.
- Submission package decision: `not submission ready`, with four unresolved blockers.
