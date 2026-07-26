# Page-by-page visual and scientific QA

Decision: submission ready.

## Main manuscript

| Page | Content inspected | Finding | Action |
|---:|---|---|---|
| 1 | Title, abstract, introduction start | Negative-result identity and numerical abstract present; margins OK | No action |
| 2 | Introduction and related work | No unsupported broad claim or float intrusion | No action |
| 3 | Method | Score target displayed cleanly; no probability claim for ranks | Reflowed target equation |
| 4 | Leakage audit and simulation start | Leakage table fits page; terminal statuses shown | No action |
| 5 | Dataset role table | Main table readable without resizebox | Replaced registry/status tables |
| 6 | Protocol and simulation results | Compact stability table fits; claims narrowed | No action |
| 7 | Simulation figure and failed-detection tables | Uniform matching heat map absent; failed-detection summary compact | Removed former Figure 4 |
| 8 | Industrial result and root-cause tables | Compact result table readable; root-cause explanations visible | Combined related metrics |
| 9 | Failed-detection figure and score-stratification text | No calibration diagonal; no frontier language | No action |
| 10 | Score-stratification figures and baseline table | Figures use score/rank semantics; baseline table fits | Redesigned generated plots |
| 11 | Timeline and discussion start | Timeline uses elapsed hours and separate panels | Replaced dense point timeline |
| 12 | Discussion, conclusion, availability start | Conclusion answers all research questions; no floats intrude | Expanded conclusion |
| 13 | Availability continuation and references | Bibliography continuous; no floats after References | Removed unnecessary clearpage |

## Automated checks

- Clean LaTeX rebuild: passed for main and supplement.
- Citation and cross-reference audit: passed through `dyn-evt check-paper`.
- Float-after-references audit: passed through PDF-text check.
- Duplicate-label audit: passed through manuscript checker.
- Placeholder and unsupported-claim audit: passed.
- Numerical-provenance audit: generated assets verified through `verify-paper-assets`.
