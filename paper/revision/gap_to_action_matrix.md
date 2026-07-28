# Gap-To-Action Matrix

| Prompt | Required action | Current revision action | Residual limitation |
| --- | --- | --- | --- |
| 21 | Select a coherent scientific identity | Selected negative-result identity and rewrote title, abstract, introduction, results, discussion, and conclusion | Claims remain evidence-bound rather than framed as superiority |
| 22 | Reframe hypotheses | Added versioned hypothesis registry tied to independent units | Some hypotheses remain exploratory because independent event counts are small |
| 23 | Specify method and leakage controls | Rewrote method with exact state, scaling, target-region, threshold, declustering, alarm, calibration, and leakage contract | A dedicated experiment-level leakage-audit module should be added for every runner |
| 24 | Complete simulation validation | Added bounded broad Monte Carlo validation and rewired paper builds to `artifacts/simulation_study_broad.parquet` | Large 100-rep production grid remains optional |
| 25 | Fair baselines and per-failure evaluation | Added real MetroPT/MetroPT2 event-level comparison and a traceable MetroPT failure timeline | Only one held-out failure is available per compressor dataset |
| 26 | Explain negative results | Added root-cause interpretation by dataset role plus target-region variants and negative controls | Explanations remain diagnostic, not causal proof |
| 27 | Rebuild calibration and uncertainty | Replaced probability language with score-stratification language | Full calibration requires enough positive independent calibration units |
| 28 | Restructure dataset scope | Assigned dataset roles and added manuscript checks against dataset-count overclaims | Event datasets and non-event datasets are reported in separate tables |
| 29 | Rewrite results figures and tables | Results now answer scientific questions; flat zero-recall figure was removed from the main paper | Full threshold grid remains in the supplement |
| 30 | Rewrite manuscript and review | Rewrote main narrative and refreshed submission/QA artifacts after the expanded experiments | Earlier package was ready under the Prompt 40/50 scope; Prompt 51-60 adds stricter unresolved gates |
| 51-60 | Advanced revision contract | Built high-rep Monte Carlo, complete event baselines, transferability, matched controls, failure decomposition and timeline reconciliation artifacts; kept DOI archive as an external gate | Not submission ready until the software archive DOI exists |
