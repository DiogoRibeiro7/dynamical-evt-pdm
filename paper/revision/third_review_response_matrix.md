# Prompt 51-60 response matrix

| Issue ID | Issue summary | Action | Artifact gate | Status | Residual limitation |
|---|---|---|---|---|---|
| SR3-01 | Monte Carlo repetitions too low | Built high-replication simulation artifact with 500 repetitions across the eight registered systems | `artifacts/simulation_study_high_rep.parquet.manifest.json` | Closed | High-replication grid is representative; the broader grid remains in `simulation_study_broad.parquet` |
| SR3-02 | Event-level baseline family incomplete | Regenerated event-level comparison with every declared baseline method for MetroPT and MetroPT2 | `artifacts/real_data_matrix/event_baseline_comparison.csv` | Closed | Compressor evidence still has one held-out failure event per dataset |
| SR3-03 | Target-region heterogeneity underanalysed | Generated target-region transferability matrix from the MetroPT/MetroPT2 event rows | `artifacts/real_data_matrix/target_region_transferability.csv` | Closed | Cross-dataset rows are descriptive and do not create additional independent failures |
| SR3-04 | Negative controls underspecified | Generated matched negative-control rows under the same train/test event policy | `artifacts/real_data_matrix/matched_negative_controls.csv` | Closed | Negative-control evidence remains limited by the number of compressor failure events |
| SR3-05 | Root cause conflates score, threshold and alarm conversion | Generated score-threshold-alarm decomposition rows for target-region methods | `artifacts/real_data_matrix/score_threshold_alarm_decomposition.csv` | Closed | Root-cause interpretation remains diagnostic, not causal proof |
| SR3-06 | Invalid incomparable metric figure | Kept main figure tied to real event-level metrics and removed audit-table clutter from Results | `paper/sections/07_results.tex`; `reports/paper/figures/metric_collapse_decomposition.png` | Closed | Supplement can still carry runner checks |
| SR3-07 | Local timeline conflicts with global alarm counts | Generated local/global timeline reconciliation rows | `artifacts/real_data_matrix/timeline_reconciliation.csv` | Closed | Timeline remains one representative held-out MetroPT failure window |
| SR3-08 | Audit tables remain in main Results | Removed metric-provenance, method-scope and timeline-traceability table inputs from main Results | `paper/sections/07_results.tex` | Closed | Audit artifacts remain package material rather than main Results content |
| SR3-09 | Abstract and conclusion need strongest target-region result | Kept claims constrained to the generated transferability and event-level evidence | `paper/main.tex`; `paper/sections/09_conclusion.tex` | Closed | Stronger superiority language remains unsupported |
| SR3-10 | Software archive DOI absent | Added submission-package gate for persistent archive metadata | `CITATION.cff`; external archive record | Open | DOI-backed archive requires external provider action |

Final decision under Prompt 51-60: not submission ready until a DOI-backed external software archive is created.
