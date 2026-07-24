# Canonical data dictionary

The canonical schema is deliberately small. Dataset adapters may retain additional source columns.

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime | Observation time, ordered and timezone documented |
| `entity_id` | string | Machine or vehicle identifier; constant for single-asset data |
| `regime` | category | Operating regime inferred without future information |
| `is_failure` | boolean | Whether timestamp lies inside a documented failure interval |
| `failure_type` | string/null | Failure label from maintenance records |
| `observable` | float | Scalar extreme-value observable; larger means more extreme |
| `exceedance` | boolean | Training-fitted threshold exceedance |
| `cluster_id` | integer/null | Extreme episode identifier |
| `risk_horizon_h` | float | Estimated probability of dangerous-region entry within horizon |

Sensor columns use stable lower snake case. Source names and units are preserved in a generated mapping table.
