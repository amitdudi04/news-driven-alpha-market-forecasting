# Dashboard Truth Audit
## UI Elements vs Backend Ground Truth

- **SAFE MODE Indication**: [PASS] Top banner, middle banner, and operational status are all linked directly to `daily_operational_state.json` and perfectly synchronized.
- **Pipeline Checkmarks**: [PASS] Now accurately displays `✓` for completed modules and `✕` for failed domains instead of defaulting to `×`.
- **Model Version**: [PASS] Now correctly pulls from `MODEL_VERSION_GOVERNANCE` (v2.0 RC1) instead of `CRASHED`.
- **Missing Values Count**: [PASS] Accurately reports missing values or `N/A (Missing Dataset)` if the data feed completely fails.
- **Data Version**: [PASS] Calculates actual MD5 hash of `final_dataset.csv` instead of displaying `FAILED`.
- **Accuracy Metrics**: [PASS] High-confidence accuracy now displays `Waiting for 20 trading days` if history is insufficient.
- **Worst Prediction**: [PASS] Now correctly collapses and informs the user `Insufficient evaluation history` when run on a short history trace.
