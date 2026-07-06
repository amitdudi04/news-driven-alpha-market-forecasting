# Documentation Cross-Validation Report

## Metric Validation
- **Dataset Size**: Consistent across ACADEMIC_DISCLOSURE, LIMITATIONS, and README (104 genuine trading days).
- **Training Size**: Consistent (80% chronological split = 83 days).
- **Walk-forward Sample Size**: Consistent (21 days OOS evaluation).
- **Model Name**: Consistent ('Regime-Switching Volatility-Aware XGBoost').
- **Classification**: Consistent ('PAPER_TRIAL').
- **Performance Constraints**: Consistent (Flat 0.00% return reported accurately due to SAFE MODE capital preservation bounds blocking all 21 OOS days).

**Status**: [PASS] No contradictory metrics found.
