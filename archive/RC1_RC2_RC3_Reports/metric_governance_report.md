# Performance Metric Governance
- All UI plots and high-confidence accuracy metrics are guarded behind a minimum sample size check (20 trading days).
- Empty states correctly render "Waiting for sufficient observations" instead of `0.0`.
- [PASS] Zero fabrication of PnL or evaluation metrics.
