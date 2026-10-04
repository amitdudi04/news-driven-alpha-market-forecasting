# Committed Results Snapshot

This directory contains the small, reviewable empirical artifacts underlying the public 2023–2026 findings.

These files are committed so an external reviewer can inspect the reported OOS probabilities, metric tables, uncertainty summaries, GARCH forecasts, and fixed-rule simulation summaries without needing access to the large historical GDELT/FinBERT reconstruction.

## Included artifacts

| File | Purpose |
|---|---|
| `release_manifest.json` | Release counts and frozen master-dataset SHA-256. |
| `directional_oos_metrics.csv` | Balanced accuracy, accuracy, Brier, log loss, ROC AUC, and related metrics for all four models across 2024 validation, 2025 holdout, and 2026 robustness. |
| `incremental_sentiment_comparison.csv` | Within-family market-plus-sentiment minus market-only performance differences. |
| `oos_predictions.csv` | Genuine saved OOS probabilities and targets used for the final directional evaluation. |
| `selected_hyperparameters.json` | Hyperparameters selected using 2024 chronological validation only. |
| `calibration_metrics.csv` | Calibration intercept, slope, and expected calibration error. |
| `block_bootstrap_summary.csv` | Paired 10-session circular moving-block bootstrap point differences and 95% intervals. |
| `garch_forecasts.csv` | Expanding one-step-ahead GARCH(1,1) volatility forecasts used by the risk overlay. |
| `garch_forecast_metrics.csv` | Period-level GARCH forecast diagnostics. |
| `simulation_metrics.csv` | Fixed-rule transaction-cost simulation metrics for all four directional models, with and without GARCH scaling. |
| `paired_simulation_comparison.csv` | Within-family sentiment-minus-market-only simulation differences. |

## What is not committed

The repository intentionally does not commit:

- the full historical GDELT headline store;
- per-headline FinBERT score files;
- the full timestamp-aligned article-level table;
- the generated 908-session master CSV;
- fitted model binaries;
- 5,000-draw bootstrap replication files;
- full simulation path files.

Those artifacts are reproducible from the committed source code and frozen specifications, but keeping them out of Git avoids presenting generated data as source code and keeps the repository practical to clone.

The frozen master-dataset SHA-256 is recorded both here and in `config/directional_experiment_2023_2026.json`. The directional experiment refuses to fit if the local master dataset does not match that hash.

## Interpretation

These result files are an auditable research snapshot, not evidence of a live trading system. The project reports null and negative findings alongside positive point estimates. See `docs/RESULTS.md` and `docs/LIMITATIONS.md` for the complete interpretation.
