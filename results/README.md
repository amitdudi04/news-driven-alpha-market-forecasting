# Empirical Result Snapshot

This directory contains the compact, tracked numerical evidence referenced by the research documentation.

The full historical GDELT store, per-headline FinBERT checkpoints, master-session dataset, fitted model binaries, bootstrap replication draws, and full simulation paths remain local/generated and are not committed.

## Files

| File | Contents |
|---|---|
| `results_manifest.json` | Experiment identity, frozen dataset hash, split, source commit, result counts, and file inventory. |
| `directional_oos_metrics.csv` | OOS directional metrics for all four models across 2024 validation, 2025 holdout, and 2026 robustness. |
| `incremental_sentiment_comparison.csv` | Within-family market-plus-sentiment versus market-only metric differences. |
| `oos_predictions.csv` | Saved OOS probabilities, targets, and predicted directions. |
| `selected_hyperparameters.json` | Hyperparameters selected only from the 2024 chronological validation period. |
| `calibration_metrics.csv` | Calibration intercept, slope, ECE, and probability metrics. |
| `block_bootstrap_summary.csv` | Paired moving-block-bootstrap point differences, 95% intervals, and bootstrap sign probabilities. |
| `garch_forecasts.csv` | One-step GARCH(1,1) volatility forecasts used by the risk overlay. |
| `garch_forecast_metrics.csv` | Period-level GARCH diagnostics. |
| `simulation_metrics.csv` | Fixed-rule transaction-cost simulation metrics for all four directional models, with and without GARCH scaling. |
| `paired_simulation_comparison.csv` | Within-family simulation differences between market-only and sentiment variants. |

## Why predictions and GARCH forecasts are tracked

The two row-level result files are small enough for GitHub (about 337 KB and 191 KB in this release) and materially improve auditability:

- OOS prediction rows allow direct verification of paired model evaluation dates and targets.
- GARCH forecast rows allow direct verification of forecast dates, target dates, risk scales, and convergence status.

They are therefore kept as reproducibility evidence rather than hidden as large generated artifacts.

## Provenance

The frozen master-dataset SHA-256 in `results_manifest.json` matches `config/directional_experiment_2023_2026.json`.

The result snapshot was generated from the finalized 2023-2026 research pipeline and is checked by `scripts/validate_repository.py` in CI.
