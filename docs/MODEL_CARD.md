# Model Card — CSI 300 Directional Forecasting

## Scope

The project compares market-only and market-plus-sentiment models for next-session CSI 300 direction forecasting.

## Dataset

- Research sessions: 908
- Model-ready rows: 867
- Target: next CSI 300 trading-session return sign
- Market close used for session alignment: 15:00 Asia/Shanghai
- Session-unique normalized headlines: 276,960

The one no-news session, 20 Jun 2025, retains missing sentiment.

## Chronological split

| Target year | Role | Rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | holdout evaluation | 223 |
| 2026 | temporal robustness | 181 |

## Model families

### Logistic regression

Two variants:

- market only;
- market + sentiment.

Standardization is fitted within each training window.

Selected C for both variants: 0.1.

### XGBoost

Two variants:

- market only;
- market + sentiment.

Selected specification:

- 150 trees;
- depth 2;
- learning rate 0.03;
- min child weight 5;
- subsample 0.9;
- column subsample 0.9;
- L2 regularization 5;
- L1 regularization 0.

## Features

### Market-only

- current log return;
- 20-session volatility;
- 5/20 momentum;
- momentum acceleration;
- volatility-regime indicator.

### Sentiment additions

- unique headline count;
- pooled FinBERT sentiment mean;
- sentiment dispersion;
- 5/10/20-session rolling sentiment;
- prior-20-session news intensity;
- sentiment × volatility interactions.

## Evaluation

Metrics include:

- balanced accuracy;
- accuracy;
- Brier score;
- log loss;
- ROC AUC;
- calibration intercept;
- calibration slope;
- five-bin expected calibration error.

Incremental sentiment effects are measured within model family.

## 2025 holdout

Results are mixed:

- logistic sentiment improves balanced accuracy but slightly worsens Brier score and log loss;
- XGBoost sentiment improves Brier score, log loss, and ROC AUC while slightly reducing balanced accuracy.

## 2026 temporal robustness

The 2026 evaluation uses the same 2023–2024 fitted models.

XGBoost + sentiment:

- balanced accuracy: 0.5568
- Brier score: 0.2502
- log loss: 0.6935
- ROC AUC: 0.5458

The balanced-accuracy difference versus market-only is +0.0548. Its paired 95% moving-block-bootstrap interval is approximately [-0.0020, +0.1183].

## Calibration

Calibration slopes are generally below 1.

For XGBoost + sentiment in 2026:

- intercept ≈ 0.074;
- slope ≈ 0.556;
- five-bin ECE ≈ 0.0485.

## Bootstrap uncertainty

The paired moving-block bootstrap uses:

- 10-session blocks;
- 5,000 resamples;
- 95% percentile intervals;
- identical sampled row indices for each model pair.

## GARCH risk model

GARCH(1,1) is separate from the directional models.

- zero-mean return specification;
- Normal innovations;
- expanding one-step forecasts;
- 907 forecasts;
- 0 convergence failures.

GARCH is used for position scaling.

## Simulation settings

- p(up) ≥ 0.55: long
- p(up) ≤ 0.45: short
- otherwise flat
- max absolute position = 1
- 10 bps cost per unit turnover
- zero risk-free rate for reported Sharpe

## Limitations

- The news sample is English-language and title-based.
- GDELT seen time may differ from original publisher publication time.
- FinBERT is not trained specifically for Chinese-market English news.
- Session-level title de-duplication does not remove all semantic duplication.
- Principal 2025 and 2026 bootstrap intervals for incremental sentiment effects include zero.
- Probability calibration is imperfect.
- The simulation omits several live-execution frictions, including market impact and financing.

## Reproducibility

The directional experiment records the master-dataset SHA-256, split, feature sets, candidate grids, and evaluation settings in `config/directional_experiment_2023_2026.json`.
