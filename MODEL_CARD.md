# Model Card - CSI 300 Directional Forecasting

## Scope

This project compares market-only and market-plus-sentiment models for next-session CSI 300 direction forecasting.

The central empirical question is whether timestamp-safe financial-news sentiment adds incremental information beyond market variables within the same model family.

## Dataset

- Research sessions: **908**
- Model-ready rows: **867**
- Known next-session targets: **907**
- Target: next CSI 300 trading-session return sign
- Forecast cutoff: **15:00 Asia/Shanghai**
- Session-unique normalized headlines: **276,960**

The one no-news research session, 20 Jun 2025, retains missing sentiment rather than being assigned a neutral score.

## Chronological split

| Target year | Role | Rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | untouched holdout | 223 |
| 2026 | temporal robustness | 181 |

Partitioning uses `target_session_date`, which prevents year-end predictor rows from crossing the intended train/holdout boundary through a next-year target.

## Model families

### Logistic regression

Two variants:

- market only;
- market + sentiment.

Standardization is fitted within each training window.

Selected C for both variants: **0.1**.

### XGBoost

Two variants:

- market only;
- market + sentiment.

Selected specification for both variants:

- 150 trees;
- maximum depth 2;
- learning rate 0.03;
- minimum child weight 5;
- subsample 0.9;
- column subsample 0.9;
- L2 regularization 5;
- L1 regularization 0.

## Feature groups

### Market-only predictors

- current log return;
- 20-session realized volatility;
- 5/20 momentum;
- momentum acceleration;
- causal volatility-regime indicator.

### Sentiment additions

- unique headline count;
- pooled FinBERT sentiment mean;
- sentiment dispersion;
- strict 5/10/20-session rolling sentiment;
- prior-20-session news intensity;
- current and rolling sentiment x volatility interactions.

## Evaluation

Primary metrics:

- balanced accuracy;
- accuracy;
- Brier score;
- log loss;
- ROC AUC;
- calibration intercept;
- calibration slope;
- five-bin expected calibration error.

Incremental effects are always measured within model family.

## 2025 holdout

The 2025 holdout result is mixed.

- Logistic + sentiment improves balanced accuracy but slightly worsens Brier score and log loss.
- XGBoost + sentiment improves Brier score, log loss, and ROC AUC while slightly reducing balanced accuracy.

Under the frozen paired block-bootstrap design, the principal 2025 incremental intervals include zero.

## 2026 temporal robustness

The 2026 evaluation uses the same models fitted on 2023-2024; 2025 is not added to training.

XGBoost + sentiment:

- balanced accuracy: **0.5568**
- Brier score: **0.2502**
- log loss: **0.6935**
- ROC AUC: **0.5458**

The balanced-accuracy difference versus market-only is **+0.0548**. Its paired 95% moving-block-bootstrap interval is approximately **[-0.0020, +0.1183]**, so the effect is not statistically resolved at 95%.

## Calibration

Calibration slopes are generally below 1.

For XGBoost + sentiment in 2026:

- intercept approximately 0.074;
- slope approximately 0.556;
- five-bin ECE approximately 0.0485.

Directional probabilities should therefore be treated as forecast scores rather than perfectly calibrated event probabilities.

## Bootstrap uncertainty

The frozen paired moving-block bootstrap uses:

- 10-session blocks;
- 5,000 resamples;
- 95% percentile intervals;
- identical sampled row indices for each market-only / market-plus-sentiment pair.

If an interval includes zero, the incremental effect is described as unresolved.

A null or negative sentiment result is retained.

## GARCH risk overlay

GARCH(1,1) is separate from the directional models.

- zero-mean return specification;
- Normal innovations;
- expanding one-step forecasts;
- **907** forecasts;
- **0** convergence failures.

GARCH is used only for position scaling.

## Simulation settings

- p(up) >= 0.55: long
- p(up) <= 0.45: short
- otherwise: flat
- maximum absolute position = 1
- transaction cost = 10 bps per unit turnover
- zero risk-free rate for reported Sharpe

These values are fixed research conventions, not optimized trading parameters.

## Limitations

1. The news sample is English-language and title-based.
2. GDELT seen time may differ from the publisher's original publication time.
3. FinBERT is not trained specifically for Chinese-market English news.
4. Session-level normalized-title de-duplication does not eliminate all semantic duplication.
5. Principal 2025 and 2026 incremental sentiment intervals include zero.
6. Probability calibration is imperfect.
7. The simulation omits live-execution frictions such as market impact, financing, and shorting constraints.
8. The study covers one equity index and one historical period.

## Reproducibility

The frozen split, features, candidate grids, and evaluation settings are stored in:

- `config/directional_experiment_2023_2026.json`
- `config/oos_uncertainty_spec.json`
- `config/garch_simulation_spec.json`

The master-dataset SHA-256 is recorded in the directional experiment specification and in the tracked result manifest.
