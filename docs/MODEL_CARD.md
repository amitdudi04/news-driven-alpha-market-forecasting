# Model Card — Directional Forecasting and GARCH Risk Overlay

## Intended use

This repository is a **research prototype** for testing whether timestamp-safe China-focused financial-news sentiment adds incremental next-session forecasting information for the CSI 300.

It is not a broker-connected trading system and is not intended for live-capital deployment.

## Dataset

- Research market sessions: **908**
- Strict model-ready rows: **867**
- Modeling target: next genuine CSI 300 trading-session return sign
- Timestamp cutoff: **15:00 Asia/Shanghai**
- News timestamp semantics: GDELT seen timestamp
- Session-unique normalized headlines used for pooled sentiment: **276,960**

Missing news remains missing. The one no-news session, 20 Jun 2025, is not imputed to neutral sentiment.

## Frozen split

Partitioning is based on **target_session_date**.

| Target year | Role | Rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | untouched holdout | 223 |
| 2026 | locked-model robustness | 181 |

The split and candidate grids were committed before fitting.

## Model families

### Logistic regression

Two variants:

- market only;
- market + sentiment.

Training-window-only standardization is performed with `StandardScaler`.

Selected C for both variants: **0.1**.

### XGBoost

Two variants:

- market only;
- market + sentiment.

Selected 2024-validation specification for both variants:

- 150 trees;
- depth 2;
- learning rate 0.03;
- min child weight 5;
- subsample 0.9;
- column subsample 0.9;
- L2 regularization 5;
- L1 regularization 0.

## Feature groups

Market-only:

- current log return;
- 20-session volatility;
- 5/20 momentum;
- momentum acceleration;
- causal volatility-regime indicator.

Sentiment additions:

- unique headline count;
- pooled FinBERT sentiment mean;
- sentiment dispersion;
- 5/10/20-session strict rolling sentiment;
- prior-20-session news intensity;
- current and rolling sentiment × volatility interactions.

## Evaluation

Primary metrics:

- balanced accuracy;
- Brier score;
- log loss;
- ROC AUC;
- calibration intercept;
- calibration slope;
- 5-bin expected calibration error.

Incremental value is always measured **within model family**.

### 2025 holdout

Sentiment evidence is mixed:

- logistic: threshold classification improves, probability loss slightly worsens;
- XGBoost: Brier/log loss/AUC improve, balanced accuracy slightly worsens.

### 2026 robustness

The strongest point result is XGBoost + sentiment:

- balanced accuracy: **0.5568**
- Brier: **0.2502**
- ROC AUC: **0.5458**

Paired balanced-accuracy improvement versus market-only is **+0.0548**, but its 95% paired block-bootstrap interval is approximately **[-0.0020, +0.1183]**.

Therefore the improvement is **not statistically resolved at 95%**.

## Calibration limitations

Calibration is imperfect. Slopes are generally below 1, indicating that probability magnitudes should not be interpreted as perfectly calibrated event probabilities.

The best observed 2026 calibration among the principal XGBoost variants is the sentiment model:

- intercept ≈ 0.074;
- slope ≈ 0.556;
- 5-bin ECE ≈ 0.0485.

This remains meaningfully different from ideal calibration (0, 1, 0).

## Uncertainty

The frozen uncertainty design uses:

- paired circular moving-block bootstrap;
- 10-session blocks;
- 5,000 resamples;
- 95% percentile intervals;
- identical sampled row indices for each market-only / sentiment pair.

If an interval includes zero, the incremental effect is described as unresolved.

A null or negative sentiment result is retained.

## GARCH risk model

GARCH(1,1) is completely separate from the directional fit.

- zero-mean return model;
- Normal innovations;
- expanding one-step forecasts;
- 907 forecasts;
- 0 convergence failures.

GARCH does not create direction probabilities. It only scales fixed directional positions.

## Simulation conventions

- p(up) >= 0.55: long
- p(up) <= 0.45: short
- otherwise flat
- max |position| = 1
- 10 bps cost per unit turnover
- risk-free rate = 0 for reported Sharpe
- each period starts flat

These values are **fixed research conventions**, not optimized trading parameters.

## Key limitations

1. GDELT seen time is not guaranteed to equal publisher-original publication time.
2. The news query is English-language and therefore samples international coverage of China rather than the full Chinese-language information set.
3. FinBERT is not China-specific and is not fine-tuned here.
4. Exact normalized-title de-duplication does not eliminate all semantically duplicated syndication.
5. Directional effects are not statistically resolved in the 2025 holdout or 2026 robustness period under the frozen block-bootstrap test.
6. Calibration is weak.
7. Fixed-rule simulation does not establish live profitability, persistence, scalability, causality, or Jensen alpha.

## Reproducibility controls

- frozen dataset SHA-256 is stored in the directional experiment specification;
- experiment design was committed before fitting;
- holdout predictions are generated only after 2024 model selection;
- 2026 uses the same 2023–2024 fit without 2025 retraining;
- all saved prediction rows are paired exactly between market-only and sentiment variants;
- 13 repository methodology tests pass.
