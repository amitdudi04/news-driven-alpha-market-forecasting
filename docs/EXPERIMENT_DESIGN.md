# Experiment Design

## Research question

Does China-focused financial-news sentiment add incremental next-session forecasting information for the CSI 300 beyond market-only predictors?

Each market-plus-sentiment model is compared with the corresponding market-only model from the same model family.

## Unit of observation

The forecasting unit is one CSI 300 trading session.

For each session:

- market and news features use information available by the Shanghai close;
- the target is the return of the next CSI 300 trading session;
- direction equals 1 when the next-session log return is positive and 0 otherwise.

The master dataset contains **908** sessions, **907** known next-session targets, and **867** rows with the complete feature set used by the principal models.

## Chronological split

Partitioning is based on `target_session_date`, not predictor-row calendar year.

| Target year | Role | Model-ready rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | chronological validation | 242 |
| 2025 | untouched holdout | 223 |
| 2026 | locked-model temporal robustness | 181 |

The exact split and candidate grids are stored in `config/directional_experiment_2023_2026.json` and were frozen before model fitting.

## 2024 validation procedure

The 2024 period is divided into 12 monthly expanding-window folds.

For each month:

1. training uses only rows with earlier target-session dates;
2. preprocessing is fitted on the training portion only;
3. candidate models produce probabilities for the held-out month;
4. candidate selection is based on combined 2024 out-of-sample predictions.

The primary selection metric is Brier score. Log loss and candidate identifier are deterministic tie-breakers.

The 2025 holdout is excluded from model selection.

## Feature groups

### Market-only predictors

- `return_t`
- `volatility_20_t`
- `momentum_5_20_t`
- `momentum_acceleration_t`
- `regime_dummy_t`

### Sentiment additions

- `unique_headline_count_t`
- `sentiment_mean_t`
- `sentiment_std_t`
- `sentiment_roll_5`
- `sentiment_roll_10`
- `sentiment_roll_20`
- `news_intensity_20`
- `sentiment_x_volatility`
- `sentiment_roll_5_x_volatility`
- `sentiment_roll_20_x_volatility`

## Model families

### Logistic regression

- L2 penalty
- `lbfgs` solver
- training-window standardization
- candidate C values: 0.1, 1, 10

Both variants select **C = 0.1** using the 2024 validation period.

### XGBoost

The candidate grid uses shallow, regularized trees. Both variants select:

- 150 trees
- maximum depth 2
- learning rate 0.03
- minimum child weight 5
- subsample 0.9
- column subsample 0.9
- L2 regularization 5
- L1 regularization 0

## Final fit and temporal robustness

After candidate selection, each model is fitted once on all eligible 2023-2024 observations.

- 2025 is used for untouched holdout evaluation.
- 2026 uses the same 2023-2024 fitted model without retraining on 2025.

This provides a separate temporal robustness check.

## Evaluation metrics

Directional performance:

- balanced accuracy;
- accuracy;
- Brier score;
- log loss;
- ROC AUC;
- mean predicted up probability;
- observed up rate.

Calibration:

- calibration intercept;
- calibration slope;
- five-bin expected calibration error;
- reliability-bin tables.

## Bootstrap uncertainty

Incremental model differences are evaluated with a paired circular moving-block bootstrap:

- 5,000 replications;
- 10-session blocks;
- 95% percentile intervals;
- identical sampled indices for each market-only / market-plus-sentiment pair.

Positive increments favor the sentiment specification. Intervals that include zero are described as unresolved.

## GARCH and simulation

The GARCH volatility model is estimated separately from the directional models.

GARCH(1,1):

- uses expanding historical returns;
- forecasts one-step-ahead volatility;
- scales position size only;
- does not generate direction probabilities.

The fixed transaction-cost simulation uses saved out-of-sample directional probabilities and the conventions stored in `config/garch_simulation_spec.json`.
