# Experiment Design

## Research question

The empirical question is:

> Does timestamp-safe China-focused financial-news sentiment add incremental next-session forecasting information for the CSI 300 beyond market-only predictors?

The comparison is therefore paired **within model family**. A market-plus-sentiment model is evaluated against the corresponding market-only model, rather than against an unrelated benchmark.

## Unit of observation

The forecasting unit is one genuine CSI 300 trading session.

For a feature row dated `t`:

- all news information is restricted to the timestamp-safe information window ending at the 15:00 Asia/Shanghai close on session `t`;
- the target is the log return of the **next genuine CSI 300 trading session**;
- direction equals 1 when that next-session log return is positive and 0 otherwise.

The final master dataset contains 908 sessions, 907 known next-session targets, and 867 rows satisfying the strict complete-feature modeling rule.

## Frozen chronological split

Partitioning uses **target_session_date**, not the predictor-row calendar year. This prevents year-end feature rows from crossing the intended train/holdout boundary when their return target belongs to the next calendar year.

| Target year | Role | Strict rows |
|---|---|---:|
| 2023 | initial development training | 221 |
| 2024 | chronological model-selection validation | 242 |
| 2025 | untouched final holdout | 223 |
| 2026 | locked-model post-sample robustness | 181 |

The exact split and candidate grids are stored in `config/directional_experiment_2023_2026.json`.

## 2024 validation procedure

The 2024 validation period is divided into 12 monthly expanding-window folds.

For each validation month:

1. training uses only eligible rows with earlier `target_session_date`;
2. the scaler, where applicable, is fitted on that training window only;
3. the candidate model produces probabilities for the held-out month;
4. candidate selection uses aggregate 2024 OOS predictions.

The primary selection metric is Brier score. Lower log loss and then candidate identifier are deterministic tie-breakers.

The 2025 holdout is not used in model selection.

## Feature groups

### Market-only predictors

- `return_t`
- `volatility_20_t`
- `momentum_5_20_t`
- `momentum_acceleration_t`
- `regime_dummy_t`

### Incremental sentiment predictors

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

The sentiment feature set is evaluated only as an addition to the same market feature base.

## Principal model families

### Logistic regression

- L2 penalty
- `lbfgs` solver
- training-window-only standardization
- frozen candidate values for `C`: 0.1, 1, 10

Both market-only and market-plus-sentiment variants selected `C = 0.1` using 2024 validation.

### XGBoost

The frozen candidate grid is intentionally shallow and regularized. Both variants selected the same 2024-validation configuration:

- 150 trees
- maximum depth 2
- learning rate 0.03
- minimum child weight 5
- subsample 0.9
- column subsample 0.9
- L2 regularization 5
- L1 regularization 0

Using the same selected complexity in both variants makes the incremental sentiment comparison easier to interpret.

## Final fit and post-sample rule

After candidate selection, each selected model is fitted once on all eligible 2023–2024 targets.

- 2025 is evaluated as the untouched final holdout.
- 2026 uses the **same 2023–2024 fitted model**. The model is not retrained on 2025 before the robustness evaluation.

This makes 2026 a locked-model temporal robustness check rather than an adaptive rolling retraining exercise.

## Evaluation metrics

Directional OOS performance is reported with:

- balanced accuracy;
- accuracy;
- Brier score;
- log loss;
- ROC AUC;
- mean predicted up probability;
- observed up rate.

Calibration is assessed separately using:

- calibration intercept;
- calibration slope;
- five-bin expected calibration error;
- reliability-bin tables.

## Incremental uncertainty

The uncertainty design is frozen in `config/oos_uncertainty_spec.json`.

For each model family and OOS period, market-only and market-plus-sentiment predictions are resampled with identical indices using a paired circular moving-block bootstrap:

- 5,000 replications;
- 10-session blocks;
- 95% percentile intervals.

Positive increments are defined to favor sentiment. If the 95% interval includes zero, the incremental effect is described as **unresolved under the frozen block-bootstrap analysis**.

The bootstrap probability that an increment is positive is descriptive and is not reported as a classical p-value.

## GARCH and simulation separation

The directional experiment is completed before the volatility overlay is applied.

GARCH(1,1):

- forecasts one-step-ahead volatility;
- uses expanding historical returns only;
- does not produce the directional probability;
- only scales the absolute position size.

The transaction-cost simulation uses the already-saved OOS directional probabilities under fixed, non-optimized conventions defined in `config/garch_simulation_spec.json`.

This separation prevents risk scaling or trading-rule results from changing the directional model-selection process.

## Interpretation rule

A positive point estimate alone is not sufficient for a claim of persistent predictive advantage.

The project distinguishes:

1. directional point performance;
2. probability quality and calibration;
3. paired uncertainty;
4. descriptive fixed-rule simulation.

No causal, Jensen-alpha, or live-trading claim is inferred from the directional experiment.
