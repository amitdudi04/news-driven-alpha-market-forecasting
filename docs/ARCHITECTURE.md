# Architecture

The repository is an auditable historical research workflow, not a live trading system.

~~~text
GDELT GAL / BigQuery
        ↓
headline-level China-finance filter
        ↓
session-safe de-duplication + FinBERT
        ↓
GDELT seen timestamps
        ↓
15:00 Asia/Shanghai trading-close alignment
        │
        ├──────────────┐
        ↓              ↓
session sentiment   CSI 300 history (2022 warm-up)
        │              ↓
        └──────→ market/session features
                       ↓
              next-session targets
                       ↓
              master session dataset
                       ↓
        frozen target-date experiment
              /                 \
      logistic regression      XGBoost
       market / +sentiment   market / +sentiment
              \                 /
                genuine OOS probabilities
                         ↓
        calibration + paired block bootstrap

CSI 300 returns ──→ expanding GARCH(1,1)
                         ↓
                 volatility risk scale

frozen OOS direction probability + GARCH risk scale
                         ↓
            fixed-rule costed simulation
                         ↓
                 saved local outputs
                         ↓
               read-only dashboard
~~~

## Separation of roles

### FinBERT

FinBERT maps individual retained headlines to sentiment probabilities/scores.

It is not the forecasting model.

### Market/session builder

The master builder creates causal trading-session features and genuine next-session targets.

It preserves missing sentiment and uses prior-only definitions for news-intensity baselines.

### Logistic regression and XGBoost

These are the principal directional forecasting models.

Each has a market-only and market+sentiment specification so incremental sentiment value can be measured within model family.

### Uncertainty layer

The paired moving-block bootstrap is applied only to saved OOS predictions.

It does not refit or retune the directional models.

### GARCH

GARCH(1,1) forecasts one-step-ahead volatility for position scaling.

It is intentionally separate from the directional alpha question.

### Simulation

The simulation applies fixed probability thresholds, risk scaling, turnover, and transaction costs to genuine OOS predictions.

It is descriptive historical research, not a broker execution engine.

### Dashboard

The Streamlit dashboard reads saved local result artifacts.

It is a presentation layer, not a training path.

## Frozen research controls

Three committed specifications control the final experiment:

~~~text
config/directional_experiment_2023_2026.json
config/oos_uncertainty_spec.json
config/garch_simulation_spec.json
~~~

This separates design decisions from later empirical outcomes.
