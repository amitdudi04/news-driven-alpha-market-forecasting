# Architecture

The project is organized as a historical research pipeline with separate data, forecasting, uncertainty, volatility, and presentation layers.

```text
GDELT GAL / BigQuery
        ↓
China-finance headline filter
        ↓
FinBERT headline scoring
        ↓
Shanghai-local trading-session alignment
        ↓
session-level headline de-duplication and pooling
        ↓
CSI 300 market/session features
        ↓
next-session targets
        ↓
master session dataset
        ↓
chronological directional experiment
     /                             \
logistic regression             XGBoost
market / +sentiment         market / +sentiment
     \                             /
       out-of-sample probabilities
                 ↓
      calibration + block bootstrap

CSI 300 returns ──→ expanding GARCH(1,1)
                         ↓
                 volatility risk scale

out-of-sample direction probability + risk scale
                         ↓
              transaction-cost simulation
                         ↓
                  dashboard and result tables
```

## Component roles

### FinBERT

FinBERT maps retained headlines to sentiment probabilities and a scalar sentiment score. It is a feature-generation model, not the final directional forecasting model.

### Session alignment

News is mapped to CSI 300 trading sessions using Shanghai-local timestamps and the 15:00 market close. After-close, weekend, and holiday news is carried forward to the next eligible session.

### Master dataset

The master dataset combines market variables, pooled sentiment, rolling sentiment, news intensity, interaction features, and next-session targets.

### Directional models

Logistic regression and XGBoost are estimated in paired market-only and market-plus-sentiment specifications.

### Uncertainty analysis

Calibration diagnostics and paired moving-block bootstrap intervals are computed from saved out-of-sample predictions.

### GARCH

GARCH(1,1) provides one-step volatility forecasts used for position scaling. It does not generate direction probabilities.

### Simulation

The simulation applies fixed probability thresholds, GARCH scaling, turnover, and transaction costs to saved out-of-sample probabilities.

### Dashboard

The Streamlit dashboard reads generated result files and does not retrain models.

## Experiment configuration

The final experiment is defined by:

```text
config/directional_experiment_2023_2026.json
config/oos_uncertainty_spec.json
config/garch_simulation_spec.json
```

These files contain the chronological split, model candidate grids, bootstrap settings, and simulation conventions.
