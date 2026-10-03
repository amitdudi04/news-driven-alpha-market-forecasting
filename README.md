# News-Driven Alpha: Financial Sentiment and CSI 300 Forecasting

This project studies a simple research question: **does daily financial-news sentiment add useful next-session information beyond market-only features?**

The pipeline combines GDELT headlines, ProsusAI FinBERT sentiment, CSI 300 market data and a shallow XGBoost classifier. A separate GARCH(1,1) model forecasts next-session volatility for paper-strategy risk scaling.

## Research design

The public experiment is organized around one timeline:

```text
GDELT headlines
      ↓
FinBERT article sentiment
      ↓
daily sentiment aggregates
      ↓
trading-session alignment
      +
CSI 300 return / volatility / momentum
      ↓
expanding-window XGBoost evaluation
      ├── market-only baseline
      └── market + sentiment model
      ↓
next-session direction probability

CSI 300 returns
      ↓
one-step-ahead GARCH(1,1)
      ↓
volatility risk forecast

direction probability + volatility forecast
      ↓
paper-trading rule + turnover costs
```

The target is the **next CSI 300 trading-session return direction**. Weekend and holiday news is aggregated into the next available trading-day information set, but no artificial weekend market rows are created.

## Features

The directional model uses:

- 20-day market volatility
- 5-day and 10-day rolling FinBERT sentiment
- rolling sentiment z-score
- sentiment momentum and sentiment dispersion
- news intensity
- market momentum and momentum acceleration
- sentiment × volatility interaction terms
- an ex-ante high/low-volatility indicator based only on earlier observations

The market-only baseline uses volatility, momentum, momentum acceleration and the same ex-ante regime indicator.

## Time-series safeguards

The research code deliberately avoids several common sources of optimistic backtests:

- no random train/test split;
- no full-sample StandardScaler;
- no full-sample target-based feature selection;
- no future volatility median for regime assignment;
- no use of realized returns to manufacture historical trading signals;
- no in-sample GARCH conditional volatility presented as a one-step forecast.

XGBoost evaluation uses expanding chronological splits. The final model used for the next paper-trading prediction is refit only **after** the historical out-of-sample evaluation has been produced.

## Current empirical status

The committed clean news/sentiment sample begins on **22 April 2026**. The earlier development seed rows are not part of the public research sample.

The clean history is still too short for a defensible claim of persistent alpha. For that reason, the repository does not publish the old high-accuracy/high-Sharpe development trace as a research result. The canonical runner builds the feature dataset and stops cleanly until at least 61 labelled rows remain after feature construction (60 initial training rows plus at least one out-of-sample forecast).

The intended empirical comparison is saved to:

```text
outputs/model_evaluation.csv
outputs/oos_predictions.csv
outputs/garch_oos_forecasts.csv
outputs/oos_backtest.csv
outputs/oos_backtest_metrics.csv
```

These files are generated locally and are intentionally not committed.

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the saved-data research experiment:

```bash
python run_research_pipeline.py
```

With the current short clean sample, this command builds the feature dataset and reports that more history is required before OOS results are published.

An optional daily research refresh is available once a canonical model has been trained:

```bash
python run_daily_pipeline.py
```

Launch the read-only Streamlit dashboard:

```bash
streamlit run app.py
```

## Repository structure

```text
data/
  news_daily.csv
  sentiment_features.csv
  csi300_features.csv

config/
  research_config.py
  asset_registry.json

module1_news.py          GDELT ingestion
module2_sentiment.py     FinBERT scoring
module3_market.py        CSI 300 market data
module4_features.py      trading-session alignment + feature engineering
module5_xgboost.py       expanding-window direction model + ablation
module6_garch.py         one-step-ahead volatility forecasts
module7_backtesting.py   OOS paper-strategy evaluation
module12_inference.py    latest next-session direction inference
module13_signal_engine.py paper-signal construction

run_research_pipeline.py
run_daily_pipeline.py
app.py
tests/
```

## Interpretation

This is a research and paper-trading project, not a broker-connected trading system. A positive result would mean that sentiment features improve out-of-sample forecasting relative to the market-only baseline on the tested sample; it would not by itself establish causality or a durable exploitable anomaly.

More detail is available in `DATA_CARD.md`, `MODEL_CARD.md`, `ARCHITECTURE.md`, `PIPELINE.md` and `PROJECT_LIMITATIONS.md`.
