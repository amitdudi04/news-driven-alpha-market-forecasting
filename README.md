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

## Current empirical findings

The committed clean news/sentiment sample begins on **22 April 2026** and contains **6,919 FinBERT-scored headline/article inputs across 49 news days**.

The clean sample currently provides **31 aligned sentiment / next-session-return observations**. At the raw-sentiment level:

- Spearman rank IC with the next-session CSI 300 return: **-0.348**;
- Pearson correlation: **-0.246**;
- naive sentiment-sign directional hit rate: **45.2%**;
- lowest-sentiment tercile mean next-session return: **+0.305%**;
- highest-sentiment tercile mean next-session return: **-0.161%**.

These are descriptive finance results, not a claim of model alpha. They indicate that raw news tone is not a simple monotonic next-session signal in this short sample and motivate the project's interaction and regime features.

The full rolling feature set currently leaves **8 labelled model rows**, so model-level OOS accuracy, Sharpe and active-return statistics are not reported yet. The expanding-window evaluation starts only after 60 training observations plus at least one unseen forecast observation are available.

For the complete result interpretation, market statistics and finance discussion, see [`docs/RESULTS.md`](docs/RESULTS.md).

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
docs/
  RESULTS.md
tests/
```

## Interpretation

This is a research and paper-trading project, not a broker-connected trading system. A positive result would mean that sentiment features improve out-of-sample forecasting relative to the market-only baseline on the tested sample; it would not by itself establish causality or a durable exploitable anomaly.

For finance reporting, benchmark-relative performance is described as **active return** versus the CSI 300. The project name uses “Alpha,” but the repository does not claim Jensen’s alpha without an asset-pricing regression.

More detail is available in `DATA_CARD.md`, `MODEL_CARD.md`, `ARCHITECTURE.md`, `PIPELINE.md` and `PROJECT_LIMITATIONS.md`.
