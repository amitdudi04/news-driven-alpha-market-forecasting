# News-Driven Alpha: Financial Sentiment and CSI 300 Forecasting

This project studies whether **China-focused economic and financial news sentiment adds useful next-session information beyond market-only variables for the CSI 300**.

The pipeline combines GDELT title/headline observations, ProsusAI FinBERT sentiment, CSI 300 market data and a shallow XGBoost classifier. A separate GARCH(1,1) model provides one-step-ahead volatility forecasts for paper-strategy risk scaling.

## Research design

~~~text
GDELT titles/headlines
      ↓
FinBERT sentiment
      ↓
daily sentiment aggregates
      ↓
trading-session alignment
      +
CSI 300 return / volatility / momentum
      ↓
descriptive finance analysis
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
~~~

The target is the **next CSI 300 trading-session return direction**. Weekend and holiday news is aggregated into the next available trading-day information set; no artificial weekend market observations are created.

## Features

The directional model uses:

- 20-day market volatility;
- 5-day and 10-day rolling FinBERT sentiment;
- rolling sentiment z-score;
- sentiment momentum and dispersion;
- news intensity;
- market momentum and momentum acceleration;
- sentiment-volatility interaction terms;
- an ex-ante volatility indicator based only on earlier observations.

The market-only baseline uses volatility, momentum, momentum acceleration and the same ex-ante volatility indicator.

## Time-series safeguards

The research design uses:

- chronological expanding-window evaluation;
- no random train/test split;
- no full-sample scaler;
- no full-sample target-based feature selection;
- no future volatility median for regime assignment;
- no realized-return bootstrap for historical trading signals;
- one-step-ahead GARCH forecasts rather than in-sample conditional volatility presented as forecasts.

The final inference model may be refit on all labelled history only after the historical OOS evaluation has been produced.

## Current empirical findings

The committed sample begins on **22 April 2026** and contains **6,919 GDELT title/headline observations scored by FinBERT across 49 news days**.

Trading-session alignment produces **31 sentiment / next-session-return pairs**. In that descriptive sample:

- Spearman time-series rank correlation: **-0.348**;
- Pearson correlation: **-0.246**;
- naive sentiment-sign directional hit rate: **45.2%**;
- lowest-sentiment tercile mean next-session return: **+0.305%**;
- highest-sentiment tercile mean next-session return: **-0.161%**.

Across the 49 aligned CSI 300 sessions, the compounded return implied by the stored daily log returns is **+1.56%** and annualized realized volatility is **20.78%**.

These statistics are exploratory. They do not establish a causal relationship or a tradable sentiment effect. Instead, they motivate the pre-specified test of whether rolling sentiment, attention and sentiment-volatility interactions improve forecasting beyond the market-only baseline.

The full rolling feature set currently leaves **8 labelled model rows**. The directional model is configured to begin walk-forward forecasting after **60 training observations**, but model-performance statistics are not published until at least **30 genuine OOS forecasts** are available. The current repository therefore reports descriptive evidence, not XGBoost accuracy, Sharpe or strategy profitability.

See docs/RESULTS.md for the full result interpretation.

## Run the project

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Run the saved-data research pipeline:

~~~bash
python run_research_pipeline.py
~~~

The command always rebuilds the descriptive finance summary and model features from committed data. With sufficient history it also runs the XGBoost comparison, GARCH forecasts and OOS paper-strategy evaluation.

An optional daily research refresh is available after a canonical model has been trained:

~~~bash
python run_daily_pipeline.py
~~~

Launch the read-only dashboard with:

~~~bash
streamlit run app.py
~~~

## Repository structure

~~~text
data/
  news_daily.csv
  sentiment_features.csv
  csi300_features.csv

config/
  research_config.py
  asset_registry.json

descriptive_analysis.py   reproducible descriptive finance results
module1_news.py            GDELT ingestion
module2_sentiment.py       FinBERT scoring
module3_market.py          CSI 300 market data
module4_features.py        trading-session alignment + feature engineering
module5_xgboost.py         expanding-window direction model + ablation
module6_garch.py           one-step-ahead volatility forecasts
module7_backtesting.py     OOS paper-strategy evaluation
module12_inference.py      latest next-session direction inference
module13_signal_engine.py  paper-signal construction

run_research_pipeline.py
run_daily_pipeline.py
app.py
docs/
  RESULTS.md
tests/
~~~

## Finance interpretation

The economic benchmark is the **CSI 300**. Strategy performance, when enough OOS observations exist, is reported relative to that benchmark as **active return**. The project title uses “Alpha,” but the repository does not claim Jensen's alpha without an explicit asset-pricing regression.

This is a research and paper-trading project, not a broker-connected trading system. A future positive result would mean that the tested sentiment feature set improved OOS forecasting or benchmark-relative paper-strategy performance on the evaluated sample; it would not by itself establish causality or a durable market anomaly.

Further methodological detail is in DATA_CARD.md, MODEL_CARD.md, ARCHITECTURE.md, PIPELINE.md and PROJECT_LIMITATIONS.md.
