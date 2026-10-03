# News-Driven Alpha: Financial Sentiment and CSI 300 Forecasting Framework

This project studies whether **China-focused economic and financial news sentiment adds incremental next-session forecasting information beyond market-only variables for the CSI 300**.

## Research summary

The committed sample contains **6,919 GDELT title/headline observations across 49 news days**. After end-of-day trading-session alignment, **31 sentiment / next-session-return pairs** are available for descriptive analysis. Raw sentiment has a **Spearman time-series rank correlation of -0.348** with the next-session CSI 300 return, while a naive rule based only on the sign of sentiment is correct on **45.2%** of aligned observations.

The current empirical contribution is therefore a **descriptive feasibility study**, not a validated forecasting or trading result. The predictive modules are implemented to test the next research question once a materially longer clean history is available: whether rolling sentiment, news intensity and sentiment-volatility interactions improve a **market + sentiment XGBoost model** relative to a **market-only XGBoost baseline** under chronological expanding-window evaluation. **GARCH(1,1) is kept separate from directional forecasting and is used only as a one-step-ahead volatility risk overlay for position scaling.**

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
configured OOS reporting gate
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
simulated strategy rule + turnover costs
~~~

The directional target is the **next CSI 300 trading-session return sign**. Weekend and holiday news is aggregated into the next available trading-day information set; no artificial weekend market observations are created.

## Features

The market + sentiment model uses:

- 20-day market volatility;
- 5-day and 10-day rolling FinBERT sentiment;
- rolling sentiment z-score;
- sentiment momentum and dispersion;
- news intensity;
- market momentum and momentum acceleration;
- sentiment-volatility interaction terms;
- a volatility indicator constructed only from information available through the forecast date.

The market-only baseline uses volatility, momentum, momentum acceleration and the same volatility indicator.

## Time-series safeguards

The research design uses:

- chronological expanding-window evaluation;
- no random train/test split;
- no full-sample scaler;
- no full-sample target-based feature selection;
- no future volatility median for regime assignment;
- no realized-return bootstrap for historical trading signals;
- one-step-ahead GARCH forecasts rather than in-sample conditional volatility presented as forecasts.

Feature definitions are fixed before the walk-forward evaluation is run and are not selected using future OOS targets. A separate final inference model may be refit on all labelled history only after historical OOS evaluation.

## Current empirical evidence

The committed news/sentiment sample begins on **22 April 2026**.

| Statistic | Current evidence |
|---|---:|
| News days | 49 |
| GDELT title/headline observations scored by FinBERT | 6,919 |
| Sentiment / next-session-return pairs | 31 |
| Spearman time-series rank correlation | **-0.348** |
| Pearson correlation | **-0.246** |
| Naive sentiment-sign directional hit rate | **45.2%** |
| Lowest-sentiment tercile mean next-session return | **+0.305%** |
| Highest-sentiment tercile mean next-session return | **-0.161%** |
| CSI 300 compounded return from stored session log returns | **+1.56%** |
| CSI 300 annualized realized volatility | **20.78%** |

The **+1.56%** figure compounds the 49 stored CSI 300 session log returns and therefore includes the 22 April return measured from the preceding trading close. The first-close to last-close price change from 22 April to 3 July is **+0.89%**; the distinction is documented in the full results file.

The descriptive evidence does **not** support a simple rule that more positive news is followed by a higher next-session CSI 300 return. The negative association is exploratory and is not presented as a contrarian trading effect, causal relationship, or statistical proof of predictability.

The complete result interpretation is in [docs/RESULTS.md](docs/RESULTS.md).

## Model-evaluation status

The 49-news-day sample is too small for credible machine-learning validation. After rolling-feature construction and target formation, only **8 labelled model rows** remain, so model-level performance is not reported. The repository should therefore be read as **descriptive evidence plus a forecasting-evaluation framework**, not as evidence that the XGBoost model or simulated strategy works.

The configured evaluation design uses:

- **60 labelled observations** for the initial expanding training window; and
- at least **30 genuine OOS forecasts** before model-performance statistics are reported.

The public pipeline therefore requires at least **90 labelled model rows** before publishing XGBoost accuracy, Sharpe ratio, active return or simulated-strategy performance. The 30-OOS rule is a minimum reporting convention, not a statistical-power claim or evidence that a durable effect has been established.

## Prospective simulation framework

GARCH(1,1) provides one-step-ahead volatility forecasts for position scaling. It does not generate the directional signal.

If the OOS reporting threshold is reached, the configured simulation rule uses:

- p(up) >= 0.55 → LONG;
- p(up) <= 0.45 → SHORT;
- otherwise → NO TRADE;
- absolute position size capped at 1.0x;
- 10 bps proportional transaction cost per unit of turnover.

The probability thresholds and transaction-cost assumption are implementation conventions for a future OOS simulation; they have not been validated or optimized on the current short sample. Stored market log returns are converted to simple returns before position weighting, transaction costs and wealth compounding.

## Reproducibility

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Run the saved-data research pipeline:

~~~bash
python run_research_pipeline.py
~~~

With the current committed sample, this regenerates:

~~~text
outputs/descriptive_summary.csv
outputs/sentiment_terciles.csv
data/final_dataset.csv
~~~

Generated outputs are excluded from version control so stale local results cannot be mistaken for current evidence.

After the configured OOS reporting threshold is reached, the same pipeline also creates:

~~~text
outputs/model_evaluation.csv
outputs/oos_predictions.csv
outputs/garch_oos_forecasts.csv
outputs/oos_backtest.csv
outputs/oos_backtest_metrics.csv
~~~

An optional daily research refresh is available after a canonical model has been trained:

~~~bash
python run_daily_pipeline.py
~~~

Launch the read-only dashboard after running the research pipeline:

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

descriptive_analysis.py   reproducible descriptive finance analysis
module1_news.py            GDELT ingestion
module2_sentiment.py       FinBERT scoring
module3_market.py          CSI 300 market data
module4_features.py        trading-session alignment + feature engineering
module5_xgboost.py         expanding-window direction model + ablation
module6_garch.py           one-step-ahead volatility forecasts
module7_backtesting.py     OOS simulated-strategy evaluation
module12_inference.py      latest next-session direction inference
module13_signal_engine.py  research-signal construction

run_research_pipeline.py
run_daily_pipeline.py
app.py
docs/
  RESULTS.md
tests/
~~~

## Finance interpretation

The economic benchmark is the **CSI 300**. If sufficient OOS evidence becomes available, benchmark-relative strategy performance is reported as **active return**. The project title uses “Alpha,” but the repository does not claim **Jensen's alpha** without an explicit asset-pricing regression.

This is a research prototype with an optional simulated-strategy layer, not a broker-connected trading system. Any future positive OOS result would describe performance on the tested sample; it would not by itself establish causality, persistence, economic scalability or a durable market anomaly.

Further methodological detail is available in [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PIPELINE.md](PIPELINE.md) and [PROJECT_LIMITATIONS.md](PROJECT_LIMITATIONS.md).
