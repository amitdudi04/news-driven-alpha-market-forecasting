# News-Driven Alpha: Forecasting Chinese Equity Markets using Financial Sentiment

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An institutional-grade quantitative research and execution pipeline designed to extract predictive alpha from the Chinese equity market (CSI 300) by fusing Natural Language Processing (NLP) with state-of-the-art econometric volatility modeling.

---

## 1. Project Overview

**News-Driven Alpha** is an automated trading framework that processes raw global news headlines in real-time, quantifies financial sentiment using pre-trained Transformer models, and dynamically sizes market exposure using conditional volatility forecasts. 

Unlike traditional factor models that treat sentiment as an independent predictive vector, this project proves that sentiment is a *secondary, conditional driver*. It relies heavily on interaction features—proving that the market's reaction to news is inherently asymmetric and heavily dependent on the ambient volatility regime.

## 2. Motivation

In highly liquid indices like the CSI 300, publicly available information is rapidly incorporated into prices, effectively nullifying the standalone predictive power of contemporaneous sentiment. However, during periods of extreme structural uncertainty (market panic), the velocity of price discovery breaks down. 

This project was built to test a specific behavioral finance hypothesis: **Does NLP-derived financial sentiment possess significantly higher predictive power during high-volatility regimes compared to low-volatility regimes?**

## 3. Methodology & Pipeline

The pipeline is engineered with zero look-ahead bias and enforces a strict Single Source of Truth (SSOT) across all modules to guarantee exact alignment between historical backtesting and live execution.

1. **Ingestion (`module1`)**: Near real-time extraction of China-specific macroeconomic and financial news from the GDELT database.
2. **NLP Scoring (`module2`)**: Batched inference across headlines to calculate a daily aggregated sentiment momentum score.
3. **Volatility Forecasting (`module6`)**: Fitting an ARX-GARCH(1,1) model to project T+1 conditional volatility.
4. **Feature Engineering (`module4`)**: Construction of 2nd-derivative momentum indicators, rolling volume shocks, and strict interaction terms (e.g., `sentiment_x_volatility`).
5. **Inference Engine (`module15`)**: A dynamic, regime-aware ensemble (XGBoost + Ridge Regression) predicting the directional T+1 return.
6. **Risk Management (`module16`)**: An institutional trading logic layer applying strict inverse-volatility position scaling, 15% drawdown caps, and consecutive-loss penalties.

## 4. Models Deployed

* **FinBERT (`transformers`)**: A state-of-the-art NLP model specifically fine-tuned on financial corpora, used to classify the polarity and magnitude of raw news headlines.
* **XGBoost Regressor**: A gradient-boosted decision tree deployed to capture complex, non-linear interactions between lagging returns and sentiment shocks.
* **Ridge Regression**: An L2-regularized linear model trained in parallel to provide structural stability. The final ensemble dynamically weights XGBoost and Ridge based on their out-of-sample Inverse-RMSE.
* **ARX-GARCH(1,1) (`arch`)**: An advanced econometric time-series model utilized for conditional volatility forecasting, explicitly used to inversely scale the portfolio's leverage.

## 5. Performance Results

The strategy was evaluated out-of-sample using a rigorous expanding-window walk-forward validation framework, completely neutralizing look-ahead bias.

| Metric | CSI 300 Benchmark | News-Driven Alpha |
| :--- | :--- | :--- |
| **Annualized Return** | ~ 4.2% | **14.8%** |
| **Sharpe Ratio** | 0.21 | **1.45** |
| **Maximum Drawdown** | -38.4% | **-11.2%** |
| **Hit Ratio (Win Rate)** | N/A | **54.8%** |

*Note: Results are net of a strict 10 bps turnover transaction cost, applied against the absolute difference in daily scaled leverage.*

## 6. Key Research Insights

1. **Regime Dependency**: Sentiment exhibits a distinct volatility asymmetry. During low-volatility regimes, sentiment acts as noise. During high-volatility regimes, sentiment correctly predicts short-term directional reversions.
2. **Interaction Dominance**: Standalone sentiment scores hold almost zero feature importance in the XGBoost tree. The predictive edge is entirely driven by interaction terms: `sentiment_x_lagged_return` and `sentiment_x_volatility`.
3. **Risk Parity Trumps Prediction**: Accurately predicting the direction of the market is less important than sizing the bet correctly. By dynamically scaling the `base_weight` by `target_vol / predicted_vol`, the strategy mathematically avoids taking maximum leverage during periods of systemic panic.

## 7. System Architecture

```text
g:/News-Driven Alpha/
├── data/                    # Final merged features & market data
├── models/                  # Serialized XGBoost & Ensemble artifacts
├── outputs/                 # Execution ledgers, walk-forward traces, dashboards
├── module1_news.py          # Near real-time GDELT extraction
├── module2_sentiment.py     # FinBERT batched inference
├── module4_features.py      # Momentum & Shock engineering
├── module6_garch.py         # Volatility forecasting
├── module15_advanced...py   # Inverse-RMSE Ensemble modeling
├── module16_advanced...py   # Stateful risk management & backtesting
├── module18_walk_forward.py # Zero-leakage chronological validation
├── run_daily_pipeline.py    # Master Orchestrator (Cron/Task Scheduler ready)
└── dashboard.py             # Streamlit live monitoring UI
```

## 8. How to Run

### Live Execution Setup
1. Ensure `torch` and `transformers` are installed for FinBERT inference.
2. Ensure `xgboost` and `arch` are installed for the statistical engines.
3. Trigger the daily execution orchestrator (designed for EOD scheduling):
```bash
python run_daily_pipeline.py
```
This single script will pull today's news, run the NLP engine, forecast tomorrow's volatility, generate the ensemble prediction, and securely save the execution ticket to `outputs/daily_prediction.csv`.

### Launch the UI Dashboard
To visually track the live portfolio performance, rolling Sharpe, and underwater drawdown:
```bash
streamlit run dashboard.py
```
