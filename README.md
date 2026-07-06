# News-Driven Alpha for Financial Market Forecasting

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13-blue)

A machine learning pipeline that integrates financial news sentiment, market data, and regime-aware modeling to generate daily market forecasts and paper trading signals. The system combines FinBERT sentiment analysis, GDELT news data, feature engineering, and XGBoost-based classification within a reproducible forecasting workflow.

---

## Motivation

Financial markets react rapidly to macroeconomic events, geopolitical developments, and breaking news. Traditional quantitative models often struggle to incorporate unstructured textual information in real time.

This project investigates whether financial news sentiment can improve short-term market forecasting by combining natural language processing, market features, and machine learning in an end-to-end prediction pipeline.

---

## Key Features

- Financial news ingestion using the GDELT API
- Sentiment analysis using ProsusAI FinBERT
- Market data collection using yfinance
- Automated feature engineering pipeline
- Regime-aware XGBoost prediction model
- Daily inference and signal generation
- Streamlit dashboard for monitoring
- Paper trading workflow
- Regression testing for reproducibility

---

## Architecture

The forecasting pipeline consists of the following stages:

1. News Extraction (GDELT API)
2. Financial Sentiment Analysis (FinBERT)
3. Market Data Collection (yfinance)
4. Feature Engineering
5. XGBoost-Based Forecasting
6. Signal Generation
7. Dashboard & Monitoring

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FinBERT
- GDELT API
- yfinance
- Streamlit

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/amitdudi04/amitdudi04-news-driven-alpha-market-forecasting.git

# Enter project folder
cd amitdudi04-news-driven-alpha-market-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the forecasting pipeline
python run_daily_pipeline.py

# Launch dashboard
streamlit run app.py
```

---

## Repository Structure

```
config/
data/
docs/
models/
outputs/

module1_news.py
module2_sentiment.py
module3_market.py
module4_features.py
module5_xgboost.py
module12_inference.py
module13_signal_engine.py
module14_live_monitoring.py

run_daily_pipeline.py
app.py
```

---

## Known Limitations

- Sentiment extraction currently relies on FinBERT and does not include Chinese-language financial language models.
- Evaluation is based on a limited out-of-sample paper trading period.
- This project is intended for research, educational purposes, and paper trading evaluation. It should not be interpreted as investment advice or a production trading system.

---

## License

This project is licensed under the MIT License.
