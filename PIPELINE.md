# Pipeline

## Reproducible saved-data experiment
Run:

```bash
python run_research_pipeline.py
```

This performs:

1. trading-session alignment and feature engineering;
2. expanding-window XGBoost out-of-sample evaluation;
3. market-only versus market+sentiment comparison;
4. one-step-ahead GARCH volatility forecasting;
5. paper-strategy backtesting with turnover costs.

## Optional daily refresh
Run:

```bash
python run_daily_pipeline.py
```

The daily path refreshes market/news data, updates FinBERT sentiment and features, creates the latest GARCH forecast, runs the canonical XGBoost artifact, and writes a paper-trading signal.

The daily path is a research workflow, not a broker-connected execution system.
