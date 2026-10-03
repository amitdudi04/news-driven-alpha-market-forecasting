# Pipeline

## Reproducible saved-data experiment

Run:

~~~bash
python run_research_pipeline.py
~~~

The command uses only committed local data. It:

1. reproduces the descriptive finance statistics;
2. rebuilds trading-session-aligned features;
3. checks the pre-specified sample-size requirement;
4. runs expanding-window XGBoost evaluation only when sufficient history exists;
5. compares market-only and market+sentiment models;
6. estimates one-step-ahead GARCH volatility forecasts;
7. evaluates the OOS paper strategy with turnover costs.

The directional model uses **60 observations** for the initial expanding training window. Public model-performance statistics require at least **30 subsequent genuine OOS forecasts**, for a minimum of 90 labelled model rows before those metrics are reported.

With the current shorter sample, the pipeline still produces the descriptive result files and feature dataset but stops before publishing model-performance or strategy-performance statistics.

## Optional daily refresh

Run:

~~~bash
python run_daily_pipeline.py
~~~

The daily path refreshes market/news data, updates FinBERT sentiment and features, creates the latest GARCH forecast, runs the canonical XGBoost artifact, and writes a paper-trading signal.

The daily path is a research workflow, not a broker-connected execution system.
