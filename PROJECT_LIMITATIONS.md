# Project Limitations

## Short news history

The committed GDELT/FinBERT sample begins on 2026-04-22 and contains only 49 news days. Public-API availability and coverage gaps leave the present sample too short for credible machine-learning validation or claims about a persistent forecasting or trading effect.

## News measurement

FinBERT is applied to GDELT title/headline text retained by the pipeline rather than full article bodies. The GDELT query is China-focused but is not a complete measure of all information reaching investors. Headline volume is used as a **news-intensity measure**, not as a direct observation of investor attention.

## End-of-day timing convention

The saved sentiment data are daily aggregates rather than article-level timestamp panels. Each trading-day feature vector is therefore interpreted as an end-of-day information set used to forecast the next trading session. The project does not claim intraday or pre-open timing precision.

## Model evaluation

XGBoost evaluation uses expanding chronological splits. Feature definitions are fixed before the walk-forward evaluation is run and are not selected using future OOS targets. The pipeline does not fit a scaler or perform target-based feature selection on the full sample.

The expanding-window design uses a 60-observation initial training window, while public performance statistics require at least 30 subsequent OOS forecasts. That reporting rule is a minimum safeguard, not a statistical-power guarantee or evidence that the resulting sample would establish a durable anomaly.

## Descriptive inference

The current correlations and sentiment-sorted returns are descriptive statistics from 31 aligned observations. They are not presented as statistically significant estimates, causal effects, a contrarian anomaly, or proof that sentiment is predictively useful.

## Risk overlay and backtest

GARCH(1,1) is used as a one-step-ahead volatility forecast for position scaling. It is a risk overlay, not evidence that sentiment causes volatility.

Simulated-strategy calculations convert stored market log returns to simple returns before applying position weights and transaction costs. Sharpe ratios, when reported, use a zero risk-free-rate convention unless otherwise stated.

## Interpretation

This is a research prototype with an optional simulated-strategy layer. It does not trade live capital, and descriptive associations or future simulation metrics should not be interpreted as investment advice, causal evidence, or proof of a durable market anomaly.
