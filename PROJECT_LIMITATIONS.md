# Project Limitations

## Short news history

The committed GDELT/FinBERT sample begins on 2026-04-22. Public-API availability and coverage gaps leave the present sample too short for claims about a persistent forecasting or trading effect.

## News measurement

FinBERT is applied to GDELT title/headline text retained by the pipeline rather than full article bodies. The GDELT query is China-focused but is not a complete measure of all information reaching investors.

## End-of-day timing convention

The saved sentiment data are daily aggregates rather than article-level timestamp panels. Each trading-day feature vector is therefore interpreted as an end-of-day information set used to forecast the next trading session. The project does not claim intraday or pre-open timing precision.

## Model evaluation

XGBoost evaluation uses expanding chronological splits. Feature definitions are fixed ex ante, and the pipeline does not fit a scaler or perform target-based feature selection on the full sample.

The first forecast can be produced after a 60-observation initial training window, but public performance statistics require at least 30 subsequent OOS forecasts. That reporting rule is a minimum safeguard, not evidence that the resulting sample would be sufficient to establish a durable anomaly.

## Risk overlay and backtest

GARCH(1,1) is used as a one-step-ahead volatility forecast for position scaling. It is a risk overlay, not evidence that sentiment causes volatility.

Paper-strategy calculations convert stored market log returns to simple returns before applying position weights and transaction costs. Sharpe ratios, when reported, use a zero risk-free-rate convention unless otherwise stated.

## Interpretation

This is a research and paper-trading project. It does not trade live capital, and descriptive associations or future paper-strategy metrics should not be interpreted as investment advice, causal evidence, or proof of a durable market anomaly.
