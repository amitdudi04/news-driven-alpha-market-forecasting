# Project Limitations

## Short clean news history
The current committed GDELT/FinBERT sample begins on 2026-04-22 after removal of earlier non-empirical development seed rows. Public-API rate limits and gaps make the present sample too short for strong claims about persistent alpha.

## End-of-day timing convention
The saved sentiment data are daily aggregates rather than article-level timestamp panels. The public research design therefore treats each trading-day feature vector as an end-of-day information set and forecasts the next trading session. It does not claim intraday or pre-open timing precision.

## Model evaluation
XGBoost evaluation uses expanding chronological splits. Feature definitions are fixed before the split, and the public pipeline does not fit a scaler or perform target-based feature selection on the full sample.

## Risk overlay
GARCH(1,1) is used as a one-step-ahead volatility forecast for position scaling. It is a risk overlay, not evidence that sentiment causes volatility.

## Interpretation
The system is a research and paper-trading prototype. It does not trade live capital, and no performance metric should be interpreted as investment advice or as proof of a durable market anomaly.
