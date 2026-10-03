# Model Card

## Direction model

- **Model:** shallow XGBoost binary classifier.
- **Target:** whether the next CSI 300 trading-session return is positive.
- **Features:** market volatility/momentum, rolling FinBERT sentiment, news intensity, sentiment-volatility interactions and an ex-ante volatility indicator.
- **Evaluation:** expanding-window chronological out-of-sample prediction.
- **Ablation:** market+sentiment XGBoost versus a market-only XGBoost baseline.
- **Preprocessing:** no full-sample scaler and no full-sample target-based feature selection.

The initial expanding training window contains **60 labelled observations**. Model-performance statistics are published only after at least **30 genuine OOS forecasts** are available. This is a minimum reporting rule, not a claim that 30 forecasts establish a durable effect.

## Final inference model

After historical OOS evaluation, a separate final model may be refit on all currently labelled history for the next paper-trading prediction. That refit is not used to score the historical OOS test.

## Risk model

A separately estimated GARCH(1,1) one-step-ahead volatility forecast is used for position scaling. It is not combined with the directional target during XGBoost training.
