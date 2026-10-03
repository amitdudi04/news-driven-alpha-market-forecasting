# Model Card

## Direction model

- **Model:** shallow XGBoost binary classifier.
- **Target:** whether the next CSI 300 trading-session return is positive.
- **Features:** market volatility/momentum, rolling FinBERT sentiment, news intensity, sentiment-volatility interactions and a volatility indicator constructed only from information available through the forecast date.
- **Evaluation:** expanding-window chronological out-of-sample prediction.
- **Ablation:** market + sentiment XGBoost versus a market-only XGBoost baseline.
- **Preprocessing:** no full-sample scaler and no full-sample target-based feature selection.

Feature definitions are fixed before the walk-forward evaluation is run and are not selected using future OOS targets.

The configured design uses **60 labelled observations** for the initial expanding training window. Model-performance statistics are reported only after at least **30 genuine OOS forecasts** are available, requiring at least **90 labelled model rows** in the public reporting pipeline. The 30-OOS threshold is a minimum reporting convention, not a statistical-power guarantee or evidence of a durable effect. With only 8 labelled rows in the current sample, no model-performance claim is made.

## Final inference model

After historical OOS evaluation, a separate final model may be refit on all currently labelled history for the next research prediction. That refit is not used to score the historical OOS test.

## Risk model

A separately estimated GARCH(1,1) one-step-ahead volatility forecast is used for position scaling. It does not generate the directional target and is not combined with the directional model during XGBoost training.
