# Model Card

## Direction model
- **Model**: shallow XGBoost binary classifier.
- **Target**: whether the next CSI 300 trading-session return is positive.
- **Features**: market volatility/momentum, rolling FinBERT sentiment, news intensity, sentiment-volatility interactions, and an ex-ante volatility-regime indicator.
- **Evaluation**: expanding-window chronological out-of-sample prediction.
- **Ablation**: the full market+sentiment model is evaluated alongside a market-only XGBoost baseline.
- **Preprocessing**: no full-sample StandardScaler and no full-sample target-based feature selection.

## Final inference model
After out-of-sample evaluation, a separate final model is refit on all currently labelled history for the next paper-trading prediction. That refit is not used to score the historical OOS test.

## Risk model
A separately estimated GARCH(1,1) one-step-ahead volatility forecast is used for risk scaling. It is not combined with the directional target during XGBoost training.
