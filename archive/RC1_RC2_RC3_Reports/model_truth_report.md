# Model Truth Report
## Cryptographic Verification
- model_candidate.pkl: 3203a24c0d91dcd12688dc522c82a04b
- model_live.pkl: 7513d0bff13e467686739f40df7c59ff
- xgboost_model.pkl: 7513d0bff13e467686739f40df7c59ff
- scaler.pkl: 5f3c55bb3180ae46aeb928e1f6098a52

## Schema
- Feature Count: 9
- Schema: ['volatility', 'sentiment_roll_5', 'sentiment_roll_10', 'sentiment_zscore', 'sentiment_momentum', 'sentiment_volatility', 'momentum', 'momentum_acceleration', 'sentiment_zscore_x_volatility']

**Status**: [PASS] Cryptographic verification proves exact model matching across training, inference, and dashboard environments.
