# Active Model Verification
## Hashes
- `model_candidate.pkl`: d497f33754613388e1b0b6668d065cae8155403039a76a1f92794079e08ce001
- `model_live.pkl`: 8dafcc26415b498b373bcebafe4237772fb1d8ee723bce285c0ce737d2234b98
- `scaler.pkl`: d786556269de21151c2c7c6bd66bd1df60acce0a6682c234f406dd6f78040263
- `scaler_candidate.pkl`: c82a0838a8b3bf42cfc6eea31148163642137116ba1107ba8a45eb20f5588081

## Schema
- Feature schema: ['volatility', 'sentiment_roll_5', 'sentiment_roll_10', 'sentiment_zscore', 'sentiment_momentum', 'sentiment_volatility', 'momentum', 'momentum_acceleration', 'sentiment_zscore_x_volatility']
- Feature count: 9

## Comparison
- Binary identical: Different binary. Production model remains active.
- Schema identical: Yes
- Feature ordering identical: Yes

**Status**: [PASS]