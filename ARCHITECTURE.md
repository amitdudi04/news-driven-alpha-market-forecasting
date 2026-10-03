# Architecture

The public project is organized as a research pipeline rather than a production trading system.

```text
GDELT news
   ↓
FinBERT daily sentiment
   ↓
Trading-session alignment
   +
CSI 300 close-to-close market features
   ↓
Expanding-window XGBoost evaluation
   ├─ market-only baseline
   └─ market + sentiment model
   ↓
Next-session direction probability

CSI 300 returns
   ↓
Walk-forward GARCH(1,1)
   ↓
Next-session volatility forecast

Direction probability + volatility forecast
   ↓
Paper-trading rule + transaction costs
   ↓
Saved research outputs
   ↓
Streamlit presentation layer
```

Model fitting and performance evaluation belong in the research pipeline. The dashboard should read saved outputs rather than manufacture historical signals.
