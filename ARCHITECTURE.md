# Architecture

The repository is organized as a research pipeline rather than a live trading system.

~~~text
GDELT title/headline observations
   ↓
FinBERT sentiment
   ↓
Daily sentiment aggregates
   ↓
Trading-session alignment
   +
CSI 300 market features
   ↓
Descriptive finance analysis
   ├─ sentiment / next-session-return association
   └─ sentiment-sorted return summaries
   ↓
Configured OOS reporting gate
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
Simulated strategy rule + turnover costs
   ↓
Saved research outputs
   ↓
Read-only Streamlit presentation layer
~~~

## Separation of roles

- **FinBERT** converts retained GDELT title/headline text into daily sentiment measures.
- **XGBoost** addresses the directional forecasting question.
- **GARCH(1,1)** forecasts volatility for risk scaling and does not generate the direction target.
- **The dashboard** reads saved research outputs; it does not fit models or manufacture historical signals.

## Reporting gate

The expanding-window design uses 60 labelled observations for the initial training window. Model-performance statistics are presented only when at least 30 subsequent genuine OOS forecasts are available. The current committed sample does not meet that threshold, so the public evidence is descriptive rather than model-performance evidence. The strategy layer is therefore a prospective simulation framework, not a validated trading result.
