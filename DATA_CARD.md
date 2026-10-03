# Data Card

## Sources
- **News**: GDELT 2.0 document API, using an English-language China/macroeconomy/financial-markets query.
- **Market data**: Yahoo Finance CSI 300 index series (`000300.SS`).

## Public sample
The committed research sample starts on **2026-04-22**. Earlier development seed rows were removed because they were not empirical news observations.

## Time alignment
The model is defined as an **end-of-day trading-session forecast**. For CSI 300 trading day *t*, calendar-day news after the previous trading date and through day *t* is aggregated into the information set for *t*. The target is the return on the **next CSI 300 trading session**.

Weekend and holiday news can therefore enter the next available trading-day information set, but the pipeline does not create artificial weekend market prices. A calendar interval with no committed news observation is treated as missing coverage, not as neutral sentiment.

## Current limitation
The clean public news history is still short and contains gaps caused by public-API availability. The repository therefore does not treat the current sample as evidence of a stable long-horizon trading edge.
