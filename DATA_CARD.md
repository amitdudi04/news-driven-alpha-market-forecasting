# Data Card

## Sources

- **News**: GDELT 2.0 document API, using an English-language China/economy/financial-markets query.
- **Market data**: Yahoo Finance CSI 300 index series (000300.SS).

FinBERT is applied to the GDELT title/headline text retained by the research pipeline. The count of 6,919 therefore refers to scored **title/headline observations**, not full article bodies or a claim of 6,919 globally unique articles.

## Public sample

The committed news/sentiment sample begins on **2026-04-22** and currently contains **49 news days**, yielding **31 aligned sentiment / next-session-return pairs** and only **8 labelled rows** after the full rolling-feature construction.

## Time alignment

The model is defined as an **end-of-day trading-session forecast**. For CSI 300 trading day t, calendar-day news after the previous trading date and through day t is aggregated into the information set for t. The target is the return on the **next CSI 300 trading session**.

Weekend and holiday news can therefore enter the next available trading-day information set, but the pipeline does not create artificial weekend market prices. A calendar interval with no committed news observation is treated as missing coverage, not as neutral sentiment.

## Scope and limitations

The GDELT query is China-focused and includes macroeconomic, financial-market and regulatory terms. It is not a comprehensive archive of all China-related news.

The public sample is short and contains gaps associated with public-API availability. It is sufficient for a descriptive feasibility analysis but not for credible machine-learning validation. The repository therefore reports the current sentiment-return statistics as exploratory and does not infer predictive model performance, a stable long-horizon trading effect, or a durable anomaly from them.
