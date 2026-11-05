# Academic Disclosure

This repository presents an empirical finance research project on whether timestamp-safe financial-news sentiment adds incremental next-session forecasting information for the CSI 300.

The project is presented with the following boundaries:

- The historical research period is **2023-2026**, with 2022 used only as market/GARCH warm-up.
- The recorded protocol uses 2023 initial training, 2024 chronological validation, a designated 2025 holdout excluded from fitting, and a 2026 evaluation using the 2023–2024 fitted models. Although the configuration declares a pre-fit freeze, independently dated preregistration evidence is not publicly available.
- The main comparison is paired within model family: market-only versus market + sentiment.
- Paired moving-block-bootstrap uncertainty is reported for incremental out-of-sample differences.
- The evidence is mixed. Some point estimates favor sentiment, especially XGBoost in 2026, but the principal 2025/2026 incremental effects are not statistically resolved at the 95% block-bootstrap level.
- GARCH(1,1) is a separate volatility-risk overlay and does not generate the directional signal.
- The fixed-rule transaction-cost simulation assumes idealized same-close entry after the current closing information is observed. This is not demonstrated to be executable and is not evidence of live-capital profitability.
- No causal effect, Jensen alpha, guaranteed trading edge, or investment recommendation is claimed.
- Null and negative results are retained rather than optimized away.

The repository is for research and educational use only.
