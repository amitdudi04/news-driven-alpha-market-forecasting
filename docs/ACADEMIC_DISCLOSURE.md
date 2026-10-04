# Academic Disclosure

This repository presents an empirical research project on whether timestamp-safe financial-news sentiment adds incremental next-session forecasting information for the CSI 300.

- The historical research period covers **2023–2026**, with 2022 used only as market/GARCH warm-up.
- The predictive design was frozen before model fitting: 2023 initial development, 2024 chronological validation, 2025 untouched holdout, and 2026 locked-model robustness.
- The main comparison is paired within model family: market-only versus market + sentiment.
- Paired moving-block-bootstrap uncertainty is reported for incremental OOS differences.
- The evidence is mixed. Some point estimates favor sentiment, especially XGBoost in 2026, but the principal 2025/2026 incremental effects are not statistically resolved at the 95% block-bootstrap level.
- GARCH(1,1) is a separate volatility-risk overlay and does not generate the directional signal.
- The transaction-cost simulation uses fixed research conventions and is not evidence of live-capital profitability.
- No causal effect, Jensen alpha, guaranteed trading edge, or investment recommendation is claimed.
- Negative and null results are retained rather than optimized away.

The repository is for research and educational use only.
