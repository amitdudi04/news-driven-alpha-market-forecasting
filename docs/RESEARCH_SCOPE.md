# Research Scope

This repository presents an empirical study of whether China-focused financial-news sentiment contributes incremental next-session forecasting information for the CSI 300.

## Coverage

- Market history: 4 Jan 2022 – 30 Sep 2026
- Research/news period: 1 Jan 2023 – 3 Oct 2026
- 2022 is used only for rolling market-variable and GARCH warm-up.
- The principal forecasting sample contains 867 fully usable session/target observations.

## Research structure

The empirical design separates four components:

1. market-only directional forecasting;
2. market + sentiment directional forecasting;
3. probability calibration and bootstrap uncertainty;
4. a separate GARCH volatility overlay and costed simulation.

The directional comparison is always made within the same model family.

## Interpretation

The 2025 holdout results are mixed. The 2026 XGBoost sentiment specification shows the strongest point improvement, while its paired bootstrap interval for balanced-accuracy improvement includes zero.

The study therefore reports model- and period-specific incremental effects rather than a single universal sentiment effect.

The GARCH and simulation sections are reported separately from the forecasting results.
