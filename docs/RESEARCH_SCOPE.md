# Research Scope

This repository presents an empirical study of whether China-focused financial-news sentiment contributes incremental next-session forecasting information for the CSI 300 beyond market-only predictors.

## Coverage

- Market history: **4 Jan 2022 - 30 Sep 2026**
- Research/news period: **1 Jan 2023 - 3 Oct 2026**
- 2022 is used only for rolling market-variable and GARCH warm-up.
- The principal forecasting sample contains **867** fully usable session/target observations.

## Research structure

The empirical design separates four components:

1. market-only directional forecasting;
2. market + sentiment directional forecasting;
3. calibration and paired moving-block-bootstrap uncertainty;
4. a separate GARCH volatility overlay and fixed-rule transaction-cost simulation.

The directional comparison is always made within the same model family.

## Timing rule

Forecasts are defined at the **15:00 Asia/Shanghai CSI 300 close**.

Headlines enter a trading-session information set only when:

```text
previous CSI 300 trading close < GDELT seen timestamp <= current CSI 300 trading close
```

After-close, weekend, and holiday news moves forward to the next eligible session.

## Inference boundaries

The 2025 holdout results are mixed. The 2026 XGBoost sentiment specification shows the strongest point improvement, but its paired 95% moving-block-bootstrap interval for balanced-accuracy improvement includes zero.

The study therefore reports model- and period-specific incremental effects rather than a universal sentiment effect.

The GARCH and simulation sections are reported separately from the directional forecasting evidence.

No causal effect, Jensen alpha, guaranteed trading edge, or live-capital profitability is claimed.
