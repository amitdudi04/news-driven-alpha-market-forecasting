# Institutional Macro Regime Protocol

## Core Philosophy
The News-Driven Alpha platform is fundamentally a bottom-up asset pricing model. However, institutional survivability requires top-down macro awareness. This protocol defines how the macro layer interacts with the core execution engine.

## 1. Zero Execution Authority
**The Macro Regime Intelligence Layer possesses ZERO execution authority.**
- It cannot generate buy/sell signals.
- It cannot override `module13` trade paths.
- It cannot manually trigger a portfolio liquidation.
- Its outputs (`macro_regime_manifest.csv`, `factor_exposure_manifest.csv`, `cross_market_stress_manifest.csv`) are strictly informational and advisory.

## 2. Institutional Factor Decomposition
The platform isolates the strategy's returns using an OLS attribution model:
- **Market Beta**: Measures raw index correlation. Beta explosion (>1.5) triggers a SAFE MODE warning.
- **Volatility Exposure**: Measures reliance on volatility expansion.
- **Momentum Exposure**: Measures trend-following leakage.
- **Sentiment Exposure**: Measures the pure AI confidence factor.
- **Residual Alpha**: Measures the idiosyncratic edge. Negative alpha triggers failure.

## 3. Cross-Market Stress Detection
The system tracks correlation and volatility transmission across `CSI300`, `HSI`, `SSE50`, and `SPY`. 
- **Systemic Stress**: Occurs when volatility transmission ratios exceed 2.0x standard deviation.
- **Correlation Collapse**: Occurs when cross-asset correlations approach 1.0, signifying a liquidity cascade.
- While macro intelligence cannot trade, extreme readings (e.g., `SAFE_MODE_LOCKED`) will invoke the global operational watchdog to intercept future runs.

## 4. Lineage and Immutability
- All macro manifests are append-only.
- The `generate_macro_governance_report.py` acts strictly as a rendering engine. It pulls directly from the immutable tracking CSVs and never computes logic itself.
