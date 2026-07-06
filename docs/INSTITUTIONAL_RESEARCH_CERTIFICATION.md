# Institutional Research Certification

## 1. Executive Summary
This document certifies that the News-Driven Alpha platform is formally established as a fully insulated, deterministic institutional research and paper-trading environment. All research components (Macro, Ensemble, Shadow Models, Multi-Asset sandboxes) are mathematically prevented from contaminating the core execution pipeline.

## 2. Research Isolation & Governance Boundaries
- **Research ≠ Execution Authority**: Under no circumstances can the research layer emit trade signals to `module13` or `live_tracking.csv`.
- **Sandbox Containment**: `CSI300` is the sole `PRODUCTION` asset. All others (`HSI`, `SSE50`, `SPY`) are contained in a parallel orchestrator (`research_orchestrator.py`).
- **No Autonomous Deployment**: All models remain in `SHADOW` or `PAPER_TRIAL` states natively. Model promotion requires manual governance override.

## 3. Replay Guarantees & Reproducibility
- The platform maintains perfect cryptographic determinism (Hash: `e1e0d7de6487519dc109b6f09e82f8da`).
- `research_reproducibility_audit.py` formally guarantees that any repeated execution of the macro or ensemble layers produces identical manifests without introducing stochastic variance.

## 4. SAFE MODE & Watchdog Protections
- `SAFE MODE` reigns supreme. Stale cross-market data, NaN outputs, exploding beta (>1.5), and uncontrollable ensemble disagreement (>0.3) physically halt execution paths.
- The `continuous_operational_watchdog.py` persistently surveys these boundaries and classifies operational health without mutating pipeline state.

## 5. Statistical & Production Limitations
- The system evaluates probabilities via XGBoost and XGBoost ensembles. Extreme OOD (Out of Distribution) events are managed by SAFE MODE clipping.
- The platform is classified as `PAPER_TRADING_ONLY`. It does not possess latency engineering for High-Frequency Trading (HFT) nor active fix protocol connections for live capital execution.

## 6. Future Model Onboarding Rules
Any future alpha model introduced into the registry MUST:
1. Pass shadow validation (`shadow_comparison_audit.py`).
2. Exist in the ensemble sandbox to test correlation.
3. Be manually promoted via the rigid institutional ladder outlined in `docs/MODEL_PROMOTION_PROTOCOL.md`.
