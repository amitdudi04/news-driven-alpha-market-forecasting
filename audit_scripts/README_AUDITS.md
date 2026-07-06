# Institutional Audit Lineage

This directory contains the historical phase-gated audit scripts executed during the hardening of the News-Driven Alpha Forecasting system. These scripts were strictly used to validate infrastructure, execution parity, and deterministic risk topology without leaking future data or weakening execution boundaries.

## Audit Classifications

### 1. Infrastructure & Governance Audits (Phases 1-4, Phase 15)
- **Objective**: Mathematically verify the integrity of the data pipeline, the centralization of execution authority (`module13`), and the parity between frontend (render-only) and backend logic.
- **Key Files**: `phase1_audit.py`, `phase2_audit.py`, `phase3_audit.py`, `phase4_audit.py`, `phase15_audit.py` (Final Systems Audit)

### 2. Explainability & Attribution Audits (Phases 11)
- **Objective**: Certify the deterministic stability of the WHY SIGNAL engine, ensuring feature attribution does not drift under identical execution states.
- **Key Files**: `phase11_audit.py`

### 3. Regime & Temporal Stability Audits (Phases 12-14)
- **Objective**: Validate rolling inference continuity, regime boundaries (HIGH_VOL vs LOW_VOL), and forensic execution lineage (UUID hashing).
- **Key Files**: `phase12_audit.py`, `phase13_audit.py`, `phase14_audit.py`

### 4. Alpha & Risk Topology Audits (Module 5 Audits)
- **Objective**: Evaluate the actual predictive edge of the certified engine on historical data, verifying Sharpe ratios, downside deviation, and capital utilization.
- **Key Files**: Relocated Module 5 audits (previously `phase1`, `phase2`, `phase3`, `phase4`).

> [!WARNING]
> These scripts are strictly for historical execution lineage and institutional auditing. They DO NOT interact with the live trading or paper trading environments. The production pipeline is managed exclusively through `run_daily_pipeline.py`.
