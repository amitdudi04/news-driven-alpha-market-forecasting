# Institutional Model Promotion Protocol

## Core Philosophy
Under no circumstances will the News-Driven Alpha platform autonomously deploy new predictive logic to live paper-trading. Every modification to `models/model_live.pkl` must strictly traverse the following certification ladder.

## 1. Governance Promotion Ladder

### Stage 1: SHADOW_ONLY
- **Condition**: New model trained via `shadow_retraining_pipeline.py`.
- **Requirements**:
  - Exact feature schema match with production.
  - Generates immutable JSON manifest binding config and data hashes.
  - Initial Offline Sharpe must statistically outperform Incumbent Sharpe.

### Stage 2: PAPER_TRIAL
- **Condition**: Promoted via `shadow_comparison_audit.py` offline test.
- **Requirements**:
  - The model runs invisibly in parallel to the live inference engine.
  - Generates out-of-sample predictions bound to `live_tracking.csv`.
  - Must survive a minimum of **20 forward-looking rolling windows** (Statistical Sufficiency Gate).

### Stage 3: EXTENDED_PAPER_VALIDATION
- **Condition**: Passed 20 days out-of-sample without Confidence Inversion or Brier decay.
- **Requirements**:
  - Model must undergo a **Regime Transition** (e.g., crossing median historical volatility).
  - Expectancy across transitions must remain strictly positive.
  - Tracking Error vs CSI300 must remain within designated bounds.

### Stage 4: LIMITED_CAPITAL_REVIEW
- **Condition**: Passed regime persistence.
- **Requirements**:
  - Replay Certification: The shadow model is forced through `full_pipeline_regression.py` to prove deterministic scaling.
  - SAFE MODE Certification: Z-Score escalations are mathematically verified.
  - Fully compiled governance and calibration report generated.

### Stage 5: MANUAL_EXECUTIVE_APPROVAL
- **Condition**: All statistical bounds satisfied.
- **Requirements**:
  - Absolute ban on automated elevation.
  - Manual execution of `promote_shadow_to_incumbent()` required by authorized portfolio manager.
  - Instant snapshot created for Rollback & Recovery Governance.

## 2. Mandatory Statistical Sufficiency Gates
Promotion is **FORBIDDEN** if:
1. **Sample Size**: Out-of-sample track record < 20 days.
2. **Calibration Collapse**: Brier Score > 0.25 on rolling basis.
3. **Regime Collapse**: Expected Return drops below 0 during `HIGH_VOL` regimes.
4. **Degradation**: Strategy Sharpe triggers `persistent_degradation_alert` during evaluation.

## 3. Rollback Governance
Upon manual promotion, the incumbent `model_live.pkl` is exactly copied to `models/shadow_registry/archive_model_live_YYYY_MM_DD.pkl` with identical JSON hashing to allow instant, non-destructive restoration via `rollback_recovery_audit.py`.
