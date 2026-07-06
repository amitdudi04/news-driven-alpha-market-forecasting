# Institutional Shadow Model Registry

## Overview
This registry establishes a mathematically defensible, immutable pipeline for tracking model evolution. Live production models are **never** directly replaced. All new models must enter this registry in a shadow state and advance through rigorous statistical gates.

## Mandatory Model Metadata
Every shadow candidate injected into `models/shadow_registry/` MUST be accompanied by a JSON manifest containing:
- `model_uuid`: Cryptographically unique identifier.
- `training_timestamp`: Exact ISO-8601 creation time.
- `feature_schema_hash`: MD5 of the exact features used during training.
- `training_dataset_hash`: MD5 of the training data artifact.
- `config_hash`: MD5 of `institutional_config.py` at time of training.
- `scaler_hash`: MD5 of the fitted scaler object.
- `performance_metrics`: Initial in-sample / cross-validation Sharpe and Hit Rate.
- `calibration_metrics`: Initial in-sample Brier and ECE scores.
- `regime_metrics`: Performance explicitly broken down by HIGH_VOL and LOW_VOL regimes.
- `deployment_status`: The current stage in the promotion ladder.
- `promotion_stage`: Current integer tier of certification.

## Allowed Promotion States

| State | Description | Restrictions |
|-------|-------------|--------------|
| `SHADOW_ONLY` | Initial state. Evaluated strictly offline against historical data. | Zero live execution authority. |
| `PAPER_TRIAL` | Permitted to generate forward-looking predictions alongside live pipeline. | Outputs logged to shadow ledger. Strict isolation from `module13` execution. |
| `EXTENDED_PAPER_VALIDATION` | Must survive regime shifts and market transitions for >60 days. | Continuous regime persistence tracking. |
| `REVIEW_REQUIRED` | Passed statistical thresholds; awaiting manual executive review. | Manual intervention mandatory. |
| `APPROVED_FOR_LIMITED_CAPITAL` | Authorized to replace incumbent model for live execution. | Rollback path must be fully intact. |
| `REJECTED` | Model failed a statistical boundary. | Permanently archived. |

## Registry Immutability Rules
1. **Append-Only**: Models placed in the registry cannot be modified.
2. **Hash Sealing**: The JSON manifest binds the model to its training dataset and feature schema. Any manual tampering invalidates the model.
3. **No Silent Replacements**: A model may only replace `models/model_live.pkl` through the `audit_scripts/shadow_comparison_audit.py` and strict `MODEL_PROMOTION_PROTOCOL.md` procedures.
