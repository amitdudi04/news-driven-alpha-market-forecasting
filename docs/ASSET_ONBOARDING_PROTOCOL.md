# Institutional Asset Onboarding Protocol

## Overview
This protocol strictly governs the expansion of the News-Driven Alpha platform from a monolithic CSI300 engine into a scalable, multi-asset research environment. 

## 1. Asset Registry Governance
All assets must be formally declared within `config/asset_registry.json`.
The registry enforces:
- `status`: `PRODUCTION` (CSI300 only) vs `RESEARCH_ONLY`.
- `required_features`: A mandatory schema validation. If an asset is missing an expected feature, the `SAFE MODE` will mathematically quarantine the asset.
- `max_stale_days`: Prevents research leakage by barring inference on stale cross-market data.

## 2. Research Isolation
- **Non-Contamination**: Research datasets (`HSI`, `SSE50`, `SPY`) must never pollute the training pool of the `PRODUCTION` asset.
- **Lineage Separation**: Each asset generates its own independent execution manifest (`research_hsi_manifest.csv`, etc.). Execution UUIDs are partitioned by asset prefix.
- **Model Sandboxing**: Research models are strictly restricted to `models/research_<asset>.pkl`. They cannot interact with the production SSOT (`live_tracking.csv`).

## 3. Mandatory SAFE MODE Cross-Market Validations
To successfully orchestrate multi-asset research, the following `SAFE MODE` bounds are dynamically enforced:
1. **Incomplete Schema**: If an asset lacks the `required_features` defined in the registry, execution for that specific asset is halted immediately.
2. **Missing Feature Mappings**: If derived targets (e.g., `target_return_t+1`) cannot be aligned, the asset is dropped from the research batch.
3. **Stale Data**: Timezone-adjusted timestamps are verified. If an asset's data exceeds `max_stale_days`, it is embargoed to prevent silent look-ahead bias in relative-value studies.

## 4. Operational Lineage Rule
No multi-asset operation can bypass the `research_orchestrator.py`. The orchestrator validates the config schema before invoking deterministic research pipelines. At no point does the orchestrator possess live execution authority.
