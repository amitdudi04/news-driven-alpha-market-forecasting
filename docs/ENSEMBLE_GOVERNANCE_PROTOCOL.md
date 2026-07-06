# Institutional Ensemble Governance Protocol

## Core Philosophy
The News-Driven Alpha platform allows for the orchestration of multiple machine-learning models (e.g., regime-specific models, macro-aware variants, base shadow models) to run simultaneously. However, this ensemble must operate exclusively as an isolated research sandbox. This protocol mathematically insulates the core production engine from ensemble stochasticity.

## 1. Zero Execution Authority
**Ensembles possess ZERO live execution authority.**
- Ensembles cannot generate or override the `module13` prediction paths.
- Ensembles cannot elevate themselves to production status autonomously.
- Ensembles cannot emit signals to the `live_tracking.csv` or `daily_prediction.csv` ledgers.

## 2. The Ensemble Registry (`models/ensemble_registry/`)
Any ensemble evaluation must be registered with a manifest containing:
- `ensemble_uuid`: Unique identifier.
- `component_models`: List of UUIDs of the models participating.
- `feature_schema_hash`: Must exactly match the production lineage.
- `state`: Ensembles exist in `SANDBOX`, `RESEARCH_ONLY`, `PAPER_TRIAL`, or `REJECTED`. 
- **Rule**: An ensemble in `PAPER_TRIAL` remains passive. It merely records its theoretical outputs for tracking, requiring manual executive approval before replacing a singular production model.

## 3. Disagreement and Entropy Analysis
Ensemble governance relies on calculating consensus.
- **Majority Vote**: Directional signal aggregation.
- **Confidence Averaging**: Probability smoothing across the ensemble.
- **Disagreement Scoring**: Measures the variance in predictions. Extremely high disagreement (Entropy) indicates regime uncertainty.
- If disagreement becomes chaotic or outputs resolve to NaN/Inf, the ensemble research engine halts to prevent logic degradation.

## 4. Operational Lineage Rule
- All ensemble manifests are append-only.
- The `full_pipeline_regression.py` validates that the deterministic replay hashes of the SSOT pipeline remain identical, mathematically proving that the ensemble logic has not leaked into the production execution stream.
