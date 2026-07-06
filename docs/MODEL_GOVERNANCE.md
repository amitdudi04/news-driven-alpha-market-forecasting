# Institutional Model Governance

## 1. Shadow Model Governance
To prevent alpha hallucination and silent model degradation, **all retrained models must first be deployed in Shadow Mode.** 
Institutional platforms NEVER hot-swap live models without out-of-sample forward verification.

**Shadow Protocol:**
- New retrained models (e.g., v1.1.0) MUST run in shadow mode parallel to the incumbent model.
- Shadow predictions MUST NOT affect live paper-trading execution decisions.
- Shadow model outputs are logged separately in `outputs/shadow_inference.csv`.
- **Promotion Criteria**: The shadow model requires a statistically significant improvement in Rolling Sharpe and Brier Score over the incumbent across a minimum of 60 live forward-tested days.
- **Rollback Path**: The incumbent model binary and scalar must remain locked in `archive/` for immediate rollback.

## 2. Long-Horizon Statistical Certification Ladder
Statistical validity requires sufficient sample sizes. Deployment is strictly gated by the number of out-of-sample live predictions (N).

### Stage 1: PAPER TRADING ONLY (N < 250)
- **Status**: The current state.
- **Constraints**: No capital allocation allowed. Model edge is statistically fragile and prone to regime-specific overfitting. 
- **Goal**: Accumulate 250 days of live, immutable execution manifests.

### Stage 2: LIMITED CAPITAL SIMULATION ELIGIBLE (250 <= N < 750)
- **Status**: Edge is statistically observable.
- **Constraints**: Eligible for minimum capital allocation ($X capital bounds). 
- **Validation**: Requires Expected Calibration Error (ECE) < 0.15 and Rolling Sharpe > 1.5 across both HIGH_VOL and LOW_VOL regimes.

### Stage 3: LIVE CAPITAL REVIEW ELIGIBLE (N >= 750)
- **Status**: Edge is institutionally hardened.
- **Constraints**: Not automatically approved. Requires full committee review of maximum drawdowns, Sharpe decay, and latency slippage.

## 3. Live Data Quality Governance
Real-world paper trading fails more frequently from bad data than bad models. The pipeline natively intercepts:
- **Stale Data**: `max_stale_data_days = 1`. Stops execution if API payload is frozen.
- **Zero-Volume Anomalies**: Missing market sessions are trapped to avoid division-by-zero crashes.
- **Corrupted Payloads**: Any `NaN` value forces `SAFE MODE` default split tracking.

## 4. Manifest Immutability & Forensics
- Every daily execution appends to an immutable ledger (`outputs/live_inference.csv`).
- Each entry contains a cryptographic `manifest_hash` verifying the data inputs, configuration state, and model versions.
- If historical replay yields a mismatched hash, the execution sequence is deemed corrupted.
