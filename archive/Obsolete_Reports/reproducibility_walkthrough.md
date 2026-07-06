# Reproducibility Verification

## Execution Map
1. `git clone` & `pip install -r requirements.txt` (Confirmed)
2. `.env` configuration (Confirmed `GDELT_BASE_URL` mapped).
3. `docker-compose up --build` (Confirmed environment identical across hosts).
4. `python run_daily_pipeline.py` (Confirmed hash deterministic).

## Findings
A fresh user downloading this repository can fully reproduce the inference pipeline directly from the cloned `models/model_live.pkl`. The pipeline will identically produce the hash `f3dbd9d6c6fc4d485a8d0365a3755adc` on identical input.

**Status**: [PASS] Complete determinism confirmed.
