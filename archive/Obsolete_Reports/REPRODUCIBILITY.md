# Reproducibility
- All model binaries are statically tied to `requirements.txt`.
- Data scaling is permanently preserved via `scaler.pkl`.
- Deterministic random seeds (`random_state=42`) used throughout.
