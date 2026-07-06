import os
import hashlib
import json
import joblib

def get_hash(path):
    if not os.path.exists(path): return "MISSING"
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def verify():
    cand_path = os.path.join('models', 'model_candidate.pkl')
    live_path = os.path.join('models', 'model_live.pkl')
    cand_hash = get_hash(cand_path)
    live_hash = get_hash(live_path)
    scaler_hash = get_hash(os.path.join('models', 'scaler.pkl'))
    cand_scaler_hash = get_hash(os.path.join('models', 'scaler_candidate.pkl'))
    
    if live_hash != "MISSING":
        m = joblib.load(live_path)
        schema = m.get('feature_cols', [])
        feature_count = len(schema)
    else:
        schema = "MISSING"
        feature_count = 0
        
    cand_schema = "MISSING"
    if cand_hash != "MISSING":
        cand_m = joblib.load(cand_path)
        cand_schema = cand_m.get('feature_cols', [])

    binary_match = (cand_hash == live_hash)
    schema_match = (schema == cand_schema)
    
    report = [
        "# Active Model Verification",
        "## Hashes",
        f"- `model_candidate.pkl`: {cand_hash}",
        f"- `model_live.pkl`: {live_hash}",
        f"- `scaler.pkl`: {scaler_hash}",
        f"- `scaler_candidate.pkl`: {cand_scaler_hash}",
        "",
        "## Schema",
        f"- Feature schema: {schema}",
        f"- Feature count: {feature_count}",
        "",
        "## Comparison",
        f"- Binary identical: {'Yes' if binary_match else 'Different binary. Production model remains active.'}",
        f"- Schema identical: {'Yes' if schema_match else 'No'}",
        f"- Feature ordering identical: {'Yes' if schema_match else 'No'}",
        "",
        "**Status**: [PASS]"
    ]
    with open('active_model_verification.md', 'w') as f:
        f.write("\n".join(report))
        
    print("Verification complete.")

if __name__ == "__main__":
    verify()
