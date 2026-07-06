import pandas as pd
import numpy as np
import hashlib
import json
import os
import sys
import datetime
import importlib
from pprint import pprint

os.environ['MAX_STALE_DAYS'] = '9999'

# Force deterministic random state for numpy if used
np.random.seed(42)

def hash_dict(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

def main():
    print("==================================================")
    print("MODULE 4 — PHASE 11: EXPLAINABILITY STABILITY & CONTRIBUTION GOVERNANCE AUDIT")
    print("==================================================")
    
    import module12_inference
    from module4_features import load_datasets, generate_latest_features
    
    sent_df, market_df = load_datasets()
    if sent_df is None or market_df is None:
        print("FAILED to load datasets.")
        return
        
    latest_features = generate_latest_features(sent_df, market_df)
    model = module12_inference.load_inference_model()
    
    print("\n--- REPLAY VALIDATION & CONTRIBUTION VALIDATION ---")
    hashes = []
    outputs = []
    for i in range(3):
        res = module12_inference.predict_next_day(model, latest_features.copy())
        outputs.append(res)
        h = hash_dict({
            'top_pos': res['top_positive'], 
            'top_neg': res['top_negative'],
            'dir_prob': res['dir_prob'],
            'confidence': res['confidence']
        })
        hashes.append(h)
    
    print(f"Replay Hashes: {hashes}")
    if len(set(hashes)) == 1:
        print("REPLAY VALIDATION: PASS - Deterministic Explanations")
    else:
        print("REPLAY VALIDATION: FAIL - Nondeterministic Explanations")
        sys.exit(1)
        
    print("\nFeature Contributions Dump (Run 1):")
    res1 = outputs[0]
    print(f"Top Positive Features: {res1['top_positive']}")
    print(f"Top Negative Features: {res1['top_negative']}")
    print(f"Directional Prob: {res1['dir_prob']:.4f}")
    print(f"Confidence: {res1['confidence']:.4f}")
    print(f"Distance Penalty: {res1['distance_penalty']:.4f}")
    print(f"Regime: {res1['regime']}")
    
    # Check coherence
    print("\n--- COHERENCE VALIDATION ---")
    is_coherent = True
    # Basic coherence: top positive should have positive values, top negative negative values
    for k,v in res1['top_positive'].items():
        if v <= 0: is_coherent = False
    for k,v in res1['top_negative'].items():
        if v >= 0: is_coherent = False
    if is_coherent:
        print("COHERENCE VALIDATION: PASS - Contributions align with math.")
    else:
        print("COHERENCE VALIDATION: FAIL - Misaligned contributions.")
        sys.exit(1)

    print("\n--- GOVERNANCE & PARITY VALIDATION ---")
    # Run full signal engine to dump why_signal
    print("Running module12 main...")
    module12_inference.main(execution_uuid="PHASE11-TEST")
    
    print("Running module13 main...")
    import module13_signal_engine
    module13_signal_engine.main()
    
    daily_pred_path = os.path.join(os.getcwd(), 'outputs', 'daily_prediction.csv')
    df = pd.read_csv(daily_pred_path)
    latest_pred = df.iloc[-1]
    
    backend_explanation = latest_pred['explanation']
    backend_why = latest_pred['why_signal']
    
    print(f"Backend Explanation: {backend_explanation}")
    print(f"Backend WHY_SIGNAL JSON:")
    pprint(json.loads(backend_why))
    
    if "N/A" in backend_explanation or "Unavailable" in backend_explanation:
        print("GOVERNANCE VALIDATION: FAIL - Placeholders detected in explanation.")
        sys.exit(1)
        
    # Check frontend/backend parity
    import app
    pipeline_data = app.run_live_pipeline()
    frontend_explanation = pipeline_data['explanation_text']
    
    print(f"Frontend Explanation: {frontend_explanation}")
    if frontend_explanation == backend_explanation:
        print("FRONTEND/BACKEND PARITY: PASS - Explanations match exactly.")
    else:
        print("FRONTEND/BACKEND PARITY: FAIL - Explanations diverge.")
        sys.exit(1)
        
    print("\n--- OOD VALIDATION (SAFE MODE) ---")
    ood_features = latest_features.copy()
    # Inject massive drift to trigger SAFE MODE ESCALATION
    ood_features.iloc[0, 1] = 20.0  # Max z-score > 15
    
    try:
        module12_inference.predict_next_day(model, ood_features)
        print("OOD VALIDATION: FAIL - Did not trigger SAFE MODE exception.")
        sys.exit(1)
    except RuntimeError as e:
        if "SAFE MODE ESCALATION" in str(e):
            print("OOD VALIDATION: PASS - SAFE MODE correctly triggered.")
            print(f"OOD Trace: {str(e)}")
        else:
            print(f"OOD VALIDATION: FAIL - Unexpected exception: {e}")
            sys.exit(1)

    print("\n==================================================")
    print("PHASE 11: ALL CHECKS PASSED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()
