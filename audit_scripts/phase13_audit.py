import os
import sys
import numpy as np
import pandas as pd
import json

def main():
    print("==================================================")
    print("MODULE 4 — PHASE 13: GOVERNANCE FAILURE INJECTION & SAFE MODE RECOVERY AUDIT")
    print("==================================================")
    
    import module13_signal_engine
    
    # ---------------------------------------------------------
    # 1. CORRUPTED INFERENCE VALIDATION
    # ---------------------------------------------------------
    print("\n--- CORRUPTED INFERENCE VALIDATION ---")
    
    # Test 1: NaN dir_prob
    res_nan = module13_signal_engine.generate_signal(
        dir_prob=float('nan'),
        confidence=0.8,
        meta_prob=0.8,
        pred_vol=0.15,
        target_vol=0.15,
        is_high_vol=False
    )
    if res_nan['signal'] == 'NO TRADE' and 'CORRUPTED' in res_nan['explanation']:
        print("PASS: NaN dir_prob cleanly triggers SAFE MODE.")
    else:
        print(f"FAIL: NaN dir_prob bypassed SAFE MODE. Output: {res_nan}")
        sys.exit(1)
        
    # Test 2: Inf confidence
    res_inf = module13_signal_engine.generate_signal(
        dir_prob=0.8,
        confidence=float('inf'),
        meta_prob=0.8,
        pred_vol=0.15,
        target_vol=0.15,
        is_high_vol=False
    )
    if res_inf['signal'] == 'NO TRADE' and 'CORRUPTED' in res_inf['explanation']:
        print("PASS: Inf confidence cleanly triggers SAFE MODE.")
    else:
        print(f"FAIL: Inf confidence bypassed SAFE MODE. Output: {res_inf}")
        sys.exit(1)
        
    # Test 3: Corrupted inputs directly from dataframe
    # We simulate this by rewriting live_inference.csv to contain NaN
    print("\n--- BROKEN MODEL & RECOVERY VALIDATION ---")
    inf_path = os.path.join('outputs', 'live_inference.csv')
    daily_pred = os.path.join('outputs', 'daily_prediction.csv')
    
    # Backup files
    import shutil
    shutil.copy(inf_path, inf_path + ".bak") if os.path.exists(inf_path) else None
    shutil.copy(daily_pred, daily_pred + ".bak") if os.path.exists(daily_pred) else None
    
    try:
        # Corrupt the inference file
        df_inf = pd.read_csv(inf_path)
        df_inf['Confidence'] = float('nan')
        df_inf.to_csv(inf_path, index=False)
        
        # Run module13
        module13_signal_engine.main()
        
        # Check output
        df_pred = pd.read_csv(daily_pred)
        latest = df_pred.iloc[-1]
        
        if latest['signal'] == 'NO TRADE' and 'CORRUPTED' in latest['explanation']:
            print("PASS: Corrupted inference file cleanly handled by module13.")
        else:
            print(f"FAIL: Corrupted inference file bypassed governance. Output: {latest['explanation']}")
            sys.exit(1)
            
        # Corrupt prediction file to simulate catastrophic storage failure
        os.remove(daily_pred)
        import app
        
        # Run app pipeline
        pipeline_data = app.run_live_pipeline()
        
        if pipeline_data['is_safe_mode'] and pipeline_data['signal'] == 'NO TRADE':
            print("PASS: Frontend pipeline recovered gracefully from missing prediction file.")
            print(f"Frontend Explanation: {pipeline_data['explanation_text']}")
        else:
            print(f"FAIL: Frontend pipeline failed to enter SAFE MODE. Output: {pipeline_data}")
            sys.exit(1)
            
        # Simulate broken inference model file
        model_path = os.path.join('models', 'xgboost_model.pkl')
        if os.path.exists(model_path):
            shutil.move(model_path, model_path + '.broken')
            import module12_inference
            try:
                # Should abort inference gracefully
                res = module12_inference.load_inference_model()
                if res is None:
                    print("PASS: Missing model handled gracefully.")
                else:
                    print("FAIL: Missing model did not return None.")
                    sys.exit(1)
            finally:
                shutil.move(model_path + '.broken', model_path)
                
    finally:
        # Restore files
        shutil.move(inf_path + ".bak", inf_path) if os.path.exists(inf_path + ".bak") else None
        shutil.move(daily_pred + ".bak", daily_pred) if os.path.exists(daily_pred + ".bak") else None

    print("\n--- RECOVERY REPLAY VALIDATION ---")
    # Verify app still runs perfectly with restored files
    pipeline_data_restored = app.run_live_pipeline()
    if not pipeline_data_restored['is_safe_mode']:
        print("PASS: Deterministic recovery pathway verified. System fully functional after failure recovery.")
    else:
        print("FAIL: System remained stuck in SAFE MODE despite restored files.")
        sys.exit(1)

    print("\n==================================================")
    print("PHASE 13: ALL CHECKS PASSED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()
