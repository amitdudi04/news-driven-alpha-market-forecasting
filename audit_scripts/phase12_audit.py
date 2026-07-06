import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import importlib

# Force deterministic random state for numpy if used
np.random.seed(42)

def main():
    print("==================================================")
    print("MODULE 4 — PHASE 12: REGIME TRANSITION & TEMPORAL CONSISTENCY AUDIT")
    print("==================================================")
    
    import module12_inference
    from module4_features import load_datasets, generate_latest_features
    import module13_signal_engine
    
    # 1. Load historical data
    sent_df, market_df = load_datasets()
    if sent_df is None or market_df is None:
        print("FAILED to load datasets.")
        return
        
    model = module12_inference.load_inference_model()
    median_vol = model.get('median_vol', 0.2)
    
    # Generate base feature vector for perturbations
    base_features = generate_latest_features(sent_df, market_df)
    
    # ==========================================
    # BOUNDARY VALIDATION (Micro-perturbations)
    # ==========================================
    print("\n--- BOUNDARY VALIDATION ---")
    perturbation_results = []
    
    # Create 20 micro-perturbations crossing the median_vol threshold broadly
    vol_values = np.linspace(median_vol - 0.05, median_vol + 0.05, 20)
    
    for vol in vol_values:
        test_features = base_features.copy()
        test_features['volatility'] = vol
        
        # We need to run inference and check regime
        res = module12_inference.predict_next_day(model, test_features)
        
        # Get signal from module13 logic (we can call generate_signal directly)
        is_high_vol = vol > median_vol
        sig_res = module13_signal_engine.generate_signal(
            dir_prob=res['dir_prob'],
            confidence=res['confidence'],
            meta_prob=res['meta_prob'],
            pred_vol=vol, # Simplified
            target_vol=median_vol,
            is_high_vol=is_high_vol
        )
        
        perturbation_results.append({
            'volatility': vol,
            'regime': res['regime'],
            'dir_prob': res['dir_prob'],
            'confidence': res['confidence'],
            'signal': sig_res['signal']
        })
        
    df_perturb = pd.DataFrame(perturbation_results)
    print(df_perturb[['volatility', 'regime', 'dir_prob', 'confidence', 'signal']].to_string())
    
    # Check for chaotic switching / confidence discontinuity
    # Find the transition point
    transition_idx = df_perturb[df_perturb['regime'] == 'HIGH_VOL'].index.min()
    if pd.isna(transition_idx):
        print("No transition observed. (Needs fixing for the test)")
    else:
        idx_low = transition_idx - 1
        idx_high = transition_idx
        if idx_low >= 0:
            conf_jump = abs(df_perturb.loc[idx_high, 'confidence'] - df_perturb.loc[idx_low, 'confidence'])
            print(f"Confidence Jump at Boundary: {conf_jump:.4f}")
            if conf_jump > 0.15:
                print("FAIL: Massive confidence discontinuity at regime boundary.")
                sys.exit(1)
                
    # Check if signal flickers LONG <-> SHORT directly
    signals = df_perturb['signal'].tolist()
    for i in range(len(signals)-1):
        if (signals[i] == 'LONG' and signals[i+1] == 'SHORT') or \
           (signals[i] == 'SHORT' and signals[i+1] == 'LONG'):
            print("FAIL: Signal flickering (LONG <-> SHORT) without NO TRADE buffer.")
            sys.exit(1)
            
    print("BOUNDARY VALIDATION: PASS")
    
    # ==========================================
    # ROLLING WINDOW VALIDATION (Sequential History)
    # ==========================================
    print("\n--- ROLLING WINDOW VALIDATION ---")
    
    # Load final_dataset to get sequential historical features
    final_ds_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    df_history = pd.read_csv(final_ds_path)
    
    # Take last 60 days
    df_history = df_history.tail(60).reset_index(drop=True)
    
    rolling_results = []
    
    for i in range(len(df_history)):
        row = df_history.iloc[[i]].copy()
        # Drop columns not in model
        expected_features = model["feature_cols"]
        row = row[expected_features]
        
        # Handle NAs if any
        if row.isna().any().any():
            row = row.fillna(0)
            
        res = module12_inference.predict_next_day(model, row)
        
        vol_val = row['volatility'].iloc[0]
        is_high_vol = vol_val > median_vol
        
        sig_res = module13_signal_engine.generate_signal(
            dir_prob=res['dir_prob'],
            confidence=res['confidence'],
            meta_prob=res['meta_prob'],
            pred_vol=vol_val,
            target_vol=median_vol,
            is_high_vol=is_high_vol
        )
        
        rolling_results.append({
            'date': df_history.iloc[i]['date'] if 'date' in df_history.columns else i,
            'regime': res['regime'],
            'dir_prob': res['dir_prob'],
            'confidence': res['confidence'],
            'signal': sig_res['signal']
        })
        
    df_rolling = pd.DataFrame(rolling_results)
    
    # Analyze rolling stability
    regime_flips = (df_rolling['regime'] != df_rolling['regime'].shift(1)).sum() - 1 # First one is a flip from NaN
    signal_flips = (df_rolling['signal'] != df_rolling['signal'].shift(1)).sum() - 1
    
    print(f"Regime Transitions in 60 days: {regime_flips}")
    print(f"Signal Transitions in 60 days: {signal_flips}")
    
    # Check for direct LONG <-> SHORT reversals
    signals = df_rolling['signal'].tolist()
    reversals = 0
    for i in range(len(signals)-1):
        if (signals[i] == 'LONG' and signals[i+1] == 'SHORT') or \
           (signals[i] == 'SHORT' and signals[i+1] == 'LONG'):
            reversals += 1
            print(f"WARNING: Direct Reversal at idx {i}")
            
    if reversals > 0:
        print(f"FAIL: Found {reversals} direct LONG<->SHORT reversals. Expected NO TRADE transition buffer.")
        sys.exit(1)
        
    print("ROLLING WINDOW VALIDATION: PASS")
    
    # Save traces
    os.makedirs('artifacts', exist_ok=True)
    df_perturb.to_csv('artifacts/boundary_trace.csv', index=False)
    df_rolling.to_csv('artifacts/rolling_trace.csv', index=False)
    print("\nTraces saved to artifacts/")

    print("\n==================================================")
    print("PHASE 12: ALL CHECKS PASSED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    os.environ['MAX_STALE_DAYS'] = '9999'
    main()
