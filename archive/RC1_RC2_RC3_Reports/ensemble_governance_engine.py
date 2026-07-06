import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ensemble_governance_engine():
    logging.info("Starting Ensemble Governance Engine...")
    
    track_path = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if not os.path.exists(track_path):
        logging.warning("Missing live_tracking.csv. Cannot simulate ensemble without base predictions.")
        return
        
    df = pd.read_csv(track_path)
    if df.empty or 'confidence_raw' not in df.columns:
        logging.warning("Insufficient data in live_tracking.csv for ensemble simulation.")
        return
        
    # Simulate Ensemble Predictions
    # Model 1: The Incumbent (live)
    # Model 2: A shadow variant (slightly noisier)
    # Model 3: A macro-aware variant (more conservative)
    
    np.random.seed(42) # For deterministic simulation
    df['model_1_conf'] = df['confidence_raw']
    df['model_2_conf'] = df['confidence_raw'] + np.random.normal(0, 0.05, len(df))
    df['model_3_conf'] = df['confidence_raw'] * 0.9 + 0.05
    
    # Clip between 0 and 1
    for col in ['model_1_conf', 'model_2_conf', 'model_3_conf']:
        df[col] = df[col].clip(0, 1)
        
    # Ensemble Aggregation
    conf_cols = ['model_1_conf', 'model_2_conf', 'model_3_conf']
    
    # 1. Confidence Averaging
    df['ensemble_confidence'] = df[conf_cols].mean(axis=1)
    
    # 2. Majority Vote
    # Signal is 1 if conf > 0.5 else 0
    votes = df[conf_cols].applymap(lambda x: 1 if x > 0.5 else 0)
    df['ensemble_majority_vote'] = votes.sum(axis=1) >= 2
    
    # 3. Disagreement Scoring (Entropy / Variance)
    df['ensemble_disagreement'] = df[conf_cols].std(axis=1)
    
    # SAFE MODE Constraints
    failures = []
    
    if df['ensemble_confidence'].isna().any() or np.isinf(df['ensemble_confidence']).any():
        failures.append("Ensemble outputs contain NaN or Inf.")
        
    latest_disagreement = df['ensemble_disagreement'].iloc[-1]
    # If standard deviation of probabilities across 3 models > 0.3, it's highly chaotic
    if latest_disagreement > 0.3:
        failures.append(f"Disagreement exploded uncontrollably: {latest_disagreement:.3f}")
        
    # Generate Output Manifest
    manifest_cols = ['date', 'model_1_conf', 'model_2_conf', 'model_3_conf', 
                     'ensemble_confidence', 'ensemble_majority_vote', 'ensemble_disagreement']
    
    manifest_df = df[manifest_cols].copy()
    out_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_research_manifest.csv')
    manifest_df.to_csv(out_path, index=False)
    
    logging.info(f"Ensemble Research Manifest generated at {out_path}")
    logging.info("Ensemble outputs are advisory only. Execution logic remains unmutated.")
    
    if failures:
        logging.error(f"SAFE MODE ESCALATION: {failures}")
        raise RuntimeError(f"SAFE MODE ESCALATION: Ensemble boundaries violated: {failures}")
        
if __name__ == "__main__":
    run_ensemble_governance_engine()
