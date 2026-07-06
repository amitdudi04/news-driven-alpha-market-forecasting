import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_cross_market_stress_engine():
    logging.info("Starting Cross-Market Stress Engine...")
    
    registry_path = os.path.join(os.getcwd(), 'config', 'asset_registry.json')
    if not os.path.exists(registry_path):
        logging.error("Asset registry missing. Cannot run cross-market stress.")
        return
        
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    assets = registry.get("assets", {})
    
    dfs = {}
    for name, config in assets.items():
        dp = os.path.join(os.getcwd(), config['data_path'])
        if os.path.exists(dp):
            df = pd.read_csv(dp)
            df['date'] = pd.to_datetime(df['date'])
            dfs[name] = df.set_index('date').sort_index()
            
    if len(dfs) < 2:
        logging.warning("Insufficient multi-asset data for cross-market stress analysis. Need at least 2.")
        return
        
    # Align all on date
    aligned_df = pd.DataFrame()
    for name, df in dfs.items():
        if 'return' in df.columns:
            aligned_df[f"{name}_return"] = df['return']
        if 'volatility' in df.columns:
            aligned_df[f"{name}_vol"] = df['volatility']
            
    aligned_df = aligned_df.dropna()
    
    if aligned_df.empty:
        logging.warning("No overlapping dates found across assets.")
        return
        
    # 1. Correlation Spikes
    returns_cols = [c for c in aligned_df.columns if '_return' in c]
    roll_corr = aligned_df[returns_cols[0]].rolling(20).corr(aligned_df[returns_cols[1]])
    # A proxy for global correlation is the average pairwise correlation
    aligned_df['cross_asset_correlation'] = roll_corr
    
    # 2. Volatility Transmission
    vol_cols = [c for c in aligned_df.columns if '_vol' in c]
    aligned_df['global_volatility'] = aligned_df[vol_cols].mean(axis=1)
    roll_vol_mean = aligned_df['global_volatility'].rolling(60).mean()
    aligned_df['vol_transmission_ratio'] = aligned_df['global_volatility'] / roll_vol_mean
    
    # 3. Stress Classification
    def classify_stress(row):
        stress_level = 0
        if pd.isna(row['cross_asset_correlation']): return "NORMAL"
        
        # High correlation across assets usually implies systemic panic (equities move together)
        if row['cross_asset_correlation'] > 0.8: stress_level += 1
        
        if row['vol_transmission_ratio'] > 2.0: stress_level += 2
        elif row['vol_transmission_ratio'] > 1.5: stress_level += 1
            
        if stress_level == 0: return "NORMAL"
        elif stress_level <= 1: return "ELEVATED_STRESS"
        elif stress_level == 2: return "SYSTEMIC_STRESS"
        else: return "SAFE_MODE_LOCKED"
        
    aligned_df['stress_state'] = aligned_df.apply(classify_stress, axis=1)
    
    latest_state = aligned_df['stress_state'].iloc[-1]
    
    failures = []
    if latest_state == "SAFE_MODE_LOCKED":
        failures.append("Extreme cross-market volatility contagion detected.")
    if aligned_df['cross_asset_correlation'].iloc[-1] > 0.95:
        failures.append("Correlation collapse (correlation -> 1.0) detected.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'cross_market_stress_manifest.csv')
    aligned_df.reset_index()[['date', 'cross_asset_correlation', 'global_volatility', 'vol_transmission_ratio', 'stress_state']].to_csv(out_path, index=False)
    
    logging.info(f"Cross Market Stress Analysis Complete. State: {latest_state}")
    
    if failures:
        logging.error(f"SAFE MODE ESCALATION: {failures}")
        raise RuntimeError(f"SAFE MODE ESCALATION: Cross-Market stress bounds exceeded: {failures}")

if __name__ == "__main__":
    run_cross_market_stress_engine()
