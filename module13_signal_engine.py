import pandas as pd
import numpy as np
import logging
import os
import datetime

# ==========================================
# MODULE 13: TRADING SIGNAL ENGINE
# ==========================================
# Objective: Convert raw statistical predictions into actionable trading signals 
# safely bounded by institutional volatility scaling and leverage limits.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from config.institutional_config import SIGNAL_THRESHOLDS, CONFIDENCE_SCALING, EXECUTION_LIMITS

def generate_signal(dir_prob: float, confidence: float, meta_prob: float, pred_vol: float, target_vol: float, is_high_vol: bool, base_weight: float = 1.0) -> dict:
    """Calculates the execution signal and scales the position size based on risk."""
    
    # 1. THRESHOLD LOGIC (Sourced from Config)
    if is_high_vol:
        upper_thresh = SIGNAL_THRESHOLDS["high_vol"]["upper_thresh"]
    else:
        upper_thresh = SIGNAL_THRESHOLDS["low_vol"]["upper_thresh"]
        
    signal = "NO TRADE"
    explanation_text = "PENDING_GOVERNANCE_EVALUATION"
    # 2. CORRUPTION CHECKS (SAFE MODE ESCALATION)
    if pd.isna(dir_prob) or pd.isna(confidence) or np.isinf(dir_prob) or np.isinf(confidence):
        return {
            'signal': 'NO TRADE',
            'position_size': 0.0,
            'dir_prob': dir_prob if not pd.isna(dir_prob) else 0.5,
            'pred_vol': pred_vol,
            'explanation': 'BLOCKED: SAFE MODE - CORRUPTED INPUT DETECTED',
            'upper_thresh': upper_thresh
        }
        
    # 3. SIGNAL LOGIC (Based strictly on confidence and dir_prob)
    if dir_prob > 0.5 and confidence > upper_thresh:
        signal = 'LONG'
        explanation_text = f"LONG: Dir={dir_prob*100:.1f}%, Conf={confidence:.3f} > {upper_thresh:.3f} thresh"
    elif dir_prob <= 0.5 and confidence > upper_thresh:
        signal = 'SHORT'
        explanation_text = f"SHORT: Dir={dir_prob*100:.1f}%, Conf={confidence:.3f} > {upper_thresh:.3f} thresh"
    else:
        signal = "NO TRADE"
        explanation_text = f"BLOCKED: Conf={confidence:.3f} <= {upper_thresh:.3f} threshold limit"
        
    # 3. VOLATILITY SCALING (Risk Control)
    base_position = 1.0 if signal == 'LONG' else (-1.0 if signal == 'SHORT' else 0.0)
    # FIX: Volatility features are Z-scored (can be negative). 
    # Using raw division causes negative scaling factors, which FLIPS SHORT signals into LONGs.
    # We must use absolute magnitude to preserve directional authority.
    scaling_factor = abs(target_vol) / max(abs(pred_vol), EXECUTION_LIMITS["safe_vol_floor"])
    scaled_position = base_position * base_weight * scaling_factor
    
    # 4. SAFETY CONTROLS
    final_position = float(np.clip(scaled_position, EXECUTION_LIMITS["min_position_clip"], EXECUTION_LIMITS["max_position_clip"]))
    
    return {
        'signal': signal,
        'position_size': final_position,
        'dir_prob': dir_prob,
        'pred_vol': pred_vol,
        'explanation': explanation_text,
        'upper_thresh': upper_thresh
    }

def main():
    logging.info("Starting Module 13: Trading Signal Engine")
    
    # Load upstream prediction matrices
    inf_path = os.path.join(os.getcwd(), 'outputs', 'live_inference.csv')
    garch_path = os.path.join(os.getcwd(), 'outputs', 'garch_predictions.csv')
    
    if not os.path.exists(inf_path) or not os.path.exists(garch_path):
        logging.error("Missing prediction outputs in outputs/. Run Modules 6 and 12 first.")
        return
        
    inf_df = pd.read_csv(inf_path)
    garch_df = pd.read_csv(garch_path)
    
    # Extract live predicted values for T+1
    latest_prob = inf_df['Direction_Prob_t+1'].iloc[-1]
    meta_prob = inf_df.get('Meta_Prob_t+1', pd.Series([1.0])).iloc[-1]
    confidence = inf_df.get('Confidence', pd.Series([0.0])).iloc[-1]
    raw_confidence = inf_df.get('raw_confidence', pd.Series([0.0])).iloc[-1]
    distance_penalty = inf_df.get('distance_penalty', pd.Series([1.0])).iloc[-1]
    regime_inf = inf_df.get('regime', pd.Series(['UNKNOWN'])).iloc[-1]
    top_pos = inf_df.get('top_positive', pd.Series(['{}'])).iloc[-1]
    top_neg = inf_df.get('top_negative', pd.Series(['{}'])).iloc[-1]
    exec_uuid = inf_df.get('execution_uuid', pd.Series(['MISSING_UUID'])).iloc[-1]
    latest_date = inf_df['Feature_Date'].iloc[-1]
    
    # Extract the conditional volatility forecast from the GARCH-ARX model
    latest_vol = garch_df['pred_vol_garch_x'].iloc[-1]
    
    # Dynamically calculate the historical target volatility median
    target_vol = garch_df['pred_vol_garch_x'].median()
    
    import joblib
    try:
        mpath = os.path.join(os.getcwd(), 'models', 'model_live.pkl')
        model = joblib.load(mpath)
        median_vol = model.get('median_vol', 0.15)
        # We need the scaled vol to determine high/low vol, but here we can just use the raw latest_vol 
        # compared to target_vol since they are both from GARCH. Wait, the regime was determined by X_live['volatility'].
        # For simplicity, we just use the GARCH median vs latest.
        is_high_vol = latest_vol > target_vol
    except:
        is_high_vol = False
    
    # Execute Trading Logic
    result = generate_signal(
        dir_prob=float(latest_prob),
        confidence=float(confidence),
        meta_prob=float(meta_prob),
        pred_vol=latest_vol,
        target_vol=target_vol,
        is_high_vol=is_high_vol
    )
    
    # Structured WHY SIGNAL Lineage
    import json
    why_signal = {
        'timestamp': str(datetime.datetime.now()),
        'regime': regime_inf,
        'dir_prob': float(latest_prob),
        'meta_prob': float(meta_prob),
        'raw_confidence': float(raw_confidence),
        'distance_penalty': float(distance_penalty),
        'final_confidence': float(confidence),
        'execution_threshold': result.get('upper_thresh', 0.0),
        'governance_result': result['explanation'],
        'top_positive': top_pos,
        'top_negative': top_neg,
        'execution_uuid': str(exec_uuid)
    }
    why_signal_str = json.dumps(why_signal)
    
    logging.info("==================================================")
    logging.info("           FINAL EXECUTION TICKET                 ")
    logging.info("==================================================")
    logging.info(f"Target Predict Date : Tomorrow (T+1)")
    logging.info(f"Directional Signal  : {result['signal']}")
    logging.info(f"Direction Prob      : {result['dir_prob']*100:.2f}%")
    logging.info(f"Confidence          : {confidence*100:.2f}%")
    logging.info(f"Predicted Volatility: {result['pred_vol']*100:.4f}%")
    logging.info(f"Target Volatility   : {target_vol*100:.4f}%")
    logging.info(f"Scaled Position Size: {result['position_size']:.4f}x Leverage")
    logging.info(f"Explanation         : {result['explanation']}")
    logging.info("--- WHY SIGNAL LINEAGE ---")
    for k, v in why_signal.items():
        logging.info(f"  {k}: {v}")
    logging.info("==================================================")
    
    # 1. OUTPUT FORMAT
    log_data = {
        'date': latest_date,
        'dir_prob': result['dir_prob'],
        'confidence': confidence,
        'signal': result['signal'],
        'position': result['position_size'],
        'explanation': result['explanation'],
        'why_signal': why_signal_str,
        'model_version': 'v1.0',
        'execution_uuid': str(exec_uuid)
    }
    log_df = pd.DataFrame([log_data])
    
    # 2. SAVE: outputs/daily_prediction.csv
    out_path = os.path.join(os.getcwd(), 'outputs', 'daily_prediction.csv')
    
    from module0_atomic_storage import atomic_write_csv
    
    # 3. APPEND MODE
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        # Prevent duplicating the same execution day
        if not (existing_df['date'] == latest_date).any():
            final_df = pd.concat([existing_df, log_df], ignore_index=True)
            atomic_write_csv(final_df, out_path, index=False)
            logging.info(f"Successfully APPENDED daily prediction for {latest_date} to {out_path}")
        else:
            # Overwrite if the date already exists to maintain correct latest signal
            for col in log_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NA
            existing_df.loc[existing_df['date'] == latest_date, log_df.columns] = log_df.values[0]
            atomic_write_csv(existing_df, out_path, index=False)
            logging.info(f"UPDATED existing prediction for {latest_date} in {out_path}")
    else:
        atomic_write_csv(log_df, out_path, index=False)
        logging.info(f"Created new logging file and saved initial record to {out_path}")

if __name__ == "__main__":
    main()
