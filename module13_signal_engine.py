import pandas as pd
import numpy as np
import logging
import os

# ==========================================
# MODULE 13: TRADING SIGNAL ENGINE
# ==========================================
# Objective: Convert raw statistical predictions into actionable trading signals 
# safely bounded by institutional volatility scaling and leverage limits.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_signal(pred_return: float, pred_vol: float, target_vol: float, base_weight: float = 1.0) -> dict:
    """Calculates the execution signal and scales the position size based on risk."""
    
    # 1. SIGNAL LOGIC
    # Strict binary directional logic
    signal = 'LONG' if pred_return > 0 else 'FLAT'
    base_position = 1.0 if signal == 'LONG' else 0.0
    
    # 2. VOLATILITY SCALING (Risk Control)
    # Volatility is mathematically inversely proportional to position size
    # Prevent fatal division by zero via safety floor
    safe_vol = max(pred_vol, 1e-6)
    scaled_position = base_position * base_weight * (target_vol / safe_vol)
    
    # 3. SAFETY CONTROLS
    # Strictly clip leverage to prevent margin calls or overexposure during low-volatility anomalies
    final_position = float(np.clip(scaled_position, -1.0, 1.0))
    
    return {
        'signal': signal,
        'position_size': final_position,
        'pred_return': pred_return,
        'pred_vol': pred_vol
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
    latest_return = inf_df['Predicted_Return_t+1'].iloc[-1]
    latest_date = inf_df['Feature_Date'].iloc[-1]
    
    # Extract the conditional volatility forecast from the GARCH-ARX model
    latest_vol = garch_df['pred_vol_garch_x'].iloc[-1]
    
    # Dynamically calculate the historical target volatility median
    target_vol = garch_df['pred_vol_garch_x'].median()
    
    # Execute Trading Logic
    result = generate_signal(
        pred_return=latest_return, 
        pred_vol=latest_vol, 
        target_vol=target_vol
    )
    
    logging.info("==================================================")
    logging.info("           FINAL EXECUTION TICKET                 ")
    logging.info("==================================================")
    logging.info(f"Target Predict Date : Tomorrow (T+1)")
    logging.info(f"Directional Signal  : {result['signal']}")
    logging.info(f"Predicted Return    : {result['pred_return']*100:+.4f}%")
    logging.info(f"Predicted Volatility: {result['pred_vol']*100:.4f}%")
    logging.info(f"Target Volatility   : {target_vol*100:.4f}%")
    logging.info(f"Scaled Position Size: {result['position_size']:.4f}x Leverage")
    logging.info("==================================================")
    
    # 1. OUTPUT FORMAT
    log_data = {
        'date': latest_date,
        'predicted_return': result['pred_return'],
        'predicted_volatility': result['pred_vol'],
        'signal': result['signal'],
        'position': result['position_size'],
        'model_version': 'v1.0'
    }
    log_df = pd.DataFrame([log_data])
    
    # 2. SAVE: outputs/daily_prediction.csv
    out_path = os.path.join(os.getcwd(), 'outputs', 'daily_prediction.csv')
    
    # 3. APPEND MODE
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        # Prevent duplicating the same execution day
        if not (existing_df['date'] == latest_date).any():
            final_df = pd.concat([existing_df, log_df], ignore_index=True)
            final_df.to_csv(out_path, index=False)
            logging.info(f"Successfully APPENDED daily prediction for {latest_date} to {out_path}")
        else:
            # Overwrite if the date already exists to maintain correct latest signal
            existing_df.loc[existing_df['date'] == latest_date, :] = log_df.values
            existing_df.to_csv(out_path, index=False)
            logging.info(f"UPDATED existing prediction for {latest_date} in {out_path}")
    else:
        log_df.to_csv(out_path, index=False)
        logging.info(f"Created new logging file and saved initial record to {out_path}")

if __name__ == "__main__":
    main()
