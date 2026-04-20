import pandas as pd
import numpy as np
import logging
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# MODULE 18: WALK-FORWARD VALIDATION
# ==========================================
# Objective: Implement a strict expanding-window walk-forward backtest.
# This mathematically guarantees zero data leakage by physically enforcing 
# that the model is only ever trained on the past before predicting the future.

def perform_walk_forward(df: pd.DataFrame, train_window: int = 252, step_size: int = 20) -> pd.DataFrame:
    """
    Executes walk-forward testing.
    train_window: Initial training size (e.g., 252 days = 1 year)
    step_size: Retraining frequency (e.g., 20 days = 1 month)
    """
    # 1. STRICT CHRONOLOGICAL SORTING
    df = df.sort_values('date').reset_index(drop=True)
    
    X = df.drop(columns=['date', 'target_return_t+1', 'target_volatility_t+1'])
    y = df['target_return_t+1']
    dates = df['date']
    
    predictions = []
    actuals = []
    pred_dates = []
    
    # We use the baseline XGBoost architecture for the walk-forward
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    
    logging.info(f"Starting Walk-Forward Loop (Initial Window: {train_window} days, Retrain Step: {step_size} days)")
    
    # 2. WALK-FORWARD LOOP
    for i in range(train_window, len(df), step_size):
        # SPLIT DATA (Expanding Window)
        # Train on EVERYTHING up to day i
        train_X = X.iloc[:i] 
        train_y = y.iloc[:i]
        
        # Test exclusively on the unknown future (i to i + step_size)
        test_end = min(i + step_size, len(df))
        test_X = X.iloc[i:test_end]
        test_y = y.iloc[i:test_end]
        test_dates = dates.iloc[i:test_end]
        
        # RETRAIN & PREDICT NEXT PERIOD
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        
        # Store chronologically pure predictions
        predictions.extend(preds)
        actuals.extend(test_y)
        pred_dates.extend(test_dates)
        
        if (i - train_window) % (step_size * 6) == 0:
            progress = (i / len(df)) * 100
            logging.info(f"Walk-Forward Progress: {progress:.1f}% -> Retrained on {len(train_X)} historical rows.")
            
    # Combine results
    results_df = pd.DataFrame({
        'date': pred_dates,
        'predicted_return': predictions,
        'actual_return': actuals
    })
    
    return results_df

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the chronologically pure out-of-sample edge."""
    # Strict directional signal (ignoring volatility sizing to purely isolate predictive accuracy)
    df['signal'] = np.where(df['predicted_return'] > 0, 1.0, 0.0)
    
    # Strategy Return (without transaction costs for pure edge isolation)
    df['strategy_return'] = df['signal'] * df['actual_return']
    df['benchmark_return'] = df['actual_return']
    
    # 3. TRACK ACCURACY
    df['correct_direction'] = np.sign(df['predicted_return']) == np.sign(df['actual_return'])
    accuracy = df['correct_direction'].mean() * 100
    
    # 3. TRACK SHARPE
    days = 252
    ann_ret = df['strategy_return'].mean() * days
    ann_vol = df['strategy_return'].std() * np.sqrt(days)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # 3. TRACK DRAWDOWN
    cum_ret = np.exp(df['strategy_return'].cumsum())
    peak = cum_ret.cummax().replace(0, 1e-6)
    mdd = ((cum_ret - peak) / peak).min() * 100
    
    logging.info("==================================================")
    logging.info("       WALK-FORWARD VALIDATION RESULTS            ")
    logging.info("==================================================")
    logging.info(f"Total Out-of-Sample Days : {len(df)}")
    logging.info(f"Pure Directional Accuracy: {accuracy:.2f}%")
    logging.info(f"Annualized Raw Return    : {ann_ret*100:.2f}%")
    logging.info(f"Predictive Sharpe Ratio  : {sharpe:.2f}")
    logging.info(f"Unscaled Maximum Drawdown: {mdd:.2f}%")
    logging.info("==================================================")
    
    return df

def main():
    logging.info("Starting Module 18: Walk-Forward Validation Engine")
    
    data_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(data_path):
        logging.error("Missing final_dataset.csv. Run module 4 first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)
    
    # Execute the walk-forward simulation
    # Initial train window: 252 days (1 trading year)
    # Step size: 20 days (retrain the model every ~1 trading month)
    wf_results = perform_walk_forward(df, train_window=252, step_size=20)
    
    wf_results = compute_metrics(wf_results)
    
    # 4. OUTPUT TIME-SERIES PERFORMANCE
    out_path = os.path.join(os.getcwd(), 'outputs', 'walk_forward_trace.csv')
    wf_results.to_csv(out_path, index=False)
    logging.info(f"Chronologically pure time-series performance saved to {out_path}")

if __name__ == "__main__":
    main()
