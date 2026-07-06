import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_advanced_predictions() -> pd.DataFrame:
    pred_path = os.path.join(os.getcwd(), 'outputs', 'advanced_predictions.csv')
    garch_path = os.path.join(os.getcwd(), 'outputs', 'garch_predictions.csv')
    
    if not os.path.exists(pred_path) or not os.path.exists(garch_path):
        logging.error("Missing advanced_predictions.csv or garch_predictions.csv")
        return None
        
    df_pred = pd.read_csv(pred_path)
    df_vol = pd.read_csv(garch_path)
    
    # Merge predictions with conditional volatility
    df = pd.merge(df_pred, df_vol[['date', 'pred_vol_garch_x']], on='date', how='inner')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def simulate_advanced_strategy(df: pd.DataFrame, transaction_cost: float = 0.001) -> pd.DataFrame:
    """Simulates the enhanced strategy with adaptive signal filtering and stateful risk management."""
    logging.info("Simulating Advanced Strategy with Institutional Risk Filters...")
    
    # Pre-calculate global constants
    target_vol = df['pred_vol_garch_x'].median()
    median_vol = target_vol
    
    # Initialize state variables
    current_peak = 1.0
    cum_strategy = 1.0
    consecutive_losses = 0
    
    positions = []
    strategy_returns = []
    t_costs = []
    
    # We must iterate row by row to apply stateful path-dependent risk filters
    # Note: For speed in massive datasets, this would be numba-compiled, but Python loops are fine for 3000 rows.
    prev_position = 0.0
    
    for i in range(len(df)):
        pred_return = df.at[i, 'advanced_pred_return']
        pred_vol = max(df.at[i, 'pred_vol_garch_x'], 1e-6)
        actual_return = df.at[i, 'actual_return_t+1']
        
        # 1. ADAPTIVE SIGNAL FILTERING
        # Ignore weak predictions buried in the noise
        threshold = 0.5 * pred_vol
        if abs(pred_return) < threshold:
            base_position = 0.0
        else:
            base_position = 1.0 if pred_return > 0 else 0.0 # Strict LONG or FLAT as per original logic
            
        # 2. BASE VOLATILITY SCALING
        base_weight = 1.0
        raw_position = base_position * base_weight * (target_vol / pred_vol)
        
        # 3. STATEFUL RISK MANAGEMENT (Protective Overlays)
        # Calculate current drawdown
        drawdown = (cum_strategy - current_peak) / current_peak
        
        # A) Drawdown Control
        if drawdown < -0.15:  # -15% drawdown
            raw_position *= 0.5
            
        # B) Consecutive Loss Filter
        if consecutive_losses >= 3:
            raw_position *= 0.5
            
        # C) Volatility Spike Protection
        if pred_vol > (2.0 * median_vol):
            raw_position *= 0.5
            
        # 4. SAFETY CLIPPING
        final_position = np.clip(raw_position, -1.0, 1.0)
        positions.append(final_position)
        
        # 5. EXECUTION & TRANSACTION COST (T to T+1)
        turnover = abs(final_position - prev_position)
        cost = turnover * transaction_cost
        
        strat_ret = (final_position * actual_return) - cost
        
        strategy_returns.append(strat_ret)
        t_costs.append(cost)
        
        # 6. UPDATE STATES FOR NEXT ITERATION
        cum_strategy *= np.exp(strat_ret)
        if cum_strategy > current_peak:
            current_peak = cum_strategy
            
        # Update consecutive losses based on the realized trade
        if final_position != 0:
            if strat_ret < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
        else:
            # If flat, decay or maintain loss count (we maintain it here until a winning trade breaks it)
            pass
            
        prev_position = final_position

    # Attach to dataframe
    df['position'] = positions
    df['t_cost'] = t_costs
    df['strategy_return'] = strategy_returns
    df['benchmark_return'] = df['actual_return_t+1']
    
    df['cum_strategy'] = np.exp(df['strategy_return'].cumsum())
    df['cum_benchmark'] = np.exp(df['benchmark_return'].cumsum())
    
    return df

def main():
    logging.info("Starting Module 16: Advanced Backtesting Engine")
    df = load_advanced_predictions()
    if df is None: return
    
    adv_bt_df = simulate_advanced_strategy(df)
    
    out_path = os.path.join(os.getcwd(), 'outputs', 'advanced_backtest_trace.csv')
    adv_bt_df.to_csv(out_path, index=False)
    logging.info(f"Advanced backtest trace safely stored to {out_path}")

if __name__ == "__main__":
    main()
