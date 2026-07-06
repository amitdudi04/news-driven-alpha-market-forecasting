import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# ==========================================
# MODULE 7: STRATEGY BACKTESTING
# ==========================================
# Objective: Backtest the ML predictions using rigorous volatility scaling, 
# strict turnover transaction costs, and accurate performance calculations.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions() -> pd.DataFrame:
    """Loads and strictly aligns predictions from Modules 5 and 6."""
    pred_path = os.path.join(os.getcwd(), 'outputs', 'predictions.csv')
    garch_path = os.path.join(os.getcwd(), 'outputs', 'garch_predictions.csv')
    
    if not os.path.exists(pred_path) or not os.path.exists(garch_path):
        logging.error("Missing prediction files in outputs/. Please run modules 5 and 6.")
        return None
        
    df_ret = pd.read_csv(pred_path)
    df_vol = pd.read_csv(garch_path)
    
    # Inner join on date
    df = pd.merge(df_ret, df_vol[['date', 'pred_vol_garch_x']], on='date', how='inner')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def simulate_strategy(df: pd.DataFrame, transaction_cost: float = 0.001) -> pd.DataFrame:
    """Applies trading logic, risk control, and transaction costs."""
    logging.info("Simulating Strategy with Volatility Risk Control...")
    
    # 1. Base Signal
    # Long if predicted return > 0. Flat if <= 0
    df['signal'] = np.where(df['predicted_return_t+1'] > 0, 1, 0)
    
    # 2. Risk Control (Volatility Scaling)
    target_vol = df['pred_vol_garch_x'].median()
    
    # Prevent division by zero
    safe_vol = np.where(df['pred_vol_garch_x'] == 0, 1e-6, df['pred_vol_garch_x'])
    raw_position = (target_vol / safe_vol) * df['signal']
    
    # Enforce leverage limit (clip to [-1, 1])
    df['position'] = np.clip(raw_position, -1.0, 1.0)
    
    # 3. Transaction Costs
    # turnover = abs(position_t - position_{t-1})
    df['turnover'] = df['position'].diff().fillna(df['position']).abs()
    df['t_cost'] = df['turnover'] * transaction_cost
    
    # 4. Strategy Return (Net of Costs)
    df['strategy_return'] = (df['position'] * df['actual_return_t+1']) - df['t_cost']
    df['benchmark_return'] = df['actual_return_t+1']
    
    # 5. Cumulative Returns
    df['cum_strategy'] = np.exp(df['strategy_return'].cumsum())
    df['cum_benchmark'] = np.exp(df['benchmark_return'].cumsum())
    
    return df

def compute_metrics(df: pd.DataFrame):
    """Calculates institutional-grade absolute and risk-adjusted metrics."""
    days = 252 # Annualization factor
    
    strat_tot = df['cum_strategy'].iloc[-1] - 1
    bench_tot = df['cum_benchmark'].iloc[-1] - 1
    
    strat_ann_ret = df['strategy_return'].mean() * days
    bench_ann_ret = df['benchmark_return'].mean() * days
    
    strat_ann_vol = df['strategy_return'].std() * np.sqrt(days)
    bench_ann_vol = df['benchmark_return'].std() * np.sqrt(days)
    
    # Sharpe Ratio: mean / std * sqrt(252)
    strat_sharpe = strat_ann_ret / strat_ann_vol if strat_ann_vol > 0 else 0
    bench_sharpe = bench_ann_ret / bench_ann_vol if bench_ann_vol > 0 else 0
    
    def max_dd(cum):
        peak = cum.cummax()
        return ((cum - peak) / peak).min()
        
    strat_mdd = max_dd(df['cum_strategy'])
    bench_mdd = max_dd(df['cum_benchmark'])
    
    # Hit Ratio: percentage of winning trades when a position is held
    active = df[df['position'] != 0]
    if len(active) > 0:
        hit_ratio = (active['actual_return_t+1'] * np.sign(active['position']) > 0).mean() * 100
    else:
        hit_ratio = 0
        
    logging.info("==================================================")
    logging.info("       STRATEGY BACKTEST PERFORMANCE REPORT       ")
    logging.info("==================================================")
    logging.info(f"Cumulative Return  | Strat: {strat_tot*100:6.2f}% | Bench: {bench_tot*100:6.2f}%")
    logging.info(f"Annualized Return  | Strat: {strat_ann_ret*100:6.2f}% | Bench: {bench_ann_ret*100:6.2f}%")
    logging.info(f"Annualized Vol     | Strat: {strat_ann_vol*100:6.2f}% | Bench: {bench_ann_vol*100:6.2f}%")
    logging.info(f"Sharpe Ratio       | Strat: {strat_sharpe:6.2f}  | Bench: {bench_sharpe:6.2f}")
    logging.info(f"Maximum Drawdown   | Strat: {strat_mdd*100:6.2f}% | Bench: {bench_mdd*100:6.2f}%")
    logging.info(f"Hit Ratio (Active) | Strat: {hit_ratio:6.2f}%  | Bench: N/A")
    logging.info("==================================================")

def plot_performance(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['date'], df['cum_strategy'], label='News-Driven Strategy', color='#1f77b4', linewidth=2)
    plt.plot(df['date'], df['cum_benchmark'], label='CSI 300 Benchmark', color='#7f7f7f', alpha=0.7)
    
    plt.title('News-Driven Alpha vs CSI 300 Benchmark', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Base 1.0)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'outputs', 'strategy_performance.png')
    plt.savefig(out_path, dpi=300)
    logging.info(f"Performance plot saved to {out_path}")

def main():
    logging.info("Starting Module 7: Backtesting Validation")
    df = load_predictions()
    if df is None: return
    
    # 10 bps transaction cost
    bt_df = simulate_strategy(df, transaction_cost=0.001)
    
    compute_metrics(bt_df)
    plot_performance(bt_df)
    
    # Save Trade Trace
    out_path = os.path.join(os.getcwd(), 'outputs', 'backtest_trace.csv')
    bt_df.to_csv(out_path, index=False)
    logging.info(f"Backtest trace safely stored to {out_path}")

if __name__ == "__main__":
    main()
