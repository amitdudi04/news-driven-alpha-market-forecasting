import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def regime_persistence_audit():
    tracking_path = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if not os.path.exists(tracking_path):
        logging.warning("Missing live_tracking.csv")
        return
        
    df = pd.read_csv(tracking_path)
    if df.empty or 'position' not in df.columns:
        return
        
    # Reconstruct High/Low Vol from config logic or we can just infer it if we saved 'regime'
    # Actually, in module14 we didn't save 'regime', but we saved 'dir_prob' and 'position'.
    # We can infer regime from the market data or load daily_prediction.csv which has regime.
    pred_path = os.path.join(os.getcwd(), 'outputs', 'daily_prediction.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        df = pd.merge(df, pred_df[['date', 'regime']], on='date', how='left')
    else:
        # Fallback if no regime column
        df['regime'] = 'UNKNOWN'
        
    # Shift regime to align with strategy return (since position T generates return at T+1)
    df['regime_shifted'] = df['regime'].shift(1)
    
    # Transition periods
    df['is_transition'] = df['regime_shifted'] != df['regime_shifted'].shift(1)
    
    report_lines = ["# Regime Persistence & Alpha Decay Audit\n"]
    
    overall_status = "PASS"
    failures = []
    
    for r in ['HIGH_VOL', 'LOW_VOL']:
        regime_df = df[df['regime_shifted'] == r]
        if not regime_df.empty and len(regime_df) > 5:
            mean_ret = regime_df['pnl'].mean()
            std_ret = regime_df['pnl'].std() + 1e-9
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            expectancy = mean_ret
            
            p = (np.sign(regime_df['prediction'].shift(1) - 0.5) == np.sign(regime_df['actual_return'])).mean()
            mean_pred = regime_df['prediction'].shift(1).mean()
            ece = abs(mean_pred - p)
            
            report_lines.append(f"## Regime: {r}")
            report_lines.append(f"- **Samples**: {len(regime_df)}")
            report_lines.append(f"- **Annualized Sharpe**: {sharpe:.2f}")
            report_lines.append(f"- **Expectancy**: {expectancy:.4f}")
            report_lines.append(f"- **ECE (Calibration Error)**: {ece:.4f}\n")
            
            if expectancy < 0:
                failures.append(f"{r} Expectancy is persistently negative ({expectancy:.4f}).")
            if ece > 0.25:
                failures.append(f"{r} Calibration diverges catastrophically (ECE: {ece:.4f}).")
        else:
            report_lines.append(f"## Regime: {r}\n- Insufficient data.\n")
            
    # Transition Periods
    trans_df = df[df['is_transition'] & df['regime_shifted'].notna()]
    if not trans_df.empty and len(trans_df) > 5:
        trans_mean = trans_df['pnl'].mean()
        trans_std = trans_df['pnl'].std() + 1e-9
        trans_sharpe = (trans_mean / trans_std) * np.sqrt(252)
        report_lines.append(f"## Transition Periods")
        report_lines.append(f"- **Samples**: {len(trans_df)}")
        report_lines.append(f"- **Annualized Sharpe**: {trans_sharpe:.2f}")
        report_lines.append(f"- **Expectancy**: {trans_mean:.4f}\n")
        
        if trans_mean < -0.01:
            failures.append("Transition instability detected (High costs or structural failure on boundary crossings).")
            
    if failures:
        overall_status = "FAIL"
        report_lines.insert(1, f"## STATUS: {overall_status} ❌\n")
        report_lines.append("### Critical Failures:")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Regime Persistence Audit FAILED.")
    else:
        report_lines.insert(1, f"## STATUS: {overall_status} ✅\n")
        logging.info("Regime Persistence Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'regime_persistence_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError("SAFE MODE ESCALATION: Regime structural collapse detected.")

if __name__ == "__main__":
    regime_persistence_audit()
