import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calibration_decay_audit():
    tracking_path = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if not os.path.exists(tracking_path):
        logging.warning("Missing live_tracking.csv")
        return
        
    df = pd.read_csv(tracking_path)
    if df.empty or 'confidence_raw' not in df.columns:
        return
        
    # We need prediction (dir_prob), confidence, and actual return
    df['is_correct'] = (np.sign(df['prediction'].shift(1) - 0.5) == np.sign(df['actual_return'])).astype(int)
    
    report_lines = ["# Long-Horizon Calibration Decay Audit\n"]
    overall_status = "PASS"
    failures = []
    
    # 1. Brier & ECE Stability
    # We can just look at the latest rolling metrics or compute globally
    brier = ((df['prediction'].shift(1) - (df['actual_return'] > 0).astype(int)) ** 2).mean()
    ece = abs(df['prediction'].shift(1).mean() - df['is_correct'].mean())
    
    report_lines.append("## 1. Global Calibration Stability")
    report_lines.append(f"- **Global Brier Score**: {brier:.4f}")
    report_lines.append(f"- **Global ECE**: {ece:.4f}\n")
    
    if brier > 0.25:
        failures.append(f"Global Brier Score exceeds 0.25 ({brier:.4f})")
    
    # 2. Confidence Usefulness (Terciles)
    # Does higher confidence lead to higher accuracy?
    df_valid = df.dropna(subset=['confidence_raw', 'is_correct'])
    if len(df_valid) > 10:
        try:
            df_valid['conf_tercile'] = pd.qcut(df_valid['confidence_raw'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            acc_by_tercile = df_valid.groupby('conf_tercile')['is_correct'].mean()
            
            report_lines.append("## 2. Confidence Tercile Usefulness")
            report_lines.append(f"- **Low Confidence Accuracy**: {acc_by_tercile.get('Low', 0)*100:.1f}%")
            report_lines.append(f"- **Medium Confidence Accuracy**: {acc_by_tercile.get('Medium', 0)*100:.1f}%")
            report_lines.append(f"- **High Confidence Accuracy**: {acc_by_tercile.get('High', 0)*100:.1f}%\n")
            
            if acc_by_tercile.get('High', 0) < acc_by_tercile.get('Low', 0):
                failures.append("Confidence Inversion: High confidence has LOWER accuracy than Low confidence.")
        except Exception as e:
            report_lines.append("## 2. Confidence Tercile Usefulness\n- Not enough variance to compute terciles.\n")
    
    # 3. Probability Saturation
    prob_saturation = (df['prediction'] > 0.95).sum() + (df['prediction'] < 0.05).sum()
    pct_saturation = prob_saturation / len(df)
    report_lines.append("## 3. Probability Saturation")
    report_lines.append(f"- **Saturated Predictions (<0.05 or >0.95)**: {pct_saturation*100:.1f}%\n")
    if pct_saturation > 0.5:
        failures.append(f"Probability Saturation detected: {pct_saturation*100:.1f}% of predictions are extreme.")
        
    if failures:
        overall_status = "FAIL"
        report_lines.insert(1, f"## STATUS: {overall_status} ❌\n")
        report_lines.append("### Critical Failures:")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Calibration Decay Audit FAILED.")
    else:
        report_lines.insert(1, f"## STATUS: {overall_status} ✅\n")
        logging.info("Calibration Decay Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'calibration_decay_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError("SAFE MODE ESCALATION: Severe Calibration Collapse detected.")

if __name__ == "__main__":
    calibration_decay_audit()
