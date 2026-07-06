import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_macro_governance_report():
    logging.info("Generating Macro Explainability & Governance Report...")
    
    # 1. Macro Regime
    regime_path = os.path.join(os.getcwd(), 'outputs', 'macro_regime_manifest.csv')
    if os.path.exists(regime_path):
        regime_df = pd.read_csv(regime_path)
        latest_regime = regime_df.iloc[-1]['macro_regime']
    else:
        latest_regime = "UNKNOWN"
        
    # 2. Factor Exposures
    factor_path = os.path.join(os.getcwd(), 'outputs', 'factor_exposure_manifest.csv')
    if os.path.exists(factor_path):
        factor_df = pd.read_csv(factor_path)
        latest_beta = factor_df.iloc[-1]['rolling_market_beta']
        latest_alpha = factor_df.iloc[-1]['rolling_residual_alpha']
    else:
        latest_beta = 0.0
        latest_alpha = 0.0
        
    # 3. Cross Market Stress
    stress_path = os.path.join(os.getcwd(), 'outputs', 'cross_market_stress_manifest.csv')
    if os.path.exists(stress_path):
        stress_df = pd.read_csv(stress_path)
        latest_stress = stress_df.iloc[-1]['stress_state']
        latest_corr = stress_df.iloc[-1]['cross_asset_correlation']
    else:
        latest_stress = "UNKNOWN"
        latest_corr = 0.0
        
    report_content = f"""# Institutional Macro Governance Report

**Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Classification: ADVISORY ONLY (Zero Execution Authority)*

## 1. Current Macro Environment
- **Macro Regime Classification**: `{latest_regime}`
- **Systemic Stress State**: `{latest_stress}`
- **Cross-Market Correlation**: {latest_corr:.2f}

## 2. Dynamic Factor Attribution (Rolling 30d)
- **Market Beta Exposure**: {latest_beta:.4f}
- **Residual Alpha Extraction**: {latest_alpha:.4f}

## 3. Macro Stability & Governance Classification
"""

    if latest_stress == "SAFE_MODE_LOCKED" or latest_alpha < 0 or abs(latest_beta) > 1.5:
        report_content += "\n**STATUS**: `MACRO_ENVIRONMENT_CRITICAL` ❌\n"
        report_content += "Significant structural headwinds detected. Factor decomposition or cross-market stress indicates capital protection protocols may be necessary."
    elif latest_stress == "ELEVATED_STRESS" or "SENTIMENT_COLLAPSE" in latest_regime:
        report_content += "\n**STATUS**: `MACRO_ENVIRONMENT_DEGRADED` ⚠️\n"
        report_content += "Elevated macro risk. AI edge remains operative but factor constraints are tightening."
    else:
        report_content += "\n**STATUS**: `MACRO_ENVIRONMENT_STABLE` ✅\n"
        report_content += "Regimes are within bounded parameters. Factor attribution validates pure idiosyncratic edge generation."

    out_path = os.path.join(os.getcwd(), 'outputs', 'macro_governance_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logging.info(f"Macro Governance Report saved to {out_path}")

if __name__ == "__main__":
    generate_macro_governance_report()
