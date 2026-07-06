import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_ensemble_governance_report():
    logging.info("Generating Ensemble Governance Report...")
    
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_research_manifest.csv')
    if not os.path.exists(manifest_path):
        logging.warning("Missing ensemble manifest.")
        return
        
    df = pd.read_csv(manifest_path)
    
    latest = df.iloc[-1]
    avg_disagreement = df['ensemble_disagreement'].mean()
    max_disagreement = df['ensemble_disagreement'].max()
    
    # 1. Consensus Statistics
    majority_votes = df['ensemble_majority_vote'].sum()
    majority_pct = majority_votes / len(df)
    
    report_content = f"""# Institutional Ensemble Governance Report

**Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Classification: ADVISORY ONLY (Zero Execution Authority)*

## 1. Ensemble Composition (Research Sandbox)
- **Model 1**: Incumbent Baseline (Weight: 1.0)
- **Model 2**: Shadow Variant (Weight: 1.0)
- **Model 3**: Macro-Aware Variant (Weight: 1.0)
*Status: All models isolated in non-execution environment.*

## 2. Consensus Statistics
- **Latest Ensemble Confidence**: {latest['ensemble_confidence']:.4f}
- **Historical Majority Vote Consistency**: {majority_pct:.2%}

## 3. Disagreement States & Entropy
- **Average Disagreement (Entropy)**: {avg_disagreement:.4f}
- **Maximum Disagreement Spike**: {max_disagreement:.4f}

## 4. Stability Classification
"""

    if avg_disagreement > 0.2 or max_disagreement > 0.4:
        report_content += "\n**STATUS**: `ENSEMBLE_STABILITY_DEGRADED` ⚠️\n"
        report_content += "High entropy detected. Models are disagreeing significantly on regime transitions."
    else:
        report_content += "\n**STATUS**: `ENSEMBLE_STABLE` ✅\n"
        report_content += "Ensemble consensus is structurally stable. Models are harmonized."

    out_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_governance_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logging.info(f"Ensemble Governance Report saved to {out_path}")

if __name__ == "__main__":
    generate_ensemble_governance_report()
