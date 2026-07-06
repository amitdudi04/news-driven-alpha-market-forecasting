import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensemble_disagreement_audit():
    logging.info("Starting Ensemble Disagreement Audit...")
    
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_research_manifest.csv')
    if not os.path.exists(manifest_path):
        logging.warning("Missing ensemble manifest.")
        return
        
    df = pd.read_csv(manifest_path)
    
    failures = []
    
    # 1. Macro Disagreement Spikes (Is disagreement > 0.3 ?)
    max_disagreement = df['ensemble_disagreement'].max()
    avg_disagreement = df['ensemble_disagreement'].mean()
    
    if avg_disagreement > 0.2:
        failures.append(f"Ensemble disagreement is structurally chaotic (Avg: {avg_disagreement:.3f})")
        
    if max_disagreement > 0.4:
        failures.append(f"Severe regime disagreement spike detected (Max: {max_disagreement:.3f})")
        
    # Generate Report
    report_lines = [
        "# Ensemble Disagreement Report\n",
        f"- **Average Disagreement (Entropy)**: {avg_disagreement:.4f}",
        f"- **Maximum Disagreement Spike**: {max_disagreement:.4f}\n",
        "## Validation Status"
    ]
    
    if failures:
        report_lines.append("### STATUS: FAIL ❌")
        for f in failures: report_lines.append(f"- {f}")
        logging.error("Ensemble Disagreement Audit FAILED.")
    else:
        report_lines.append("### STATUS: PASS ✅")
        report_lines.append("Ensemble consensus is maintained across stress regimes.")
        logging.info("Ensemble Disagreement Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_disagreement_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Disagreement Failed. {failures}")

if __name__ == "__main__":
    ensemble_disagreement_audit()
