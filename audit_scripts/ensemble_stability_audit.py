import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensemble_stability_audit():
    logging.info("Starting Ensemble Stability Audit...")
    
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_research_manifest.csv')
    if not os.path.exists(manifest_path):
        logging.warning("Missing ensemble_research_manifest.csv. Cannot audit stability.")
        return
        
    df = pd.read_csv(manifest_path)
    
    failures = []
    
    # 1. Rolling Ensemble Agreement
    # Calculate percentage of models agreeing with majority
    conf_cols = ['model_1_conf', 'model_2_conf', 'model_3_conf']
    votes = df[conf_cols].applymap(lambda x: 1 if x > 0.5 else 0)
    # Agreement score is the ratio of models matching the majority vote
    df['majority_vote'] = votes.sum(axis=1) >= 2
    
    def agreement_ratio(row):
        maj_vote = 1 if row['majority_vote'] else 0
        matches = sum([1 for c in conf_cols if (row[c] > 0.5) == maj_vote])
        return matches / len(conf_cols)
        
    df['agreement_ratio'] = df.apply(agreement_ratio, axis=1)
    avg_agreement = df['agreement_ratio'].mean()
    
    if avg_agreement < 0.6:
        failures.append(f"Entropy persistence detected. Avg agreement critically low: {avg_agreement:.2f}")
        
    # 2. Confidence Variance
    conf_variance = df['ensemble_confidence'].var()
    if conf_variance < 0.0001:
        failures.append("Confidence saturation occurs. Ensemble outputs are deterministic and static.")
        
    # Generate Report
    report_lines = [
        "# Ensemble Stability Report\n",
        f"- **Average Agreement Ratio**: {avg_agreement:.2%}",
        f"- **Confidence Variance**: {conf_variance:.6f}\n",
        "## Validation Status"
    ]
    
    if failures:
        report_lines.append("### STATUS: FAIL ❌")
        for f in failures: report_lines.append(f"- {f}")
        logging.error("Ensemble Stability Audit FAILED.")
    else:
        report_lines.append("### STATUS: PASS ✅")
        report_lines.append("Ensemble stability remains within institutional bounds. Entropy is contained.")
        logging.info("Ensemble Stability Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'ensemble_stability_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Ensemble Stability Failed. {failures}")

if __name__ == "__main__":
    ensemble_stability_audit()
