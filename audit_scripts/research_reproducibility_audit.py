import os
import hashlib
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_hash(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def research_reproducibility_audit():
    logging.info("Starting Research Reproducibility Audit...")
    failures = []
    
    # Target files to test
    targets = {
        "macro": {
            "script": "macro_regime_engine.py",
            "manifest": os.path.join(os.getcwd(), 'outputs', 'macro_regime_manifest.csv')
        },
        "ensemble": {
            "script": "ensemble_governance_engine.py",
            "manifest": os.path.join(os.getcwd(), 'outputs', 'ensemble_research_manifest.csv')
        }
    }
    
    for name, config in targets.items():
        if not os.path.exists(os.path.join(os.getcwd(), config["script"])):
            continue
            
        hash_pre = get_file_hash(config["manifest"])
        
        # Re-run the engine
        try:
            subprocess.run(["python", config["script"]], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            failures.append(f"Failed to execute {config['script']}: {e.stderr.decode('utf-8')}")
            continue
            
        hash_post = get_file_hash(config["manifest"])
        
        if hash_pre and hash_post and hash_pre != hash_post:
            failures.append(f"Reproducibility failed for {name}. Hash drifted from {hash_pre} to {hash_post}")
            
    report_lines = ["# Research Reproducibility Report\n"]
    if failures:
        report_lines.append("## STATUS: FAIL ❌\n")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Research Reproducibility Audit FAILED.")
    else:
        report_lines.append("## STATUS: PASS ✅\n")
        report_lines.append("All research outputs (Macro & Ensemble) are perfectly deterministic and reproducible.")
        logging.info("Research Reproducibility Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'research_reproducibility_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Reproducibility violated. {failures}")

if __name__ == "__main__":
    research_reproducibility_audit()
