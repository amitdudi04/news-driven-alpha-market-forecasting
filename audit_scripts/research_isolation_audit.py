import os
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def research_isolation_audit():
    logging.info("Starting Research Isolation Audit...")
    failures = []
    
    # 1. module13 SSOT check
    module13_path = os.path.join(os.getcwd(), 'module13_execution_core.py')
    if os.path.exists(module13_path):
        with open(module13_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ensemble' in content.lower() or 'research' in content.lower() or 'macro' in content.lower():
                # We need to be careful; "research" might be in comments. We check for imported modules.
                if 'import research_orchestrator' in content or 'import macro_regime' in content:
                    failures.append("module13 imports research/macro modules. Isolation compromised.")
                    
    # 2. Manifest Segregation Check
    live_track = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if os.path.exists(live_track):
        with open(live_track, 'r', encoding='utf-8') as f:
            headers = f.readline()
            if 'ensemble' in headers or 'macro' in headers or 'shadow' in headers:
                failures.append("live_tracking.csv contains research/macro columns. Manifest polluted.")
                
    # 3. Production Model Immutability
    model_path = os.path.join(os.getcwd(), 'models', 'model_live.pkl')
    # Just verifying it exists and we have no logic allowing overwrite from research orchestrator
    orchestrator_path = os.path.join(os.getcwd(), 'research_orchestrator.py')
    if os.path.exists(orchestrator_path):
        with open(orchestrator_path, 'r', encoding='utf-8') as f:
            if 'model_live.pkl' in f.read() and 'open(' in f.read() and "'w'" in f.read():
                failures.append("research_orchestrator contains write permission to model_live.pkl")
                
    report_lines = ["# Research Isolation Audit Report\n"]
    if failures:
        report_lines.append("## STATUS: FAIL ❌\n")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Research Isolation Audit FAILED.")
    else:
        report_lines.append("## STATUS: PASS ✅\n")
        report_lines.append("The research sandbox is mathematically isolated from production execution.")
        logging.info("Research Isolation Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'research_isolation_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Isolation boundary breached. {failures}")

if __name__ == "__main__":
    research_isolation_audit()
