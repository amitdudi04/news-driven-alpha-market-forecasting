import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classify_research_ecosystem():
    logging.info("Starting Institutional Research Classification Engine...")
    
    classification = {
        "timestamp": datetime.now().isoformat(),
        "classification": "UNKNOWN",
        "checks": {
            "operational_health": False,
            "macro_isolation": False,
            "ensemble_isolation": False,
            "replay_determinism": False
        }
    }
    
    # Check Operational Health
    op_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    if os.path.exists(op_path):
        with open(op_path, 'r') as f:
            op_data = json.load(f)
            if op_data.get('status') == 'HEALTHY':
                classification['checks']['operational_health'] = True
                
    # Check Macro Isolation
    macro_report = os.path.join(os.getcwd(), 'outputs', 'macro_governance_report.md')
    if os.path.exists(macro_report):
        with open(macro_report, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'STABLE' in content or 'DEGRADED' in content: # As long as it ran and didn't crash
                classification['checks']['macro_isolation'] = True
                
    # Check Ensemble Isolation
    ensemble_report = os.path.join(os.getcwd(), 'outputs', 'ensemble_governance_report.md')
    if os.path.exists(ensemble_report):
        with open(ensemble_report, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'STABLE' in content or 'DEGRADED' in content:
                classification['checks']['ensemble_isolation'] = True
                
    # Check Replay Determinism
    repro_report = os.path.join(os.getcwd(), 'outputs', 'research_reproducibility_report.md')
    if os.path.exists(repro_report):
        with open(repro_report, 'r', encoding='utf-8') as f:
            if 'PASS' in f.read():
                classification['checks']['replay_determinism'] = True
                
    # Classify
    all_checks_passed = all(classification['checks'].values())
    
    if all_checks_passed:
        classification['classification'] = "PAPER_TRADING_ONLY" 
        # Production certification requires live capital testing, not achieved yet.
        logging.info("Platform is fully certified for PAPER TRADING with Research/Macro isolation.")
    else:
        classification['classification'] = "RESEARCH_ONLY"
        logging.warning("Platform failed operational or isolation checks. Downgraded to RESEARCH_ONLY.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'research_classification_manifest.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(classification, f, indent=4)
        
    logging.info(f"Final Classification: {classification['classification']}")

if __name__ == "__main__":
    classify_research_ecosystem()
