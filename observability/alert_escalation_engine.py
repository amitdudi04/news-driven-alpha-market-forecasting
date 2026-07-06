import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_alert_escalation():
    logging.info("Starting Institutional Alert Escalation Framework...")
    
    alerts = []
    
    # Check Telemetry Snapshot
    snapshot_path = os.path.join(os.getcwd(), 'outputs', 'runtime_health_snapshot.json')
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as f:
            telemetry = json.load(f)
            
            if telemetry.get('watchdog_state') in ['CRITICAL', 'SAFE_MODE_LOCKED']:
                alerts.append({
                    "level": telemetry.get('watchdog_state'),
                    "trigger": "watchdog_critical_state",
                    "message": "Operational watchdog is reporting a systemic failure or SAFE MODE lock."
                })
                
            if telemetry.get('pipeline_latency_sec', 0) > 120.0:
                alerts.append({
                    "level": "WARNING",
                    "trigger": "latency_sla_breach",
                    "message": f"Pipeline latency {telemetry.get('pipeline_latency_sec')}s exceeded SLA of 120s."
                })
                
    # Check Reproducibility
    repro_path = os.path.join(os.getcwd(), 'outputs', 'research_reproducibility_report.md')
    if os.path.exists(repro_path):
        with open(repro_path, 'r', encoding='utf-8') as f:
            if 'FAIL' in f.read():
                alerts.append({
                    "level": "CRITICAL",
                    "trigger": "replay_drift",
                    "message": "Research reproducibility audit failed. Deterministic hash drift detected."
                })
                
    active_alerts = {
        "timestamp": datetime.now().isoformat(),
        "total_alerts": len(alerts),
        "alerts": alerts
    }
    
    out_path = os.path.join(os.getcwd(), 'outputs', 'active_alerts.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(active_alerts, f, indent=4)
        
    for alert in alerts:
        logging.warning(f"ALERT [{alert['level']}]: {alert['message']}")
        
    logging.info("Alert escalation evaluation complete. System holds NO execution authority.")

if __name__ == "__main__":
    run_alert_escalation()
