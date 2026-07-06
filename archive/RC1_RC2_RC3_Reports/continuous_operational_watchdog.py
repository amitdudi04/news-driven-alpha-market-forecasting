import os
import json
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def continuous_operational_watchdog():
    logging.info("Initializing Continuous Operational Watchdog...")
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "HEALTHY",
        "alerts": []
    }
    
    def add_alert(msg, severity="WARNING"):
        health_status["alerts"].append({"severity": severity, "message": msg})
        
    # 1. Check Core Artifacts
    core_files = [
        'models/model_live.pkl',
        'config/institutional_config.py',
        'outputs/rolling_surveillance_manifest.csv',
        'outputs/daily_prediction.csv',
        'outputs/live_tracking.csv'
    ]
    
    for f_path in core_files:
        if not os.path.exists(os.path.join(os.getcwd(), f_path)):
            add_alert(f"Missing critical artifact: {f_path}", "CRITICAL")
            
    # 2. Check Latency
    latency_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        recent_lat = lat_df.tail(3)
        if (recent_lat['latency_sec'] > 120.0).any():
            add_alert("Latency explosions detected in recent runs (>120s).", "CRITICAL")
            
    # 3. Check Surveillance Manifest
    surv_path = os.path.join(os.getcwd(), 'outputs', 'rolling_surveillance_manifest.csv')
    if os.path.exists(surv_path):
        surv_df = pd.read_csv(surv_path)
        latest = surv_df.iloc[-1]
        
        # Stale manifests
        last_date = pd.to_datetime(latest['date'])
        days_stale = (datetime.now() - last_date).days
        if days_stale > 5:
            add_alert(f"Stale surveillance manifest (Last updated {days_stale} days ago).", "CRITICAL")
            
        # UUID Chains
        if 'execution_uuid' in surv_df.columns:
            uuids = surv_df[surv_df['execution_uuid'] != 'PRE_DEPLOYMENT_N/A']['execution_uuid']
            if uuids.duplicated().any():
                add_alert("Broken UUID chains detected (Duplicates).", "CRITICAL")
                
        # Degradation & Calibration
        if latest.get('persistent_degradation_alert', False):
            add_alert("Persistent degradation alert active.", "DEGRADED")
            
        if latest.get('calibration_alert', False):
            add_alert("Calibration collapse detected (Brier/ECE bounds breached).", "DEGRADED")
            
        if latest.get('safe_mode_flag', False):
            add_alert("SAFE MODE is currently locked.", "SAFE_MODE_LOCKED")
            
    # State Evaluation
    severity_levels = [alert["severity"] for alert in health_status["alerts"]]
    
    if "SAFE_MODE_LOCKED" in severity_levels:
        health_status["status"] = "SAFE_MODE_LOCKED"
    elif "CRITICAL" in severity_levels:
        health_status["status"] = "CRITICAL"
    elif "DEGRADED" in severity_levels:
        health_status["status"] = "DEGRADED"
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(health_status, f, indent=4)
        
    logging.info(f"Watchdog evaluation complete. Status: {health_status['status']}")
    
    if health_status["status"] in ["CRITICAL", "SAFE_MODE_LOCKED"]:
        logging.error("Watchdog halting further execution. Escalation required.")
        # Note: Watchdog halts itself, but does not mutate the pipeline. The pipeline must check this status.
        return False
        
    return True

if __name__ == "__main__":
    continuous_operational_watchdog()
