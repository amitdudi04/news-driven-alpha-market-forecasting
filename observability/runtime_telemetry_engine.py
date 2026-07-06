import os
import psutil
import pandas as pd
import json
import logging
from datetime import datetime
from structured_logger import logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def capture_runtime_telemetry():
    logging.info("Capturing Runtime Telemetry...")
    
    telemetry = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "watchdog_state": "UNKNOWN",
        "safe_mode_trigger_count": 0,
        "pipeline_latency_sec": 0.0
    }
    
    # 1. Watchdog State
    watchdog_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    if os.path.exists(watchdog_path):
        with open(watchdog_path, 'r') as f:
            wd_data = json.load(f)
            telemetry["watchdog_state"] = wd_data.get("status", "UNKNOWN")
            
    # 2. Pipeline Latency
    latency_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        if not lat_df.empty:
            telemetry["pipeline_latency_sec"] = lat_df.iloc[-1]['latency_sec']
            
    # 3. SAFE MODE Trigger Count
    safe_log_path = os.path.join(os.getcwd(), 'logs', 'safe_mode_events.jsonl')
    if os.path.exists(safe_log_path):
        with open(safe_log_path, 'r') as f:
            telemetry["safe_mode_trigger_count"] = sum(1 for _ in f)
            
    # Write JSON Snapshot
    snapshot_path = os.path.join(os.getcwd(), 'outputs', 'runtime_health_snapshot.json')
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(telemetry, f, indent=4)
        
    # Append to CSV Manifest
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'runtime_telemetry_manifest.csv')
    df_new = pd.DataFrame([telemetry])
    
    if os.path.exists(manifest_path):
        df_new.to_csv(manifest_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(manifest_path, index=False)
        
    logger.log_infrastructure_event("INFO", "Runtime telemetry captured.", telemetry)
    logging.info("Telemetry capture complete. Read-only limits strictly enforced.")

if __name__ == "__main__":
    capture_runtime_telemetry()
