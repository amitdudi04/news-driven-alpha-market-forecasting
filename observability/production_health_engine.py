import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compile_production_health():
    logging.info("Compiling Production Health Dashboard...")
    
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "global_status": "UNKNOWN",
        "subsystems": {
            "telemetry": "UNKNOWN",
            "alerts": "UNKNOWN",
            "watchdog": "UNKNOWN"
        }
    }
    
    # Check Telemetry
    telemetry_path = os.path.join(os.getcwd(), 'outputs', 'runtime_health_snapshot.json')
    if os.path.exists(telemetry_path):
        with open(telemetry_path, 'r') as f:
            t_data = json.load(f)
            dashboard['subsystems']['telemetry'] = "HEALTHY" if t_data.get('memory_percent', 100) < 90 else "DEGRADED"
            
    # Check Alerts
    alerts_path = os.path.join(os.getcwd(), 'outputs', 'active_alerts.json')
    if os.path.exists(alerts_path):
        with open(alerts_path, 'r') as f:
            a_data = json.load(f)
            if a_data.get('total_alerts', 0) == 0:
                dashboard['subsystems']['alerts'] = "HEALTHY"
            else:
                levels = [a['level'] for a in a_data.get('alerts', [])]
                if 'SAFE_MODE_LOCKED' in levels or 'CRITICAL' in levels:
                    dashboard['subsystems']['alerts'] = "CRITICAL"
                else:
                    dashboard['subsystems']['alerts'] = "DEGRADED"
                    
    # Check Watchdog
    watchdog_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    if os.path.exists(watchdog_path):
        with open(watchdog_path, 'r') as f:
            w_data = json.load(f)
            dashboard['subsystems']['watchdog'] = w_data.get('status', 'UNKNOWN')
            
    # Determine Global Status
    vals = list(dashboard['subsystems'].values())
    if 'SAFE_MODE_LOCKED' in vals:
        dashboard['global_status'] = "SAFE_MODE_LOCKED"
    elif 'CRITICAL' in vals:
        dashboard['global_status'] = "CRITICAL"
    elif 'DEGRADED' in vals:
        dashboard['global_status'] = "DEGRADED"
    else:
        dashboard['global_status'] = "HEALTHY"
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'production_health_dashboard.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=4)
        
    logging.info(f"Production Health Dashboard compiled. Global Status: {dashboard['global_status']}")

if __name__ == "__main__":
    compile_production_health()
