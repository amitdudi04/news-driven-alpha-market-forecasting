import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compile_production_survivability():
    logging.info("Compiling Production Survivability Engine...")
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "classification": "UNKNOWN",
        "health_checks": {
            "deployment_readiness": False,
            "rollback_integrity": False,
            "multi_instance_determinism": False,
            "operational_health": False
        }
    }
    
    # 1. Deployment Readiness
    dr_path = os.path.join(os.getcwd(), 'outputs', 'deployment_readiness_report.md')
    if os.path.exists(dr_path):
        with open(dr_path, 'r', encoding='utf-8') as f:
            if 'PASS' in f.read():
                manifest['health_checks']['deployment_readiness'] = True
                
    # 2. Rollback Integrity
    rb_path = os.path.join(os.getcwd(), 'outputs', 'rollback_integrity_report.md')
    if os.path.exists(rb_path):
        with open(rb_path, 'r', encoding='utf-8') as f:
            if 'PASS' in f.read():
                manifest['health_checks']['rollback_integrity'] = True
                
    # 3. Multi-Instance Determinism
    mid_path = os.path.join(os.getcwd(), 'outputs', 'multi_instance_determinism_report.md')
    if os.path.exists(mid_path):
        with open(mid_path, 'r', encoding='utf-8') as f:
            if 'PASS' in f.read():
                manifest['health_checks']['multi_instance_determinism'] = True
                
    # 4. Operational Health
    oh_path = os.path.join(os.getcwd(), 'outputs', 'production_health_dashboard.json')
    if os.path.exists(oh_path):
        with open(oh_path, 'r') as f:
            data = json.load(f)
            if data.get('global_status') in ['HEALTHY', 'DEGRADED']:
                manifest['health_checks']['operational_health'] = True
                
    # Determine Survivability Status
    all_checks = list(manifest['health_checks'].values())
    if all(all_checks):
        manifest['classification'] = "STABLE"
    elif sum(all_checks) >= len(all_checks) - 1:
        manifest['classification'] = "DEGRADED"
    elif sum(all_checks) > 0:
        manifest['classification'] = "CRITICAL"
    else:
        manifest['classification'] = "SAFE_MODE_LOCKED"
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'production_survivability_manifest.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    logging.info(f"Production Survivability Compiled. Classification: {manifest['classification']}")

if __name__ == "__main__":
    compile_production_survivability()
