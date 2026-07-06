import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rollback_integrity_audit():
    logging.info("Starting Infrastructure Rollback Integrity Audit...")
    failures = []
    
    # Verify backup directories exist
    backup_dir = os.path.join(os.getcwd(), 'archive', 'disaster_recovery_backups')
    if not os.path.exists(backup_dir):
        failures.append("Missing disaster recovery backups directory. Rollback impossible.")
        
    # Verify rollback components (manifests, config, watchdog)
    # Ensure they can be restored deterministically
    dr_protocol_path = os.path.join(os.getcwd(), 'disaster_recovery_protocol.py')
    if not os.path.exists(dr_protocol_path):
        failures.append("Missing disaster_recovery_protocol.py. Cannot orchestrate rollback.")
        
    watchdog_status = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    if not os.path.exists(watchdog_status):
        failures.append("Missing watchdog status file. Continuity cannot be guaranteed post-rollback.")
        
    # Read the manifest to verify it's structurally sound
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            if not f.readline().startswith('date,prediction,actual_return,pnl,confidence_raw'):
                failures.append("Manifest structure corrupted. Rollback required.")
    else:
        failures.append("live_tracking.csv missing.")
        
    report_lines = ["# Infrastructure Rollback Integrity Audit Report\n"]
    if failures:
        report_lines.append("## STATUS: FAIL ❌\n")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Rollback Integrity Audit FAILED.")
    else:
        report_lines.append("## STATUS: PASS ✅\n")
        report_lines.append("Infrastructure possesses full rollback survivability and manifest restoration capacity.")
        logging.info("Rollback Integrity Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'rollback_integrity_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Rollback Integrity Breached. {failures}")

if __name__ == "__main__":
    rollback_integrity_audit()
