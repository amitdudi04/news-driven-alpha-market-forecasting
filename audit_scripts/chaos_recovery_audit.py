import os
import shutil
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chaos_recovery_audit():
    report_lines = ["# Operational Chaos & Recovery Test Report\n"]
    failures = []
    
    # We will simulate missing a file and running the watchdog and DR
    target_file = os.path.join(os.getcwd(), 'config', 'institutional_config.py')
    backup_path = os.path.join(os.getcwd(), 'config', 'institutional_config_backup.py')
    
    # 1. Take a backup of the real file
    shutil.copy2(target_file, backup_path)
    
    try:
        # 2. Inject Chaos (Delete the config)
        os.remove(target_file)
        logging.info("Chaos Injected: Deleted institutional_config.py")
        
        # 3. Test Watchdog
        import sys
        sys.path.insert(0, os.getcwd())
        import continuous_operational_watchdog
        watchdog_passed = continuous_operational_watchdog.continuous_operational_watchdog()
        if watchdog_passed:
            failures.append("Watchdog failed to halt execution after config deletion.")
            
        with open(os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json'), 'r') as f:
            status = json.load(f)
            if status['status'] != "CRITICAL":
                failures.append("Watchdog failed to classify missing config as CRITICAL.")
                
        # 4. Test Disaster Recovery
        import disaster_recovery_protocol
        recovered = disaster_recovery_protocol.restore_manifest('institutional_config.py')
        if not recovered:
            failures.append("Disaster Recovery Protocol failed to restore institutional_config.py")
            
        # 5. Verify Restoration
        if not os.path.exists(target_file):
            failures.append("File was not actually restored to the disk.")
            
    finally:
        # 6. Clean up
        if not os.path.exists(target_file):
            shutil.copy2(backup_path, target_file)
        os.remove(backup_path)
        
    if failures:
        report_lines.insert(1, "## STATUS: FAIL ❌\n")
        report_lines.append("### Chaos Testing Failures:")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Chaos Recovery Audit FAILED.")
    else:
        report_lines.insert(1, "## STATUS: PASS ✅\n")
        report_lines.append("### Tests Passed:")
        report_lines.append("- [x] Watchdog intercepted missing artifacts.")
        report_lines.append("- [x] Watchdog correctly classified CRITICAL state.")
        report_lines.append("- [x] Disaster Recovery successfully restored from immutable backup.")
        report_lines.append("- [x] Operational state fully recoverable.")
        logging.info("Chaos Recovery Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'chaos_recovery_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    chaos_recovery_audit()
