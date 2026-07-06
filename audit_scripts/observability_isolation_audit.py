import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def observability_isolation_audit():
    logging.info("Starting Observability Isolation Audit...")
    failures = []
    
    # 1. module13 isolation
    module13_path = os.path.join(os.getcwd(), 'module13_execution_core.py')
    if os.path.exists(module13_path):
        with open(module13_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'telemetry' in content.lower() or 'structured_logger' in content.lower():
                failures.append("module13 imports observability layers. Execution path contaminated.")
                
    # 2. Manifest isolation
    # Make sure we didn't add anything to live_tracking.csv or daily_prediction.csv
    live_track = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if os.path.exists(live_track):
        with open(live_track, 'r', encoding='utf-8') as f:
            headers = f.readline()
            if 'telemetry' in headers or 'cpu_percent' in headers or 'latency_sec' in headers:
                failures.append("live_tracking.csv contains observability columns. Manifest polluted.")
                
    # 3. Observability Read-Only Check
    # Verify that telemetry engine does not contain pandas to_csv over live_tracking
    tele_path = os.path.join(os.getcwd(), 'observability', 'runtime_telemetry_engine.py')
    if os.path.exists(tele_path):
        with open(tele_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'live_tracking.csv' in content and 'to_csv' in content:
                failures.append("Runtime telemetry attempts to modify live_tracking.csv")
                
    report_lines = ["# Observability Isolation Audit Report\n"]
    if failures:
        report_lines.append("## STATUS: FAIL ❌\n")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Observability Isolation Audit FAILED.")
    else:
        report_lines.append("## STATUS: PASS ✅\n")
        report_lines.append("Observability and telemetry systems strictly adhere to read-only boundaries.")
        logging.info("Observability Isolation Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'observability_isolation_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Observability leaked into execution. {failures}")

if __name__ == "__main__":
    observability_isolation_audit()
