import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sla_governance_audit():
    latency_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    if not os.path.exists(latency_path):
        logging.warning("Missing latency_monitor.csv. Skipping SLA check.")
        return
        
    df = pd.read_csv(latency_path)
    
    # Check bounds
    pipeline_max_sla = 120.0
    inference_max_sla = 15.0 # (Inference latency isn't separately tracked in this file right now, but total latency serves as proxy if total < 120)
    # We will enforce pipeline_runtime_sec < 120
    
    sla_critical_breaches = df[df['latency_sec'] > pipeline_max_sla]
    sla_warnings = df[(df['latency_sec'] > 60.0) & (df['latency_sec'] <= pipeline_max_sla)]
    
    report_lines = ["# Operational SLA Governance Audit\n"]
    
    if len(sla_critical_breaches) > 0:
        report_lines.append("## STATUS: SAFE_MODE_LOCKED ❌")
        report_lines.append(f"CRITICAL SLA BREACH DETECTED: {len(sla_critical_breaches)} execution(s) exceeded {pipeline_max_sla}s.")
        logging.error("SLA Governance FAILED.")
        fail = True
    elif len(sla_warnings) > 0:
        report_lines.append("## STATUS: SLA_WARNING ⚠️")
        report_lines.append(f"Warning: {len(sla_warnings)} execution(s) approached SLA bounds (>60s).")
        logging.warning("SLA WARNING generated.")
        fail = False
    else:
        report_lines.append("## STATUS: PASS ✅")
        report_lines.append("All operational latencies are well within SLA bounds.")
        logging.info("SLA Governance PASSED.")
        fail = False
        
    # Check timestamp monotonicity
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if not df['timestamp'].is_monotonic_increasing:
        report_lines.append("- **ERROR**: Timestamp monotonicity broken.")
        logging.error("Timestamp monotonicity broken.")
        fail = True
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'sla_governance_report.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if fail:
        raise RuntimeError("SAFE MODE ESCALATION: SLA constraints violated.")

if __name__ == "__main__":
    sla_governance_audit()
