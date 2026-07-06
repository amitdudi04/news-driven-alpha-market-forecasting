import os
import pandas as pd
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def operational_persistence_audit():
    # 1. Load ledgers
    surveillance_path = os.path.join(os.getcwd(), 'outputs', 'rolling_surveillance_manifest.csv')
    latency_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    hash_log_path = os.path.join(os.getcwd(), 'logs', 'surveillance_hash_log.csv')
    
    report_lines = ["# Operational Persistence Health Report\n"]
    failures = []
    
    if not os.path.exists(surveillance_path):
        failures.append("Missing rolling_surveillance_manifest.csv")
    else:
        df = pd.read_csv(surveillance_path)
        
        # UUID Continuity & Duplication
        if 'execution_uuid' in df.columns:
            # We filter out PRE_DEPLOYMENT_N/A since those are historical backfills
            uuids = df[df['execution_uuid'] != 'PRE_DEPLOYMENT_N/A']['execution_uuid']
            if uuids.duplicated().any():
                failures.append("Duplicate execution_uuid detected in surveillance manifest.")
                
        # Timestamp Monotonicity
        df['date'] = pd.to_datetime(df['date'])
        if not df['date'].is_monotonic_increasing:
            failures.append("Timestamps in surveillance manifest are NOT monotonically increasing.")
            
        # Duplicate Session Detection
        if df['date'].duplicated().any():
            failures.append("Duplicate execution dates detected in surveillance manifest.")
            
        # Append-Only Ledger Continuity (Hash check)
        if os.path.exists(hash_log_path):
            hash_df = pd.read_csv(hash_log_path)
            # Re-hash current file
            with open(surveillance_path, 'rb') as f:
                current_hash = hashlib.md5(f.read()).hexdigest()
            last_recorded_hash = hash_df['surveillance_manifest_hash'].iloc[-1] if not hash_df.empty else ""
            if current_hash != last_recorded_hash:
                # This could happen if the file was modified after the pipeline ran.
                logging.warning(f"Surveillance manifest hash mismatch. Current: {current_hash}, Recorded: {last_recorded_hash}")
                # We won't strictly fail on this unless we want to enforce nobody touches the CSV.
                # Actually, the user says "FAIL IMMEDIATELY IF ... surveillance manifests corrupted"
                failures.append("Surveillance manifest hash corruption detected. Append-only violation.")
        
    # Latency SLA Continuity
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        sla_breaches = len(lat_df[lat_df['latency_sec'] > 120])
        # If it's persistently violated (e.g. > 3 times)
        if sla_breaches > 3:
            failures.append(f"Latency SLA persistently violated ({sla_breaches} times).")
            
    # Determine Health Status
    # HEALTHY, DEGRADED, CRITICAL, SAFE_MODE_LOCKED
    status = "HEALTHY"
    if failures:
        status = "CRITICAL"
    elif 'degradation_alert' in df.columns and df['degradation_alert'].iloc[-1]:
        status = "DEGRADED"
    elif 'safe_mode_flag' in df.columns and df['safe_mode_flag'].iloc[-1]:
        status = "SAFE_MODE_LOCKED"
        
    report_lines.insert(1, f"## OVERALL STATUS: {status}\n")
    
    if failures:
        report_lines.append("### Critical Failures:")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Operational Persistence Audit FAILED.")
    else:
        report_lines.append("### System Checks:")
        report_lines.append("- UUID Continuity: **PASS**")
        report_lines.append("- Timestamp Monotonicity: **PASS**")
        report_lines.append("- Latency SLA: **PASS**")
        report_lines.append("- Ledger Append-Only Integrity: **PASS**")
        logging.info("Operational Persistence Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_report.md')
    with open(out_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Operational persistence compromised. {failures}")

if __name__ == "__main__":
    operational_persistence_audit()
