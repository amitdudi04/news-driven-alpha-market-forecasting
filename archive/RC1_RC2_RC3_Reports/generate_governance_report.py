import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_governance_report():
    logging.info("Generating Monthly Governance Report...")
    
    surveillance_path = os.path.join(os.getcwd(), 'outputs', 'rolling_surveillance_manifest.csv')
    latency_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    lineage_report_path = os.path.join(os.getcwd(), 'outputs', 'lineage_verification_report.md')
    
    if not os.path.exists(surveillance_path):
        logging.warning("Missing rolling_surveillance_manifest.csv. Run pipeline first.")
        return
        
    df = pd.read_csv(surveillance_path)
    latest = df.iloc[-1]
    
    # Process Latency
    latency_sla_status = "No Data"
    max_latency = 0
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        max_latency = lat_df['latency_sec'].max()
        sla_breaches = len(lat_df[lat_df['latency_sec'] > 120])
        latency_sla_status = f"{sla_breaches} breaches detected" if sla_breaches > 0 else "PASS (100% compliance)"
        
    # Process Lineage Status
    lineage_status = "No Report Found"
    if os.path.exists(lineage_report_path):
        with open(lineage_report_path, 'r') as f:
            if 'Status: PASS' in f.read():
                lineage_status = "PASS"
            else:
                lineage_status = "FAIL"
                
    # Process Operational Health Status
    op_health_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_report.md')
    op_health_status = "UNKNOWN"
    if os.path.exists(op_health_path):
        with open(op_health_path, 'r') as f:
            content = f.read()
            if 'OVERALL STATUS: HEALTHY' in content: op_health_status = "HEALTHY"
            elif 'OVERALL STATUS: DEGRADED' in content: op_health_status = "DEGRADED"
            elif 'OVERALL STATUS: SAFE_MODE_LOCKED' in content: op_health_status = "SAFE_MODE_LOCKED"
            elif 'OVERALL STATUS: CRITICAL' in content: op_health_status = "CRITICAL"
                
    # Compute aggregates from the surveillance manifest
    drift_incidents = df['drift_alert'].sum()
    calibration_incidents = df['calibration_alert'].sum()
    safe_mode_incidents = df['safe_mode_flag'].sum()
    
    report_content = f"""# Monthly Institutional Governance Report

**Date Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Latest Execution UUID**: `{latest.get('execution_uuid', 'N/A')}`
**Operational Health Classification**: `{op_health_status}`
**Total Tracked Executions**: {len(df)}

## 1. Rolling Performance Metrics
*Derived entirely from immutable surveillance manifest.*
- **Rolling Sharpe (20d)**: {latest.get('rolling_sharpe', 0):.2f} (95% CI Lower: {latest.get('rolling_sharpe_ci_lower', 0):.2f})
- **Rolling Sortino (20d)**: {latest.get('rolling_sortino', 0):.2f}
- **Rolling Calmar Ratio**: {latest.get('rolling_calmar_ratio', 0):.2f}
- **Rolling Max Drawdown**: {latest.get('rolling_max_drawdown', 0)*100:.2f}%
- **Rolling Hit Rate (20d)**: {latest.get('rolling_hit_rate', 0):.1f}% (95% CI Lower: {latest.get('rolling_hit_rate_ci_lower', 0):.1f}%)
- **Rolling Expectancy (20d)**: {latest.get('rolling_expectancy', 0):.4f}
- **Rolling Profit Factor**: {latest.get('rolling_profit_factor', 0):.2f}
- **Rolling Win/Loss Ratio**: {latest.get('rolling_win_loss_ratio', 0):.2f}
- **Rolling Signal Density**: {latest.get('rolling_signal_density', 0)*100:.1f}%
- **Rolling Turnover Efficiency**: {latest.get('rolling_turnover_efficiency', 0):.2f}
- **Benchmark Relative Edge (Sharpe)**: {latest.get('relative_edge_vs_bench', 0):.2f}
- **Rolling Tracking Error**: {latest.get('rolling_tracking_error', 0):.4f}
- **Rolling Information Ratio**: {latest.get('rolling_information_ratio', 0):.2f}

## 2. Calibration & Edge Decay
- **Rolling Brier Score**: {latest.get('rolling_brier', 0):.4f}
- **Rolling ECE**: {latest.get('rolling_ece', 0):.4f}
- **Calibration Status**: {'ALERT' if latest.get('calibration_alert', False) else 'PASS'}
- **Degradation Alert**: {'ALERT' if latest.get('degradation_alert', False) else 'PASS'}
- **Persistent Degradation Alert**: {'ESCALATED' if latest.get('persistent_degradation_alert', False) else 'PASS'}

## 3. Governance Escalations (Historical)
- **Total Drift Escalations**: {drift_incidents}
- **Total SAFE MODE Incidents**: {safe_mode_incidents}
- **Total Calibration Collapses**: {calibration_incidents}

## 4. Operational Infrastructure
- **Latency SLA Status**: {latency_sla_status} (Max recorded: {max_latency:.2f}s)
- **Manifest Integrity Status**: {lineage_status}
- **UUID Continuity Status**: {lineage_status}

---
*End of Report*
"""

    report_path = os.path.join(os.getcwd(), 'outputs', 'monthly_governance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logging.info(f"Governance Report saved to {report_path}")

if __name__ == "__main__":
    generate_governance_report()
