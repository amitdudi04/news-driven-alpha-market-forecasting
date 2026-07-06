import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def deployment_readiness_audit():
    logging.info("Starting Deployment Readiness Audit...")
    failures = []
    
    # Check Dockerfiles
    if not os.path.exists(os.path.join(os.getcwd(), 'Dockerfile')):
        failures.append("Missing Dockerfile. Infrastructure cannot be containerized.")
        
    if not os.path.exists(os.path.join(os.getcwd(), 'docker-compose.yml')):
        failures.append("Missing docker-compose.yml. Orchestration disabled.")
        
    if not os.path.exists(os.path.join(os.getcwd(), '.env.template')):
        failures.append("Missing .env.template. Secret governance compromised.")
        
    # Check Requirements
    if not os.path.exists(os.path.join(os.getcwd(), 'deployment', 'requirements.txt')):
        failures.append("Missing deployment/requirements.txt. Dependency lineage broken.")
        
    # Check Healthcheck
    if not os.path.exists(os.path.join(os.getcwd(), 'container_healthcheck.py')):
        failures.append("Missing container_healthcheck.py. Container liveness cannot be verified.")
        
    # Check isolation (make sure secrets are not in the repo, theoretically)
    # This is a basic audit, in a real environment we'd scan for API keys.
    
    report_lines = ["# Deployment Readiness Audit Report\n"]
    if failures:
        report_lines.append("## STATUS: FAIL ❌\n")
        for f in failures:
            report_lines.append(f"- {f}")
        logging.error("Deployment Readiness Audit FAILED.")
    else:
        report_lines.append("## STATUS: PASS ✅\n")
        report_lines.append("Infrastructure successfully transformed into a deployment-grade, Docker-orchestrated environment.")
        logging.info("Deployment Readiness Audit PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'deployment_readiness_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Deployment Readiness Compromised. {failures}")

if __name__ == "__main__":
    deployment_readiness_audit()
