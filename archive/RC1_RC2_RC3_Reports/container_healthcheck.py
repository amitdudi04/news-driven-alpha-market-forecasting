import os
import sys

def check_health():
    """
    Validates that the container is fully operational and has access to required mounts.
    Exits 0 if healthy, exits 1 if degraded.
    """
    required_dirs = ['data', 'outputs', 'logs', 'models', 'archive']
    
    for d in required_dirs:
        path = os.path.join(os.getcwd(), d)
        if not os.path.exists(path):
            print(f"HEALTHCHECK FAILED: Missing mounted volume {path}")
            sys.exit(1)
            
    # Check if the operational watchdog status is CRITICAL or SAFE_MODE_LOCKED
    status_path = os.path.join(os.getcwd(), 'outputs', 'operational_health_status.json')
    if os.path.exists(status_path):
        import json
        try:
            with open(status_path, 'r') as f:
                data = json.load(f)
                if data.get('status') in ['CRITICAL', 'SAFE_MODE_LOCKED']:
                    print(f"HEALTHCHECK FAILED: Watchdog status is {data.get('status')}")
                    sys.exit(1)
        except Exception as e:
            print(f"HEALTHCHECK ERROR: Could not parse watchdog status. {e}")
            sys.exit(1)
            
    print("HEALTHCHECK PASSED: System is nominal.")
    sys.exit(0)

if __name__ == "__main__":
    check_health()
