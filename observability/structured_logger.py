import os
import json
import uuid
from datetime import datetime, timezone

class StructuredLogger:
    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.deployment_id = os.environ.get('HOSTNAME', 'local_instance')
        
    def _write_log(self, filename, level, event_type, message, metadata=None):
        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event_type": event_type,
            "message": message,
            "deployment_instance": self.deployment_id,
            "execution_uuid": metadata.get('execution_uuid', 'SYSTEM_LEVEL') if metadata else 'SYSTEM_LEVEL',
            "metadata": metadata or {}
        }
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_runtime_event(self, level, message, metadata=None):
        self._write_log('runtime_events.jsonl', level, 'RUNTIME', message, metadata)
        
    def log_safe_mode_event(self, message, metadata=None):
        self._write_log('safe_mode_events.jsonl', 'CRITICAL', 'SAFE_MODE_TRIGGER', message, metadata)
        
    def log_infrastructure_event(self, level, message, metadata=None):
        self._write_log('infrastructure_events.jsonl', level, 'INFRASTRUCTURE', message, metadata)

# Global instance
logger = StructuredLogger()

if __name__ == "__main__":
    logger.log_infrastructure_event("INFO", "Structured Logger Initialized.")
