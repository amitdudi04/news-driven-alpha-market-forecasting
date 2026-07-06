import os
import shutil
import hashlib
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backup_manifests():
    """Takes an immutable snapshot of all critical operational manifests."""
    backup_dir = os.path.join(os.getcwd(), 'archive', 'disaster_recovery_backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files to backup
    files = [
        'outputs/rolling_surveillance_manifest.csv',
        'outputs/live_tracking.csv',
        'outputs/daily_prediction.csv',
        'logs/latency_monitor.csv',
        'logs/surveillance_hash_log.csv',
        'config/institutional_config.py'
    ]
    
    for f_path in files:
        full_path = os.path.join(os.getcwd(), f_path)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                f_hash = hashlib.md5(f.read()).hexdigest()
            dest_name = f"{os.path.basename(f_path)}_{f_hash}.bak"
            dest_path = os.path.join(backup_dir, dest_name)
            if not os.path.exists(dest_path):
                shutil.copy2(full_path, dest_path)
                logging.info(f"Backed up {f_path} to {dest_name}")
                
def restore_manifest(filename_prefix):
    """Restores the latest backup of a specific file prefix."""
    backup_dir = os.path.join(os.getcwd(), 'archive', 'disaster_recovery_backups')
    backups = glob.glob(os.path.join(backup_dir, f"{filename_prefix}_*.bak"))
    
    if not backups:
        logging.error(f"No backups found for {filename_prefix}")
        return False
        
    # Get the latest by modification time
    latest_backup = max(backups, key=os.path.getmtime)
    
    # Determine target path
    target_map = {
        'rolling_surveillance_manifest.csv': 'outputs/rolling_surveillance_manifest.csv',
        'live_tracking.csv': 'outputs/live_tracking.csv',
        'daily_prediction.csv': 'outputs/daily_prediction.csv',
        'latency_monitor.csv': 'logs/latency_monitor.csv',
        'surveillance_hash_log.csv': 'logs/surveillance_hash_log.csv',
        'institutional_config.py': 'config/institutional_config.py'
    }
    
    target_rel = target_map.get(filename_prefix)
    if not target_rel:
        logging.error("Unknown restoration target.")
        return False
        
    target_abs = os.path.join(os.getcwd(), target_rel)
    
    # If the file exists, we don't overwrite if it's considered uncorrupted. 
    # The prompt mandates: "corrupted manifests NEVER overwritten". We append-only or restore if completely missing/empty.
    if os.path.exists(target_abs) and os.path.getsize(target_abs) > 0:
        logging.error(f"Refusing to overwrite existing non-empty file {target_abs}. Move it manually if corrupted.")
        return False
        
    shutil.copy2(latest_backup, target_abs)
    logging.info(f"Successfully restored {target_rel} from {latest_backup}")
    return True

if __name__ == "__main__":
    # By default, take a backup snapshot
    backup_manifests()
