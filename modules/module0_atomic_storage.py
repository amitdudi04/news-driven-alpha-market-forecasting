import os
import json
import logging
import tempfile
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def atomic_write_csv(df: pd.DataFrame, out_path: str, index: bool = False):
    """
    Writes a pandas DataFrame to CSV atomically using a temporary shadow file.
    Prevents partial/corrupted files if the process crashes mid-write.
    """
    dir_name = os.path.dirname(out_path)
    os.makedirs(dir_name, exist_ok=True)
    
    # Create a temporary file in the same directory to ensure atomic os.replace across filesystems
    fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="tmp_shadow_", suffix=".csv")
    os.close(fd) # Close the file descriptor, pandas will open it
    
    try:
        df.to_csv(temp_path, index=index)
        # Atomically swap the temporary file with the target file
        os.replace(temp_path, out_path)
    except Exception as e:
        logging.error(f"Atomic write failed for {out_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def atomic_write_json(data: dict, out_path: str):
    """
    Writes a dictionary to JSON atomically using a temporary shadow file.
    """
    dir_name = os.path.dirname(out_path)
    os.makedirs(dir_name, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="tmp_shadow_", suffix=".json")
    os.close(fd)
    
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno()) # Force write to disk before rename
            
        os.replace(temp_path, out_path)
    except Exception as e:
        logging.error(f"Atomic JSON write failed for {out_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
