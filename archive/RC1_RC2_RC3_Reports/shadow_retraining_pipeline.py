import os
import uuid
import datetime
import hashlib
import json
import logging
import joblib
import pandas as pd
from module5_xgboost import train_and_evaluate, load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_shadow_retraining():
    logging.info("Starting Controlled Shadow Retraining Pipeline...")
    
    shadow_dir = os.path.join(os.getcwd(), 'models', 'shadow_registry')
    os.makedirs(shadow_dir, exist_ok=True)
    
    df = load_data()
    if df is None or df.empty:
        logging.error("Failed to load dataset. Aborting retraining.")
        return
        
    logging.info("Training Shadow Candidate...")
    best_model, _ = train_and_evaluate(df)
    
    if best_model is None:
        logging.error("Model training failed. Aborting.")
        return
        
    model_uuid = str(uuid.uuid4())
    shadow_model_path = os.path.join(shadow_dir, f'model_{model_uuid}.pkl')
    
    # Save Model
    joblib.dump(best_model, shadow_model_path)
    
    # Hash dataset, config
    data_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    config_path = os.path.join(os.getcwd(), 'config', 'institutional_config.py')
    
    # We do not have a scaler object explicitly since it's just xgboost trees, but we hash the features
    feature_cols = best_model.get("feature_cols", [])
    feature_schema_hash = hashlib.md5(json.dumps(feature_cols).encode()).hexdigest()
    
    manifest = {
        "model_uuid": model_uuid,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "feature_schema_hash": feature_schema_hash,
        "training_dataset_hash": hash_file(data_path),
        "config_hash": hash_file(config_path),
        "scaler_hash": "N/A_XGBOOST",
        "performance_metrics": {
            "cv_accuracy_proxy": "N/A (Derived in comparison audit)"
        },
        "calibration_metrics": {},
        "regime_metrics": {},
        "deployment_status": "SHADOW_ONLY",
        "promotion_stage": 1
    }
    
    manifest_path = os.path.join(shadow_dir, f'manifest_{model_uuid}.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    logging.info(f"Shadow Model {model_uuid} successfully placed in registry.")
    logging.info("Live production models remain unmutated.")
    
if __name__ == "__main__":
    run_shadow_retraining()
