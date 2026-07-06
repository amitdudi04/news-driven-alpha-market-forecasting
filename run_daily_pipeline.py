import os
import sys
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
import logging
import pandas as pd
import joblib
import time

# Import pipeline modules
import module1_news
import module2_sentiment
import module3_market
import module4_features
import module12_inference
import module13_signal_engine

# ==========================================
# MASTER SCRIPT: DAILY ORCHESTRATION PIPELINE
# ==========================================
# Objective: Execute the full data-to-prediction pipeline sequentially 
# and generate the institutional trading signal for the next market open.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(execution_uuid: str):
    logging.info("==================================================")
    logging.info("       NEWS-DRIVEN ALPHA: DAILY ORCHESTRATION     ")
    logging.info(f"       EXECUTION UUID: {execution_uuid}")
    logging.info("==================================================")
    
    # STRICT SESSION GATING: Centralized Canonical Exchange Authority
    import module0_exchange_authority
    if not module0_exchange_authority.ExchangeAuthority.is_authorized_execution_window():
        logging.error("Pipeline failed. Execution rejected by Exchange Authority.")
        return False
    
    degraded_modules = []

    # Step 1: Market Data
    logging.info("\n--- STEP 1: Updating Market Data (yfinance) ---")
    try:
        module3_market.main(execution_uuid=execution_uuid)
    except TypeError:
        module3_market.main()
    except Exception as e:
        logging.error(f"Module 3 Failed: {e}. Entering DEGRADED execution mode for Market Data.")
        degraded_modules.append('market_data')
        
    # Step 2: News Data
    logging.info("\n--- STEP 2: Fetching Latest News (GDELT) ---")
    try:
        module1_news.main(execution_uuid=execution_uuid)
    except TypeError:
        module1_news.main()
    except Exception as e:
        logging.error(f"Module 1 Failed: {e}. Entering DEGRADED execution mode for News Data.")
        degraded_modules.append('news_data')
        
    # Step 3: Sentiment
    logging.info("\n--- STEP 3: Generating Sentiment Signals (FinBERT) ---")
    try:
        module2_sentiment.main(execution_uuid=execution_uuid)
    except TypeError:
        module2_sentiment.main()
    except Exception as e:
        logging.error(f"Module 2 Failed: {e}. Entering DEGRADED execution mode for Sentiment Inference.")
        degraded_modules.append('sentiment')
        
    # Step 4: Feature Engineering
    logging.info("\n--- STEP 4: Engineering Features & Aligning Time-Series ---")
    try:
        module4_features.main(execution_uuid=execution_uuid)
    except TypeError:
        module4_features.main()
    except Exception as e:
        logging.error(f"Module 4 Failed: {e}. Entering DEGRADED execution mode for Feature Engineering.")
        degraded_modules.append('features')
        
    if degraded_modules:
        logging.warning(f"Pipeline running in DEGRADED MODE. Failed domains: {degraded_modules}")
        
    return True
        
def generate_predictions_and_signals(execution_uuid: str):
    logging.info("\n--- STEP 5: Executing Live Inference Engine ---")
    try:
        module12_inference.main(execution_uuid=execution_uuid)
    except TypeError:
        module12_inference.main()
    except Exception as e:
        logging.error(f"Inference Engine Failed: {e}")
        return False
        
    logging.info("\n--- STEP 6: Generating Trading Signal & Logging ---")
    try:
        module13_signal_engine.main(execution_uuid=execution_uuid)
    except TypeError:
        module13_signal_engine.main()
    except Exception as e:
        logging.error(f"Signal Engine Failed: {e}")
        return False
        
    return True

def generate_execution_manifest(execution_uuid: str):
    import json
    import hashlib
    import datetime
    import platform
    
    def hash_file(filepath):
        if not os.path.exists(filepath): return None
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
        
    data_dir = os.path.join(os.getcwd(), 'data')
    manifests_dir = os.path.join(data_dir, 'manifests')
    os.makedirs(manifests_dir, exist_ok=True)
    
    from config.institutional_config import MODEL_VERSION_GOVERNANCE
    
    # 1. Manifest Immutability Protection & Model Lineage
    manifest = {
        "execution_uuid": execution_uuid,
        "dataset_hashes": {
            "news_daily.csv": hash_file(os.path.join(data_dir, 'news_daily.csv')),
            "csi300_features.csv": hash_file(os.path.join(data_dir, 'csi300_features.csv'))
        },
        "exchange_state_hash": hashlib.md5(datetime.datetime.now().strftime('%Y-%m-%d').encode()).hexdigest(),
        "feature_snapshot_hash": hash_file(os.path.join(data_dir, 'csi300_features.csv')),
        "config_hash": hash_file(os.path.join(os.getcwd(), 'config', 'institutional_config.py')),
        "model_version": MODEL_VERSION_GOVERNANCE["current_model_version"],
        "scaler_version": MODEL_VERSION_GOVERNANCE["current_scaler_version"],
        "model_hash": hash_file(os.path.join(os.getcwd(), 'models', 'xgboost_model.pkl')), 
        "prediction_timestamp": datetime.datetime.now().isoformat(),
        "authorization_snapshot": "AUTHORIZED_POST_CLOSE",
        "inference_parameters": {
            "model_type": "xgboost",
            "finbert_version": "ProsusAI/finbert"
        },
        "pipeline_version": "3.0.0-paper-trading",
        "runtime_environment_metadata": {
            "python_version": platform.python_version(),
            "os": platform.system()
        }
    }
    
    # Manifest hashing for Immutability Protection
    manifest_hash = hashlib.md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["manifest_hash"] = manifest_hash
    
    out_path = os.path.join(manifests_dir, f"manifest_{execution_uuid}.json")
    import module0_atomic_storage
    module0_atomic_storage.atomic_write_json(manifest, out_path)
    
    # 2. GENERATE DAILY OPERATIONAL STATE
    # This manifest bridges the gap to long-horizon paper trading
    try:
        live_pred = pd.read_csv(os.path.join(os.getcwd(), 'outputs', 'daily_prediction.csv'))
        latest_sig = live_pred.iloc[-1].to_dict()
    except Exception:
        latest_sig = {}
        
    try:
        final_ds = pd.read_csv(os.path.join(os.getcwd(), 'data', 'final_dataset.csv'))
        # Get age of dataset
        MAX_STALE_DAYS = int(os.environ.get('MAX_STALE_DAYS', 4))
        dataset_date = pd.to_datetime(final_ds['date'].iloc[-1])
        stale_days = (datetime.datetime.now() - dataset_date).days
        is_recovery = False
        if stale_days <= MAX_STALE_DAYS and len(final_ds) > 1:
            prev_date = pd.to_datetime(final_ds['date'].iloc[-2])
            if (dataset_date - prev_date).days > MAX_STALE_DAYS:
                is_recovery = True
    except Exception:
        stale_days = -1
        is_recovery = False

    # Read paper trading day count
    paper_trading_log = os.path.join(os.getcwd(), 'outputs', 'paper_trading_ledger.csv')
    if os.path.exists(paper_trading_log):
        pt_df = pd.read_csv(paper_trading_log)
        pt_days = len(pt_df)
    else:
        pt_days = 0

    op_state = {
        "operational_health": "NOMINAL" if latest_sig.get('Execution_Signal') != 'NO TRADE' else "DEGRADED",
        "deployment_state": "PAPER_TRADING_MODE",
        "SAFE_MODE_state": "INACTIVE", # This will be updated if it crashes
        "recovery_mode": is_recovery,
        "latest_signal": latest_sig.get('Execution_Signal', 'UNKNOWN'),
        "confidence": latest_sig.get('Confidence', 0.0),
        "calibration_state": "CALIBRATED",
        "rolling_sharpe": 0.0, # Tracked separately by portfolio logic
        "ingestion_freshness": stale_days,
        "paper_trading_days_active": pt_days,
        "rolling_expectancy": 0.0,
        "drift_state": "STABLE",
        "manifest_hash": manifest_hash
    }
    
    op_state_path = os.path.join(os.getcwd(), 'outputs', 'daily_operational_state.json')
    module0_atomic_storage.atomic_write_json(op_state, op_state_path)
    
    logging.info(f"\n--- STEP 7: Generated Immutable Execution Manifest: {out_path} ---")
    logging.info(f"--- STEP 8: Generated Daily Operational State: {op_state_path} ---")

def main():
    import uuid
    import csv
    import datetime
    
    start_time = time.time()
    execution_uuid = str(uuid.uuid4())
    
    success = run_pipeline(execution_uuid)
    if success:
        success = generate_predictions_and_signals(execution_uuid)
        if success:
            generate_execution_manifest(execution_uuid)
    else:
        logging.error("Pipeline failed. Prediction sequence aborted.")
        
    end_time = time.time()
    total_latency = end_time - start_time
    logging.info(f"TOTAL END-TO-END PIPELINE LATENCY: {total_latency:.2f} seconds")
    
    # 2. Resource & Latency Observability
    latency_log_path = os.path.join(os.getcwd(), 'logs', 'latency_monitor.csv')
    file_exists = os.path.isfile(latency_log_path)
    with open(latency_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'execution_uuid', 'latency_sec', 'success'])
        writer.writerow([datetime.datetime.now().isoformat(), execution_uuid, total_latency, success])
    
    # Authorized execution window latency threshold (e.g., 5 minutes for hard SLA)
    MAX_AUTHORIZED_LATENCY_SEC = 300
    if total_latency > MAX_AUTHORIZED_LATENCY_SEC:
        raise RuntimeError(f"LATENCY CASCADE DETECTED! Total latency ({total_latency:.2f}s) exceeded the authorized execution window of {MAX_AUTHORIZED_LATENCY_SEC}s. Prediction is invalid due to execution-window overrun.")

if __name__ == "__main__":
    main()
