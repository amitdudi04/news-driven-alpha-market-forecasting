import pandas as pd
import numpy as np
import joblib
import logging
import os

# ==========================================
# MODULE 12: LIVE INFERENCE ENGINE
# ==========================================
# Objective: Standalone prediction module designed for production environments.
# Handles feature validation, missing values, and executing the XGBoost model.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_inference_model():
    """Loads the serialized, cross-validated XGBoost model from disk."""
    model_path = os.path.join(os.getcwd(), 'models', 'xgboost_model.pkl')
    if not os.path.exists(model_path):
        logging.error(f"Missing model at {model_path}. Train module 5 first.")
        return None
    return joblib.load(model_path)

def predict_next_day(model, latest_features: pd.DataFrame) -> float:
    """Executes live inference with strict input validation."""
    if latest_features.empty:
        logging.error("Empty feature DataFrame provided for inference. Aborting.")
        return None
        
    logging.info("Validating inference inputs...")
    
    # 1. VALIDATION: Feature Order & Existence
    # Extract the exact feature names the XGBoost model was trained on
    try:
        expected_features = model.feature_names_in_
    except AttributeError:
        logging.error("Model does not contain feature_names_in_. Ensure scikit-learn/XGBoost versions match.")
        return None
    
    missing_features = [f for f in expected_features if f not in latest_features.columns]
    if missing_features:
        logging.error(f"FATAL: Missing required features for inference: {missing_features}")
        return None
        
    # 2. VALIDATION: Missing Values
    if latest_features[expected_features].isna().any().any():
        logging.warning("NaN values detected in live inference row. XGBoost will safely map these using default split directions.")
        
    # 3. ENFORCE ORDER: Align columns strictly to the trained schema
    X_live = latest_features[expected_features]
    
    # 4. PREDICT
    logging.info("Executing XGBoost Inference...")
    predicted_return = model.predict(X_live)[0]
    
    return float(predicted_return)

def main():
    """Standalone wrapper to execute the live inference cycle."""
    # Import the feature generator we built in Module 4
    try:
        from module4_features import load_datasets, generate_latest_features
    except ImportError:
        logging.error("Failed to import module4_features. Ensure it exists in the active directory.")
        return
        
    logging.info("Starting Module 12: Live Inference Deployment")
    
    sent_df, market_df = load_datasets()
    if sent_df is None or market_df is None: return
    
    # Generate precisely one row (the latest market date)
    latest_features = generate_latest_features(sent_df, market_df)
    
    model = load_inference_model()
    if model is None: return
    
    pred_return = predict_next_day(model, latest_features)
    
    if pred_return is not None:
        logging.info("==================================================")
        logging.info("            LIVE INFERENCE COMPLETE               ")
        logging.info("==================================================")
        logging.info(f"Base Feature Date   : {latest_features['date'].iloc[0]}")
        logging.info(f"Target Predict Date : Tomorrow (T+1)")
        logging.info(f"Predicted Return    : {pred_return*100:.4f}%")
        logging.info("==================================================")
        
        # Save output
        out_df = pd.DataFrame({
            'Feature_Date': [latest_features['date'].iloc[0]],
            'Predicted_Return_t+1': [pred_return]
        })
        out_path = os.path.join(os.getcwd(), 'outputs', 'live_inference.csv')
        out_df.to_csv(out_path, index=False)
        logging.info(f"Live prediction safely written to {out_path}")

if __name__ == "__main__":
    main()
