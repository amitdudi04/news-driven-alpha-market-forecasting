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
        expected_features = model["feature_cols"]
    except KeyError:
        logging.error("Model does not contain feature_cols. Ensure scikit-learn/XGBoost versions match.")
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
    
    # STATISTICAL DRIFT GOVERNANCE (PSI & Z-SCORE OUTLIER DETECTION)
    # The scaled values are effectively Z-scores against the training distribution.
    # We validate them against catastrophic drift thresholds to prevent generating predictions on alien data.
    
    from config.institutional_config import DRIFT_LIMITS, CONFIDENCE_SCALING, DATA_QUALITY_LIMITS, MODEL_VERSION_GOVERNANCE
    
    # 2b. LIVE DATA QUALITY GOVERNANCE
    if DATA_QUALITY_LIMITS["require_all_features"] and latest_features[expected_features].isna().any().any():
        logging.warning("Missing features detected. Default split directions will be used, but this impacts alpha quality.")
    if not DATA_QUALITY_LIMITS["allow_zero_volume"] and 'volume' in latest_features.columns:
        if (latest_features['volume'] == 0).any():
            raise RuntimeError("SAFE MODE ESCALATION: Zero-volume anomaly detected in live data. Possible missing market session.")
            
    scaled_values_array = X_live.values[0]
    max_drift_z = np.max(np.abs(scaled_values_array))
    
    if max_drift_z > DRIFT_LIMITS["max_z_score_drift"]:
        # Extreme feature drift detected (Flash Crash / Distribution Collapse)
        # In a full PSI implementation, this simulates a PSI > 0.25 bin shift.
        raise RuntimeError(f"SAFE MODE ESCALATION: Catastrophic data drift detected. Max Z-score = {max_drift_z:.2f}. "
                           f"Threshold = {DRIFT_LIMITS['max_z_score_drift']}. Execution Halted to protect capital.")
                           
    # 4. PREDICT
    logging.info("Executing XGBoost Inference...")
    median_vol = model.get('median_vol', 0.2)
    vol_value = X_live['volatility'].iloc[0]
    is_high_vol = vol_value > median_vol
    
    base_prob_high = model['model_high'].predict_proba(X_live)[0][1]
    base_prob_low = model['model_low'].predict_proba(X_live)[0][1]
    
    # REGIME BOUNDARY GOVERNANCE (Epistemic Uncertainty Suppression)
    boundary_margin = 0.10
    near_boundary = abs(vol_value - median_vol) < boundary_margin
    agree_direction = (base_prob_high > 0.5) == (base_prob_low > 0.5)
    
    # 4. META-MODEL CONFIDENCE SCORING & PROBABILITY ROUTING
    if is_high_vol:
        base_prob = base_prob_high
        
        X_meta = X_live.copy()
        X_meta['direction_prob'] = base_prob
        
        meta_prob = model['meta_model'].predict_proba(X_meta)[0][1]
        # FIX: The meta-model is overfit and causing severe confidence inversion (HIGH conf = -66 bps).
        # We replace it with strict probabilistic entropy (distance from 0.5), scaled to financial ML bounds.
        raw_confidence = min(abs(base_prob - 0.5) * CONFIDENCE_SCALING["entropy_scalar"], 1.0)
        dir_prob = base_prob
    else:
        base_prob = base_prob_low
        dir_prob = base_prob
        meta_prob = 1.0
        raw_confidence = min(abs(base_prob - 0.5) * CONFIDENCE_SCALING["entropy_scalar"], 1.0)
        
    # DISTANCE PENALTY (CONFIDENCE DECAY)
    l2_distance = np.linalg.norm(X_live.values[0])
    tau_threshold = np.sqrt(len(X_live.columns)) * 1.5
    distance_penalty = np.exp(-0.5 * (l2_distance / tau_threshold)**2)
    
    # BOUNDARY UNCERTAINTY PENALTY
    # If we are near the regime threshold and the two regime models disagree,
    # it indicates high epistemic uncertainty. We collapse confidence to 0.
    regime_penalty = 0.0 if (near_boundary and not agree_direction) else 1.0
    
    final_confidence = raw_confidence * distance_penalty * regime_penalty
        
    # FEATURE CONTRIBUTIONS
    try:
        active_model = model['model_high'] if is_high_vol else model['model_low']
        importances = None
        
        # Handle CalibratedClassifierCV which stores fitted models in calibrated_classifiers_
        if hasattr(active_model, 'calibrated_classifiers_'):
            est_importances = []
            for clf in active_model.calibrated_classifiers_:
                base_est = getattr(clf, 'estimator', getattr(clf, 'base_estimator', None))
                if base_est and hasattr(base_est, 'feature_importances_'):
                    est_importances.append(base_est.feature_importances_)
            if est_importances:
                importances = np.mean(est_importances, axis=0)
        elif hasattr(active_model, 'feature_importances_'):
            importances = active_model.feature_importances_
        
        if importances is not None:
            feature_values = X_live.values[0]
            contributions = feature_values * importances
            contrib_series = pd.Series(contributions, index=X_live.columns)
            top_positive = contrib_series[contrib_series > 0].nlargest(2).to_dict()
            top_negative = contrib_series[contrib_series < 0].nsmallest(2).to_dict()
        else:
            top_positive = {}
            top_negative = {}
    except:
        top_positive = {}
        top_negative = {}

    return {
        'dir_prob': float(dir_prob),
        'meta_prob': float(meta_prob),
        'raw_confidence': float(raw_confidence),
        'distance_penalty': float(distance_penalty),
        'confidence': float(final_confidence),
        'regime': 'HIGH_VOL' if is_high_vol else 'LOW_VOL',
        'top_positive': top_positive,
        'top_negative': top_negative,
        'model_version': MODEL_VERSION_GOVERNANCE["current_model_version"],
        'scaler_version': MODEL_VERSION_GOVERNANCE["current_scaler_version"]
    }

def main(execution_uuid: str = None):
    """Standalone wrapper to execute the live inference cycle."""
    # Import the feature generator we built in Module 4
    import datetime
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
    
    # SNAPSHOT FRESHNESS VALIDATION
    feature_date_str = latest_features['date'].iloc[0]
    feature_date = pd.to_datetime(feature_date_str)
    now = datetime.datetime.now()
    age = (now - feature_date).days
    
    import os
    MAX_STALE_DAYS = int(os.environ.get('MAX_STALE_DAYS', 4))
    if age > MAX_STALE_DAYS:
        raise ValueError(f"FATAL: Dataset is {age} days old (Date: {feature_date_str}), exceeding MAX_STALE_DAYS={MAX_STALE_DAYS}. Inference aborted to prevent infinite stale persistence.")
    
    model = load_inference_model()
    if model is None: return
    
    probs = predict_next_day(model, latest_features)
    
    if probs is not None:
        dir_prob = probs['dir_prob']
        meta_prob = probs['meta_prob']
        raw_confidence = probs['raw_confidence']
        distance_penalty = probs['distance_penalty']
        confidence = probs['confidence']
        regime = probs['regime']
        import json
        top_pos = json.dumps(probs['top_positive'])
        top_neg = json.dumps(probs['top_negative'])
        
        logging.info("==================================================")
        logging.info("            LIVE INFERENCE COMPLETE               ")
        logging.info("==================================================")
        logging.info(f"Base Feature Date   : {feature_date_str}")
        logging.info(f"Target Predict Date : Tomorrow (T+1)")
        logging.info(f"Regime Assessed     : {regime}")
        logging.info(f"Direction Prob      : {dir_prob*100:.2f}%")
        logging.info(f"Meta Prob           : {meta_prob*100:.2f}%")
        logging.info(f"Raw Confidence      : {raw_confidence*100:.2f}%")
        logging.info(f"Distance Penalty    : {distance_penalty:.4f}")
        logging.info(f"Final Confidence    : {confidence*100:.2f}%")
        logging.info("==================================================")
        
        # Save output
        out_df = pd.DataFrame({
            'Feature_Date': [feature_date_str],
            'Direction_Prob_t+1': [dir_prob],
            'Meta_Prob_t+1': [meta_prob],
            'raw_confidence': [raw_confidence],
            'distance_penalty': [distance_penalty],
            'Confidence': [confidence],
            'regime': [regime],
            'top_positive': [top_pos],
            'top_negative': [top_neg],
            'execution_uuid': [execution_uuid] if execution_uuid else ['unknown']
        })
        out_path = os.path.join(os.getcwd(), 'outputs', 'live_inference.csv')
        
        import module0_atomic_storage
        module0_atomic_storage.atomic_write_csv(out_df, out_path, index=False)
        logging.info(f"Live prediction safely written atomically to {out_path}")

if __name__ == "__main__":
    main()
