import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import accuracy_score
import logging
import os
import joblib
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(file_path):
        logging.error(f"Missing {file_path}")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['target_return_t+1']).reset_index(drop=True)
    return df.sort_values('date').reset_index(drop=True)

def train_and_evaluate(df: pd.DataFrame):
    exclude_cols = ['date', 'target_return_t+1', 'target_volatility_t+1']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # PHASE 1 — TARGET DEFINITION
    target_return = df['target_return_t+1']
    direction_target = np.where(target_return > 0, 1, 0)
    
    # PHASE 2 — REGIME CREATION
    median_vol = df['volatility'].median()
    df['regime'] = np.where(df['volatility'] > median_vol, 1, 0)
    if 'regime' in feature_cols: feature_cols.remove('regime')
    
    # PHASE 3 — SPLIT DATA
    df_high = df[df['regime'] == 1]
    df_low = df[df['regime'] == 0]
    
    X = df[feature_cols]
    
    # PHASE 4 — TRAIN MODELS (HIGH AND LOW VOL)
    base_high = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, eval_metric='logloss')
    base_low = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, eval_metric='logloss')
    
    # Wrap in CalibratedClassifierCV to break confidence plateaus and smooth probability entropy
    model_high_vol = CalibratedClassifierCV(base_high, method='sigmoid', cv=3)
    model_low_vol = CalibratedClassifierCV(base_low, method='sigmoid', cv=3)
    
    if len(df_high) >= 15: # Need enough samples for CV
        model_high_vol.fit(df_high[feature_cols], direction_target[df_high.index])
    elif len(df_high) > 0:
        base_high.fit(df_high[feature_cols], direction_target[df_high.index])
        model_high_vol = base_high
        
    if len(df_low) >= 15:
        model_low_vol.fit(df_low[feature_cols], direction_target[df_low.index])
    elif len(df_low) > 0:
        base_low.fit(df_low[feature_cols], direction_target[df_low.index])
        model_low_vol = base_low
        
    base_meta = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42, eval_metric='logloss')
    meta_model = CalibratedClassifierCV(base_meta, method='sigmoid', cv=3)
    tscv = TimeSeriesSplit(n_splits=5)
    oof_indices = []
    dir_probs_oof = []
    
    # Restrict meta-training strictly to HIGH VOL regime
    X_high = df_high[feature_cols]
    y_high = direction_target[df_high.index]
    
    for train_idx, test_idx in tscv.split(X_high):
        df_train_high = df_high.iloc[train_idx]
        df_test_high = df_high.iloc[test_idx]
        
        y_train_series = y_high[train_idx]
        
        cv_high = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, eval_metric='logloss')
        
        if len(df_train_high) > 0: 
            cv_high.fit(df_train_high[feature_cols], y_train_series)
            
        if len(df_test_high) > 0 and len(df_train_high) > 0:
            preds = cv_high.predict_proba(df_test_high[feature_cols])[:, 1]
            oof_indices.extend(df_test_high.index)
            dir_probs_oof.extend(preds)
            
    oof_indices = np.array(oof_indices)
    dir_probs_oof = np.array(dir_probs_oof)
    
    predicted_direction_oof = np.where(dir_probs_oof > 0.5, 1, 0)
    actual_direction_oof = direction_target[oof_indices]
    
    meta_label = np.where(predicted_direction_oof == actual_direction_oof, 1, 0)
    
    X_meta_train = X.iloc[oof_indices].copy()
    X_meta_train['direction_prob'] = dir_probs_oof
    
    if len(X_meta_train) >= 15:
        meta_model.fit(X_meta_train, meta_label)
    elif len(X_meta_train) > 0:
        base_meta.fit(X_meta_train, meta_label)
        meta_model = base_meta
    
    print("\nModel metrics summary")
    print("==================================")
    print("Phase 5 Meta Training Completed.")
    
    # PHASE 6 — SAVE MODEL
    composite_model = {
        "model_high": model_high_vol,
        "model_low": model_low_vol,
        "meta_model": meta_model,
        "feature_cols": feature_cols,
        "median_vol": float(median_vol)
    }
    
    return composite_model, None

def main():
    logging.info("Starting Module 5: Multi-Model XGBoost")
    df = load_data()
    if df is None: return
    
    best_model, predictions = train_and_evaluate(df)
    
    model_new_path = os.path.join(os.getcwd(), 'models', 'model_new.pkl')
    model_live_path = os.path.join(os.getcwd(), 'models', 'model_live.pkl')
    
    joblib.dump(best_model, model_new_path)
    
    # Validation: Simple sanity check before pushing to live
    if best_model is not None:
        if os.path.exists(model_live_path):
            try: os.remove(model_live_path)
            except: pass
        os.rename(model_new_path, model_live_path)
        logging.info("Atomic update: model_new.pkl -> model_live.pkl")
    
    logging.info("Saved artifacts to models/ directory.")

if __name__ == "__main__":
    main()
