import pandas as pd
import numpy as np
import logging
import os
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> pd.DataFrame:
    data_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(data_path):
        logging.error("Missing final_dataset.csv. Run module 4 first.")
        return None
        
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)
    return df

def generate_cv_predictions(X, y, model, n_splits=5):
    """Generates contiguous out-of-sample predictions strictly using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.full(len(y), np.nan)
    
    # Simple strategy: for each split, train on train, predict on test.
    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oof_preds[test_idx] = preds
        
    # For the very first fold (which has no prior training data), we can backfill or use a simple mean
    first_test_idx = tscv.split(X).__next__()[1][0]
    oof_preds[:first_test_idx] = y.mean()
    
    return oof_preds

def train_regime_ensemble(df_regime: pd.DataFrame, regime_name: str):
    """Trains an XGBoost + Ridge ensemble and calculates dynamic inverse-RMSE weights."""
    X = df_regime.drop(columns=['date', 'target_return_t+1', 'target_volatility_t+1'])
    y = df_regime['target_return_t+1']
    
    logging.info(f"Training [{regime_name}] Regime Ensemble (N={len(X)})...")
    
    # 1. XGBoost
    xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    xgb_preds = generate_cv_predictions(X, y, xgb_model)
    
    # 2. Linear Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_preds = generate_cv_predictions(X, y, ridge_model)
    
    # 3. Dynamic Ensemble Weighting (Inverse RMSE)
    # Calculate RMSE strictly on the out-of-sample predictions
    valid_mask = ~np.isnan(xgb_preds) & ~np.isnan(ridge_preds)
    
    rmse_xgb = np.sqrt(mean_squared_error(y[valid_mask], xgb_preds[valid_mask]))
    rmse_ridge = np.sqrt(mean_squared_error(y[valid_mask], ridge_preds[valid_mask]))
    
    weight_xgb = 1.0 / (rmse_xgb + 1e-6)
    weight_ridge = 1.0 / (rmse_ridge + 1e-6)
    
    total_weight = weight_xgb + weight_ridge
    w_xgb = weight_xgb / total_weight
    w_ridge = weight_ridge / total_weight
    
    logging.info(f"[{regime_name}] Dynamic Weights -> XGBoost: {w_xgb*100:.1f}% | Ridge: {w_ridge*100:.1f}%")
    
    # Final Weighted Prediction
    final_preds = (w_xgb * xgb_preds) + (w_ridge * ridge_preds)
    
    # Retrain on full regime data for production serialization
    xgb_model.fit(X, y)
    ridge_model.fit(X, y)
    
    return final_preds, xgb_model, ridge_model, w_xgb, w_ridge

def main():
    logging.info("Starting Module 15: Regime-Aware Ensemble Modeling")
    df = load_data()
    if df is None: return
    
    # Splitting logic: High vs Low Volatility
    # Note: We must use the 'volatility' feature (which is known at time t) to split
    median_vol = df['volatility'].median()
    
    high_vol_mask = df['volatility'] > median_vol
    df_high = df[high_vol_mask].copy().reset_index(drop=True)
    df_low = df[~high_vol_mask].copy().reset_index(drop=True)
    
    # Train High Volatility Ensemble
    preds_high, xgb_high, ridge_high, w_xgb_high, w_ridge_high = train_regime_ensemble(df_high, "HIGH VOLATILITY")
    df_high['advanced_pred_return'] = preds_high
    
    # Train Low Volatility Ensemble
    preds_low, xgb_low, ridge_low, w_xgb_low, w_ridge_low = train_regime_ensemble(df_low, "LOW VOLATILITY")
    df_low['advanced_pred_return'] = preds_low
    
    # Recombine and sort chronologically
    df_final = pd.concat([df_high, df_low]).sort_values('date').reset_index(drop=True)
    
    # Save Out-Of-Sample Predictions
    out_df = df_final[['date', 'target_return_t+1', 'advanced_pred_return']]
    out_df.rename(columns={'target_return_t+1': 'actual_return_t+1'}, inplace=True)
    
    out_path = os.path.join(os.getcwd(), 'outputs', 'advanced_predictions.csv')
    out_df.to_csv(out_path, index=False)
    logging.info(f"Advanced ensemble predictions securely saved to {out_path}")
    
    # Serialize Models and Weights for Live Inference
    ensemble_package = {
        'median_vol_threshold': median_vol,
        'high_vol': {
            'xgb': xgb_high,
            'ridge': ridge_high,
            'w_xgb': w_xgb_high,
            'w_ridge': w_ridge_high
        },
        'low_vol': {
            'xgb': xgb_low,
            'ridge': ridge_low,
            'w_xgb': w_xgb_low,
            'w_ridge': w_ridge_low
        }
    }
    
    model_path = os.path.join(os.getcwd(), 'models', 'regime_ensemble_model.pkl')
    joblib.dump(ensemble_package, model_path)
    logging.info(f"Regime-Aware Ensemble package serialized to {model_path}")

if __name__ == "__main__":
    main()
