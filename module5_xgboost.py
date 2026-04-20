import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import os
import joblib

# ==========================================
# MODULE 5: XGBOOST PREDICTIVE MODEL
# ==========================================
# Objective: Train an XGBoost model using rigorous time-series cross-validation.
# Target: target_return_t+1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> pd.DataFrame:
    """Loads feature-engineered dataset."""
    file_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(file_path):
        logging.error(f"Missing {file_path}")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def train_and_evaluate(df: pd.DataFrame):
    """Performs strict grid search and outputs optimal model and predictions."""
    exclude_cols = ['date', 'target_return_t+1', 'target_volatility_t+1']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_return_t+1']
    
    # ---------------------------------------------------------
    # 1. Time Series Split
    # ---------------------------------------------------------
    # Ensures strictly chronologically expanding windows (no look-ahead leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # ---------------------------------------------------------
    # 2. Hyperparameter Grid
    # ---------------------------------------------------------
    param_grid = {
        'max_depth': [3, 5, 7],               # Shallow trees to prevent noise overfitting
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300]
    }
    
    logging.info("Running GridSearchCV (TimeSeriesSplit)...")
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    logging.info(f"Optimal Parameters: {grid_search.best_params_}")
    
    # ---------------------------------------------------------
    # 3. Evaluation
    # ---------------------------------------------------------
    predictions = best_model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    dir_acc = np.mean(np.sign(y) == np.sign(predictions)) * 100
    
    logging.info(f"RMSE: {rmse:.6f}")
    logging.info(f"Directional Accuracy: {dir_acc:.2f}%")
    
    # ---------------------------------------------------------
    # 4. Feature Importance Extraction
    # ---------------------------------------------------------
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    logging.info("Top 3 Drivers of Alpha:")
    for i, row in feat_imp.head(3).iterrows():
        logging.info(f" - {row['Feature']}: {row['Importance']:.4f}")
        
    return best_model, predictions

def main():
    logging.info("Starting Module 5: XGBoost Modeling")
    df = load_data()
    if df is None: return
    
    best_model, predictions = train_and_evaluate(df)
    
    # Export predictions for Module 7 (Backtesting)
    pred_df = pd.DataFrame({
        'date': df['date'].dt.strftime('%Y-%m-%d'),
        'actual_return_t+1': df['target_return_t+1'],
        'predicted_return_t+1': predictions
    })
    
    pred_path = os.path.join(os.getcwd(), 'outputs', 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    
    # Export Model
    model_path = os.path.join(os.getcwd(), 'models', 'xgboost_model.pkl')
    joblib.dump(best_model, model_path)
    
    logging.info(f"Saved artifacts to models/ and outputs/ directories.")

if __name__ == "__main__":
    main()
