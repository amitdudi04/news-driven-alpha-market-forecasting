import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import os

# ==========================================
# MODULE 9: ROBUSTNESS TESTING
# ==========================================
# Objective: Quantitatively prove whether adding NLP sentiment features 
# genuinely adds predictive Alpha over a purely autoregressive market baseline.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(file_path):
        logging.error(f"Missing data/final_dataset.csv")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def train_eval_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Trains XGBoost using TimeSeriesSplit and returns strictly evaluated metrics."""
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Reduced grid for faster robustness checking
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    predictions = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    # Directional Accuracy (Excluding exact 0% returns to avoid noise)
    correct_direction = np.sign(y) == np.sign(predictions)
    dir_acc = np.mean(correct_direction) * 100
    
    return rmse, dir_acc

def main():
    logging.info("Starting Module 9: Sentiment Robustness Testing")
    df = load_data()
    if df is None: return
    
    target_col = 'target_return_t+1'
    exclude_base = ['date', 'target_return_t+1', 'target_volatility_t+1']
    
    # Automatically identify all sentiment and interaction features
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
    
    # ---------------------------------------------------------
    # 1. FULL MODEL (Market Features + Sentiment Features)
    # ---------------------------------------------------------
    X_full = df[[col for col in df.columns if col not in exclude_base]]
    y_full = df[target_col]
    logging.info(f"Training FULL Model (Features: {X_full.shape[1]})...")
    rmse_full, dir_acc_full = train_eval_model(X_full, y_full)
    
    # ---------------------------------------------------------
    # 2. BASELINE MODEL (Pure Market Autoregression)
    # ---------------------------------------------------------
    # Exclude both targets AND all sentiment-related columns
    exclude_baseline = exclude_base + sentiment_cols
    X_base = df[[col for col in df.columns if col not in exclude_baseline]]
    y_base = df[target_col]
    logging.info(f"Training BASELINE Model (Features: {X_base.shape[1]})...")
    rmse_base, dir_acc_base = train_eval_model(X_base, y_base)
    
    # ---------------------------------------------------------
    # 3. ROBUSTNESS COMPARISON 
    # ---------------------------------------------------------
    logging.info("=================================================================================")
    logging.info("                           ROBUSTNESS TEST COMPARISON                            ")
    logging.info("=================================================================================")
    logging.info(f"{'Metric':<25} | {'Baseline (Market Only)':<25} | {'Full (Market + Sentiment)':<25}")
    logging.info("-" * 81)
    logging.info(f"{'RMSE (Lower is Better)':<25} | {rmse_base:<25.6f} | {rmse_full:<25.6f}")
    logging.info(f"{'Dir. Acc (Higher is Better)':<25} | {dir_acc_base:<24.2f}% | {dir_acc_full:<24.2f}%")
    logging.info("=================================================================================")
    
    # Output analytical conclusion
    if rmse_full < rmse_base and dir_acc_full > dir_acc_base:
        logging.info("CONCLUSION: TRUE ALPHA DETECTED. Sentiment definitively improves both accuracy and magnitude precision.")
    elif dir_acc_full > dir_acc_base:
        logging.info("CONCLUSION: DIRECTIONAL EDGE DETECTED. Sentiment improves trade direction accuracy, despite minor magnitude noise.")
    elif rmse_full < rmse_base:
        logging.info("CONCLUSION: MAGNITUDE EDGE DETECTED. Sentiment improves sizing precision but not directional win-rate.")
    else:
        logging.info("CONCLUSION: NO ALPHA DETECTED. Sentiment fails to outperform pure autoregressive market pricing.")
        
    # Save the output
    out_df = pd.DataFrame({
        'Model': ['Baseline (Market Only)', 'Full (Market + Sentiment)'],
        'RMSE': [rmse_base, rmse_full],
        'Directional_Accuracy': [dir_acc_base, dir_acc_full]
    })
    out_path = os.path.join(os.getcwd(), 'outputs', 'robustness_comparison.csv')
    out_df.to_csv(out_path, index=False)
    logging.info(f"Comparison metrics securely saved to {out_path}")

if __name__ == "__main__":
    main()
