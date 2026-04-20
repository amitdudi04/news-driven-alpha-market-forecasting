import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import logging
import os

# ==========================================
# MODULE 8: MODEL INTERPRETABILITY
# ==========================================
# Objective: Extract Gini importance from the XGBoost Regressor to 
# quantitatively evaluate the financial drivers of CSI 300 returns.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts() -> tuple:
    model_path = os.path.join(os.getcwd(), 'models', 'xgboost_model.pkl')
    data_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        logging.error("Missing model or data. Please run modules 4 and 5 first.")
        return None, None
        
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df

def extract_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and maps feature importance from the XGBoost model."""
    exclude_cols = ['date', 'target_return_t+1', 'target_volatility_t+1']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle both standalone XGB models and models wrapped in GridSearchCV
    if hasattr(model, 'best_estimator_'):
        xgb_model = model.best_estimator_
    else:
        xgb_model = model
        
    importances = xgb_model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    
    # Sort ascending for horizontal bar plot
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True).reset_index(drop=True)
    return feat_imp

def plot_importance(feat_imp: pd.DataFrame):
    """Generates a publication-ready horizontal bar chart of feature importance."""
    plt.figure(figsize=(10, 8))
    
    # Use color mapping to highlight sentiment-related features
    colors = ['#ff7f0e' if 'sentiment' in feat else '#1f77b4' for feat in feat_imp['Feature']]
    
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='black')
    
    plt.xlabel('Gini Importance (Information Gain)', fontsize=12, fontweight='bold')
    plt.ylabel('Predictive Features', fontsize=12, fontweight='bold')
    plt.title('XGBoost Drivers of CSI 300 Returns (Orange = Sentiment Signals)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'outputs', 'feature_importance.png')
    plt.savefig(out_path, dpi=300)
    logging.info(f"Feature importance visualization saved to {out_path}")

def analyze_findings(feat_imp: pd.DataFrame):
    """Outputs the ranked feature list for console review."""
    logging.info("==================================================")
    logging.info("       QUANTITATIVE INTERPRETATION REPORT         ")
    logging.info("==================================================")
    
    # Sort descending for printing
    top_features = feat_imp.sort_values(by='Importance', ascending=False)
    
    for _, row in top_features.iterrows():
        logging.info(f"{row['Feature']:25} : {row['Importance']:.4f}")
        
    logging.info("==================================================")

def main():
    logging.info("Starting Module 8: Model Interpretability")
    model, df = load_artifacts()
    if model is None or df is None: return
    
    feat_imp = extract_importance(model, df)
    plot_importance(feat_imp)
    analyze_findings(feat_imp)
    
    # Save raw importances to CSV
    csv_path = os.path.join(os.getcwd(), 'outputs', 'feature_importance.csv')
    feat_imp.sort_values('Importance', ascending=False).to_csv(csv_path, index=False)
    logging.info(f"Raw importance scores saved to {csv_path}")

if __name__ == "__main__":
    main()
