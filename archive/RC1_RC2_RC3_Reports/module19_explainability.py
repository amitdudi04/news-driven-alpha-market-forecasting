import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# MODULE 19: MODEL EXPLAINABILITY (SHAP)
# ==========================================
# Objective: Crack open the XGBoost "black box" using Shapley Additive Explanations 
# to mathematically prove how sentiment interacts with market regimes to generate alpha.

def main():
    logging.info("Starting Module 19: SHAP Explainability Analysis")
    
    data_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(data_path):
        logging.error("Missing final_dataset.csv. Run module 4 first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)
    
    X = df.drop(columns=['date', 'target_return_t+1', 'target_volatility_t+1'])
    y = df['target_return_t+1']
    
    # Train a baseline tree model specifically for interpretability
    # We use XGBoost as it handles SHAP TreeExplainer natively and extremely fast
    logging.info("Training baseline XGBoost Regressor for SHAP extraction...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    logging.info("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    out_dir = os.path.join(os.getcwd(), 'outputs')
    
    # ---------------------------------------------------------
    # 1. GLOBAL FEATURE IMPORTANCE (BAR PLOT)
    # ---------------------------------------------------------
    # Proves *which* features matter most on average
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance", fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'shap_importance_bar.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    logging.info(f"Feature importance bar chart saved to {bar_path}")
    
    # ---------------------------------------------------------
    # 2. SHAP SUMMARY PLOT (BEESWARM)
    # ---------------------------------------------------------
    # Proves the *directionality* of the features (e.g., does high sentiment push returns up or down?)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Distribution (Directional Impact)", fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(out_dir, 'shap_summary_beeswarm.png')
    plt.savefig(summary_path, dpi=300)
    plt.close()
    logging.info(f"Directional SHAP summary saved to {summary_path}")
    
    # ---------------------------------------------------------
    # 3. INTERACTION DEPENDENCE PLOT
    # ---------------------------------------------------------
    # Physically proves the thesis: how does sentiment impact returns as a function of volatility?
    # We plot the SHAP value of sentiment against the raw sentiment value, color-coded by volatility
    
    if 'sentiment_t' in X.columns and 'volatility' in X.columns:
        plt.figure(figsize=(8, 6))
        # shap.dependence_plot automatically picks the strongest interaction if interaction_index is None
        # We explicitly force it to interact with 'volatility' to prove our thesis
        shap.dependence_plot("sentiment_t", shap_values, X, interaction_index="volatility", show=False)
        plt.title("SHAP Dependence: Sentiment Interacting with Volatility", fontweight='bold')
        plt.tight_layout()
        dep_path = os.path.join(out_dir, 'shap_dependence_sentiment_volatility.png')
        plt.savefig(dep_path, dpi=300)
        plt.close()
        logging.info(f"Interaction dependence plot saved to {dep_path}")
    else:
        logging.warning("Could not generate dependence plot: missing 'sentiment_t' or 'volatility' columns.")

    logging.info("Module 19 Execution Complete. Review the SHAP outputs in the outputs/ directory.")

if __name__ == "__main__":
    main()
