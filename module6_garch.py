import pandas as pd
import numpy as np
from arch import arch_model
import logging
import os

# ==========================================
# MODULE 6: ECONOMETRIC GARCH MODELING
# ==========================================
# Objective: Implement GARCH(1,1) via the `arch` package.
# Extend via ARX model by explicitly adding sentiment to the MEAN equation.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    if not os.path.exists(file_path):
        logging.error(f"Missing data/final_dataset.csv")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def fit_garch_models(df: pd.DataFrame):
    # Scale returns for optimization stability
    returns = df['return'] * 100
    
    # Exogenous variable for Mean equation (ARX)
    sentiment_x = df['sentiment_t']
    
    # ---------------------------------------------------------
    # 1. Standard GARCH(1,1)
    # ---------------------------------------------------------
    logging.info("Fitting Standard GARCH(1,1) (Mean='Constant')...")
    am_base = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    res_base = am_base.fit(disp='off')
    
    # ---------------------------------------------------------
    # 2. ARX-GARCH(1,1) with Exogenous Sentiment in Mean Eq
    # ---------------------------------------------------------
    logging.info("Fitting ARX-GARCH(1,1) (Mean='ARX', Exog=Sentiment)...")
    am_x = arch_model(returns, x=sentiment_x, vol='Garch', p=1, q=1, mean='ARX', dist='Normal')
    res_x = am_x.fit(disp='off')
    
    # ---------------------------------------------------------
    # 3. Model Evaluation & Comparison
    # ---------------------------------------------------------
    logging.info("--------------------------------------------------")
    logging.info("GARCH Model Comparison (Information Criteria)")
    logging.info("--------------------------------------------------")
    logging.info(f"Standard GARCH | AIC: {res_base.aic:.2f} | BIC: {res_base.bic:.2f}")
    logging.info(f"ARX-GARCH      | AIC: {res_x.aic:.2f} | BIC: {res_x.bic:.2f}")
    
    if res_x.aic < res_base.aic:
        logging.info("CONCLUSION: Sentiment significantly improves the fit (Lower AIC).")
    else:
        logging.info("CONCLUSION: Sentiment does not improve the fit.")
        
    return res_base, res_x

def main():
    logging.info("Starting Module 6: GARCH Volatility Modeling")
    df = load_data()
    if df is None: return
    
    res_base, res_x = fit_garch_models(df)
    
    # Extract conditional volatility (rescaled back to unscaled log returns)
    df['pred_vol_garch'] = res_base.conditional_volatility / 100.0
    df['pred_vol_garch_x'] = res_x.conditional_volatility / 100.0
    
    # Forecast Error (Target vs Pred shifted)
    df['forecast_error_garch'] = df['target_volatility_t+1'] - df['pred_vol_garch'].shift(-1)
    df['forecast_error_garch_x'] = df['target_volatility_t+1'] - df['pred_vol_garch_x'].shift(-1)
    
    # Prepare output format for Backtester
    out_df = df[['date', 'return', 'pred_vol_garch', 'pred_vol_garch_x']]
    
    out_path = os.path.join(os.getcwd(), 'outputs', 'garch_predictions.csv')
    out_df.to_csv(out_path, index=False)
    logging.info(f"Saved Volatility Forecasts to {out_path}")

if __name__ == "__main__":
    main()
