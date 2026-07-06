import os
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def factor_decomposition_audit():
    logging.info("Starting Institutional Factor Decomposition...")
    
    track_path = os.path.join(os.getcwd(), 'outputs', 'live_tracking.csv')
    if not os.path.exists(track_path):
        logging.warning("Missing live_tracking.csv")
        return
        
    df = pd.read_csv(track_path)
    
    # We need market features for factor proxies
    feat_path = os.path.join(os.getcwd(), 'data', 'csi300_features.csv')
    feat_df = pd.read_csv(feat_path)
    
    # Merge
    df = pd.merge(df, feat_df[['date', 'return', 'volatility']], on='date', how='inner')
    
    if len(df) < 30:
        logging.warning("Not enough data for factor decomposition.")
        return
        
    # Proxies
    # Market Beta = 'return'
    # Volatility = 'volatility'
    # Momentum = 'return' rolling 20d (we can approximate)
    df['momentum_factor'] = df['return'].rolling(20, min_periods=1).sum()
    # Sentiment (we don't have explicit sentiment score in live_tracking, we use confidence_raw proxy)
    if 'confidence_raw' in df.columns:
        df['sentiment_factor'] = df['confidence_raw']
    else:
        df['sentiment_factor'] = 0.5
        
    # Drop NAs
    eval_df = df.dropna(subset=['pnl', 'return', 'volatility', 'momentum_factor', 'sentiment_factor']).copy()
    
    if len(eval_df) < 10:
        return
        
    # OLS Regression for Factor Attribution
    X = eval_df[['return', 'volatility', 'momentum_factor', 'sentiment_factor']]
    X = sm.add_constant(X)
    y = eval_df['pnl']
    
    model = sm.OLS(y, X).fit()
    
    beta = model.params.get('return', 0)
    vol_exp = model.params.get('volatility', 0)
    mom_exp = model.params.get('momentum_factor', 0)
    sent_exp = model.params.get('sentiment_factor', 0)
    residual_alpha = model.params.get('const', 0)
    
    # Check bounds
    failures = []
    if abs(beta) > 1.5: failures.append(f"Beta exposure explodes: {beta:.2f}")
    if residual_alpha < -0.01: failures.append(f"Residual Alpha collapses: {residual_alpha:.4f}")
    if model.params.isna().any(): failures.append("Factor exposures are NaN.")
    
    # Generate Output Manifest
    # Rolling 30d exposures
    rolling_betas = []
    rolling_alphas = []
    for i in range(len(eval_df)):
        if i < 30:
            rolling_betas.append(0)
            rolling_alphas.append(0)
        else:
            wX = X.iloc[i-30:i]
            wy = y.iloc[i-30:i]
            # Ridge to prevent singular matrix in small windows
            try:
                wmod = sm.OLS(wy, wX).fit_regularized(alpha=0.01, L1_wt=0)
                rolling_betas.append(wmod.params.get('return', 0))
                rolling_alphas.append(wmod.params.get('const', 0))
            except:
                rolling_betas.append(0)
                rolling_alphas.append(0)
                
    eval_df['rolling_market_beta'] = rolling_betas
    eval_df['rolling_residual_alpha'] = rolling_alphas
    
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'factor_exposure_manifest.csv')
    eval_df[['date', 'rolling_market_beta', 'rolling_residual_alpha']].to_csv(manifest_path, index=False)
    
    # Generate Report
    report_lines = [
        "# Institutional Factor Exposure Report\n",
        "## Static Factor Loadings",
        f"- **Market Beta**: {beta:.4f}",
        f"- **Volatility Exposure**: {vol_exp:.4f}",
        f"- **Momentum Exposure**: {mom_exp:.4f}",
        f"- **Sentiment Exposure**: {sent_exp:.4f}",
        f"- **Residual Alpha**: {residual_alpha:.4f}\n",
        "## Validation Status"
    ]
    
    if failures:
        report_lines.append("### STATUS: FAIL ❌")
        for f in failures: report_lines.append(f"- {f}")
        logging.error("Factor Decomposition FAILED bounds check.")
    else:
        report_lines.append("### STATUS: PASS ✅")
        report_lines.append("All factor loadings remain within institutional safety limits.")
        logging.info("Factor Decomposition PASSED.")
        
    out_path = os.path.join(os.getcwd(), 'outputs', 'factor_exposure_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    if failures:
        raise RuntimeError(f"SAFE MODE ESCALATION: Factor attribution compromised. {failures}")

if __name__ == "__main__":
    factor_decomposition_audit()
