import pandas as pd
import numpy as np
import logging
import os

# ==========================================
# MODULE 4: FEATURE ENGINEERING
# ==========================================
# Objective: Merge datasets and construct predictive features,
# specifically engineering interaction terms and strictly shifting targets.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_datasets() -> tuple:
    sent_path = os.path.join(os.getcwd(), 'data', 'sentiment_features.csv')
    market_path = os.path.join(os.getcwd(), 'data', 'csi300_features.csv')
    
    if not os.path.exists(sent_path) or not os.path.exists(market_path):
        logging.error("Missing input data in data/ directory. Run modules 2 and 3.")
        return None, None
        
    sent_df = pd.read_csv(sent_path)
    market_df = pd.read_csv(market_path)
    
    sent_df['date'] = pd.to_datetime(sent_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    return sent_df, market_df

def engineer_features(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    # Outer join to preserve weekend sentiment, then ffill market data
    df = pd.merge(market_df, sent_df, on='date', how='outer')
    
    # CRITICAL: Re-sort to guarantee time integrity before applying lags
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    
    logging.info("Engineering Base Features & Lags...")
    
    # ---------------------------------------------------------
    # 1. Base Sentiment Features
    # New Alpha Features (Phase 4)
    df['sentiment_roll_5'] = df['sentiment_mean'].rolling(window=5).mean()
    df['sentiment_roll_10'] = df['sentiment_mean'].rolling(window=10).mean()
    df['sentiment_zscore'] = (df['sentiment_mean'] - df['sentiment_mean'].rolling(20).mean()) / (df['sentiment_mean'].rolling(20).std() + 1e-6)
    
    # NEW STRONG FEATURES
    df['sentiment_momentum'] = df['sentiment_mean'] - df['sentiment_mean'].shift(1)
    df['sentiment_volatility'] = df['sentiment_mean'].rolling(20).std()
    df['news_intensity'] = df['article_count'] / (df['article_count'].rolling(7).mean() + 1e-6)
    
    df['volatility'] = df['return'].rolling(20).std()
    
    ma_short = df['close'].rolling(5).mean()
    ma_long = df['close'].rolling(20).mean()
    df['momentum'] = ma_short - ma_long
    df['momentum_acceleration'] = df['momentum'] - df['momentum'].shift(1)
    
    # ---------------------------------------------------------
    # 3. INTERACTION TERMS & REGIME
    # ---------------------------------------------------------
    logging.info("Generating Interaction Terms...")
    df['sentiment_x_volatility'] = df['sentiment_roll_5'] * df['volatility']
    df['sentiment_zscore_x_volatility'] = df['sentiment_zscore'] * df['volatility']
    df['regime_dummy'] = np.where(df['volatility'] > 0.20, 1, 0)
    
    # ---------------------------------------------------------
    # 4. TARGET ALIGNMENT & NAN HANDLING
    # ---------------------------------------------------------
    # Shift(-1) maps tomorrow's return to today's row index.
    df['target_return_t+1'] = df['return'].shift(-1)
    df['target_volatility_t+1'] = df['volatility'].shift(-1)
    
    # ---------------------------------------------------------
    # 5. Strict Cleaning
    # ---------------------------------------------------------
    initial_len = len(df)
    
    exclude = ['target_return_t+1', 'target_volatility_t+1', 'date', 'return', 'close', 'sentiment_mean', 'sentiment_std', 'article_count']
    base_feature_cols = [col for col in df.columns if col not in exclude]
    
    import joblib
    from sklearn.preprocessing import StandardScaler
    scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
    
    if os.path.exists(scaler_path):
        df.dropna(subset=base_feature_cols, inplace=True)
        # ---------------------------------------------------------
        # 6. FEATURE VALIDATION (LOCKED TO SCALER)
        # ---------------------------------------------------------
        scaler = joblib.load(scaler_path)
        final_features = list(scaler.feature_names_in_)
        
        # PRE-SCALER STABILIZATION (RECOVERY_MODE for historical reconstruction)
        # Only apply clipping to rows that are considered 'recovery' (i.e. > MAX_STALE_DAYS gap)
        MAX_STALE_DAYS = int(os.environ.get('MAX_STALE_DAYS', 4))
        df['date_obj'] = pd.to_datetime(df['date'])
        df['gap'] = df['date_obj'].diff().dt.days
        
        stabilize_targets = ['news_intensity', 'article_count', 'sentiment_zscore', 'sentiment_volatility', 'sentiment_x_volatility', 'sentiment_zscore_x_volatility']
        
        for f in stabilize_targets:
            if f in df.columns:
                rolling_95 = df[f].rolling(252, min_periods=20).quantile(0.95)
                rolling_05 = df[f].rolling(252, min_periods=20).quantile(0.05)
                # Apply clipping where gap > MAX_STALE_DAYS
                mask = df['gap'] > MAX_STALE_DAYS
                df.loc[mask, f] = np.clip(df.loc[mask, f], rolling_05[mask], rolling_95[mask])
                
        df[final_features] = scaler.transform(df[final_features])
        logging.info(f"Locked scaling loaded. Applied to {len(final_features)} features.")
    else:
        df.dropna(subset=base_feature_cols + ['target_return_t+1'], inplace=True)
        # ---------------------------------------------------------
        # 6. FEATURE VALIDATION & FILTERING (IC FILTER)
        # ---------------------------------------------------------
        from scipy.stats import spearmanr
        final_features = ['volatility'] if 'volatility' in base_feature_cols else []
        for f in base_feature_cols:
            if f == 'volatility': continue
            if df[f].nunique() <= 1: continue # Drop constants
            ic, _ = spearmanr(df[f], df['target_return_t+1'])
            if abs(ic) >= 0.02:
                final_features.append(f)
                
        logging.info(f"Retained {len(final_features)} / {len(base_feature_cols)} features after IC filtering.")
        
        scaler = StandardScaler()
        df[final_features] = scaler.fit_transform(df[final_features])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        
    # Only keep the validated features + targets + date
    columns_to_keep = ['date', 'target_return_t+1', 'target_volatility_t+1'] + final_features
    df = df[columns_to_keep].copy()
    
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

def generate_latest_features(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Optimized function specifically for real-time inference. Extracts ONLY the latest row."""
    import joblib
    
    # 1. MERGE
    df = pd.merge(market_df, sent_df, on='date', how='outer')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    
    # 2. GENERATE EXACTLY ALL FEATURES FROM TRAINING
    df['sentiment_roll_5'] = df['sentiment_mean'].rolling(window=5).mean()
    df['sentiment_roll_10'] = df['sentiment_mean'].rolling(window=10).mean()
    df['sentiment_zscore'] = (df['sentiment_mean'] - df['sentiment_mean'].rolling(20).mean()) / (df['sentiment_mean'].rolling(20).std() + 1e-6)
    
    df['sentiment_momentum'] = df['sentiment_mean'] - df['sentiment_mean'].shift(1)
    df['sentiment_volatility'] = df['sentiment_mean'].rolling(20).std()
    df['news_intensity'] = df['article_count'] / (df['article_count'].rolling(7).mean() + 1e-6)
    
    df['volatility'] = df['return'].rolling(20).std()
    
    ma_short = df['close'].rolling(5).mean()
    ma_long = df['close'].rolling(20).mean()
    df['momentum'] = ma_short - ma_long
    df['momentum_acceleration'] = df['momentum'] - df['momentum'].shift(1)
    
    df['sentiment_x_volatility'] = df['sentiment_roll_5'] * df['volatility']
    df['sentiment_zscore_x_volatility'] = df['sentiment_zscore'] * df['volatility']
    df['regime_dummy'] = np.where(df['volatility'] > 0.20, 1, 0)
    
    # RECOVERY MODE DETECTION & PRE-SCALER STABILIZATION
    df['date_obj'] = pd.to_datetime(df['date'])
    stale_days = (df['date_obj'].iloc[-1] - df['date_obj'].iloc[-2]).days if len(df) > 1 else 0
    import os
    MAX_STALE_DAYS = int(os.environ.get('MAX_STALE_DAYS', 4))
    
    is_recovery_mode = stale_days > MAX_STALE_DAYS
    
    # Identify variables to stabilize
    stabilize_targets = ['news_intensity', 'article_count', 'sentiment_zscore', 'sentiment_volatility', 'sentiment_x_volatility', 'sentiment_zscore_x_volatility']
    
    if is_recovery_mode:
        logging.warning(f"RECOVERY_MODE ACTIVATED: Ingestion gap of {stale_days} days detected. Applying rolling percentile stabilization.")
        for f in stabilize_targets:
            if f in df.columns:
                # Calculate the robust historical 95th percentile (excluding the current spike row)
                rolling_95 = df[f].iloc[:-1].rolling(252, min_periods=20).quantile(0.95)
                rolling_05 = df[f].iloc[:-1].rolling(252, min_periods=20).quantile(0.05)
                
                upper_bound = rolling_95.iloc[-1] if not pd.isna(rolling_95.iloc[-1]) else df[f].iloc[:-1].max()
                lower_bound = rolling_05.iloc[-1] if not pd.isna(rolling_05.iloc[-1]) else df[f].iloc[:-1].min()
                
                # Clip the recovery burst to historical norms to prevent artificial drift
                df.loc[df.index[-1], f] = np.clip(df.loc[df.index[-1], f], lower_bound, upper_bound)
                
    # Extract ONLY the latest row BEFORE applying scaler so we don't scale NaNs in early history
    latest_features = df.iloc[[-1]].copy()
    
    # Store recovery mode status for operational manifest
    latest_features['recovery_mode'] = is_recovery_mode
    latest_features['ingestion_spike_ratio'] = (df['article_count'].iloc[-1] / (df['article_count'].iloc[:-1].rolling(20).mean().iloc[-1] + 1e-6)) if is_recovery_mode else 1.0
    
    # 3. ENFORCE SCHEMA FROM SCALER/MODEL
    scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.pkl not found. Run Module 4 engineer_features first.")
        
    scaler = joblib.load(scaler_path)
    
    # The scaler expects exactly the features it was fit on, in the same order.
    # scikit-learn StandardScaler stores feature names in `feature_names_in_` if fitted on pandas
    expected_features = list(scaler.feature_names_in_)
    
    # Verify all expected features exist
    missing = [f for f in expected_features if f not in latest_features.columns]
    if missing:
        raise ValueError(f"FATAL: Missing features required by scaler: {missing}")
        
    # Strictly select and order columns
    X_live = latest_features[expected_features].copy()
    
    # Check for NaNs
    if X_live.isna().any().any():
        logging.warning("NaNs detected in live feature vector before scaling. This indicates insufficient history for rolling windows.")
    
    # Apply scaling
    scaled_values = scaler.transform(X_live)
    
    # Reassign scaled values
    for i, col in enumerate(expected_features):
        latest_features[col] = scaled_values[0, i]
        
    # Prepare final output format (date + scaled expected features)
    final_cols = ['date'] + expected_features
    # Also carry operational metadata if present
    for extra_col in ['execution_uuid', 'recovery_mode', 'ingestion_spike_ratio']:
        if extra_col in latest_features.columns:
            final_cols.append(extra_col)
        
    latest_features = latest_features[final_cols]
    latest_features['date'] = latest_features['date'].dt.strftime('%Y-%m-%d')
    
    return latest_features

def main():
    logging.info("Starting Module 4: Feature Engineering")
    sent_df, market_df = load_datasets()
    if sent_df is None or market_df is None:
        return
        
    final_df = engineer_features(sent_df, market_df)
    
    out_path = os.path.join(os.getcwd(), 'data', 'final_dataset.csv')
    final_df.to_csv(out_path, index=False)
    logging.info(f"Saved {len(final_df)} final predictive records to {out_path}")

if __name__ == "__main__":
    main()
