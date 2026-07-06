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
    
    # Ensure datetime alignment
    sent_df['date'] = pd.to_datetime(sent_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    return sent_df, market_df

def engineer_features(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    # Inner join strictly by date to perfectly align market and sentiment
    df = pd.merge(market_df, sent_df, on='date', how='inner')
    
    # CRITICAL: Re-sort to guarantee time integrity before applying lags
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logging.info("Engineering Base Features & Lags...")
    
    # ---------------------------------------------------------
    # 1. Base Sentiment Features
    # ---------------------------------------------------------
    df['sentiment_t'] = df['sentiment_mean']
    df['sentiment_t-1'] = df['sentiment_t'].shift(1)
    df['sentiment_t-2'] = df['sentiment_t'].shift(2)
    df['sentiment_roll_3'] = df['sentiment_t'].rolling(window=3).mean()
    df['sentiment_roll_7'] = df['sentiment_t'].rolling(window=7).mean()
    df['sentiment_volatility_7d'] = df['sentiment_t'].rolling(window=7).std()
    
    # Advanced Momentum & Acceleration Features
    df['sentiment_momentum'] = df['sentiment_t'] - df['sentiment_t-1']
    df['sentiment_acceleration'] = df['sentiment_t'] - 2 * df['sentiment_t-1'] + df['sentiment_t-2']
    
    # Extreme Sentiment Indicator (1 if abs(sentiment) > rolling 20-day std)
    rolling_sentiment_std = df['sentiment_t'].rolling(window=20).std()
    df['extreme_sentiment'] = np.where(df['sentiment_t'].abs() > rolling_sentiment_std, 1, 0)
    
    # News Volume Shock
    # Using safe division handling if rolling mean is 0
    rolling_article_mean = df['article_count'].rolling(window=7).mean()
    df['volume_shock'] = np.where(rolling_article_mean > 0, df['article_count'] / rolling_article_mean, 1.0)
    
    # ---------------------------------------------------------
    # 2. Base Market Features
    # ---------------------------------------------------------
    df['return_t-1'] = df['return'].shift(1)
    df['return_t-2'] = df['return'].shift(2)
    # df['volatility'] natively holds the 20-day market volatility
    
    # ---------------------------------------------------------
    # 3. Interaction Terms (Critical Signals)
    # ---------------------------------------------------------
    logging.info("Generating Interaction Terms...")
    # Does sentiment magnitude matter more in highly volatile markets?
    df['sentiment_x_volatility'] = df['sentiment_t'] * df['volatility']
    
    # Does sentiment act as momentum or mean-reversion following a specific return?
    df['sentiment_x_lag_return'] = df['sentiment_t'] * df['return_t-1']
    
    # ---------------------------------------------------------
    # 4. Target Variables (t+1) - NO LEAKAGE
    # ---------------------------------------------------------
    # Shift(-1) maps tomorrow's return to today's row index.
    df['target_return_t+1'] = df['return'].shift(-1)
    df['target_volatility_t+1'] = df['volatility'].shift(-1)
    
    # ---------------------------------------------------------
    # 5. Strict Cleaning
    # ---------------------------------------------------------
    initial_len = len(df)
    # Drop rows where *features* are NaN, but allow target variables to be NaN for the latest day (production inference)
    feature_cols = [col for col in df.columns if col not in ['target_return_t+1', 'target_volatility_t+1']]
    df.dropna(subset=feature_cols, inplace=True) # Drops NA induced by shifts and rolls
    logging.info(f"Dropped {initial_len - len(df)} NaN rows to preserve integrity.")
    
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

def generate_latest_features(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Optimized function specifically for real-time inference. Extracts ONLY the latest row."""
    # 1. MERGE
    df = pd.merge(market_df, sent_df, on='date', how='inner')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2. CREATE FEATURES (Identical logic to training to prevent feature drift)
    df['sentiment_t'] = df['sentiment_mean']
    df['sentiment_t-1'] = df['sentiment_t'].shift(1)
    df['sentiment_t-2'] = df['sentiment_t'].shift(2)
    df['sentiment_roll_3'] = df['sentiment_t'].rolling(window=3).mean()
    df['sentiment_roll_7'] = df['sentiment_t'].rolling(window=7).mean()
    df['sentiment_volatility_7d'] = df['sentiment_t'].rolling(window=7).std()
    
    df['sentiment_momentum'] = df['sentiment_t'] - df['sentiment_t-1']
    df['sentiment_acceleration'] = df['sentiment_t'] - 2 * df['sentiment_t-1'] + df['sentiment_t-2']
    rolling_sentiment_std = df['sentiment_t'].rolling(window=20).std()
    df['extreme_sentiment'] = np.where(df['sentiment_t'].abs() > rolling_sentiment_std, 1, 0)
    rolling_article_mean = df['article_count'].rolling(window=7).mean()
    df['volume_shock'] = np.where(rolling_article_mean > 0, df['article_count'] / rolling_article_mean, 1.0)
    
    df['return_t-1'] = df['return'].shift(1)
    df['return_t-2'] = df['return'].shift(2)
    
    # Interaction terms
    df['sentiment_x_volatility'] = df['sentiment_t'] * df['volatility']
    df['sentiment_x_lag_return'] = df['sentiment_t'] * df['return_t-1']
    
    # Clean only the features (target not needed for inference)
    feature_cols = [col for col in df.columns if col not in ['target_return_t+1', 'target_volatility_t+1', 'date']]
    df.dropna(subset=feature_cols, inplace=True)
    
    # 3. EXTRACT LATEST ROW
    latest_features = df.iloc[[-1]].copy()
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
