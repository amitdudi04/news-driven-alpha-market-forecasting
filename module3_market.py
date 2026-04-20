import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import datetime

# ==========================================
# MODULE 3: MARKET DATA EXTRACTION (yfinance)
# ==========================================
# Objective: Download daily CSI 300 index data, ensure strict chronological order, 
# and compute econometric features (Log Returns & Rolling Volatility).

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_csi300(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Downloads historical data via Yahoo Finance."""
    logging.info(f"Downloading {ticker} market data from {start} to {end}...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            logging.error(f"No data returned for ticker {ticker}.")
            return df
            
        df = df.reset_index()
        
        # Ensure flat columns if yfinance returns a MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns, keeping the variable name
            df.columns = df.columns.get_level_values(0)
            
        # Standardize column naming
        col_map = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=col_map, inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Error fetching yfinance data: {e}")
        return pd.DataFrame()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes log returns and rolling volatility mathematically."""
    if df.empty:
        return df
        
    logging.info("Computing Market Features...")
    
    # CRITICAL: Strict chronological sort to prevent look-ahead bias
    df = df.sort_values('date').reset_index(drop=True)
    df.dropna(subset=['close'], inplace=True)
    
    # 1. Continuous Log Returns (Standard in academic finance)
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Rolling Volatility (20-day standard deviation)
    df['volatility'] = df['return'].rolling(window=20).std()
    
    # Trim the warm-up period
    df.dropna(subset=['return', 'volatility'], inplace=True)
    
    # Format dates
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    return df[['date', 'close', 'return', 'volatility']]

def main():
    logging.info("Starting Module 3: Market Data")
    ticker = "000300.SS" # CSI 300
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365) # Fetch 1 year to allow rolling computation
    
    raw_df = download_csi300(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if raw_df.empty:
        return
        
    final_df = compute_features(raw_df)
    
    out_path = os.path.join(os.getcwd(), 'data', 'csi300_features.csv')
    final_df.to_csv(out_path, index=False)
    logging.info(f"Successfully generated market features and saved to {out_path}")

if __name__ == "__main__":
    main()
