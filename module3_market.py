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

import time

def download_csi300(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Downloads historical data via Yahoo Finance with exponential backoff."""
    logging.info(f"Downloading {ticker} market data from {start} to {end}...")
    max_retries = 3
    base_wait = 2
    
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                logging.error(f"No data returned for ticker {ticker}.")
                return df
                
            df = df.reset_index()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            col_map = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            df.rename(columns=col_map, inplace=True)
            
            return df
        except Exception as e:
            wait_time = base_wait * (2 ** attempt)
            logging.warning(f"Error fetching yfinance data (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Network failure.")
                return pd.DataFrame()
            time.sleep(wait_time)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes log returns and rolling volatility mathematically."""
    if df.empty:
        return df
        
    logging.info("Computing Market Features...")
    
    df = df.sort_values('date').reset_index(drop=True)
    df.dropna(subset=['close'], inplace=True)
    
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(window=20).std()
    
    df.dropna(subset=['return', 'volatility'], inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    return df[['date', 'close', 'return', 'volatility']]

def main(execution_uuid: str = None):
    logging.info("Starting Module 3: Market Data")
    ticker = "000300.SS" # CSI 300
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    raw_df = download_csi300(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if raw_df.empty:
        return
        
    final_df = compute_features(raw_df)
    
    if execution_uuid:
        final_df['execution_uuid'] = execution_uuid
        
    out_path = os.path.join(os.getcwd(), 'data', 'csi300_features.csv')
    
    import module0_atomic_storage
    module0_atomic_storage.atomic_write_csv(final_df, out_path, index=False)
    
    logging.info(f"Successfully generated market features and saved atomically to {out_path}")

if __name__ == "__main__":
    main()
