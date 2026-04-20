import pandas as pd
import requests
import datetime
import time
import logging
import os

# ==========================================
# MODULE 1: NEWS DATA EXTRACTION (GDELT)
# ==========================================
# Objective: Extract, filter, and process China-related financial news.
# Timezone converted strictly to China Standard Time (UTC+8).
# Supports both historical rebuilding and near real-time appending.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_gdelt_api(start_date: datetime.date, end_date: datetime.date, keywords: list) -> pd.DataFrame:
    """Extracts articles using GDELT v2 Doc API by iterating through days."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
    query = f"({keyword_query}) sourcelang:english"
    
    all_articles = []
    current_date = start_date
    
    while current_date <= end_date:
        logging.info(f"Fetching GDELT for {current_date}...")
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 250,
            "startdatetime": current_date.strftime("%Y%m%d000000"),
            "enddatetime": current_date.strftime("%Y%m%d235959")
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "articles" in data and data["articles"]:
                    all_articles.extend(data["articles"])
            elif response.status_code == 429:
                logging.warning(f"Rate limited on {current_date}. Waiting...")
                time.sleep(5)
                continue
        except Exception as e:
            logging.error(f"Error fetching {current_date}: {e}")
            
        current_date += datetime.timedelta(days=1)
        time.sleep(1.5) # Rate limiting respect
        
    return pd.DataFrame(all_articles)

def get_latest_news(keywords: list) -> pd.DataFrame:
    """Fetches near real-time news for the last 24 hours via GDELT API."""
    logging.info("Executing real-time fetch (Last 24 hours)...")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
    query = f"({keyword_query}) sourcelang:english"
    
    now = datetime.datetime.utcnow()
    yesterday = now - datetime.timedelta(days=1)
    
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": yesterday.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S")
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if "articles" in data and data["articles"]:
                return pd.DataFrame(data["articles"])
    except Exception as e:
        logging.error(f"Error fetching latest news: {e}")
        
    return pd.DataFrame()

def process_news(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans text, converts time to CST (UTC+8), and aggregates daily."""
    if df.empty:
        return df
        
    expected_cols = ['seendate', 'title', 'domain']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
            
    df = df[expected_cols].copy()
    df.rename(columns={'seendate': 'timestamp', 'title': 'text', 'domain': 'source'}, inplace=True)
    
    # Strict Timezone processing to align with Chinese Equity Markets
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df['datetime_cst'] = df['datetime_utc'] + pd.Timedelta(hours=8)
    df['date'] = df['datetime_cst'].dt.strftime('%Y-%m-%d')
    
    # Noise and duplicate reduction
    df.dropna(subset=['date', 'text'], inplace=True)
    df.drop_duplicates(subset=['date', 'text'], inplace=True)
    df = df[df['text'].str.strip().str.len() > 20]
    
    # Daily aggregation
    agg_df = df.groupby('date').agg(
        article_count=('text', 'count'),
        raw_text=('text', lambda x: ' || '.join(x.astype(str)))
    ).reset_index()
    
    return agg_df.sort_values('date')

def combine_and_deduplicate(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Safely appends new aggregated data to existing data, deduplicating texts at the article level."""
    if old_df is None or old_df.empty:
        return new_df
    if new_df.empty:
        return old_df
        
    combined = pd.concat([old_df, new_df])
    
    def merge_texts(series):
        # Unpack all articles for the specific day to deduplicate
        all_articles = []
        for text_block in series:
            articles = [a.strip() for a in str(text_block).split(' || ') if a.strip()]
            all_articles.extend(articles)
        
        unique_articles = list(set(all_articles))
        return ' || '.join(unique_articles), len(unique_articles)
        
    final_rows = []
    for date, group in combined.groupby('date'):
        merged_text, count = merge_texts(group['raw_text'])
        final_rows.append({
            'date': date,
            'article_count': count,
            'raw_text': merged_text
        })
        
    return pd.DataFrame(final_rows).sort_values('date')

def main():
    logging.info("Starting Module 1: News Extraction")
    keywords = ["China", "PBOC", "economy", "regulation", "financial markets", "stock market"]
    out_path = os.path.join(os.getcwd(), 'data', 'news_daily.csv')
    
    # Attempt to load existing data to append to
    if os.path.exists(out_path):
        old_df = pd.read_csv(out_path)
        logging.info(f"Loaded existing database with {len(old_df)} days of history.")
        # Near real-time fetch
        raw_new_df = get_latest_news(keywords)
    else:
        # Full historical rebuild if no file exists
        old_df = pd.DataFrame()
        logging.info("No existing database found. Running full 60-day historical fetch...")
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=60)
        raw_new_df = fetch_gdelt_api(start_date, end_date, keywords)
        
    if raw_new_df.empty:
        logging.warning("No new data extracted.")
        return
        
    processed_new_df = process_news(raw_new_df)
    final_df = combine_and_deduplicate(old_df, processed_new_df)
    
    final_df.to_csv(out_path, index=False)
    logging.info(f"Database successfully updated. Total active days tracked: {len(final_df)}")

if __name__ == "__main__":
    main()
