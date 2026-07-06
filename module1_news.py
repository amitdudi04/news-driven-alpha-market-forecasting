import pandas as pd
import requests
import datetime
import time
import logging
import os
import json
import hashlib
import uuid
import numpy as np

# ==========================================
# MODULE 1: INSTITUTIONAL NEWS DATA EXTRACTION (GDELT)
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_hash(data: str) -> str:
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def log_api_diagnostic(query: str, status_code: int, response_bytes: int, article_count: int, url: str):
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'gdelt_response_audit.jsonl')
    
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "url_hash": get_hash(url),
        "query_hash": get_hash(query),
        "status_code": status_code,
        "response_bytes": response_bytes,
        "extracted_article_count": article_count
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def fetch_gdelt_api(start_date: datetime.date, end_date: datetime.date, query: str) -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
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
                articles = data.get("articles", [])
                log_api_diagnostic(query, 200, len(response.content), len(articles), url)
                if articles:
                    all_articles.extend(articles)
            elif response.status_code == 429:
                logging.warning(f"Rate limited on {current_date}. Waiting...")
                time.sleep(5)
                continue
            else:
                log_api_diagnostic(query, response.status_code, len(response.content), 0, url)
        except Exception as e:
            logging.error(f"Error fetching {current_date}: {e}")
            
        current_date += datetime.timedelta(days=1)
        time.sleep(1.5)
        
    return pd.DataFrame(all_articles)

def get_latest_news(query: str) -> pd.DataFrame:
    logging.info("Executing real-time fetch (Last 24 hours)...")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    now = datetime.datetime.now(datetime.timezone.utc)
    yesterday = now - datetime.timedelta(days=3)
    
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": yesterday.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S")
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logging.warning("Rate limited on real-time fetch. Waiting 5s...")
            time.sleep(5)
            continue
        elif response.status_code != 200:
            log_api_diagnostic(query, response.status_code, len(response.content), 0, url)
            raise RuntimeError(f"SAFE MODE ESCALATION: API Error {response.status_code}")
        break
    else:
        log_api_diagnostic(query, 429, 0, 0, url)
        raise RuntimeError("SAFE MODE ESCALATION: Persistent API 429 Rate Limit.")
        
    try:
        data = response.json()
    except json.JSONDecodeError:
        log_api_diagnostic(query, response.status_code, len(response.content), 0, url)
        raise RuntimeError("SAFE MODE ESCALATION: Malformed JSON from GDELT API")
        
    articles = data.get("articles", [])
    log_api_diagnostic(query, response.status_code, len(response.content), len(articles), url)
    
    if not articles:
        raise RuntimeError("SAFE MODE ESCALATION: Zero articles extracted. Silent empty returns are strictly prohibited.")
        
    df = pd.DataFrame(articles)
    
    # Schema Drift Detection
    expected_schema = ['seendate', 'title', 'domain']
    for col in expected_schema:
        if col not in df.columns:
            raise RuntimeError(f"SAFE MODE ESCALATION: Schema Drift Detected. Missing column: {col}")
            
    # Timestamp Format Validation
    invalid_dates = df[~df['seendate'].str.match(r'^\d{8}T\d{6}Z$')]
    if not invalid_dates.empty:
        raise RuntimeError("SAFE MODE ESCALATION: Malformed timestamp formats detected in API payload.")
        
    return df

def process_news(df: pd.DataFrame, query: str, execution_uuid: str = None) -> pd.DataFrame:
    if df.empty:
        return df
        
    initial_count = len(df)
    
    df.rename(columns={'seendate': 'timestamp', 'title': 'text', 'domain': 'source'}, inplace=True)
    
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M%SZ', errors='coerce', utc=True)
    df['datetime_cst'] = df['datetime_utc'] + pd.Timedelta(hours=8)
    df['date'] = df['datetime_cst'].dt.strftime('%Y-%m-%d')
    
    # FUTURE DATA FILTER
    current_cst = pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=8)
    df = df[df['datetime_cst'] <= current_cst]
    
    # NLP CLEANING
    df.dropna(subset=['date', 'text'], inplace=True)
    
    # Calculate duplicate ratio
    unique_count = len(df.drop_duplicates(subset=['date', 'text']))
    duplicate_ratio = 1.0 - (unique_count / len(df)) if len(df) > 0 else 0
    df.drop_duplicates(subset=['date', 'text'], inplace=True)
    
    # HTML & Boilerplate Rejection
    df = df[~df['text'].str.contains(r'<[^>]+>', regex=True, na=False)]
    boilerplate_terms = ['subscribe', 'newsletter', 'click here', 'error 404', 'not found', 'access denied']
    df = df[~df['text'].str.lower().str.contains('|'.join(boilerplate_terms))]
    
    after_boilerplate_count = len(df)
    boilerplate_rejection_rate = 1.0 - (after_boilerplate_count / unique_count) if unique_count > 0 else 0
    
    df = df[df['text'].str.strip().str.len() > 30]
    df = df[df['text'].str.count(r'[!?\-]') < 5]
    
    avg_length = df['text'].str.len().mean()
    
    # Source Concentration Governance
    source_counts = df['source'].value_counts(normalize=True)
    if not source_counts.empty and source_counts.iloc[0] > 0.7:
        logging.warning(f"NEWS_SOURCE_CONCENTRATION_ALERT: Domain {source_counts.index[0]} controls {source_counts.iloc[0]:.1%} of flow.")
        
    if df.empty:
        raise RuntimeError("SAFE MODE ESCALATION: Post-processing resulted in zero usable articles. Persistent noise environment.")
        
    # Stale News Surveillance
    if len(df['timestamp'].unique()) == 1 and len(df) > 10:
        raise RuntimeError("SAFE MODE ESCALATION: Frozen timestamps detected. Suspected stale feed recycling.")
        
    # Daily aggregation
    agg_df = df.groupby('date').agg(
        article_count=('text', 'count'),
        raw_text=('text', lambda x: ' || '.join(x.astype(str)))
    ).reset_index()
    
    # Generate Extraction Manifest
    latest_timestamp = df['datetime_utc'].max()
    stale_days = (pd.Timestamp.now(tz='UTC') - latest_timestamp).days
    
    manifest_entry = {
        "execution_uuid": execution_uuid or str(uuid.uuid4()),
        "extraction_timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
        "article_count": len(df),
        "latest_article_timestamp": latest_timestamp.isoformat() if pd.notna(latest_timestamp) else None,
        "stale_days": stale_days,
        "query_hash": get_hash(query),
        "gdelt_response_hash": get_hash(df.to_json())
    }
    
    manifest_path = os.path.join(os.getcwd(), 'outputs', 'news_ingestion_manifest.csv')
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    pd.DataFrame([manifest_entry]).to_csv(manifest_path, mode='a', header=not os.path.exists(manifest_path), index=False)
    
    # Generate Quality Manifest
    source_probs = source_counts.values
    entropy = -np.sum(source_probs * np.log2(source_probs)) if len(source_probs) > 0 else 0
    
    quality_entry = {
        "extraction_timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
        "duplicate_ratio": duplicate_ratio,
        "boilerplate_rejection_rate": boilerplate_rejection_rate,
        "average_article_length": avg_length,
        "max_source_concentration": source_counts.iloc[0] if not source_counts.empty else 0,
        "daily_article_entropy": entropy
    }
    
    quality_path = os.path.join(os.getcwd(), 'outputs', 'news_quality_manifest.csv')
    pd.DataFrame([quality_entry]).to_csv(quality_path, mode='a', header=not os.path.exists(quality_path), index=False)
    
    return agg_df.sort_values('date')

def combine_and_deduplicate(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df
    if new_df.empty:
        return old_df
        
    combined = pd.concat([old_df, new_df])
    
    def merge_texts(series):
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

def main(execution_uuid: str = None):
    logging.info("Starting Module 1: Institutional News Extraction")
    
    query = '(China OR PBOC OR Beijing) (economy OR "stock market" OR "financial markets" OR regulation) sourcelang:english'
    out_path = os.path.join(os.getcwd(), 'data', 'news_daily.csv')
    
    if os.path.exists(out_path):
        old_df = pd.read_csv(out_path)
        logging.info(f"Loaded existing database with {len(old_df)} days of history.")
        raw_new_df = get_latest_news(query)
    else:
        old_df = pd.DataFrame()
        logging.info("No existing database found. Running historical fetch...")
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=730)
        raw_new_df = fetch_gdelt_api(start_date, end_date, query)
        
    if raw_new_df.empty:
        raise RuntimeError("SAFE MODE ESCALATION: Silent empty returns are strictly prohibited.")
        
    processed_new_df = process_news(raw_new_df, query, execution_uuid)
    final_df = combine_and_deduplicate(old_df, processed_new_df)
    
    final_df.to_csv(out_path, index=False)
    logging.info(f"Database successfully updated. Total active days tracked: {len(final_df)}")

if __name__ == "__main__":
    main()
