import pandas as pd
import logging
import os
import torch
from transformers import pipeline
import warnings

# Suppress HuggingFace warnings for cleaner execution
warnings.filterwarnings('ignore')

# ==========================================
# MODULE 2: LIVE NLP SENTIMENT (FinBERT)
# ==========================================
# Objective: Convert new financial text into quantitative sentiment signals.
# Upgraded for Live Inference: Efficiently checks existing features and 
# only processes days where new articles have been appended.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data() -> tuple:
    news_path = os.path.join(os.getcwd(), 'data', 'news_daily.csv')
    sent_path = os.path.join(os.getcwd(), 'data', 'sentiment_features.csv')
    
    if not os.path.exists(news_path):
        logging.error(f"Missing input file: {news_path}")
        return None, None
        
    news_df = pd.read_csv(news_path)
    
    # Load existing sentiment features if available
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)
    else:
        sent_df = pd.DataFrame(columns=['date', 'sentiment_mean', 'sentiment_std', 'article_count'])
        
    return news_df, sent_df

def identify_unprocessed_days(news_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """Isolates specific days that are entirely missing or have new appended articles."""
    if sent_df.empty:
        return news_df
        
    # Merge current news with existing sentiment records
    merged = pd.merge(news_df, sent_df[['date', 'article_count']], on='date', how='left', suffixes=('', '_old'))
    
    # Filter for completely new days (NaN) OR days where the article count has increased
    needs_processing = merged[(merged['article_count_old'].isna()) | (merged['article_count'] != merged['article_count_old'])]
    
    if needs_processing.empty:
        logging.info("No new articles detected. Sentiment database is fully up to date.")
        return pd.DataFrame()
        
    logging.info(f"Identified {len(needs_processing)} days requiring sentiment inference update.")
    return needs_processing[['date', 'article_count', 'raw_text']]

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Runs batched FinBERT inference on newly ingested text blocks."""
    if df.empty:
        return pd.DataFrame()
        
    logging.info("Initializing FinBERT pipeline...")
    device = 0 if torch.cuda.is_available() else -1
    
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert", 
        device=device, 
        top_k=None
    )
    
    # Unpack daily texts into individual articles for inference
    expanded_rows = []
    for _, row in df.iterrows():
        date = row['date']
        articles = str(row['raw_text']).split(' || ')
        for article in articles:
            if article.strip() and len(article.strip()) > 15:
                expanded_rows.append({'date': date, 'text': article.strip()})
                
    article_df = pd.DataFrame(expanded_rows)
    if article_df.empty:
        return article_df
        
    texts = article_df['text'].tolist()
    dates = article_df['date'].tolist()
    
    logging.info(f"Running batched inference on {len(texts)} new articles...")
    pipeline_outputs = sentiment_model(texts, batch_size=32, truncation=True, max_length=512)
    
    results = []
    for date, preds in zip(dates, pipeline_outputs):
        probs = {pred['label']: pred['score'] for pred in preds}
        
        prob_positive = probs.get('positive', 0.0)
        prob_negative = probs.get('negative', 0.0)
        
        # Calculate quantitative signal
        sentiment_score = prob_positive - prob_negative
        
        results.append({
            'date': date,
            'sentiment_score': sentiment_score
        })
        
    if not results:
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    
    # Re-aggregate to daily metrics
    daily_df = results_df.groupby('date').agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        article_count=('sentiment_score', 'count')
    ).reset_index()
    
    daily_df['sentiment_std'] = daily_df['sentiment_std'].fillna(0.0)
    return daily_df

def update_sentiment_database(sent_df: pd.DataFrame, new_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Replaces old records and appends new daily records while maintaining chronological order."""
    if new_daily_df.empty:
        return sent_df
        
    if not sent_df.empty:
        # Drop rows for the dates we just re-calculated
        sent_df = sent_df[~sent_df['date'].isin(new_daily_df['date'])]
        
    combined = pd.concat([sent_df, new_daily_df], ignore_index=True)
    return combined.sort_values('date').reset_index(drop=True)

def main():
    logging.info("Starting Module 2: Live Sentiment Inference")
    news_df, sent_df = load_data()
    if news_df is None: return
    
    # Efficient delta processing
    new_data_df = identify_unprocessed_days(news_df, sent_df)
    
    if not new_data_df.empty:
        new_features = analyze_sentiment(new_data_df)
        final_df = update_sentiment_database(sent_df, new_features)
        
        out_path = os.path.join(os.getcwd(), 'data', 'sentiment_features.csv')
        final_df.to_csv(out_path, index=False)
        logging.info(f"Successfully updated sentiment features at {out_path}")

if __name__ == "__main__":
    main()
