import logging
import os

import pandas as pd
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = os.path.join(os.getcwd(), "data", "news_daily.csv")
    sent_path = os.path.join(os.getcwd(), "data", "sentiment_features.csv")
    if not os.path.exists(news_path):
        raise FileNotFoundError("Missing data/news_daily.csv. Run module1_news.py first.")

    news_df = pd.read_csv(news_path)
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)
    else:
        sent_df = pd.DataFrame(
            columns=["date", "sentiment_mean", "sentiment_std", "article_count"]
        )
    return news_df, sent_df


def identify_unprocessed_days(
    news_df: pd.DataFrame,
    sent_df: pd.DataFrame,
) -> pd.DataFrame:
    if sent_df.empty:
        return news_df.copy()

    merged = news_df.merge(
        sent_df[["date", "article_count"]],
        on="date",
        how="left",
        suffixes=("", "_sentiment"),
    )
    needs_update = merged[
        merged["article_count_sentiment"].isna()
        | (merged["article_count"] != merged["article_count_sentiment"])
    ]
    return needs_update[["date", "article_count", "raw_text"]]


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["date", "sentiment_mean", "sentiment_std", "article_count"]
        )

    rows = []
    for _, row in df.iterrows():
        for article in str(row["raw_text"]).split(" || "):
            article = article.strip()
            if len(article) > 15:
                rows.append({"date": row["date"], "text": article})

    article_df = pd.DataFrame(rows)
    if article_df.empty:
        return pd.DataFrame(
            columns=["date", "sentiment_mean", "sentiment_std", "article_count"]
        )

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=device,
        top_k=None,
    )
    outputs = model(
        article_df["text"].tolist(),
        batch_size=32,
        truncation=True,
        max_length=512,
    )

    scores = []
    for date, predictions in zip(article_df["date"], outputs):
        probs = {item["label"].lower(): item["score"] for item in predictions}
        scores.append(
            {
                "date": date,
                "sentiment_score": probs.get("positive", 0.0)
                - probs.get("negative", 0.0),
            }
        )

    scored = pd.DataFrame(scores)
    daily = (
        scored.groupby("date")
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    return daily


def update_sentiment_database(
    existing: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    if updates.empty:
        return existing.sort_values("date").reset_index(drop=True)

    if not existing.empty:
        existing = existing[~existing["date"].isin(updates["date"])]
    return (
        pd.concat([existing, updates], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def main(execution_uuid: str | None = None):
    del execution_uuid
    news_df, sent_df = load_data()
    unprocessed_days = identify_unprocessed_days(news_df, sent_df)
    if unprocessed_days.empty:
        logging.info("Sentiment features are already up to date")
        return

    updates = analyze_sentiment(unprocessed_days)
    final_df = update_sentiment_database(sent_df, updates)
    out_path = os.path.join(os.getcwd(), "data", "sentiment_features.csv")
    final_df.to_csv(out_path, index=False)
    logging.info("Updated sentiment features for %s days", len(updates))


if __name__ == "__main__":
    main()
