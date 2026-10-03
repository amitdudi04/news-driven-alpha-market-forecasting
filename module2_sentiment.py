import hashlib
import logging
import os

import pandas as pd
import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FINBERT_MODEL = "ProsusAI/finbert"


def _normalize_headline_text(raw_text: str) -> str:
    headlines = [
        " ".join(item.casefold().split())
        for item in str(raw_text).split(" || ")
        if item.strip()
    ]
    return " || ".join(headlines)


def headline_hash(raw_text: str) -> str:
    normalized = _normalize_headline_text(raw_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prepare_news_metadata(news_df: pd.DataFrame) -> pd.DataFrame:
    news = news_df.copy()
    if "news_volume_raw" not in news.columns:
        # Backward-compatible fallback for the existing short sample.
        news["news_volume_raw"] = news["article_count"]
    if "news_volume_norm" not in news.columns:
        # A denominator of one preserves the legacy raw-count intensity when
        # historical GDELT normalization metadata are unavailable.
        news["news_volume_norm"] = 1.0
    if "headline_hash" not in news.columns:
        news["headline_hash"] = news["raw_text"].map(headline_hash)

    news["article_count"] = pd.to_numeric(
        news["article_count"],
        errors="raise",
    ).astype(int)
    news["news_volume_raw"] = pd.to_numeric(
        news["news_volume_raw"],
        errors="coerce",
    )
    news["news_volume_norm"] = pd.to_numeric(
        news["news_volume_norm"],
        errors="coerce",
    )
    return news


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = os.path.join(os.getcwd(), "data", "news_daily.csv")
    sent_path = os.path.join(
        os.getcwd(),
        "data",
        "sentiment_features.csv",
    )
    if not os.path.exists(news_path):
        raise FileNotFoundError(
            "Missing data/news_daily.csv. Run the news-data module first."
        )

    news_df = _prepare_news_metadata(pd.read_csv(news_path))
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)
    else:
        sent_df = pd.DataFrame(
            columns=[
                "date",
                "sentiment_mean",
                "sentiment_std",
                "article_count",
                "news_volume_raw",
                "news_volume_norm",
                "headline_hash",
            ]
        )
    return news_df, sent_df


def identify_unprocessed_days(
    news_df: pd.DataFrame,
    sent_df: pd.DataFrame,
) -> pd.DataFrame:
    news = _prepare_news_metadata(news_df)
    if sent_df.empty:
        return news.copy()

    sent_meta = sent_df.copy()
    if "headline_hash" not in sent_meta.columns:
        sent_meta["headline_hash"] = ""
    sent_meta = sent_meta[
        ["date", "article_count", "headline_hash"]
    ].rename(
        columns={
            "article_count": "article_count_sentiment",
            "headline_hash": "headline_hash_sentiment",
        }
    )

    merged = news.merge(sent_meta, on="date", how="left")
    needs_update = merged[
        merged["article_count_sentiment"].isna()
        | (
            merged["article_count"]
            != merged["article_count_sentiment"]
        )
        | (
            merged["headline_hash"]
            != merged["headline_hash_sentiment"].fillna("")
        )
    ]
    return needs_update[
        [
            "date",
            "article_count",
            "news_volume_raw",
            "news_volume_norm",
            "headline_hash",
            "raw_text",
        ]
    ]


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sentiment_mean",
                "sentiment_std",
                "article_count",
            ]
        )

    rows = []
    for _, row in df.iterrows():
        for headline in str(row["raw_text"]).split(" || "):
            headline = headline.strip()
            if len(headline) > 15:
                rows.append(
                    {
                        "date": row["date"],
                        "text": headline,
                    }
                )

    article_df = pd.DataFrame(rows)
    if article_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sentiment_mean",
                "sentiment_std",
                "article_count",
            ]
        )

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    default_batch = 32 if use_cuda else 16
    batch_size = int(
        os.environ.get("FINBERT_BATCH_SIZE", str(default_batch))
    )
    logging.info(
        "Scoring %s headlines with %s (device=%s, batch=%s)",
        len(article_df),
        FINBERT_MODEL,
        "cuda" if use_cuda else "cpu",
        batch_size,
    )

    model = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        device=device,
        top_k=None,
    )
    outputs = model(
        article_df["text"].tolist(),
        batch_size=batch_size,
        truncation=True,
        max_length=128,
    )

    scores = []
    for date, predictions in zip(article_df["date"], outputs):
        probs = {
            item["label"].lower(): item["score"]
            for item in predictions
        }
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
        existing = existing[
            ~existing["date"].isin(updates["date"])
        ]
    return (
        pd.concat([existing, updates], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def attach_news_metadata(
    sentiment_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    news = _prepare_news_metadata(news_df)
    metadata = news[
        [
            "date",
            "news_volume_raw",
            "news_volume_norm",
            "headline_hash",
        ]
    ].drop_duplicates(subset=["date"])

    sent = sentiment_df.drop(
        columns=[
            "news_volume_raw",
            "news_volume_norm",
            "headline_hash",
        ],
        errors="ignore",
    )
    sent = sent.merge(metadata, on="date", how="left")
    return sent[
        [
            "date",
            "sentiment_mean",
            "sentiment_std",
            "article_count",
            "news_volume_raw",
            "news_volume_norm",
            "headline_hash",
        ]
    ].sort_values("date").reset_index(drop=True)


def main(execution_uuid: str | None = None):
    del execution_uuid
    news_df, sent_df = load_data()
    unprocessed_days = identify_unprocessed_days(news_df, sent_df)

    updates = analyze_sentiment(unprocessed_days)
    final_df = update_sentiment_database(sent_df, updates)
    final_df = attach_news_metadata(final_df, news_df)

    out_path = os.path.join(
        os.getcwd(),
        "data",
        "sentiment_features.csv",
    )
    final_df.to_csv(out_path, index=False)

    if unprocessed_days.empty:
        logging.info("Sentiment features are already up to date")
    else:
        logging.info(
            "Updated sentiment features for %s days",
            len(updates),
        )


if __name__ == "__main__":
    main()
