"""Consolidate and validate historical FinBERT daily outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--news",
        default="data/news_daily_historical.csv",
    )
    parser.add_argument(
        "--input-dir",
        default="data/finbert_scores",
    )
    parser.add_argument(
        "--output",
        default="data/sentiment_features_historical.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/finbert_scores_summary.json",
    )
    args = parser.parse_args()

    news = pd.read_csv(args.news)
    news["date"] = news["date"].astype(str)

    daily_dir = Path(args.input_dir) / "daily"
    files = sorted(daily_dir.glob("sentiment_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No sentiment daily files found under {daily_dir}"
        )

    sentiment = pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
    )
    sentiment["date"] = sentiment["date"].astype(str)
    sentiment = (
        sentiment.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    overlap = news.merge(
        sentiment,
        on="date",
        how="inner",
        suffixes=("_news", "_sentiment"),
    )
    if overlap.empty:
        raise ValueError("No overlapping news and sentiment dates.")

    count_mismatch = overlap[
        overlap["article_count_news"]
        != overlap["article_count_sentiment"]
    ]
    hash_mismatch = overlap[
        overlap["headline_hash_news"]
        != overlap["headline_hash_sentiment"]
    ]
    if len(count_mismatch):
        raise ValueError(
            "News/sentiment article-count mismatch on: "
            + ", ".join(count_mismatch["date"].tolist())
        )
    if len(hash_mismatch):
        raise ValueError(
            "News/sentiment headline-hash mismatch on: "
            + ", ".join(hash_mismatch["date"].tolist())
        )

    share_sum = (
        sentiment[
            ["positive_share", "negative_share", "neutral_share"]
        ]
        .sum(axis=1)
    )
    max_share_error = float((share_sum - 1.0).abs().max())
    if max_share_error > 1e-9:
        raise ValueError(
            f"FinBERT class shares do not sum to one; "
            f"max error={max_share_error}"
        )

    output_columns = [
        "date",
        "sentiment_mean",
        "sentiment_std",
        "article_count",
        "positive_share",
        "negative_share",
        "neutral_share",
        "headline_hash",
    ]
    output = sentiment[output_columns].copy()

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    summary = {
        "sentiment_days": int(len(output)),
        "news_days_available": int(len(news)),
        "overlapping_validated_days": int(len(overlap)),
        "total_headlines_scored": int(
            output["article_count"].sum()
        ),
        "article_count_mismatches": int(len(count_mismatch)),
        "headline_hash_mismatches": int(len(hash_mismatch)),
        "max_class_share_sum_error": max_share_error,
        "sentiment_mean_min": float(
            output["sentiment_mean"].min()
        ),
        "sentiment_mean_median": float(
            output["sentiment_mean"].median()
        ),
        "sentiment_mean_mean": float(
            output["sentiment_mean"].mean()
        ),
        "sentiment_mean_max": float(
            output["sentiment_mean"].max()
        ),
        "mean_positive_share": float(
            output["positive_share"].mean()
        ),
        "mean_negative_share": float(
            output["negative_share"].mean()
        ),
        "mean_neutral_share": float(
            output["neutral_share"].mean()
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Candidate sentiment dataset: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
