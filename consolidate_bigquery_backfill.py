"""Consolidate daily BigQuery GDELT headline partitions.

The script builds a resumable historical daily-news candidate dataset from
BigQuery partitions. Exact syndicated headline repeats are removed within each
Shanghai calendar day before aggregation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

NON_EMPIRICAL_PATTERNS = (
    "China PBOC announces new liquidity measures to stabilize markets on",
    "China PBOC economy stock market financial markets regulation",
)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def normalize_title(value: str) -> str:
    return " ".join(str(value).casefold().split())


def build_headline_hash(titles: list[str]) -> str:
    payload = " || ".join(normalize_title(value) for value in titles)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument(
        "--input-dir",
        default="data/bigquery_backfill_2023_2025",
    )
    parser.add_argument(
        "--output",
        default="data/news_daily_bigquery_2023_2025.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/bigquery_backfill_2023_2025_summary.json",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    input_dir = Path(args.input_dir)
    daily_dir = input_dir / "daily"
    article_files = sorted(daily_dir.glob("articles_*.csv.gz"))
    if not article_files:
        raise FileNotFoundError(
            f"No BigQuery daily article files found under {daily_dir}"
        )

    frames = []
    for path in article_files:
        frame = pd.read_csv(path)
        required = {
            "date",
            "datetime_utc",
            "title",
            "url",
            "domain",
            "language",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} missing required columns: {sorted(missing)}"
            )
        frames.append(frame)

    articles = pd.concat(frames, ignore_index=True)
    rows_before_filter = int(len(articles))

    articles["date"] = articles["date"].astype(str)
    articles["datetime_utc"] = pd.to_datetime(
        articles["datetime_utc"],
        utc=True,
        errors="coerce",
    )
    articles["title"] = (
        articles["title"].fillna("").astype(str).str.strip()
    )
    articles["url"] = (
        articles["url"].fillna("").astype(str).str.strip()
    )
    articles["domain"] = (
        articles["domain"].fillna("").astype(str).str.strip()
    )

    articles = articles.dropna(subset=["datetime_utc"])
    articles = articles[
        (articles["date"] >= args.start.isoformat())
        & (articles["date"] <= args.end.isoformat())
        & (articles["title"].str.len() > 15)
    ].copy()

    seed_mask = articles["title"].apply(
        lambda value: any(
            pattern in value for pattern in NON_EMPIRICAL_PATTERNS
        )
    )
    if seed_mask.any():
        raise ValueError(
            "Known non-empirical development text found in historical data."
        )

    articles["title_key"] = articles["title"].map(normalize_title)
    articles = (
        articles.sort_values(
            ["date", "datetime_utc", "title"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["date", "title_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    rows = []
    for date_value, group in articles.groupby("date", sort=True):
        titles = group["title"].tolist()
        rows.append(
            {
                "date": date_value,
                "article_count": int(len(titles)),
                "headline_hash": build_headline_hash(titles),
                "headlines_json": json.dumps(
                    titles,
                    ensure_ascii=False,
                ),
                "raw_text": " || ".join(titles),
            }
        )
    daily = pd.DataFrame(
        rows,
        columns=[
            "date",
            "article_count",
            "headline_hash",
            "headlines_json",
            "raw_text",
        ],
    )

    requested_dates = pd.date_range(
        args.start,
        args.end,
        freq="D",
    )
    observed = set(pd.to_datetime(daily["date"])) if len(daily) else set()
    missing_dates = [
        value.strftime("%Y-%m-%d")
        for value in requested_dates
        if value not in observed
    ]

    manifest_path = input_dir / "query_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else pd.DataFrame()
    )

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)

    summary = {
        "requested_start": args.start.isoformat(),
        "requested_end": args.end.isoformat(),
        "daily_partition_files_found": int(len(article_files)),
        "article_rows_loaded_before_date_filter": rows_before_filter,
        "unique_retained_headlines": int(len(articles)),
        "calendar_days_requested": int(len(requested_dates)),
        "calendar_days_with_news": int(len(daily)),
        "calendar_days_without_news": int(len(missing_dates)),
        "missing_dates": missing_dates,
        "headline_count_min": (
            int(daily["article_count"].min()) if len(daily) else 0
        ),
        "headline_count_median": (
            float(daily["article_count"].median()) if len(daily) else 0.0
        ),
        "headline_count_mean": (
            float(daily["article_count"].mean()) if len(daily) else 0.0
        ),
        "headline_count_max": (
            int(daily["article_count"].max()) if len(daily) else 0
        ),
        "unique_domains": int(articles["domain"].nunique()),
        "duplicate_title_date_pairs_remaining": int(
            articles.duplicated(["date", "title_key"]).sum()
        ),
        "bigquery_rows_returned": (
            int(manifest["rows_returned"].sum())
            if "rows_returned" in manifest.columns
            else None
        ),
        "bigquery_rows_retained_before_consolidation": (
            int(manifest["rows_retained"].sum())
            if "rows_retained" in manifest.columns
            else None
        ),
        "bigquery_bytes_processed": (
            int(manifest["total_bytes_processed"].sum())
            if "total_bytes_processed" in manifest.columns
            else None
        ),
        "bigquery_bytes_billed": (
            int(manifest["total_bytes_billed"].sum())
            if "total_bytes_billed" in manifest.columns
            else None
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Candidate daily dataset: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
