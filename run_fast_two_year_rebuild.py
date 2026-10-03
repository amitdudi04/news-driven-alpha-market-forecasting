"""Fast, single-process 2024-2025 rebuild with weekly checkpoints.

This continues the already validated 2023 historical dataset through the
remaining two calendar years while preserving weekly resumability.

Optimizations:
- one persistent BigQuery client;
- one persistent FinBERT model/tokenizer for the entire run;
- one BigQuery query per contiguous missing weekly segment;
- CUDA FinBERT inference with a larger batch on the RTX 3050;
- in-process cumulative news/sentiment consolidation;
- no repeated Python subprocess startup/model reload per week;
- weekly durable checkpoints.

The canonical public short-sample files are never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from google.cloud import bigquery
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bigquery_gdelt_backfill import append_manifest
from finbert_backfill import MODEL_NAME, score_day


BASE_START = dt.date(2023, 1, 1)
PERIOD_START = dt.date(2024, 1, 1)
PERIOD_END = dt.date(2025, 12, 31)
PROJECT_ID = "news-alpha-amit-2026"

WEEK_QUERY = r"""
SELECT DISTINCT
  g.date AS datetime_utc,
  g.url,
  g.domain,
  g.title,
  g.lang AS language
FROM `gdelt-bq.gdeltv2.gal` AS g
WHERE g.date >= TIMESTAMP(@start_date, 'Asia/Shanghai')
  AND g.date < TIMESTAMP(
    DATE_ADD(@end_date, INTERVAL 1 DAY),
    'Asia/Shanghai'
  )
  AND g.lang = 'en'
  AND g.title IS NOT NULL
  AND LENGTH(TRIM(g.title)) > 15
  AND REGEXP_CONTAINS(
    LOWER(g.title),
    r'(^|[^a-z])(china|chinese|pboc|beijing)([^a-z]|$)'
  )
  AND REGEXP_CONTAINS(
    LOWER(g.title),
    r'(^|[^a-z])(economy|economic|growth|gdp|inflation|deflation|trade|exports?|imports?|markets?|stocks?|shares|equity|equities|yuan|renminbi|interest rates?|rates|liquidity|property|real estate|banks?|banking|regulation|regulatory|stimulus|debt|investment|investors?|financial|finance)([^a-z]|$)'
  )
ORDER BY datetime_utc, url
"""

NON_EMPIRICAL_PATTERNS = (
    "China PBOC announces new liquidity measures to stabilize markets on",
    "China PBOC economy stock market financial markets regulation",
)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def normalize_title(value: str) -> str:
    return " ".join(str(value).casefold().split())


def headline_hash(titles: list[str]) -> str:
    payload = " || ".join(normalize_title(value) for value in titles)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def week_ranges(start: dt.date, end: dt.date):
    week_number = 1
    cursor = start
    while cursor <= end:
        week_end = min(cursor + dt.timedelta(days=6), end)
        yield week_number, cursor, week_end
        cursor = week_end + dt.timedelta(days=1)
        week_number += 1


def contiguous_ranges(dates: list[dt.date]):
    if not dates:
        return []
    dates = sorted(dates)
    ranges: list[tuple[dt.date, dt.date]] = []
    start = previous = dates[0]
    for current in dates[1:]:
        if current == previous + dt.timedelta(days=1):
            previous = current
            continue
        ranges.append((start, previous))
        start = previous = current
    ranges.append((start, previous))
    return ranges


def query_config(start_date: dt.date, end_date: dt.date):
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_date",
                "DATE",
                start_date,
            ),
            bigquery.ScalarQueryParameter(
                "end_date",
                "DATE",
                end_date,
            ),
        ],
        use_query_cache=True,
    )


def normalize_week_rows(rows) -> pd.DataFrame:
    """Normalize and deduplicate within each Shanghai calendar day."""
    records = [dict(row.items()) for row in rows]
    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "datetime_utc",
                "title",
                "url",
                "domain",
                "language",
            ]
        )

    frame = pd.DataFrame(records)
    frame["datetime_utc"] = pd.to_datetime(
        frame["datetime_utc"],
        utc=True,
        errors="coerce",
    )
    frame["title"] = (
        frame["title"].fillna("").astype(str).str.strip()
    )
    frame["url"] = (
        frame["url"].fillna("").astype(str).str.strip()
    )
    frame["domain"] = (
        frame["domain"].fillna("").astype(str).str.strip()
    )
    frame["language"] = (
        frame["language"].fillna("").astype(str).str.strip()
    )
    frame = frame.dropna(subset=["datetime_utc"])
    frame = frame[frame["title"].str.len() > 15].copy()

    shanghai = frame["datetime_utc"].dt.tz_convert(
        "Asia/Shanghai"
    )
    frame["date"] = shanghai.dt.strftime("%Y-%m-%d")

    url_key = frame["url"].str.casefold()
    fallback_key = (
        frame["datetime_utc"].astype(str)
        + "|"
        + frame["title"].str.casefold()
    )
    frame["dedup_key"] = url_key.where(
        url_key.str.len() > 0,
        fallback_key,
    )
    frame = frame.drop_duplicates(
        subset=["date", "dedup_key"],
        keep="first",
    ).copy()

    frame["title_key"] = (
        frame["title"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame = frame.drop_duplicates(
        subset=["date", "title_key"],
        keep="first",
    ).copy()

    return (
        frame[
            [
                "date",
                "datetime_utc",
                "title",
                "url",
                "domain",
                "language",
            ]
        ]
        .sort_values(["date", "datetime_utc", "title"])
        .reset_index(drop=True)
    )


def collect_week(
    client: bigquery.Client,
    week_start: dt.date,
    week_end: dt.date,
    output_dir: Path,
) -> None:
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "query_manifest.csv"

    dates = [
        value.date()
        for value in pd.date_range(week_start, week_end, freq="D")
    ]
    missing = [
        day
        for day in dates
        if not (
            daily_dir / f"articles_{day:%Y%m%d}.csv.gz"
        ).exists()
    ]

    if not missing:
        print(
            f"BigQuery: all daily checkpoints already exist for "
            f"{week_start} -> {week_end}",
            flush=True,
        )
        return

    for range_start, range_end in contiguous_ranges(missing):
        print(
            f"BigQuery range query: {range_start} -> {range_end}",
            flush=True,
        )
        job = client.query(
            WEEK_QUERY,
            job_config=query_config(range_start, range_end),
        )
        raw_rows = list(job.result())
        frame = normalize_week_rows(raw_rows)

        range_dates = [
            value.date()
            for value in pd.date_range(
                range_start,
                range_end,
                freq="D",
            )
        ]
        retained_written = 0
        for day in range_dates:
            day_text = day.isoformat()
            day_frame = frame[
                frame["date"] == day_text
            ].copy()
            retained_written += len(day_frame)
            path = daily_dir / f"articles_{day:%Y%m%d}.csv.gz"
            day_frame.to_csv(
                path,
                index=False,
                compression="gzip",
            )
            print(
                f"  {day}: retained={len(day_frame)}",
                flush=True,
            )

        append_manifest(
            manifest_path,
            {
                "run_date": (
                    range_start.isoformat()
                    if range_start == range_end
                    else (
                        f"{range_start.isoformat()}/"
                        f"{range_end.isoformat()}"
                    )
                ),
                "job_id": job.job_id,
                "rows_returned": int(len(raw_rows)),
                "rows_retained": int(retained_written),
                "syndicated_or_repeated_rows_removed": int(
                    len(raw_rows) - retained_written
                ),
                "total_bytes_processed": int(
                    job.total_bytes_processed or 0
                ),
                "total_bytes_billed": int(
                    job.total_bytes_billed or 0
                ),
                "output_file": (
                    f"daily/articles_{range_start:%Y%m%d}"
                    f"_through_{range_end:%Y%m%d}"
                ),
                "completed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            },
        )
        print(
            f"  BigQuery processed "
            f"{(job.total_bytes_processed or 0)/(1024**3):.3f} GiB",
            flush=True,
        )


def consolidate_news(
    input_dir: Path,
    start: dt.date,
    end: dt.date,
    output_path: Path,
    summary_path: Path,
) -> dict:
    daily_dir = input_dir / "daily"
    paths = []
    for day in pd.date_range(start, end, freq="D"):
        path = daily_dir / f"articles_{day:%Y%m%d}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing daily BigQuery checkpoint: {path}"
            )
        paths.append(path)

    frames = [pd.read_csv(path) for path in paths]
    articles = pd.concat(frames, ignore_index=True)

    if not articles.empty:
        articles["date"] = articles["date"].astype(str)
        articles["datetime_utc"] = pd.to_datetime(
            articles["datetime_utc"],
            utc=True,
            errors="coerce",
        )
        articles["title"] = (
            articles["title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        articles["domain"] = (
            articles["domain"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        articles = articles.dropna(
            subset=["datetime_utc"]
        ).copy()

        bad = articles["title"].apply(
            lambda value: any(
                pattern in value
                for pattern in NON_EMPIRICAL_PATTERNS
            )
        )
        if bad.any():
            raise ValueError(
                "Known non-empirical development text found."
            )

        articles["title_key"] = articles["title"].map(
            normalize_title
        )
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
    for date_value, group in articles.groupby(
        "date",
        sort=True,
    ):
        titles = group["title"].tolist()
        rows.append(
            {
                "date": date_value,
                "article_count": int(len(titles)),
                "headline_hash": headline_hash(titles),
                "headlines_json": json.dumps(
                    titles,
                    ensure_ascii=False,
                ),
                "raw_text": " || ".join(titles),
            }
        )

    news = pd.DataFrame(
        rows,
        columns=[
            "date",
            "article_count",
            "headline_hash",
            "headlines_json",
            "raw_text",
        ],
    )

    expected_dates = [
        value.strftime("%Y-%m-%d")
        for value in pd.date_range(start, end, freq="D")
    ]
    observed = set(news["date"].tolist())
    missing_dates = [
        value for value in expected_dates
        if value not in observed
    ]
    # Genuine zero-news days remain absent from the news dataset.
    # They are recorded in the summary and are never neutralized.
    manifest_path = input_dir / "query_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else pd.DataFrame()
    )

    summary = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "calendar_days_requested": len(expected_dates),
        "calendar_days_with_news": int(len(news)),
        "calendar_days_without_news": len(missing_dates),
        "missing_dates": missing_dates,
        "unique_retained_headlines": int(len(articles)),
        "unique_domains": int(
            articles["domain"].nunique()
            if len(articles)
            else 0
        ),
        "duplicate_title_date_pairs_remaining": int(
            articles.duplicated(
                ["date", "title_key"]
            ).sum()
            if len(articles)
            else 0
        ),
        "headline_count_min": int(
            news["article_count"].min()
        ),
        "headline_count_median": float(
            news["article_count"].median()
        ),
        "headline_count_mean": float(
            news["article_count"].mean()
        ),
        "headline_count_max": int(
            news["article_count"].max()
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def consolidate_sentiment(
    news_path: Path,
    finbert_dir: Path,
    output_path: Path,
    summary_path: Path,
) -> dict:
    news = pd.read_csv(news_path)
    news["date"] = news["date"].astype(str)

    sentiment_files = sorted(
        (finbert_dir / "daily").glob(
            "sentiment_*.csv"
        )
    )
    if not sentiment_files:
        raise FileNotFoundError(
            "No FinBERT daily sentiment files exist."
        )

    sentiment = pd.concat(
        [
            pd.read_csv(path)
            for path in sentiment_files
        ],
        ignore_index=True,
    )
    sentiment["date"] = sentiment["date"].astype(str)
    sentiment = (
        sentiment[
            sentiment["date"].isin(
                set(news["date"])
            )
        ]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    merged = news.merge(
        sentiment,
        on="date",
        how="left",
        suffixes=("_news", "_sentiment"),
    )

    missing_sentiment = merged[
        merged["sentiment_mean"].isna()
    ]
    if len(missing_sentiment):
        raise ValueError(
            "Missing FinBERT output for dates: "
            + ", ".join(
                missing_sentiment["date"].tolist()
            )
        )

    count_mismatch = merged[
        merged["article_count_news"]
        != merged["article_count_sentiment"]
    ]
    hash_mismatch = merged[
        merged["headline_hash_news"]
        != merged["headline_hash_sentiment"]
    ]
    if len(count_mismatch):
        raise ValueError(
            "News/sentiment count mismatch."
        )
    if len(hash_mismatch):
        raise ValueError(
            "News/sentiment headline-hash mismatch."
        )

    output = sentiment[
        [
            "date",
            "sentiment_mean",
            "sentiment_std",
            "article_count",
            "positive_share",
            "negative_share",
            "neutral_share",
            "headline_hash",
        ]
    ].copy()

    share_error = float(
        (
            output[
                [
                    "positive_share",
                    "negative_share",
                    "neutral_share",
                ]
            ].sum(axis=1)
            - 1.0
        ).abs().max()
    )
    if share_error > 1e-9:
        raise ValueError(
            f"Class-share sum error={share_error}"
        )

    summary = {
        "sentiment_days": int(len(output)),
        "news_days_available": int(len(news)),
        "overlapping_validated_days": int(len(merged)),
        "total_headlines_scored": int(
            output["article_count"].sum()
        ),
        "article_count_mismatches": int(
            len(count_mismatch)
        ),
        "headline_hash_mismatches": int(
            len(hash_mismatch)
        ),
        "max_class_share_sum_error": share_error,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def load_finbert(device: torch.device):
    print(
        f"Loading {MODEL_NAME} ONCE on {device}",
        flush=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

    model.to(device)
    model.eval()
    return tokenizer, model


def checkpoint_path(
    directory: Path,
    week_number: int,
    week_start: dt.date,
    week_end: dt.date,
) -> Path:
    return directory / (
        f"week_{week_number:02d}_"
        f"{week_start:%Y%m%d}_{week_end:%Y%m%d}.json"
    )


def is_complete_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return False
    return data.get("status") == "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=parse_date,
        default=PERIOD_START,
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=PERIOD_END,
    )
    parser.add_argument(
        "--project",
        default=PROJECT_ID,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    if (
        args.start < PERIOD_START
        or args.end > PERIOD_END
        or args.end < args.start
    ):
        raise ValueError(
            "Fast runner is intentionally restricted to "
            "2024-01-01 through 2025-12-31."
        )

    root = Path.cwd()
    data_dir = root / "data"
    bigquery_dir = data_dir / "bigquery_backfill_2023_2025"
    finbert_dir = data_dir / "finbert_backfill_2023_2025"
    checkpoint_dir = data_dir / "two_year_2024_2025_checkpoints"
    progress_path = data_dir / "two_year_2024_2025_progress.jsonl"
    news_path = data_dir / "news_daily_bigquery_2023_2025.csv"
    news_summary_path = (
        data_dir / "bigquery_backfill_2023_2025_summary.json"
    )
    sentiment_path = (
        data_dir / "sentiment_features_bigquery_2023_2025.csv"
    )
    sentiment_summary_path = (
        data_dir / "finbert_backfill_2023_2025_summary.json"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=args.project)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if args.batch_size is None:
        batch_size = 32 if device.type == "cuda" else 16
    else:
        batch_size = args.batch_size

    print(
        f"FAST 2024-2025 RUNNER | device={device} | "
        f"batch={batch_size}",
        flush=True,
    )

    tokenizer, model = load_finbert(device)

    weeks = list(
        week_ranges(args.start, args.end)
    )
    run_started = time.time()

    for week_number, week_start, week_end in weeks:
        cp_path = checkpoint_path(
            checkpoint_dir,
            week_number,
            week_start,
            week_end,
        )

        if is_complete_checkpoint(cp_path):
            print(
                f"[{week_number}/{len(weeks)}] "
                f"SKIP completed week "
                f"{week_start} -> {week_end}",
                flush=True,
            )
            continue

        started = time.time()
        print(
            "\n"
            + "#" * 68
            + f"\nFAST WEEK {week_number}/{len(weeks)}: "
            f"{week_start} -> {week_end}"
            + "\n"
            + "#" * 68,
            flush=True,
        )

        try:
            print("STAGE 1/4: BigQuery", flush=True)
            collect_week(
                client,
                week_start,
                week_end,
                bigquery_dir,
            )

            print("STAGE 2/4: Consolidate news", flush=True)
            news_summary = consolidate_news(
                bigquery_dir,
                BASE_START,
                week_end,
                news_path,
                news_summary_path,
            )
            print(
                f"  cumulative headlines="
                f"{news_summary['unique_retained_headlines']}",
                flush=True,
            )

            print("STAGE 3/4: FinBERT", flush=True)
            news = pd.read_csv(news_path)
            news["date"] = news["date"].astype(str)
            week_news = news[
                (news["date"] >= week_start.isoformat())
                & (news["date"] <= week_end.isoformat())
            ].sort_values("date")

            if week_news.empty:
                print(
                    "  No empirical news days in this week; "
                    "nothing to score.",
                    flush=True,
                )

            for _, row in week_news.iterrows():
                result = score_day(
                    row=row,
                    output_dir=finbert_dir,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    force=False,
                )
                print(
                    f"  {row['date']}: "
                    f"{result['status']} "
                    f"n={result.get('headlines_scored', '-')}"
                    f" sec={result.get('elapsed_seconds', '-')}",
                    flush=True,
                )

            print(
                "STAGE 4/4: Validate sentiment",
                flush=True,
            )
            sentiment_summary = consolidate_sentiment(
                news_path,
                finbert_dir,
                sentiment_path,
                sentiment_summary_path,
            )

            if (
                sentiment_summary[
                    "overlapping_validated_days"
                ]
                != news_summary[
                    "calendar_days_with_news"
                ]
            ):
                raise ValueError(
                    "Cumulative sentiment/news day mismatch."
                )

            checkpoint = {
                "status": "complete",
                "week_number": week_number,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "completed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
                "week_elapsed_seconds": round(
                    time.time() - started,
                    3,
                ),
                "cumulative_calendar_days_requested": int(
                    news_summary[
                        "calendar_days_requested"
                    ]
                ),
                "cumulative_news_days": int(
                    news_summary[
                        "calendar_days_with_news"
                    ]
                ),
                "cumulative_unique_headlines": int(
                    news_summary[
                        "unique_retained_headlines"
                    ]
                ),
                "cumulative_sentiment_days": int(
                    sentiment_summary[
                        "overlapping_validated_days"
                    ]
                ),
                "cumulative_headlines_scored": int(
                    sentiment_summary[
                        "total_headlines_scored"
                    ]
                ),
                "bigquery_bytes_processed": int(
                    news_summary[
                        "bigquery_bytes_processed"
                    ]
                    or 0
                ),
                "device": str(device),
                "batch_size": int(batch_size),
            }
            cp_path.write_text(
                json.dumps(checkpoint, indent=2),
                encoding="utf-8",
            )
            with progress_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(checkpoint) + "\n"
                )

            print(
                f"WEEK {week_number} COMPLETE in "
                f"{checkpoint['week_elapsed_seconds']} sec | "
                f"days={checkpoint['cumulative_news_days']} | "
                f"headlines="
                f"{checkpoint['cumulative_headlines_scored']}",
                flush=True,
            )

        except Exception as exc:
            failure = {
                "status": "failed",
                "week_number": week_number,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "error": str(exc),
                "failed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            }
            with progress_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(failure) + "\n"
                )
            raise

    final_status = (
        "two_year_complete"
        if args.start == PERIOD_START and args.end == PERIOD_END
        else "period_complete"
    )
    final = {
        "status": final_status,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "completed_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "elapsed_seconds_this_run": round(
            time.time() - run_started,
            3,
        ),
        "device": str(device),
        "batch_size": int(batch_size),
    }
    with progress_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(json.dumps(final) + "\n")

    print(
        f"\nFAST REBUILD PERIOD COMPLETE. Stopped at {args.end}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
