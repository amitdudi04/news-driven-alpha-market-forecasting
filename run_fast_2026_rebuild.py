"""Fast incremental 2026 extension with weekly checkpoints.

The validated 2023-2025 historical store is preserved. For 2026 each week:
1. queries GDELT GAL once for the missing weekly range;
2. builds lossless JSON headline arrays from daily checkpoints;
3. scores only that week's unscored headlines with one persistent FinBERT
   model on CUDA;
4. validates daily article counts and headline hashes;
5. writes a durable weekly checkpoint.

A single full 2023-2026 consolidation and validation is performed only after
the requested 2026 period is complete. This avoids repeatedly re-reading the
entire multi-year history after every week.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import torch
from google.cloud import bigquery

from finbert_backfill import score_day
from run_fast_two_year_rebuild import (
    PROJECT_ID,
    collect_week,
    consolidate_news,
    consolidate_sentiment,
    headline_hash,
    load_finbert,
)

BASE_START = dt.date(2023, 1, 1)
BASE_END = dt.date(2025, 12, 31)
PERIOD_START = dt.date(2026, 1, 1)
PERIOD_END = dt.date(2026, 10, 3)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def week_ranges(start: dt.date, end: dt.date):
    number = 1
    cursor = start
    while cursor <= end:
        week_end = min(cursor + dt.timedelta(days=6), end)
        yield number, cursor, week_end
        cursor = week_end + dt.timedelta(days=1)
        number += 1


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


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("status") != "complete":
        return None
    return payload


def base_state(data_dir: Path) -> dict:
    """Load the already validated 2023-2025 baseline from small summaries."""
    news_summary_path = (
        data_dir / "bigquery_backfill_2023_2025_summary.json"
    )
    sentiment_summary_path = (
        data_dir / "finbert_backfill_2023_2025_summary.json"
    )
    if (
        not news_summary_path.exists()
        or not sentiment_summary_path.exists()
    ):
        raise FileNotFoundError(
            "Validated 2023-2025 summary files are required."
        )

    news = json.loads(
        news_summary_path.read_text(encoding="utf-8")
    )
    sentiment = json.loads(
        sentiment_summary_path.read_text(encoding="utf-8")
    )

    if news.get("requested_start") != BASE_START.isoformat():
        raise ValueError("Unexpected 2023-2025 baseline start date.")
    if news.get("requested_end") != BASE_END.isoformat():
        raise ValueError("Unexpected 2023-2025 baseline end date.")
    if int(news.get("duplicate_title_date_pairs_remaining", -1)) != 0:
        raise ValueError("Baseline still contains duplicate title/day pairs.")
    if int(sentiment.get("article_count_mismatches", -1)) != 0:
        raise ValueError("Baseline article-count mismatch remains.")
    if int(sentiment.get("headline_hash_mismatches", -1)) != 0:
        raise ValueError("Baseline headline-hash mismatch remains.")

    news_days = int(news["calendar_days_with_news"])
    sentiment_days = int(
        sentiment["overlapping_validated_days"]
    )
    headlines = int(news["unique_retained_headlines"])
    scored = int(sentiment["total_headlines_scored"])

    if news_days != sentiment_days:
        raise ValueError("Baseline news/sentiment day counts differ.")
    if headlines != scored:
        raise ValueError("Baseline headline/scored counts differ.")

    return {
        "calendar_days": int(news["calendar_days_requested"]),
        "news_days": news_days,
        "headlines": headlines,
    }


def candidate_row(
    daily_file: Path,
    day: dt.date,
) -> pd.Series | None:
    frame = pd.read_csv(daily_file)
    if frame.empty:
        return None

    if "date" not in frame.columns or "title" not in frame.columns:
        raise ValueError(
            f"{daily_file} missing required date/title columns."
        )
    frame["date"] = frame["date"].astype(str)
    expected_date = day.isoformat()
    if not (frame["date"] == expected_date).all():
        raise ValueError(
            f"{daily_file} contains rows outside {expected_date}."
        )

    titles = frame["title"].fillna("").astype(str).str.strip().tolist()
    titles = [value for value in titles if len(value) > 15]
    if len(titles) != len(frame):
        raise ValueError(
            f"{expected_date}: blank/short title appeared after collection."
        )

    return pd.Series(
        {
            "date": expected_date,
            "article_count": int(len(titles)),
            "headline_hash": headline_hash(titles),
            "headlines_json": json.dumps(
                titles,
                ensure_ascii=False,
            ),
            "raw_text": " || ".join(titles),
        }
    )


def validate_scored_day(
    row: pd.Series,
    finbert_dir: Path,
) -> None:
    label = str(row["date"]).replace("-", "")
    daily_path = (
        finbert_dir / "daily" / f"sentiment_{label}.csv"
    )
    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing FinBERT daily output: {daily_path}"
        )

    scored = pd.read_csv(daily_path)
    if len(scored) != 1:
        raise ValueError(
            f"{row['date']}: expected one daily sentiment row."
        )
    actual = scored.iloc[0]
    if int(actual["article_count"]) != int(row["article_count"]):
        raise ValueError(
            f"{row['date']}: article-count mismatch after scoring."
        )
    if str(actual["headline_hash"]) != str(row["headline_hash"]):
        raise ValueError(
            f"{row['date']}: headline-hash mismatch after scoring."
        )


def manifest_bytes(input_dir: Path) -> int:
    path = input_dir / "query_manifest.csv"
    if not path.exists():
        return 0
    manifest = pd.read_csv(path)
    if "total_bytes_processed" not in manifest.columns:
        return 0
    return int(
        pd.to_numeric(
            manifest["total_bytes_processed"],
            errors="coerce",
        ).fillna(0).sum()
    )


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
            "2026 fast runner is restricted to "
            "2026-01-01 through 2026-10-03."
        )

    root = Path.cwd()
    data_dir = root / "data"
    bigquery_dir = data_dir / "bigquery_backfill_2023_2025"
    finbert_dir = data_dir / "finbert_backfill_2023_2025"
    checkpoint_dir = data_dir / "year_2026_checkpoints"
    progress_path = data_dir / "year_2026_progress.jsonl"

    final_news = data_dir / "news_daily_bigquery_2023_2026.csv"
    final_news_summary = (
        data_dir / "bigquery_backfill_2023_2026_summary.json"
    )
    final_sentiment = (
        data_dir / "sentiment_features_bigquery_2023_2026.csv"
    )
    final_sentiment_summary = (
        data_dir / "finbert_backfill_2023_2026_summary.json"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline = base_state(data_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (32 if device.type == "cuda" else 16)
    )

    print(
        f"FAST 2026 RUNNER | device={device} | "
        f"batch={batch_size} | "
        f"period={args.start}->{args.end}",
        flush=True,
    )
    print(
        f"BASELINE 2023-2025 | news_days={baseline['news_days']} | "
        f"headlines={baseline['headlines']}",
        flush=True,
    )

    client = bigquery.Client(project=args.project)
    tokenizer, model = load_finbert(device)

    cumulative_news_days = baseline["news_days"]
    cumulative_headlines = baseline["headlines"]
    run_started = time.time()
    weeks = list(week_ranges(args.start, args.end))

    for week_number, week_start, week_end in weeks:
        cp_path = checkpoint_path(
            checkpoint_dir,
            week_number,
            week_start,
            week_end,
        )
        existing = load_checkpoint(cp_path)
        if existing is not None:
            cumulative_news_days = int(
                existing["cumulative_news_days"]
            )
            cumulative_headlines = int(
                existing["cumulative_headlines_scored"]
            )
            print(
                f"[{week_number}/{len(weeks)}] SKIP completed "
                f"{week_start}->{week_end}",
                flush=True,
            )
            continue

        started = time.time()
        print(
            "\n"
            + "#" * 68
            + f"\nFAST 2026 WEEK {week_number}/{len(weeks)}: "
            f"{week_start} -> {week_end}"
            + "\n"
            + "#" * 68,
            flush=True,
        )

        try:
            print("STAGE 1/3: BigQuery", flush=True)
            collect_week(
                client,
                week_start,
                week_end,
                bigquery_dir,
            )

            print(
                "STAGE 2/3: FinBERT + daily validation",
                flush=True,
            )
            week_news_days = 0
            week_headlines = 0

            for value in pd.date_range(
                week_start,
                week_end,
                freq="D",
            ):
                day = value.date()
                daily_file = (
                    bigquery_dir
                    / "daily"
                    / f"articles_{day:%Y%m%d}.csv.gz"
                )
                if not daily_file.exists():
                    raise FileNotFoundError(
                        f"Missing BigQuery checkpoint: {daily_file}"
                    )

                row = candidate_row(daily_file, day)
                if row is None:
                    print(
                        f"  {day}: zero matching headlines",
                        flush=True,
                    )
                    continue

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
                validate_scored_day(row, finbert_dir)
                week_news_days += 1
                week_headlines += int(row["article_count"])
                print(
                    f"  {day}: {result['status']} "
                    f"n={row['article_count']} "
                    f"sec={result.get('elapsed_seconds', '-')}",
                    flush=True,
                )

            print("STAGE 3/3: Weekly checkpoint", flush=True)
            cumulative_news_days += week_news_days
            cumulative_headlines += week_headlines

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
                "week_news_days": int(week_news_days),
                "week_headlines_scored": int(week_headlines),
                "cumulative_calendar_days_requested": int(
                    baseline["calendar_days"]
                    + (week_end - PERIOD_START).days
                    + 1
                ),
                "cumulative_news_days": int(
                    cumulative_news_days
                ),
                "cumulative_unique_headlines": int(
                    cumulative_headlines
                ),
                "cumulative_sentiment_days": int(
                    cumulative_news_days
                ),
                "cumulative_headlines_scored": int(
                    cumulative_headlines
                ),
                "bigquery_bytes_processed": manifest_bytes(
                    bigquery_dir
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
                handle.write(json.dumps(checkpoint) + "\n")

            print(
                f"WEEK {week_number} COMPLETE in "
                f"{checkpoint['week_elapsed_seconds']} sec | "
                f"week_headlines={week_headlines} | "
                f"cumulative={cumulative_headlines}",
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
                handle.write(json.dumps(failure) + "\n")
            raise

    print(
        "\nFINAL STAGE: one full 2023-2026 consolidation",
        flush=True,
    )
    news_summary = consolidate_news(
        bigquery_dir,
        BASE_START,
        args.end,
        final_news,
        final_news_summary,
    )
    sentiment_summary = consolidate_sentiment(
        final_news,
        finbert_dir,
        final_sentiment,
        final_sentiment_summary,
    )

    if (
        int(news_summary["calendar_days_with_news"])
        != int(
            sentiment_summary["overlapping_validated_days"]
        )
    ):
        raise ValueError(
            "Final 2023-2026 news/sentiment day mismatch."
        )
    if (
        int(news_summary["unique_retained_headlines"])
        != int(sentiment_summary["total_headlines_scored"])
    ):
        raise ValueError(
            "Final 2023-2026 headline-count mismatch."
        )
    if int(
        sentiment_summary["article_count_mismatches"]
    ) != 0:
        raise ValueError(
            "Final article-count mismatches remain."
        )
    if int(
        sentiment_summary["headline_hash_mismatches"]
    ) != 0:
        raise ValueError(
            "Final headline-hash mismatches remain."
        )

    final = {
        "status": "period_complete",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "completed_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "elapsed_seconds_this_run": round(
            time.time() - run_started,
            3,
        ),
        "final_news_days_2023_2026": int(
            news_summary["calendar_days_with_news"]
        ),
        "final_headlines_2023_2026": int(
            news_summary["unique_retained_headlines"]
        ),
        "final_sentiment_days_2023_2026": int(
            sentiment_summary["overlapping_validated_days"]
        ),
        "final_headlines_scored_2023_2026": int(
            sentiment_summary["total_headlines_scored"]
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
        f"\nFAST 2026 EXTENSION COMPLETE through {args.end}.",
        flush=True,
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
